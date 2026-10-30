"""Local STL airframe on the official Iris force/torque simulation.

Geometry is user-supplied; mass, inertia, motor reaction torque and rotor mounting
remain profile assumptions. No set-pose actuator or fixed-wing aerodynamics.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from runpy import run_path
import struct
import xml.etree.ElementTree as ET

import numpy as np

ASSET_NAMES = (
    'A_lower_body_x1', 'B_upper_tube_x1', 'C_nose_cone_x1',
    'D_electronics_module_x1', 'G_camera_bracket_x1',
    'H_pod_cap_1', 'H_pod_cap_2', 'H_pod_cap_3', 'H_pod_cap_4',
    'I_battery_hatch_x1',
)


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('finite_shape_required: ' + name)
    return array


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('positive_finite_required: ' + name)
    return float(value)


def load_profile(path):
    """Read a complete local profile; never fill missing physical inputs with Iris values."""
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('physics_JSON_object_required')
    if type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('unsupported_stl_airframe_schema')
    if config.get('aircraft') != 'stl_pacer_quad':
        raise ValueError('stl_pacer_quad_profile_required')
    for key in ('mass_kg', 'rotor_mass_kg'):
        _positive(config[key], key)
    for key in ('center_of_mass_body_m', 'home_world'):
        _array(config[key], (3,), key)
    for key in ('inertia_body_kg_m2', 'rotor_inertia_kg_m2'):
        a = _array(config[key], (3, 3), key)
        if (not np.allclose(a, a.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(a).min() <= 0
                or np.linalg.eigvalsh(np.trace(a)*np.eye(3)/2-a).min() < -1e-12):
            raise ValueError('physical_positive_inertia_required: ' + key)
    _array(config['rotor_positions_body_m'], (4, 3), 'rotor_positions_body_m')
    axes = _array(config['rotor_axes_body'], (4, 3), 'rotor_axes_body')
    # This profile is a conventional vertical-thrust MC, not a tiltrotor or VTOL.
    if not np.allclose(axes, np.tile([0., 0., 1.], (4, 1)), atol=1e-9, rtol=0):
        raise ValueError('multicopter_positive_body_z_rotor_axes_required')
    spins = config['rotor_spin_directions']
    if len(spins) != 4 or any(s not in ('cw', 'ccw') for s in spins):
        raise ValueError('four_cw_ccw_rotor_directions_required')
    if 4*config['rotor_mass_kg'] >= config['mass_kg']:
        raise ValueError('component_mass_exceeds_total_mass')
    motor = config['motor']
    for key in ('max_rpm', 'time_constant_up_s', 'time_constant_down_s'):
        _positive(motor[key], 'motor.'+key)
    if not math.isfinite(float(config.get('home_yaw_world', 0.))):
        raise ValueError('finite_home_yaw_required')
    geometry = config['geometry']
    _positive(geometry['stl_scale_m'], 'stl_scale_m')
    _positive(geometry['propeller_diameter_m'], 'propeller_diameter_m')
    _array(geometry['stl_origin_source'], (3,), 'stl_origin_source')
    flight = config.get('flight_parameters', {})
    for key, maximum in (('max_speed', 1.8), ('max_acceleration', .7), ('max_jerk', 1.4)):
        if key in flight and _positive(flight[key], key) > maximum:
            raise ValueError('unvalidated_multicopter_envelope: '+key)
    for key in ('body_radius', 'takeoff_height', 'takeoff_min_duration', 'takeoff_join_speed'):
        if key in flight:
            _positive(flight[key], key)
    for key in ('continuous_takeoff', 'target_is_virtual_waypoint', 'target_is_pass_through_gate'):
        if key in flight and type(flight[key]) is not bool:
            raise ValueError('boolean_flight_option_required: '+key)
    camera = config.get('onboard_camera')
    if camera is not None:
        for key in ('pcb_center_source_mm', 'pcb_size_mm', 'optical_center_source_mm'):
            _array(camera[key], (3,), 'onboard_camera.'+key)
        if min(camera['pcb_size_mm']) <= 0:
            raise ValueError('positive_camera_PCB_dimensions_required')
        for key in ('horizontal_fov_deg', 'update_rate_hz', 'near_clip_m', 'far_clip_m'):
            _positive(camera[key], 'onboard_camera.'+key)
        if camera['horizontal_fov_deg'] >= 180 or camera['near_clip_m'] >= camera['far_clip_m']:
            raise ValueError('valid_camera_FOV_and_clipping_required')
        for key in ('width', 'height'):
            if type(camera[key]) is not int or not 1 <= camera[key] <= 4096:
                raise ValueError('valid_camera_image_dimensions_required')
    for component in config.get('internal_visuals', {}).get('boxes_source_mm', []):
        _array(component['center'], (3,), 'internal_visual.center')
        if (_array(component['size'], (3,), 'internal_visual.size') <= 0).any():
            raise ValueError('positive_internal_visual_dimensions_required')
    return config


def _mesh_bounds(path, origin, scale):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError('invalid_binary_STL: '+str(path))
    count = struct.unpack('<I', data[80:84])[0]
    if count == 0 or len(data) != 84+50*count:
        raise ValueError('invalid_binary_STL: '+str(path))
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attribute', '<u2')])
    vertices = np.frombuffer(data[84:], dtype=dtype)['v'].reshape(-1, 3).astype(float)
    if not np.isfinite(vertices).all():
        raise ValueError('nonfinite_STL_vertices: '+str(path))
    vertices = (vertices-origin)*scale
    vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    return vertices.min(axis=0), vertices.max(axis=0), float(np.linalg.norm(vertices[:, :2], axis=1).max()), count


def _shape(parent, tag, name, shape, values, pose=(0, 0, 0, 0, 0, 0)):
    element = ET.SubElement(parent, tag, name=name)
    ET.SubElement(element, 'pose').text = ' '.join(map(str, pose))
    geometry = ET.SubElement(element, 'geometry')
    primitive = ET.SubElement(geometry, shape)
    for key, value in values.items():
        ET.SubElement(primitive, key).text = str(value)
    return element


def _material(visual, color):
    material = ET.SubElement(visual, 'material')
    for key in ('ambient', 'diffuse'):
        ET.SubElement(material, key).text = color


def _onboard_visuals(body, config):
    """Diagram-derived visual details; all masses remain in the existing 2 kg model."""
    scale = config['geometry']['stl_scale_m']
    origin = np.asarray(config['geometry']['stl_origin_source'])

    def position(raw):
        delta = (np.asarray(raw)-origin)*scale
        return delta[[0, 2, 1]]*[1, -1, 1]

    def box(name, center, size, color):
        # Source +Y is the nose direction; body +Z points toward the nose.
        dimensions = np.asarray(size)[[0, 2, 1]]*scale
        visual = _shape(body, 'visual', name, 'box',
                        dict(size=' '.join(map(str, dimensions))), (*position(center), 0, 0, 0))
        _material(visual, color)

    def cylinder(name, center, radius, length, color):
        visual = _shape(body, 'visual', name, 'cylinder',
                        dict(radius=radius*scale, length=length*scale), (*position(center), 0, 0, 0))
        _material(visual, color)

    for component in config.get('internal_visuals', {}).get('boxes_source_mm', []):
        box(component['name'], component['center'], component['size'], component['color'])
    camera = config.get('onboard_camera')
    if camera is None:
        return None
    box('onboard_camera_pcb', camera['pcb_center_source_mm'], camera['pcb_size_mm'], '.10 .40 .24 1')
    # Mechanical layout follows the official standard module drawing. The two
    # rails adapt the old bracket hole pattern, avoiding its rear connector.
    box('onboard_camera_lens_housing', [0, 1208.0575, 0], [10.8, 3.875, 10.8], '.08 .09 .10 1')
    cylinder('onboard_camera_lens_barrel', [0, 1211.5075, 0], 2.875, 3.025, '.055 .065 .075 1')
    cylinder('onboard_camera_lens_glass', [0, 1213.06, 0], 2.375, .08, '.05 .19 .28 1')
    box('onboard_camera_rear_connector', [0, 1203.625, 6.607], [19.61, 2.75, 5.71], '.76 .73 .63 1')
    # Short cable tail only; the complete cable route through the old bracket
    # and avionics needs a mechanical redesign, not a guessed solid intersection.
    box('onboard_camera_ribbon_tail', [0, 1203.5, 10.5], [12, .2, 2.1], '.72 .43 .16 1')
    for side, x in enumerate((-10.5, 10.5)):
        box(f'onboard_camera_adapter_rail_{side}', [x, 1202.5, -6.15], [3, 1, 15.5], '.58 .61 .64 1')
        cylinder(f'onboard_camera_adapter_anchor_{side}', [x, 1202.5, -5], .8, 1, '.24 .26 .28 1')
        for index, z in enumerate((-12.4, .1)):
            cylinder(f'onboard_camera_standoff_{side}_{index}', [x, 1204, z], 1.5, 2, '.64 .55 .32 1')
            cylinder(f'onboard_camera_screw_{side}_{index}', [x, 1206.45, z], 1.5, .66, '.28 .30 .32 1')
    optical_position = position(camera['optical_center_source_mm']).tolist()
    sensor = ET.SubElement(body, 'sensor', name='onboard_camera_rgb', type='camera')
    # Gazebo +X looks out of the nose; image right/down follow body +X/+Y,
    # matching the physical PCB's 25 mm horizontal edge and the optical TF.
    sensor_rpy = [math.pi/2, -math.pi/2, 0]
    ET.SubElement(sensor, 'pose').text = ' '.join(map(str, [*optical_position, *sensor_rpy]))
    for key, value in (('always_on', 'true'), ('update_rate', camera['update_rate_hz']), ('visualize', 'false')):
        ET.SubElement(sensor, key).text = str(value)
    optics = ET.SubElement(sensor, 'camera')
    ET.SubElement(optics, 'horizontal_fov').text = str(math.radians(camera['horizontal_fov_deg']))
    image = ET.SubElement(optics, 'image')
    for key, value in (('width', camera['width']), ('height', camera['height']), ('format', 'R8G8B8')):
        ET.SubElement(image, key).text = str(value)
    clip = ET.SubElement(optics, 'clip')
    for key in ('near', 'far'):
        ET.SubElement(clip, key).text = str(camera[key+'_clip_m'])
    plugin = ET.SubElement(sensor, 'plugin', name='jol_onboard_camera', filename='libgazebo_ros_camera.so')
    ros = ET.SubElement(plugin, 'ros')
    ET.SubElement(ros, 'namespace').text = camera['namespace']
    topic_root = camera['namespace'].rstrip('/')+'/'+camera['camera_name']
    for topic in ('image_raw', 'camera_info'):
        ET.SubElement(ros, 'remapping').text = f'{topic}:={topic_root}/{topic}'
    for key in ('camera_name', 'frame_name'):
        ET.SubElement(plugin, key).text = camera[key]
    for key, dimension in (('cx', 'width'), ('cy', 'height')):
        ET.SubElement(plugin, key).text = str((camera[dimension]-1)/2)
    return dict(optical_position_body_m=optical_position, sensor_rpy_rad=sensor_rpy,
                frame_name=camera['frame_name'], image_topic=topic_root+'/image_raw',
                camera_info_topic=topic_root+'/camera_info', data_status='ideal_pinhole_concept_mount',
                added_mass_kg=0., total_mass_policy='Included in existing assumed 2kg, not a measured component partition')


def build_model(sim_share, px4_root, plugins, aero_plugins, profile_path, assets_dir, *, shell_transparency=0.0):
    """Return an SDF model and explicit provenance report; build does not prove flight."""
    config = load_profile(profile_path)
    if type(shell_transparency) not in (int, float) or not math.isfinite(shell_transparency) or not 0 <= shell_transparency <= 1:
        raise ValueError('shell_transparency_between_zero_and_one_required')
    helpers = run_path(str(Path(sim_share)/'launch/tailsitter_model.py'))
    physics = run_path(str(Path(sim_share)/'launch/tailsitter_physics.py'))
    put = helpers['set_value']
    motor_config = config['motor']
    fit = physics['motor_fit'](motor_config['rpm_thrust_torque_samples'], motor_config['max_rpm'])
    mass = config['mass_kg']
    center = np.array(config['center_of_mass_body_m'])
    inertia = np.array(config['inertia_body_kg_m2'])
    positions = np.array(config['rotor_positions_body_m'])
    axes = np.array(config['rotor_axes_body'])
    spins = config['rotor_spin_directions']
    signs = np.array([-1 if s == 'ccw' else 1 for s in spins])
    torques = np.cross(positions-center, axes)+fit['moment_constant_m']*signs[:, None]*axes
    allocation = np.vstack((axes[:, 2], torques.T))
    if np.linalg.matrix_rank(allocation) != 4:
        raise ValueError('independent_hover_and_three_torques_required')
    weight = mass*9.80665
    maximum_thrust = 4*fit['thrust_coefficient']*fit['max_omega_rad_s']**2
    if maximum_thrust <= weight:
        raise ValueError('insufficient_static_hover_thrust')
    model, source = helpers['template_model'](px4_root, 'iris')
    model.set('name', 'stl_pacer_quad')
    home = config['home_world']
    yaw = float(config.get('home_yaw_world', 0.))
    put(model, 'pose', ' '.join(map(str, [*home, 0, 0, yaw])))
    # Remove Iris geometry and numerical IMU mass; reuse actual PX4 sensor plugins.
    for item in list(model):
        if (item.tag == 'include'
                or item.tag == 'link' and item.get('name') not in ('base_link', *(f'rotor_{i}' for i in range(4)))
                or item.tag == 'joint' and not item.get('name', '').startswith('rotor_')):
            model.remove(item)
    body = model.find("link[@name='base_link']")
    for link in model.findall('link'):
        for item in list(link):
            if item.tag in ('visual', 'collision'):
                link.remove(item)
    geometry_config = config['geometry']
    scale = float(geometry_config['stl_scale_m'])
    origin = np.array(geometry_config['stl_origin_source'])
    translation = np.array([-origin[0], origin[2], -origin[1]])*scale
    bounds = []
    faces = 0
    for name in ASSET_NAMES:
        path = Path(assets_dir).resolve()/(name+'.stl')
        low, high, radius, count = _mesh_bounds(path, origin, scale)
        bounds.append((low, high, radius)); faces += count
        visual = _shape(body, 'visual', name, 'mesh',
                        dict(uri=path.as_uri(), scale=f'{scale} {scale} {scale}'),
                        (*translation, math.pi/2, 0, 0))
        _material(visual, '.23 .28 .34 1' if name.startswith(('A_', 'B_', 'H_')) else '.75 .78 .80 1')
        if name.startswith(('A_', 'B_', 'C_', 'H_', 'I_')):
            ET.SubElement(visual, 'transparency').text = str(shell_transparency)
    onboard_camera = _onboard_visuals(body, config)
    low = np.min([b[0] for b in bounds], axis=0)
    high = np.max([b[1] for b in bounds], axis=0)
    # Conservative cylinder contains every supplied shell, including tail fins.
    # Report this envelope instead of pretending these are exact contact surfaces.
    collision_radius_xy = max(b[2] for b in bounds)
    collision = _shape(body, 'collision', 'stl_conservative_envelope', 'cylinder',
                       dict(radius=collision_radius_xy, length=float(high[2]-low[2])),
                       (0, 0, float((low[2]+high[2])/2), 0, 0, 0))
    ode = ET.SubElement(ET.SubElement(ET.SubElement(collision, 'surface'), 'contact'), 'ode')
    stiffness = float(config.get('physics_numerics', {}).get('contact_stiffness_N_m', 1e4))
    damping = float(config.get('physics_numerics', {}).get('contact_damping_N_s_m', 4*math.sqrt(mass*stiffness)))
    _positive(stiffness, 'contact_stiffness_N_m'); _positive(damping, 'contact_damping_N_s_m')
    for key, value in dict(kp=stiffness, kd=damping, max_vel=.05, min_depth=0).items():
        put(ode, key, value)
    rotor_mass = config['rotor_mass_kg']
    rotor_inertia = np.array(config['rotor_inertia_kg_m2'])
    prop_radius = float(geometry_config['propeller_diameter_m'])/2
    parts = []
    for index, position in enumerate(positions):
        rotor = model.find(f"link[@name='rotor_{index}']")
        put(rotor, 'pose', ' '.join(map(str, [*position, 0, 0, 0])))
        helpers['set_inertial'](rotor, rotor_mass, [0, 0, 0], rotor_inertia)
        joint = model.find(f"joint[@name='rotor_{index}_joint']")
        put(joint, 'pose', '0 0 0 0 0 0')
        put(joint.find('axis'), 'xyz', '0 0 1')
        put(joint.find('axis'), 'use_parent_model_frame', 0)
        for blade in range(3):
            angle = blade*2*math.pi/3
            visual = _shape(rotor, 'visual', f'blade_{blade}', 'box',
                            dict(size=f'{prop_radius} {prop_radius*.14} .0015'),
                            (prop_radius/2*math.cos(angle), prop_radius/2*math.sin(angle), 0, 0, 0, angle))
            _material(visual, '.1 .12 .14 1' if spins[index] == 'cw' else '.8 .32 .08 1')
        parts.append((rotor_mass, position, rotor_inertia))
    # Preserve the official noisy GPS sensor while accounting for every gram.
    partition = .001
    gps = deepcopy(ET.parse(source/'models/gps/gps.sdf').getroot().find('model/link'))
    gps.set('name', 'gps_link'); put(gps, 'pose', ' '.join(map(str, [*center, 0, 0, 0])))
    for visual in gps.findall('visual'):
        gps.remove(visual)
    put(gps.find('sensor/plugin'), 'topic', 'gps')
    helpers['set_inertial'](gps, mass*partition, [0, 0, 0], inertia*partition)
    model.append(gps)
    model.append(ET.fromstring('<joint name="gps_joint" type="fixed"><parent>base_link</parent><child>gps_link</child></joint>'))
    parts.append((mass*partition, center, inertia*partition))
    base_mass = mass-sum(p[0] for p in parts)
    if base_mass <= 0:
        raise ValueError('component_mass_exceeds_total_mass')
    base_center = (mass*center-sum(m*p for m, p, _ in parts))/base_mass
    parallel = lambda d: np.dot(d, d)*np.eye(3)-np.outer(d, d)
    base_inertia = inertia-sum(t+m*parallel(p-center) for m, p, t in parts)-base_mass*parallel(base_center-center)
    physics['inertia'](base_inertia.tolist(), 'body_inertia_after_component_subtraction')
    helpers['set_inertial'](body, base_mass, base_center, base_inertia)
    model.find("plugin[@filename='libgazebo_imu_plugin.so']/linkName").text = 'base_link'
    for motor in model.findall("plugin[@filename='libgazebo_motor_model.so']"):
        index = int(motor.findtext('motorNumber'))
        for key, value in dict(motorConstant=fit['thrust_coefficient'], momentConstant=fit['moment_constant_m'],
                               maxRotVelocity=fit['max_omega_rad_s'], turningDirection=spins[index],
                               timeConstantUp=motor_config['time_constant_up_s'], timeConstantDown=motor_config['time_constant_down_s'],
                               rotorDragCoefficient=0., rollingMomentCoefficient=0., maxRelativeAirspeed=1e30,
                               rotorVelocitySlowdownSim=50).items():
            put(motor, key, value)
    put(model.find("plugin[@name='rosbag']"), 'rotorVelocitySlowdownSim', 50)
    channels = model.find("plugin[@name='mavlink_interface']/control_channels")
    for channel in list(channels):
        if int(channel.findtext('input_index')) >= 4:
            channels.remove(channel)
        else:
            put(channel, 'input_scaling', fit['max_omega_rad_s'])
    helpers['add_rotor_joint_observer'](model)
    helpers['body_observer'](model, 'stl_quad_truth')
    helpers['plugin_paths'](model, plugins, aero_plugins)
    parameters = dict(SYS_AUTOSTART=10015, CA_AIRFRAME=0, CA_ROTOR_COUNT=4, CA_SV_CS_COUNT=0)
    flu_to_frd = np.array([1., -1., -1.])
    for index in range(4):
        for axis, value in zip('XYZ', (positions[index]-center)*flu_to_frd):
            parameters[f'CA_ROTOR{index}_P{axis}'] = float(value)
        for axis, value in zip('XYZ', axes[index]*flu_to_frd):
            parameters[f'CA_ROTOR{index}_A{axis}'] = float(value)
        parameters[f'CA_ROTOR{index}_KM'] = fit['moment_constant_m']*(1 if spins[index] == 'ccw' else -1)
    overrides = config.get('px4_parameters', {})
    for key, value in overrides.items():
        if key.startswith('CA_') or key == 'SYS_AUTOSTART':
            if key not in parameters or value != parameters[key]:
                raise ValueError('rotor_geometry_or_multicopter_configuration_override: '+key)
    parameters.update(overrides)
    if parameters['SYS_AUTOSTART'] != 10015 or parameters['CA_AIRFRAME'] != 0:
        raise ValueError('stl_airframe_requires_multicopter_autostart_and_allocation')
    # The planner envelope includes conservative contacts and rotating propellers.
    envelope = math.hypot(collision_radius_xy, max(abs(low[2]), abs(high[2])))
    envelope = max(envelope, max(math.hypot(np.linalg.norm(p[:2])+prop_radius, abs(p[2])) for p in positions))
    planning_radius = config.get('flight_parameters', {}).get('body_radius')
    if planning_radius is not None and planning_radius < envelope:
        raise ValueError('planning_body_radius_smaller_than_actual_collision_envelope')
    report = dict(model_name='stl_pacer_quad', vehicle_kind='multicopter', px4_sim_model='gazebo-classic_iris',
                  px4_parameters=parameters, home_world=home, home_yaw_world=yaw,
                  total_mass_kg=mass, body_mass_kg=base_mass, payload_mass_kg=config.get('payload_mass_kg'),
                  center_of_mass_body_m=center.tolist(), inertia_body_kg_m2=inertia.tolist(),
                  collision_bounding_sphere_radius_m=envelope, geometry_bounds_body_m=[low.tolist(), high.tolist()],
                  geometry_triangle_count=faces, motor_fit=fit, max_static_thrust_to_weight=maximum_thrust/weight,
                  hover_allocation_rank=4, fixed_sensor_numerical_mass_partition_fraction=partition,
                  bench_only=False, flight_verified=False, physics_calibrated=False, vehicle_pose_actuation=False,
                  data_status=config.get('data_status'), input_sources=config.get('sources', {}),
                  physics_profile_path=str(Path(profile_path).resolve()),
                  rotor_joint_state_topic='/jol/sim_observer/rotor_joints',
                  vehicle_rotation_actuation='PX4 differential rotor forces and reaction torques',
                  onboard_camera=onboard_camera, shell_transparency=shell_transparency,
                  limitations=[
                      'STL unit is assumed millimetres; all original source vertices are preserved.',
                      'Mass/CG/inertia and motor reaction torque are assumptions, not identified hardware.',
                      'P2207 V3 diameter 28.1 mm exceeds the approximately 27 mm pod bore; physical mounting is unresolved.',
                      'Conservative cylinder contact envelope; rotor collisions are not modeled.',
                      'Three primitive blades represent each propeller, not a manufacturer blade mesh.',
                      'Camera/avionics primitives are visual-only within the assumed total mass; mounting and cable route are unverified.',
                      'No calibrated body drag, wing lift, propwash or fixed-wing transition model.',
                      'Static thrust fit and official Iris GPS/IMU noise do not certify real flight.'])
    return model, report
