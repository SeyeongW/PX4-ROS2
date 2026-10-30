"""Offline SDF construction. Reference/measurement inputs never prove flight."""
import math
from copy import deepcopy
from pathlib import Path
from runpy import run_path
import xml.etree.ElementTree as ET

def observer_model(share, static_flight=False):
    """Reuse the existing 20cm aircraft and its local mesh copies in every mode."""
    model = ET.fromstring((share / 'worlds/observer.sdf').read_text().replace(
        'model://tailsitter_20cm/', 'file://'+str(share / 'meshes/tailsitter_20cm')+'/')).find('model')
    if static_flight:
        plugin = model.find("plugin[@name='jol_observer']")
        for key, value in dict(ned_mode='true', horizontal_limit='65', height_limit='20',
                               north_yaw_world=str(math.pi/2)).items():
            ET.SubElement(plugin, key).text = value
    return model

def mount_rotor_visuals(model, original, relocate_rotors=True):
    """Mount the 20cm visuals on motor axes; never alter the legacy Iris caller."""
    from copy import deepcopy
    import numpy as np
    from scipy.spatial.transform import Rotation

    def pose(element):
        values = np.fromstring(element.findtext('pose', '0 0 0 0 0 0'), sep=' ')
        if values.shape != (6,) or not np.isfinite(values).all():raise ValueError('invalid_rotor_visual_pose')
        return values[:3], Rotation.from_euler('xyz', values[3:])

    body = model.find("link[@name='base_link']")
    if body is None:raise ValueError('rotor_visual_base_link_required')
    body_position, body_rotation = pose(body)
    plans = []
    # Match the PX4 FLU rotor quadrants, not the display model's visual indices.
    for motor, prop in enumerate((1, 2, 3, 0)):
        rotor = model.find(f"link[@name='rotor_{motor}']")
        joint = model.find(f"joint[@name='rotor_{motor}_joint']")
        source = original.find(f"visual[@name='propeller_{prop}']")
        if rotor is None or joint is None or source is None:raise ValueError('four_motor_rotor_visuals_required')
        position, rotation = pose(source)
        position = body_position+body_rotation.apply(position)
        rotation = body_rotation*rotation
        rotor_position, rotor_rotation = (position, rotation) if relocate_rotors else pose(rotor)
        if (not np.allclose(rotor_position, position, atol=1e-10)
                or not np.allclose(rotor_rotation.apply([0, 0, 1]), rotation.apply([0, 0, 1]), atol=1e-10)):
            raise ValueError('rotor_visual_axis_alignment_required')
        visual = deepcopy(source)
        handedness = 'ccw' if motor in (0, 1) else 'cw'
        uri = visual.find('geometry/mesh/uri')
        uri.text = uri.text.rsplit('/', 1)[0]+f'/iris_prop_{handedness}.dae'
        local = np.r_[rotor_rotation.inv().apply(position-rotor_position),
                      (rotor_rotation.inv()*rotation).as_euler('xyz')]
        visual.find('pose').text = ' '.join(map(str, local))
        plans.append((rotor, joint, visual, np.r_[rotor_position, rotor_rotation.as_euler('xyz')]))
    for link in model.findall('link'):
        for visual in link.findall('visual'):link.remove(visual)
    for visual in original.findall('visual'):
        if not visual.get('name', '').startswith('propeller_'):body.append(deepcopy(visual))
    for rotor, joint, visual, values in plans:
        if relocate_rotors:rotor.find('pose').text = ' '.join(map(str, values))
        rotor.append(visual)
        # Joint pose defaults to its child link; its local +Z is also the motor force axis.
        if joint.find('pose') is not None:joint.find('pose').text = '0 0 0 0 0 0'
        joint.find('axis/xyz').text = '0 0 1'
        joint.find('axis/use_parent_model_frame').text = '0'

def add_rotor_joint_observer(model):
    """Read motor-driven joint positions/velocities without publishing commands."""
    plugin = ET.fromstring(
        '<plugin name="jol_rotor_joint_observer" filename="libgazebo_ros_joint_state_publisher.so">'
        '<ros><namespace>/jol/sim_observer</namespace><remapping>joint_states:=rotor_joints</remapping>'
        '</ros><update_rate>50</update_rate></plugin>')
    for motor in range(4):ET.SubElement(plugin, 'joint_name').text = f'rotor_{motor}_joint'
    model.append(plugin)


def template_model(px4_root, name):
    import jinja2
    source = Path(px4_root)/'Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    text = jinja2.Environment(loader=jinja2.FileSystemLoader(str(source))).get_template(
        f'models/{name}/{name}.sdf.jinja').render(mavlink_tcp_port=4560, mavlink_udp_port=14560,
        serial_enabled=0, serial_device='', serial_baudrate=921600, hil_mode=0)
    return ET.fromstring(text).find('model'), source


def plugin_paths(model, plugin_directory, aero_plugin_directory):
    for plugin in model.iter('plugin'):
        filename = plugin.get('filename')
        if filename.startswith('libgazebo_ros_'):continue
        directory = aero_plugin_directory if filename in ('libAdvancedLiftDragPlugin.so', 'libgazebo_airspeed_plugin.so') else plugin_directory
        path = Path(directory).resolve()/Path(filename).name
        if not path.is_file():raise FileNotFoundError(path)
        plugin.set('filename', str(path))


def body_observer(model, name):
    model.append(ET.fromstring(
        f'<plugin name="{name}" filename="libgazebo_ros_p3d.so"><ros>'
        '<namespace>/jol/sim_observer</namespace><remapping>odom:=ground_truth</remapping></ros>'
        '<body_name>base_link</body_name><frame_name>world</frame_name><update_rate>50</update_rate>'
        '<gaussian_noise>0</gaussian_noise></plugin>'))


def bench_model(share, px4_root, plugin_directory):
    model, _ = template_model(px4_root, 'iris')
    model.set('name', 'tailsitter_20cm_bench')
    ET.SubElement(model, 'pose').text = '0 0 .1 0 0 0'
    for element in list(model):
        if ((element.tag == 'plugin' and not element.get('filename', '').endswith('libgazebo_motor_model.so'))
                or element.tag in ('model', 'include')
                or element.tag == 'link' and element.get('name') not in ('base_link', 'rotor_0', 'rotor_1', 'rotor_2', 'rotor_3')
                or element.tag == 'joint' and element.get('name') not in tuple(f'rotor_{i}_joint' for i in range(4))):
            model.remove(element)
    for link in model.findall('link'):
        for sensor in link.findall('sensor'):link.remove(sensor)
    model.find("link[@name='base_link']/collision/geometry/box/size").text = '.47 .47 .20'
    original = observer_model(Path(share)).find("link[@name='body']")
    mount_rotor_visuals(model, original); add_rotor_joint_observer(model)
    body_observer(model, 'jol_bench_body_truth')
    model.append(ET.fromstring('<joint name="measurement_stand" type="fixed">'
                              '<parent>world</parent><child>base_link</child></joint>'))
    plugin_paths(model, plugin_directory, plugin_directory)
    return model


def set_value(element, key, value):
    child = element.find(key)
    if child is None:child = ET.SubElement(element, key)
    child.text = str(value)


def set_inertial(link, mass, center, tensor):
    inertial = link.find('inertial')
    if inertial is None:inertial = ET.SubElement(link, 'inertial')
    inertial.clear()
    set_value(inertial, 'pose', ' '.join(map(str, center))+' 0 0 0')
    set_value(inertial, 'mass', mass)
    values = ET.SubElement(inertial, 'inertia')
    for row, a in enumerate('xyz'):
        for col, b in enumerate('xyz'):
            if col >= row:set_value(values, 'i'+a+b, tensor[row][col])


def build_model(share, px4_root, plugin_directory, aero_plugin_directory, physics_profile_path, requested_speed=8.):
    """Candidate only. Every physical coefficient comes from the supplied profile."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    share = Path(share)
    physics = run_path(str(share/'launch/tailsitter_physics.py'))
    config = physics['load'](physics_profile_path)
    audit = physics['audit'](config, requested_speed=requested_speed)
    if not audit['data_complete']:raise ValueError('measured_physics_incomplete: '+', '.join(audit['missing']))
    if not audit['static_hover_thrust_available']:raise ValueError('insufficient_static_hover_thrust')
    if not audit['forward_speed_above_stall_margin']:raise ValueError('requested_speed_below_stall_margin')
    model, source = template_model(px4_root, 'quadtailsitter')
    model.set('name', 'tailsitter_20cm_px4')
    home = np.asarray(config.get('home_world', [-5., -6., .1]), dtype=float)
    yaw = float(config.get('home_yaw_world', 0.))
    if home.shape != (3,) or not np.isfinite(home).all() or not math.isfinite(yaw):
        raise ValueError('finite_3d_home_and_yaw_required')
    home = home.tolist()
    ET.SubElement(model, 'pose').text = ' '.join(map(str, [*home, 0, 0, yaw]))
    body = model.find("link[@name='base_link']"); body.find('pose').text = '0 0 0 0 0 0'
    original = observer_model(share).find("link[@name='body']")
    for item in list(model):
        if item.tag in ('include', 'joint') and (item.tag == 'include' or not item.get('name').startswith('rotor_')):
            model.remove(item)
        elif item.tag == 'link' and item.get('name') != 'base_link' and not item.get('name').startswith('rotor_'):
            model.remove(item)
    positions = np.asarray(config['rotor_positions_body_m']); axes = np.asarray(config['rotor_axes_body'])
    rotations = []
    for motor, prop in enumerate((1, 2, 3, 0)):
        axis = axes[motor]; cross = np.cross([0., 0., 1.], axis); norm = np.linalg.norm(cross)
        aligned = (Rotation.from_rotvec(cross/norm*math.atan2(norm, axis[2])) if norm > 1e-10 else
                   Rotation.identity() if axis[2] > 0 else Rotation.from_rotvec([math.pi, 0, 0]))
        visual = original.find(f"visual[@name='propeller_{prop}']")
        phase = np.fromstring(visual.findtext('pose'), sep=' ')[5]
        rotation = aligned*Rotation.from_euler('z', phase); rotations.append(rotation.as_matrix())
        visual.find('pose').text = ' '.join(map(str, np.r_[positions[motor], rotation.as_euler('xyz')]))
    mount_rotor_visuals(model, original)
    for collision in body.findall('collision'):body.remove(collision)
    body.append(deepcopy(original.find('collision')))
    # Default 1e12 N/m contacts alias micron-scale motion into the 250Hz IMU
    # on this lightweight research model. Resolve compliant, damped ground contact.
    collision = body.find('collision')
    collision_radius = float(np.linalg.norm(np.fromstring(collision.findtext('geometry/box/size'),sep=' '))/2)
    flight_radius = config.get('flight_parameters', {}).get('body_radius')
    if flight_radius is not None and flight_radius<collision_radius:
        raise ValueError('planning_body_radius_smaller_than_actual_collision_box')
    surface = collision.find('surface')
    if surface is None:surface = ET.SubElement(collision, 'surface')
    contact = surface.find('contact')
    if contact is None:contact = ET.SubElement(surface, 'contact')
    contact_ode = contact.find('ode')
    if contact_ode is None:contact_ode = ET.SubElement(contact, 'ode')
    numerics = config.get('physics_numerics', {})
    stiffness = float(numerics.get('contact_stiffness_N_m', 1e4))
    if not math.isfinite(stiffness) or stiffness<=0:
        raise ValueError('positive_finite_contact_stiffness_required')
    damping = float(numerics.get('contact_damping_N_s_m', 4*math.sqrt(config['mass_kg']*stiffness)))
    if not math.isfinite(damping) or damping<=0:
        raise ValueError('positive_finite_contact_damping_required')
    for key, value in dict(kp=stiffness, kd=damping, max_vel=.05, min_depth=0.).items():
        set_value(contact_ode, key, value)
    for motor, prop in enumerate((1, 2, 3, 0)):
        rotor = model.find(f"link[@name='rotor_{motor}']")
        for collision in rotor.findall('collision'):rotor.remove(collision)
        uri = rotor.find('visual/geometry/mesh/uri')
        uri.text = uri.text.rsplit('/', 1)[0]+f"/iris_prop_{config['rotor_spin_directions'][motor]}.dae"
    # The official airspeed plugin reads its parent LINK's +X, not sensor pose.
    # Two fixed virtual sensor bodies partition the measured aggregate exactly;
    # this numerical decomposition is not an estimate of actual sensor mass.
    partition = .001; mass = config['mass_kg']; center = np.array(config['center_of_mass_body_m'])
    full_inertia = np.asarray(config['inertia_body_kg_m2']); rotor_mass = config['rotor_mass_kg']
    rotor_inertia = np.asarray(config['rotor_inertia_kg_m2'])
    components = []
    for motor, rotation in enumerate(rotations):
        link = model.find(f"link[@name='rotor_{motor}']")
        set_inertial(link, rotor_mass, [0, 0, 0], rotor_inertia)
        components.append((rotor_mass, positions[motor], rotation@rotor_inertia@rotation.T))
    for sensor_name in ('gps', 'airspeed'):
        link = deepcopy(ET.parse(source/f'models/{sensor_name}/{sensor_name}.sdf').getroot().find('model/link'))
        link.set('name', sensor_name+'_link')
        rotation = Rotation.from_euler('y', -math.pi/2) if sensor_name == 'airspeed' else Rotation.identity()
        sensor_angles = [0., -math.pi/2, 0.] if sensor_name == 'airspeed' else [0., 0., 0.]
        link.find('pose').text = ' '.join(map(str, [*center, *sensor_angles]))
        for visual in link.findall('visual'):link.remove(visual)
        set_value(link.find('sensor/plugin'), 'topic', sensor_name)
        sensor_tensor = rotation.as_matrix().T@full_inertia@rotation.as_matrix()*partition
        set_inertial(link, mass*partition, [0, 0, 0], sensor_tensor)
        model.append(link)
        model.append(ET.fromstring(f'<joint name="{sensor_name}_joint" type="fixed">'
                                  f'<parent>base_link</parent><child>{sensor_name}_link</child></joint>'))
        components.append((mass*partition, center, full_inertia*partition))
    base_mass = mass-sum(item[0] for item in components)
    if base_mass <= 0:raise ValueError('component_mass_exceeds_complete_aircraft_mass')
    base_center = (mass*center-sum(m*p for m, p, _ in components))/base_mass
    parallel = lambda d:np.dot(d, d)*np.eye(3)-np.outer(d, d)
    base_tensor = full_inertia-sum(tensor+m*parallel(p-center) for m, p, tensor in components)
    base_tensor -= base_mass*parallel(base_center-center)
    physics['inertia'](base_tensor.tolist(), 'body_inertia_after_component_subtraction')
    set_inertial(body, base_mass, base_center, base_tensor)
    model.find("plugin[@name='gazebo_imu_plugin']/linkName").text = 'base_link'
    fit = audit['motor_fit']; motors = [p for p in model.findall('plugin') if p.get('filename') == 'libgazebo_motor_model.so']
    for motor in motors:
        index = int(motor.findtext('motorNumber'))
        for key, value in dict(motorConstant=fit['thrust_coefficient'], momentConstant=fit['moment_constant_m'],
            maxRotVelocity=fit['max_omega_rad_s'], turningDirection=config['rotor_spin_directions'][index],
            timeConstantUp=config['motor']['time_constant_up_s'], timeConstantDown=config['motor']['time_constant_down_s'],
            rotorDragCoefficient=0., rollingMomentCoefficient=0., maxRelativeAirspeed=1e30, rotorVelocitySlowdownSim=50).items():
            set_value(motor, key, value)
    set_value(model.find("plugin[@name='rosbag']"), 'rotorVelocitySlowdownSim', 50)
    for channel in model.findall("plugin[@name='mavlink_interface']/control_channels/channel"):
        set_value(channel, 'input_scaling', fit['max_omega_rad_s'])
    aero = config['aerodynamics']; aerodynamic = model.find("plugin[@name='aircraft']")
    filename = aerodynamic.get('filename'); aerodynamic.clear(); aerodynamic.set('name', 'aircraft'); aerodynamic.set('filename', filename)
    values = dict(a0=0., CL0=aero['CL0'], CLa=aero['CLa_per_rad'], CD0=aero['CD0'])
    values.update(Cem0=aero['Cm0'], Cema=aero['Cma_per_rad'], AR=aero['aspect_ratio'], eff=aero['efficiency'],
        alpha_stall=aero['alpha_stall_rad'], CLa_stall=aero['CLa_stall_per_rad'],
        CDa_stall=aero['CDa_stall_per_rad'], Cema_stall=aero['Cma_stall_per_rad'],
        area=aero['wing_area_m2'], mac=aero['mean_chord_m'], air_density=config['air_density_kg_m3'],
        ref_pt=' '.join(map(str, aero['center_of_pressure_body_m'])), forward='0 0 1', upward='-1 0 0',
        link_name='base_link', num_ctrl_surfaces=0, robotNamespace='', windSubTopic='world_wind')
    for key, value in values.items():set_value(aerodynamic, key, value)
    add_rotor_joint_observer(model); body_observer(model, 'jol_px4_truth')
    plugin_paths(model, plugin_directory, aero_plugin_directory)
    parameters = dict(SYS_AUTOSTART=1045, CA_AIRFRAME=4, CA_ROTOR_COUNT=4, CA_SV_CS_COUNT=0)
    flu_to_frd = np.array([1., -1., -1.])
    for motor in range(4):
        for xyz, value in zip('XYZ', (positions[motor]-center)*flu_to_frd):parameters[f'CA_ROTOR{motor}_P{xyz}'] = float(value)
        for xyz, value in zip('XYZ', axes[motor]*flu_to_frd):parameters[f'CA_ROTOR{motor}_A{xyz}'] = float(value)
        parameters[f'CA_ROTOR{motor}_KM'] = fit['moment_constant_m']*(1 if config['rotor_spin_directions'][motor] == 'ccw' else -1)
    parameters.update(config.get('px4_parameters', {}))
    report = dict(bench_only=False, flight_verified=False, physics_calibrated=False,
        collision_bounding_sphere_radius_m=collision_radius,
        vehicle_pose_actuation=False, data_status=config.get('data_status'),
        physics_profile_path=str(Path(physics_profile_path).resolve()), input_sources=config['sources'], audit=audit,
        model_name=model.get('name'), home_world=home, home_yaw_world=yaw,
        vehicle_rotation_actuation='PX4 differential rotor forces and reaction torques',
        rotor_joint_state_topic='/jol/sim_observer/rotor_joints',
        px4_parameters=parameters, px4_sim_model='gazebo-classic_quadtailsitter',
        flight_parameters=config.get('flight_parameters', {}),
        fixed_sensor_numerical_mass_partition_fraction=partition,
        total_mass_kg=mass, body_mass_kg=base_mass,
        limitations=['Input provenance does not authenticate measurements or prove flight.',
            'Original visual body collision envelope only; rotor collisions are not modeled.',
            'Static motor fit only; inflow thrust loss, rotor drag and rolling torque omitted.',
            'Longitudinal aerodynamic approximation: unmeasured lateral/dynamic derivatives omitted.',
            'GPS/IMU/airspeed sensor noise is the official simulation reference, not identified hardware.'])
    return model, report
