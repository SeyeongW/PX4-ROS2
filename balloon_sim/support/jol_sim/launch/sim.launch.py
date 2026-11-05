"""장면·센서 장착·반복 조건을 SDF로 만들고 Gazebo를 실행한다."""
import math
import json
import os
from pathlib import Path
from runpy import run_path
from string import Template
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def drone_model(index):
    # 보이는 외형과 충돌 형상의 크기를 동일하게 유지한다.
    pieces = [('body', '0 0 0 0 0 0', '<box><size>0.55 0.35 0.30</size></box>')]
    for sign in (-1, 1):
        pieces.append((f'arm_{sign}', f'0 0 0 0 0 {sign * math.pi / 4}',
                       '<box><size>1.2 0.075 0.075</size></box>'))
    for j, (x, y) in enumerate(((0.4, 0.4), (0.4, -0.4), (-0.4, 0.4), (-0.4, -0.4))):
        pieces.append((f'motor_{j}', f'{x} {y} 0.035 0 0 0',
                       '<cylinder><radius>0.06</radius><length>0.16</length></cylinder>'))
        # 얇은 실제 블레이드 형상으로 반사면 과장을 피한다.
        pieces.append((f'blade_{j}', f'{x} {y} 0.13 0 0 {j * math.pi / 4}',
                       '<box><size>0.56 0.055 0.025</size></box>'))
    parts = []
    for name, pose, geometry in pieces:
        color = '0.08 0.08 0.09 1' if name.startswith('blade') else '0.86 0.18 0.045 1'
        parts.append(f'<visual name="{name}_visual"><pose>{pose}</pose><geometry>{geometry}</geometry>'
                     f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>'
                     f'<collision name="{name}"><pose>{pose}</pose><geometry>{geometry}</geometry></collision>')
    return (f'<model name="drone_{index}"><pose>0 0 -100 0 0 0</pose><static>false</static>'
            '<link name="body"><gravity>false</gravity><kinematic>true</kinematic>'
            '<inertial><mass>3</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.2</izz></inertia></inertial>'
            + ''.join(parts) + '</link></model>')


ARM_CAMERA_POSE = (.126, 0., .06)

# Pure shared SDF helpers also serve the independent diagnostic/model exporter.
TAILSITTER_MODEL = run_path(str(Path(__file__).with_name('tailsitter_model.py')))
observer_model = TAILSITTER_MODEL['observer_model']
mount_rotor_visuals = TAILSITTER_MODEL['mount_rotor_visuals']
add_rotor_joint_observer = TAILSITTER_MODEL['add_rotor_joint_observer']


def px4_observer_model(share, px4_root, plugin_directory):
    """Exact 20cm visuals on an official Iris motor/sensor physics proxy."""
    import jinja2

    source = Path(px4_root).resolve() / 'Tools/simulation/gazebo-classic/sitl_gazebo-classic'
    plugins = Path(plugin_directory).resolve()
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(str(source)))
    model = ET.fromstring(environment.get_template('models/iris/iris.sdf.jinja').render(
        mavlink_tcp_port=4560, mavlink_udp_port=14560, serial_enabled=0,
        serial_device='/dev/ttyACM0', serial_baudrate=921600, hil_mode=0)).find('model')
    model.set('name', 'tailsitter_20cm_px4')
    ET.SubElement(model, 'pose').text = '-5 -6 .12 0 0 0'
    for link in model.findall('link'):
        for visual in link.findall('visual'):link.remove(visual)
    original = observer_model(share).find("link[@name='body']")
    body = model.find("link[@name='base_link']")
    for visual in original.findall('visual'):body.append(visual)
    # Keep Iris mass/inertia/rotors/motor coefficients. The 20cm height prevents
    # the preserved feet/fin visuals from penetrating this flat ground.
    body.find('collision/geometry/box/size').text = '.47 .47 .20'
    contacts = original.find("sensor[@name='contacts']")
    contacts.find('contact/collision').text = 'base_link_inertia_collision'
    body.append(contacts)
    include = model.find('include')
    if include is None or include.findtext('uri') != 'model://gps':
        raise ValueError('unexpected_PX4_Iris_GPS_include')
    gps = ET.parse(source / 'models/gps/gps.sdf').getroot().find('model')
    gps.set('name', include.findtext('name'))
    ET.SubElement(gps, 'pose').text = include.findtext('pose', '0 0 0 0 0 0')
    for visual in gps.find('link').findall('visual'):gps.find('link').remove(visual)
    model.remove(include); model.append(gps)
    for plugin in model.iter('plugin'):
        if plugin.get('filename').startswith('libgazebo_ros_'):continue
        path = plugins / plugin.get('filename')
        if not path.is_file():raise FileNotFoundError(path)
        plugin.set('filename', str(path))
    # ROS truth odometry is read-only and separate from PX4 motor actuation.
    model.append(ET.fromstring(
        '<plugin name="jol_px4_truth" filename="libgazebo_ros_p3d.so"><ros>'
        '<namespace>/jol/sim_observer</namespace><remapping>odom:=ground_truth</remapping>'
        '</ros><body_name>base_link</body_name><frame_name>world</frame_name>'
        '<update_rate>50</update_rate><gaussian_noise>0</gaussian_noise></plugin>'))
    return model


def static_target_model(azimuth_deg=-90., balloon=False):
    """한 개의 고정 시험 표적. 초기 +X 시선의 오른쪽(-Y) 40 m, 높이 8 m이다."""
    azimuth_deg = float(azimuth_deg)
    if not math.isfinite(azimuth_deg) or not -180 <= azimuth_deg <= 180:
        raise ValueError('static_target_azimuth_deg must be finite and within [-180, 180]')
    angle = math.radians(azimuth_deg)
    x, y = 40 * math.cos(angle), 40 * math.sin(angle)
    geometry = '<sphere><radius>.15</radius></sphere>' if balloon else '<box><size>1.2 1.2 1.2</size></box>'
    return (f'<model name="static_target"><pose>{x} {y} 8 0 0 0</pose><static>true</static>'
            f'<link name="body"><visual name="orange_marker"><geometry>{geometry}'
            '</geometry><material><ambient>1 .45 .05 1</ambient><diffuse>1 .45 .05 1</diffuse>'
            '<emissive>.10 .045 .005 1</emissive></material></visual><collision name="target_collision">'
            f'<geometry>{geometry}</geometry></collision></link></model>')


def static_sensor_rig(share, seed, camera_only=False):
    """Camera gimbal rig; NUB static mode omits the legacy MID360 and mast."""
    from xml.sax.saxutils import escape
    model = ET.fromstring(Template((share/'worlds/static_sensor_rig.sdf.in').read_text()).substitute(
        CAMERA=camera_model(share, (.084, 0., 0.), camera_only=camera_only),
        MID360_MESH=escape((share/'meshes/livox_mid360/mid360.dae').resolve().as_uri()),
        SCAN_FILE=escape(str(share/'scan_mode/mid360.csv')), SEED=seed))
    if camera_only:
        for tag, name in (('link', 'lidar_head'), ('joint', 'lidar_mount_joint'), ('plugin', 'jol_mid360')):
            element = model.find(f"{tag}[@name='{name}']")
            if element is not None: model.remove(element)
        base = model.find("link[@name='base']")
        for element in list(base):
            if element.get('name', '').startswith('lidar_mast'): base.remove(element)
    # The camera optics helper also contains the old D435 visual; use the
    # compact gimbal geometry authored in the SDF instead.
    camera_link = model.find("link[@name='gimbal_pitch']")
    camera_link.remove(camera_link.find("visual[@name='d435_camera']"))
    sensor = camera_link.find("sensor[@name='d435_rgb']")
    # OGRE viewport coordinates have half-integer pixel centers. The optical
    # axis at W/2,H/2 is index (W-1)/2,(H-1)/2 in the zero-based RGB array.
    # gazebo_ros_camera 3.9 defaults to (W+1)/2 without changing rendering.
    for axis, dimension in (('cx', 'width'), ('cy', 'height')):
        ET.SubElement(sensor.find('plugin'), axis).text = str(
            (int(sensor.findtext('camera/image/'+dimension))-1)/2)
    return ET.tostring(model, encoding='unicode')


def camera_model(share, xyz, camera_only=False):
    # 가상 D435 광학계이며 실물 카메라의 보정값은 아니다.
    x, y, z = xyz
    width, height, fov = (1280, 960, math.radians(12)) if camera_only else (640, 480, 1.211)
    return (f'<visual name="d435_camera"><pose>{x+.0043} {y-.0325} {z} 1.57079632679 0 1.57079632679</pose>'
            f'<geometry><mesh><uri>file://{share}/meshes/camera/d435.dae</uri></mesh></geometry></visual>'
            '<sensor name="d435_rgb" type="camera">'
            f'<pose>{x} {y} {z} 0 0 0</pose><always_on>true</always_on><update_rate>15</update_rate><visualize>false</visualize>'
            f'<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>'
            '<clip><near>0.05</near><far>150</far></clip></camera>'
            '<plugin name="jol_d435" filename="libgazebo_ros_camera.so">'
            '<ros><namespace>/jol</namespace><remapping>image_raw:=/jol/camera/image_raw</remapping>'
            '<remapping>camera_info:=/jol/camera/camera_info</remapping></ros>'
            '<camera_name>camera</camera_name><frame_name>jol/camera_optical</frame_name>'
            '</plugin></sensor>')


def arm_model(share, camera_mount='turret'):
    """제조사 형상으로 팔을 만들고 포탑 yaw에 고정한다. pitch는 따르지 않는다."""
    assets = share / 'meshes/open_manipulator_x'
    reference = ET.parse(assets / 'upstream_open_manipulator_x.urdf.xacro').getroot()[0]
    origins = ((0, 0, 0), (.012, 0, .017), (.012, 0, .0765),
               (.036, 0, .2045), (.160, 0, .2045))
    limits = ((-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97))
    mount = (-.8, .9, 1.7)  # yaw 링크 기준 (-.8, .9, 1.0); yaw 높이는 .7 m.
    parts = []
    for i, xyz in enumerate(origins, 1):
        xyz = tuple(a+b for a,b in zip(xyz, mount))
        source = reference.find(f"link[@name='${{prefix}}link{i}']")
        inertial = source.find('inertial')
        mass = inertial.find('mass').get('value')
        center = inertial.find('origin').get('xyz').replace('${0.018 + 2.9433464e-02}', '0.047433464')
        tensor = ''.join(f'<{key}>{value}</{key}>' for key, value in inertial.find('inertia').attrib.items())
        parts.append(f'<link name="link{i}"><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>'
                     '<gravity>false</gravity><kinematic>true</kinematic>'
                     f'<inertial><pose>{center} 0 0 0</pose><mass>{mass}</mass>'
                     f'<inertia>{tensor}</inertia></inertial>')
        for kind in ('visual', 'collision'):
            offset = source.find(kind).find('origin').get('xyz')
            parts.append(f'<{kind} name="manufacturer_mesh{"_visual" if kind == "visual" else ""}"><pose>{offset} 0 0 0</pose>'
                         f'<geometry><mesh><uri>file://{assets}/link{i}.stl</uri>'
                         '<scale>0.001 0.001 0.001</scale></mesh></geometry>' +
                         ('<material><ambient>0.52 0.54 0.58 1</ambient><diffuse>0.65 0.68 0.73 1</diffuse></material>'
                          if kind == 'visual' else '') + f'</{kind}>')
        if i == 1:
            # 브래킷은 기하 배치용이며 베어링·접촉 강도는 모델링하지 않는다.
            bracket = [('mount_plate', '0 0 -0.025 0 0 0', '.24 .24 .05'),
                       ('mount_post', '0 0 -0.63 0 0 0', '.12 .12 1.26'),
                       ('mount_beam', f'.4 -.45 -1.30 0 0 {math.atan2(-.9, .8)}',
                        f'{math.hypot(.8, .9)+.12} .12 .08')]
            for name, location, dimensions in bracket:
                for kind in ('visual', 'collision'):
                    parts.append(f'<{kind} name="{name}{"_visual" if kind == "visual" else ""}"><pose>{location}</pose>'
                                 f'<geometry><box><size>{dimensions}</size></box></geometry>'
                                 + ('<material><diffuse>0.25 0.31 0.36 1</diffuse></material>' if kind == 'visual' else '')
                                 + f'</{kind}>')
        if i == 5 and camera_mount == 'arm':
            parts.append(camera_model(share, ARM_CAMERA_POSE))
        parts.append('</link>')
        if i > 1:
            lower, upper = limits[i - 2]
            axis = '0 0 1' if i == 2 else '0 1 0'
            parts.append(f'<joint name="joint{i-1}" type="revolute"><parent>link{i-1}</parent>'
                         f'<child>link{i}</child><axis><xyz>{axis}</xyz><limit><lower>{lower}</lower>'
                         f'<upper>{upper}</upper><velocity>4.8</velocity><effort>1</effort></limit>'
                         '<dynamics><damping>0.1</damping></dynamics></axis></joint>')
    parts.append('<joint name="mount_fixed" type="fixed"><parent>yaw</parent><child>link1</child></joint>'
                 f'<link name="rod"><pose>{mount[0]+.286} {mount[1]} {mount[2]+.2045} 0 0 0</pose>'
                 '<gravity>false</gravity><kinematic>true</kinematic>'
                 '<inertial><mass>0.03</mass><inertia><ixx>0.000002</ixx><iyy>0.00025</iyy><izz>0.00025</izz></inertia></inertial>')
    for name, position, length, radius, color in (
            ('tool_adapter', '-0.03 0 0', .06, .018, '0.3 0.3 0.32 1'),
            ('pointing_rod', '0.15 0 0', .30, .009, '1 0.45 0.04 1')):
        for kind in ('visual', 'collision'):
            parts.append(f'<{kind} name="{name}{"_visual" if kind == "visual" else ""}"><pose>{position} 0 {math.pi/2} 0</pose>'
                         f'<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>'
                         + (f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>' if kind == 'visual' else '')
                         + f'</{kind}>')
    parts.append('<visual name="tip"><pose>0.3 0 0 0 0 0</pose><geometry><sphere><radius>0.012</radius></sphere></geometry>'
                 '<material><emissive>1 0.25 0 1</emissive></material></visual></link>'
                 '<joint name="rod_fixed" type="fixed"><parent>link5</parent><child>rod</child></joint>'
                 '<plugin name="jol_arm" filename="libjol_arm.so"><ros><namespace>/jol</namespace></ros>'
                 f'<camera_mount>{camera_mount}</camera_mount>'
                 f'<camera_pose>{" ".join(str(v) for v in ARM_CAMERA_POSE)} 0 0 0</camera_pose>'
                 '<max_velocity>1.5</max_velocity></plugin>')
    return ''.join(parts)


def lidar_model(mount_mode='payload', scan_file='mid360.csv', seed=360, lateral_offset=0.):
    """Livox 각도표에 따라 물리 틱마다 초당 20만 광선을 취득한다."""
    if mount_mode not in ('payload', 'level'):
        raise ValueError('lidar_mount_mode must be payload or level')
    # level은 기울어진 박스를 피하도록 8 cm 높인 이상적 역피치 장착이다.
    level = mount_mode == 'level'
    height = 2.127 if level else 2.047
    parts = []
    if level:
        parts.append(f'<link name="lidar_mount"><pose>0 {lateral_offset} 2 0 0 0</pose>'
                     '<gravity>false</gravity><kinematic>true</kinematic>'
                     '<inertial><mass>0.1</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>')
        for kind in ('visual', 'collision'):
            parts.append(f'<{kind} name="level_mount_post{"_visual" if kind == "visual" else ""}"><pose>0 0 0.04 0 0 0</pose>'
                         '<geometry><cylinder><radius>0.022</radius><length>0.08</length></cylinder></geometry>'
                         + ('<material><diffuse>0.25 0.31 0.36 1</diffuse></material>' if kind == 'visual' else '')
                         + f'</{kind}>')
        parts.append('</link><joint name="lidar_support_fixed" type="fixed"><parent>pitch</parent><child>lidar_mount</child></joint>')
    parts.append(f'<link name="lidar_head"><pose>0 {lateral_offset} {height} 0 0 0</pose>'
                 '<gravity>false</gravity><kinematic>true</kinematic>'
                 '<inertial><mass>0.265</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia></inertial>'
                 '<visual name="mid360_body"><pose>0 0 -0.017 0 0 0</pose><geometry><box><size>0.065 0.065 0.06</size></box></geometry><material><ambient>0.12 0.12 0.12 1</ambient><diffuse>0.16 0.16 0.16 1</diffuse></material></visual>'
                 '<visual name="mid360_window"><geometry><cylinder><radius>0.034</radius><length>0.023</length></cylinder></geometry><material><ambient>0.08 0.19 0.24 1</ambient><diffuse>0.08 0.19 0.24 1</diffuse></material></visual>'
                 '</link>')
    parent = 'lidar_mount' if level else 'pitch'
    joint_type = 'revolute' if level else 'fixed'
    parts.append(f'<joint name="lidar_mount_joint" type="{joint_type}"><parent>{parent}</parent><child>lidar_head</child>')
    if level:
        parts.append('<axis><xyz>0 1 0</xyz><limit><lower>-0.5235987755982988</lower><upper>0.5235987755982988</upper></limit></axis>')
    from xml.sax.saxutils import escape
    return (''.join(parts) + '</joint>' +
            '<plugin name="jol_mid360" filename="libjol_mid360.so">'
            '<ros><namespace>/jol</namespace></ros>'
            f'<scan_file>{escape(str(scan_file))}</scan_file><seed>{seed}</seed></plugin>')


def clock_parameters(rate):
    rate = float(rate)
    if not math.isfinite(rate) or not 0 < rate <= 200:
        raise ValueError('clock_rate must be finite and within (0, 200] Hz')
    return {'/gazebo': {'ros__parameters': {'publish_rate': rate}}}


def setup(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    clock_settings = clock_parameters(value('clock_rate'))
    seed = int(value('seed'))
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must fit an unsigned 32-bit integer')
    scenario_start = float(value('scenario_start_seconds'))
    empty_seconds = float(value('empty_seconds'))
    if not math.isfinite(empty_seconds) or empty_seconds < 0:
        raise ValueError('empty_seconds must be finite and nonnegative')
    if not math.isfinite(scenario_start) or (scenario_start != -1 and scenario_start < 0):
        raise ValueError('scenario_start_seconds must be -1 or finite and nonnegative')
    scenario = value('scenario')
    if scenario not in ('empty', 'stationary', 'single', 'multi', 'stop', 'occlusion', 'reappearance', 'all', 'static_flight'):
        raise ValueError('Unknown scenario: ' + scenario)
    trajectory = value('trajectory')
    if trajectory not in ('nonlinear', 'circle'):
        raise ValueError('trajectory must be nonlinear or circle')
    radius, speed, altitude = (float(value(k)) for k in ('radius', 'speed', 'altitude'))
    if not all(math.isfinite(x) for x in (radius, speed, altitude)) or radius <= 1 or speed < 0 or altitude <= 0:
        raise ValueError('Invalid flight settings')
    yaw_amplitude = float(value('turret_yaw_amplitude'))
    pitch_amplitude = float(value('turret_pitch_amplitude'))
    period = float(value('turret_period'))
    survey_elevation = math.radians(float(value('turret_survey_elevation_deg')))
    if not (math.isfinite(yaw_amplitude) and 0 <= yaw_amplitude <= math.pi and
            math.isfinite(pitch_amplitude) and 0 <= pitch_amplitude <= math.pi / 6 and
            math.isfinite(period) and period >= 1):
        raise ValueError('Turret amplitudes must fit yaw ±pi / pitch ±pi/6; period >= 1 second')
    if not math.isfinite(survey_elevation) or abs(survey_elevation) > math.pi / 6:
        raise ValueError('Survey elevation must be within ±30 degrees')
    if value('turret_survey').lower() == 'true' and 2 * math.pi / period > 1.2:
        raise ValueError('Survey period exceeds the turret yaw rate limit')
    lidar_mount_mode = value('lidar_mount_mode')
    camera_mount = value('camera_mount')
    arm_enabled = value('arm').lower() == 'true'
    static_flight = scenario == 'static_flight'
    flight_backend = value('flight_backend')
    if flight_backend not in ('native', 'mavros'):
        raise ValueError('flight_backend must be native or mavros')
    if flight_backend == 'mavros' and not static_flight:
        raise ValueError('flight_backend=mavros requires scenario:=static_flight')
    aircraft_type = value('aircraft_type')
    if aircraft_type not in ('iris_proxy', 'tailsitter20cm_reference'):
        raise ValueError('aircraft_type must be iris_proxy or tailsitter20cm_reference')
    if aircraft_type == 'tailsitter20cm_reference' and not (static_flight and flight_backend == 'mavros'):
        raise ValueError('tailsitter20cm_reference requires static_flight and mavros')
    model_report = None
    if static_flight:
        # 별도 시험 모드로 기존 회전 포탑/다중 표적 실험의 설정을 보존한다.
        camera_mount, arm_enabled, lidar_mount_mode = 'gimbal', False, 'payload'
    elif camera_mount == 'gimbal':
        raise ValueError('camera_mount=gimbal requires scenario:=static_flight')
    if camera_mount not in ('turret', 'arm', 'gimbal') or (camera_mount == 'arm' and not arm_enabled):
        raise ValueError('camera_mount must be turret, or arm with arm:=true')
    share = Path(get_package_share_directory('jol_sim'))
    lidar = lidar_model(lidar_mount_mode, share / 'scan_mode/mid360.csv', seed,
                        .145 if camera_mount == 'turret' else 0.)
    occluder = ''
    if scenario in ('occlusion', 'all'):
        occluder = ('<model name="occluder"><static>true</static><pose>25 0 9 0 0 0</pose><link name="body">'
                    '<visual name="v"><geometry><box><size>1 4 18</size></box></geometry>'
                    '<material><diffuse>0.32 0.36 0.4 1</diffuse></material></visual>'
                    '<collision name="c"><geometry><box><size>1 4 18</size></box></geometry></collision></link></model>')
    world = Template((share / 'worlds/monitor.world.in').read_text()).substitute(
        CAMERA=camera_model(share, (.14, -.145, 1.0125)) if camera_mount == 'turret' else '',
        CAMERA_MOUNT=camera_mount, LIDAR=lidar, LIDAR_MOUNT_MODE=lidar_mount_mode,
        DRONES=static_target_model(value('static_target_azimuth_deg'), balloon=True) if static_flight else '\n'.join(drone_model(i) for i in range(1, 4)), OCCLUDER=occluder,
        OBSERVER=(ET.tostring(observer_model(share), encoding='unicode')
                  if value('observer_flight').lower() == 'true' and not static_flight else ''),
        SCENARIO=scenario, TRAJECTORY=trajectory, RADIUS=radius, SPEED=speed, ALTITUDE=altitude,
        EMPTY_SECONDS=empty_seconds, WAIT_BACKGROUND=value('wait_for_background'),
        SCENARIO_START_SECONDS=scenario_start,
        TURRET_MOTION=value('turret_motion'), TURRET_SURVEY=value('turret_survey'), TURRET_YAW_AMPLITUDE=yaw_amplitude,
        TURRET_PITCH_AMPLITUDE=pitch_amplitude, TURRET_PERIOD=period,
        TURRET_SURVEY_ELEVATION=survey_elevation,
        ARM=arm_model(share, camera_mount) if arm_enabled else '')
    if static_flight:
        tree = ET.fromstring(world)
        scene = tree.find('world')
        scene.remove(scene.find("model[@name='fixed_turret']"))
        scene.append(ET.fromstring(static_sensor_rig(share, seed, camera_only=True)))
        # 시험 표식과 기존 갈색 상자가 같은 HSV 문턱을 통과하지 않도록 새 모드만 회색으로 둔다.
        crate = scene.find("model[@name='background_crate']/link/visual/material")
        for field in ('ambient', 'diffuse'):
            element = crate.find(field)
            if element is None:
                element = ET.SubElement(crate, field)
            element.text = '.45 .45 .45 1'
        if aircraft_type == 'tailsitter20cm_reference':
            profile = value('physics_profile_path') or str(share/'config/tailsitter_20cm_reference.json')
            aero_plugins = value('px4_aero_plugin_dir') or str(Path.home()/'ros2_nub/build/tailsitter_plugins_reference')
            aircraft, model_report = TAILSITTER_MODEL['build_model'](
                share, value('px4_root'), value('px4_plugin_dir'), aero_plugins, profile,
                requested_speed=json.loads(Path(profile).read_text())['flight_parameters']['max_speed'])
            if model_report['data_status'] != 'reference_scaled_not_measured':
                raise ValueError('tailsitter20cm_reference_requires_explicit_reference_profile')
        else:
            aircraft = (px4_observer_model(share, value('px4_root'), value('px4_plugin_dir'))
                        if flight_backend == 'mavros' else observer_model(share, static_flight=True))
        scene.append(aircraft)
        if model_report:
            virtual_marker = model_report['flight_parameters'].get('target_is_virtual_waypoint', False)
            if not isinstance(virtual_marker, bool):
                raise ValueError('target_is_virtual_waypoint_must_be_boolean')
            if virtual_marker:
                # Keep the box for MID360 ODE ray intersections; suppress contact forces only.
                collision = scene.find("model[@name='static_target']/link/collision")
                contact = ET.SubElement(ET.SubElement(collision, 'surface'), 'contact')
                ET.SubElement(contact, 'collide_without_contact').text = 'true'
            model_report.update(marker_virtual=virtual_marker,
                target_collision_response_enabled=not virtual_marker, target_raycast_geometry_preserved=True,
                target_collision_rule=('surface/contact/collide_without_contact=true; ray geometry retained'
                    if virtual_marker else 'ordinary physical contact response'))
            # Spawn registration is infrastructure metadata; flight feedback remains PX4 EKF.
            system_pose = [float(x) for x in scene.findtext("model[@name='fixed_turret']/pose").split()]
            model_report.update(system_base_frame='jol/base', system_origin_world=system_pose[:3],
                spawn_horizontal_distance_from_system_m=math.dist(model_report['home_world'][:2], system_pose[:2]),
                camera_rotation_joints=['gimbal_yaw_joint', 'gimbal_pitch_joint'],
                camera_command_topic='/jol/gimbal/command', camera_feedback_topic='/jol/gimbal/joint_states')
        # 정적 표적은 시작부터 공중에 고정되며 시나리오 플러그인은 평가 정답만 발행한다.
        world = ET.tostring(tree, encoding='unicode')
    handle, world_path = tempfile.mkstemp(prefix='jol_', suffix='.world')
    with os.fdopen(handle, 'w') as stream:
        stream.write(world)
    if value('world_output'):
        output = Path(value('world_output')).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(world)
        if model_report:output.with_suffix('.model.json').write_text(json.dumps(model_report, indent=2)+'\n')
    handle, clock_path = tempfile.mkstemp(prefix='jol_clock_', suffix='.yaml')
    with os.fdopen(handle, 'w') as stream:
        json.dump(clock_settings, stream)  # YAML과 호환되는 JSON으로 /gazebo 설정만 전달한다.
    plugin_path = os.path.join(get_package_prefix('jol_sim'), 'lib')
    plugin_path += os.pathsep + os.environ.get('GAZEBO_PLUGIN_PATH', '/opt/ros/humble/lib')
    server_command = ['gzserver', '--verbose', '--seed', str(seed)]
    if value('lockstep').lower() == 'true':
        server_command.append('--lockstep')
    server_command += [world_path, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so']
    server_command += ['--ros-args', '--params-file', clock_path, '--']
    server = ExecuteProcess(cmd=server_command, output='screen')
    watch_required_process = run_path(str(Path(__file__).with_name('required_process.py')))['watch_required_process']
    return [
        LogInfo(msg=f'Gazebo /clock publish_rate={clock_settings["/gazebo"]["ros__parameters"]["publish_rate"]} Hz'),
        LogInfo(msg=f'Gazebo/MID360 seed={seed}; scenario_start_seconds={scenario_start}; world={world_path}'),
        LogInfo(msg=(f'Camera mount={camera_mount}; simulated 12deg telephoto optics, not measured hardware calibration'
                     if static_flight else f'Camera mount={camera_mount}; nominal virtual D435 optics, not measured hardware calibration')),
        LogInfo(msg=(f'static_flight: one static 30cm orange balloon at azimuth {value("static_target_azimuth_deg")} deg, '
                     '40 m horizontal, 8 m high; '
                     'one existing 20cm tailsitter; fixed camera pedestal, no LiDAR; camera gimbal only; no target motion.'
                     if static_flight else 'Legacy scene configuration preserved.')),
        LogInfo(msg=('Flight backend: MAVROS/PX4 reference-scaled 20cm quadtailsitter; '
                     'real motor joints and aerodynamic/airspeed plugins; NOT measured hardware dynamics.'
                     if aircraft_type == 'tailsitter20cm_reference' else
                     'Flight backend: MAVROS/PX4 with official Iris 1.5kg motor/sensor physics proxy; '
                     'all 33 original 20cm aircraft visuals preserved; not calibrated 20cm aerodynamics.'
                     if flight_backend == 'mavros' else 'Flight backend: native Gazebo force/torque baseline.')),
        LogInfo(msg=f'Drone scenario={scenario}, trajectory={trajectory}, speed={speed} m/s; '
                    'paths use simulation motion elapsed time with fixed per-drone phases.'),
        LogInfo(msg=('Camera-only: simulated 1280x960, 12deg HFOV, exact 30cm sphere assumption; no LiDAR.'
                    if static_flight else 'MID360: fixed 200,000 rays/s, 10 Hz; range noise 1cm.')),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH', plugin_path),
        watch_required_process(server),
        server,
        ExecuteProcess(cmd=['gzclient'], output='screen', condition=IfCondition(LaunchConfiguration('gui'))),

    ]


def generate_launch_description():
    raise ValueError('This bundled support package is not a standalone legacy launcher. '
                     'Use: ros2 launch balloon_locator gazebo.launch.py (via tools/run_gazebo.sh)')
