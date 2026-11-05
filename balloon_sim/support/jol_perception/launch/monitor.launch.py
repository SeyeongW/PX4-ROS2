"""가상 센서 → 추적 → 영상 융합 → 관절 제어 → 실험 화면을 함께 실행한다."""
from pathlib import Path
import math
import os
import re
from runpy import run_path
import shutil
import tempfile
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from jol_perception.mavros_adapter import aircraft_profile


def prepare_px4_parameter_overrides(etc, rootfs, airframe, parameters):
    """Apply explicit SITL values after airframe defaults, before module startup."""
    values = {}
    for name, value in parameters.items():
        number = float(value)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', name) or not math.isfinite(number):
            raise ValueError('Invalid PX4 parameter override: ' + str(name))
        values[name] = format(number, '.17g')
    source = etc / 'init.d-posix/airframes' / airframe
    if not source.is_file():
        raise ValueError('Local PX4 airframe required: ' + str(source))
    # The original PX4 ROMFS stays untouched. rcS sources this private copy
    # before starting the simulator, estimator and vehicle controllers.
    local_etc = rootfs / 'etc'
    shutil.copytree(etc, local_etc, symlinks=True)
    script = local_etc / 'init.d-posix/airframes' / airframe
    lines = ['\n# JOL explicit simulation parameters: after all airframe defaults.',
             "printf 'name\\trequested\\tactual\\n' > px4_parameter_overrides.tsv"]
    for name, value in values.items():
        lines.extend([
            f'param set {name} {value} || exit 1',
            f'param compare {name} {value} || {{ echo "JOL parameter mismatch: {name}"; exit 1; }}',
            f'printf "%s\\t%s\\t%s\\n" {name} {value} "$(param show -q {name})" >> px4_parameter_overrides.tsv'])
    lines.append(f'echo "JOL explicit PX4 parameters verified: {len(values)}"')
    with script.open('a') as stream:
        stream.write('\n'.join(lines) + '\n')
    return local_etc


def validate_configuration(context):
    if LaunchConfiguration('rviz_view').perform(context) not in ('original','focus'):
        raise ValueError('rviz_view must be original or focus')
    if LaunchConfiguration('scenario').perform(context) == 'static_flight':
        if not IfCondition(LaunchConfiguration('simulation')).evaluate(context):
            raise ValueError('static_flight is Gazebo simulation only')
        if LaunchConfiguration('static_detector').perform(context) not in ('orange_marker','yolo'):
            raise ValueError('static_detector must be orange_marker or yolo')
        if LaunchConfiguration('flight_backend').perform(context) not in ('native','mavros'):
            raise ValueError('flight_backend must be native or mavros')
        aircraft=LaunchConfiguration('aircraft_type').perform(context)
        aircraft_profile(aircraft,LaunchConfiguration('physics_profile_path').perform(context))
        if aircraft=='tailsitter20cm_reference' and LaunchConfiguration('flight_backend').perform(context)!='mavros':
            raise ValueError('reference_tailsitter_requires_MAVROS')
        return []
    if (LaunchConfiguration('camera_mount').perform(context) == 'arm'
            and not IfCondition(LaunchConfiguration('arm')).evaluate(context)):
        raise ValueError('camera_mount:=arm requires arm:=true')
    if IfCondition(LaunchConfiguration('observer_flight')).evaluate(context):
        if not all(IfCondition(LaunchConfiguration(name)).evaluate(context)
                   for name in ('simulation', 'camera', 'pointing', 'coordinate_fusion', 'stabilize_frame')):
            raise ValueError('observer_flight requires simulation, camera, pointing, coordinate_fusion and stabilize_frame')
    for name, allowed in (('camera_mount', ('arm', 'turret')), ('lidar_mount_mode', ('payload', 'level'))):
        if LaunchConfiguration(name).perform(context) not in allowed:
            raise ValueError(f'{name} must be one of {allowed}')
    elevation = float(LaunchConfiguration('patrol_elevation_deg').perform(context))
    period = float(LaunchConfiguration('turret_period').perform(context))
    if not math.isfinite(elevation) or not 0 <= elevation <= 30:
        raise ValueError('patrol_elevation_deg must be an upward angle from 0 to 30 degrees')
    if IfCondition(LaunchConfiguration('pointing')).evaluate(context):
        if not math.isfinite(period) or period < 2*math.pi/1.2:
            raise ValueError('Patrol period must respect the 1.2 rad/s yaw limit (at least 5.236 s)')
        horizon = float(LaunchConfiguration('prediction_horizon').perform(context))
        lead = float(LaunchConfiguration('turret_lead').perform(context))
        target_id = int(LaunchConfiguration('target_id').perform(context))
        if not 0 <= lead <= horizon <= 3 or target_id < 0:
            raise ValueError('Pointing requires 0 <= turret_lead <= prediction_horizon <= 3 and target_id >= 0')
    if IfCondition(LaunchConfiguration('camera')).evaluate(context):
        interpreter = LaunchConfiguration('perception_python').perform(context)
        model = Path(LaunchConfiguration('model_path').perform(context))
        if shutil.which(interpreter) is None:
            raise ValueError(f'Camera perception_python is not executable: {interpreter}')
        if not model.is_file() or not os.access(model, os.R_OK):
            raise ValueError(f'Camera model_path is not a readable local file: {model}')
    if IfCondition(LaunchConfiguration('observation')).evaluate(context):
        scene = Path(LaunchConfiguration('observation_scene').perform(context))
        standoff = float(LaunchConfiguration('observation_standoff').perform(context))
        if not scene.is_file() or not os.access(scene, os.R_OK):
            raise ValueError('observation_scene must be a readable scene JSON file')
        if not math.isfinite(standoff) or standoff < 2.:
            raise ValueError('observation_standoff must be finite and at least 2m')
    return []


def start_mavros(context):
    """Start local SITL only; fresh parameters, never serial/hardware connections."""
    if (LaunchConfiguration('scenario').perform(context)!='static_flight'
            or LaunchConfiguration('flight_backend').perform(context)!='mavros'):return []
    root=Path(LaunchConfiguration('px4_root').perform(context)).resolve()
    override=LaunchConfiguration('px4_binary').perform(context)
    binary=Path(override).resolve() if override else root/'build/px4_sitl_default/bin/px4'
    etc=root/'build/px4_sitl_default/etc'
    if not binary.is_file() or not etc.is_dir():raise ValueError('Local PX4 SITL build required: '+str(binary))
    output=Path(LaunchConfiguration('static_output').perform(context));output.mkdir(parents=True,exist_ok=True)
    rootfs=Path(tempfile.mkdtemp(prefix='px4_rootfs_',dir=output))
    aircraft=LaunchConfiguration('aircraft_type').perform(context)
    profile_path=LaunchConfiguration('physics_profile_path').perform(context)
    profile=aircraft_profile(aircraft,profile_path)
    px4_env={'PX4_SIM_MODEL':'gazebo-classic_iris','PX4_SYS_AUTOSTART':'10015',
        'PX4_PARAM_COM_RC_IN_MODE':'4','PX4_PARAM_COM_RCL_EXCEPT':'4','PX4_PARAM_MPC_LAND_SPEED':'.2'}
    sim_share=Path(get_package_share_directory('jol_sim'))
    if aircraft=='tailsitter20cm_reference':
        _,report=run_path(str(sim_share/'launch/tailsitter_model.py'))['build_model'](
            sim_share,root,LaunchConfiguration('px4_plugin_dir').perform(context),
            LaunchConfiguration('px4_aero_plugin_dir').perform(context),profile_path,
            requested_speed=profile['flight_parameters']['max_speed'])
        px4_env.update(PX4_SIM_MODEL='gazebo-classic_quadtailsitter',PX4_SYS_AUTOSTART='1045')
        px4_env.update({'PX4_PARAM_'+key:str(value) for key,value in report['px4_parameters'].items()})
        etc=prepare_px4_parameter_overrides(etc,rootfs,'1045_gazebo-classic_quadtailsitter',
            {key.removeprefix('PX4_PARAM_'):value for key,value in px4_env.items()
             if key.startswith('PX4_PARAM_')})
    px4=ExecuteProcess(cmd=[str(binary),'-d',str(etc)],cwd=str(rootfs),output='screen',
        additional_env=px4_env)
    share=Path(get_package_share_directory('jol_perception'))
    mavros=Node(package='mavros',executable='mavros_node',namespace='mavros',output='screen',
        parameters=[str(share/'config/mavros_sitl.yaml')])
    adapter=Node(package='jol_perception',executable='mavros_adapter',output='screen',
        parameters=[{'use_sim_time':True,'home_world':profile.get('home_world',[-5.,-6.,.1]),
            'home_yaw_world':profile.get('home_yaw_world',0.),'aircraft_type':aircraft,'physics_profile_path':profile_path}])
    watch=run_path(str(sim_share/'launch/required_process.py'))['watch_required_process']
    return [*[watch(process) for process in (px4,mavros,adapter)],px4,mavros,adapter]


def generate_launch_description():
    raise ValueError('This bundled support package is not a standalone legacy launcher. '
                     'Use: ros2 launch balloon_locator gazebo.launch.py (via tools/run_gazebo.sh)')
