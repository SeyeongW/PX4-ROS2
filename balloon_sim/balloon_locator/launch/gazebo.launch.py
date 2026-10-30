"""Self-contained camera-only STL/PX4 simulation; no SIYI hardware IO."""
import json
import os
import tempfile
from pathlib import Path
from runpy import run_path
from ament_index_python.packages import get_package_prefix,get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess,LogInfo,OpaqueFunction,SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from balloon_locator.gazebo_scene import make_world
from balloon_locator.stl_airframe import build_model,load_profile
from balloon_locator.runtime_paths import runtime_paths


def start_stl_mavros(context,report,perception,existing,watch):
    """One model report supplies both Gazebo physics and the MC rotor allocation."""
    value=lambda key:LaunchConfiguration(key).perform(context)
    root=Path(value('px4_root')).resolve();binary=Path(value('px4_binary')).resolve()
    etc=root/'build/px4_sitl_default/etc'
    if not binary.is_file() or not etc.is_dir():raise ValueError('Local PX4 SITL build required')
    rootfs=Path(tempfile.mkdtemp(prefix='px4_rootfs_',dir=value('static_output')))
    etc=existing['prepare_px4_parameter_overrides'](etc,rootfs,'10015_gazebo-classic_iris',report['px4_parameters'])
    px4=ExecuteProcess(cmd=[str(binary),'-d',str(etc)],cwd=str(rootfs),output='screen',
        additional_env={'PX4_SIM_MODEL':'gazebo-classic_iris','PX4_SYS_AUTOSTART':'10015'})
    mavros=Node(package='mavros',executable='mavros_node',namespace='mavros',output='screen',
        parameters=[str(perception/'config/mavros_sitl.yaml')])
    adapter=Node(package='balloon_locator',executable='stl_mavros_adapter',output='screen',parameters=[{
        'use_sim_time':True,'home_world':report['home_world'],'home_yaw_world':report['home_yaw_world'],
        'aircraft_type':'stl_pacer_quad','physics_profile_path':value('physics_profile_path')}])
    return [*[watch(p) for p in (px4,mavros,adapter)],px4,mavros,adapter]


def setup(context):
    value=lambda key:LaunchConfiguration(key).perform(context)
    for key,expected in (('scenario','static_flight'),('flight_backend','mavros')):
        if value(key)!=expected:
            raise ValueError(f'balloon_gazebo_requires_{key}={expected}')
    aircraft=value('aircraft_type')
    if aircraft != 'stl_pacer_quad':
        raise ValueError('balloon_gazebo_requires_aircraft_type=stl_pacer_quad (legacy VTOL assets are not bundled)')
    sim=Path(get_package_share_directory('jol_sim'));perception=Path(get_package_share_directory('jol_perception'))
    if not value('static_output') or not Path(value('static_output')).is_absolute():
        raise ValueError('Use tools/run_gazebo.sh for private temporary output, or set absolute static_output explicitly')
    output=Path(value('static_output'));output.mkdir(parents=True,exist_ok=True)
    model=report=None
    if aircraft=='stl_pacer_quad':
        local=Path(get_package_share_directory('balloon_locator'))
        model,report=build_model(sim,value('px4_root'),value('px4_plugin_dir'),value('px4_aero_plugin_dir'),
            value('physics_profile_path'),local/'meshes/stl_airframe')
    world=make_world(sim,value('px4_root'),value('px4_plugin_dir'),value('px4_aero_plugin_dir'),value('physics_profile_path'),
        range_m=float(value('target_range')),height_m=float(value('target_height')),azimuth_deg=float(value('target_azimuth')),
        aircraft_model=model)
    world_path=output/'balloon.world';world_path.write_text(world)
    clock_path=output/'clock.yaml';clock_path.write_text(json.dumps({'/gazebo':{'ros__parameters':{'publish_rate':100.}}}))
    existing=run_path(str(perception/'launch/monitor.launch.py'))
    watch=run_path(str(sim/'launch/required_process.py'))['watch_required_process']
    server=ExecuteProcess(cmd=['gzserver','--verbose','--lockstep','--seed','360',str(world_path),
        '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so','--ros-args','--params-file',str(clock_path),'--'],output='screen')
    detector=Node(package='balloon_locator',executable='balloon_detector_node',output='screen',parameters=[{
        'use_sim_time':True,'process_scale':1.0,'hsv_ranges':[3,100,70,28,255,255]}],
        remappings=[('/camera/image_raw','/jol/camera/image_raw')])
    locator=Node(package='balloon_locator',executable='balloon_locator_node',output='screen',parameters=[{
        'use_sim_time':True,'pose_source':'tf','fixed_frame':'world','camera_info_topic':'/jol/camera/camera_info',
        'calibration_valid':True,'balloon_diameter_m':.3,'diameter_rel_sigma':0.0,
        'pixel_sigma_px':.5,'focal_rel_sigma':0.0,'angle_sigma_deg':.1,'min_samples':3,'max_sigma_m':.5}])
    tracker=Node(package='balloon_locator',executable='gimbal_tracker_node',output='screen',parameters=[{
        'use_sim_time':True,'image_width':1280.,'image_height':960.,'deadband':.003,
        'yaw_sign':1.,'pitch_sign':-1.}])
    bridge=Node(package='balloon_locator',executable='gazebo_bridge',output='screen',parameters=[{'use_sim_time':True}])
    flight=load_profile(value('physics_profile_path'))['flight_parameters'] if report else {}
    mission=Node(package='balloon_locator' if report else 'jol_perception',
        executable='stl_static_flight' if report else 'static_flight',output='screen',parameters=[{
        'use_sim_time':True,'output_dir':str(output/'mission'),'scene_file':str(perception/'config/static_flight_scene.json'),
        'flight_backend':'mavros','aircraft_type':aircraft,'physics_profile_path':value('physics_profile_path'),
        'target_source':'camera_monocular_size','takeoff_height':3.,'vertical_handle':float(value('vertical_handle')),**flight}])
    ui=Node(package='balloon_locator',executable='stl_experiment_ui',output='screen',condition=IfCondition(LaunchConfiguration('ui')),
        parameters=[{'use_sim_time':True,'static_flight_mode':True,'auto_record':False,
                     'record_dir':str(output/'ui')}])
    rviz=Node(package='rviz2',executable='rviz2',output='screen',condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d',str(perception/'config/static_flight.rviz')],parameters=[{'use_sim_time':True}])
    return [LogInfo(msg='Balloon locator: 30cm, horizontal40m/altitude10m defaults; exact simulated optics/size, NOT measured hardware.'),
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH',str(Path(get_package_prefix('jol_sim'))/'lib')+os.pathsep+os.environ.get('GAZEBO_PLUGIN_PATH','/opt/ros/humble/lib')),
        *(start_stl_mavros(context,report,perception,existing,watch) if report else existing['start_mavros'](context)),
        *[watch(node) for node in (server,detector,locator,tracker,bridge,mission)],
        server,detector,locator,tracker,bridge,mission,ui,rviz,
        ExecuteProcess(cmd=['gzclient'],output='screen',condition=IfCondition(LaunchConfiguration('gui')))]


def generate_launch_description():
    paths=runtime_paths()
    defaults=dict(gui='true',ui='true',rviz='true',target_range='40.0',target_height='10.0',target_azimuth='-90.0',
        vertical_handle='4.0',scenario='static_flight',flight_backend='mavros',aircraft_type='stl_pacer_quad',
        static_output=os.environ.get('BALLOON_RUNTIME_DIR',''),px4_root=str(paths['px4_root']),
        px4_binary=str(paths['px4_binary']),
        physics_profile_path=str(Path(get_package_share_directory('balloon_locator'))/'config/stl_pacer_quad.json'),
        px4_plugin_dir=str(paths['px4_plugin_dir']),
        px4_aero_plugin_dir=str(paths['px4_plugin_dir']))
    return LaunchDescription([*[DeclareLaunchArgument(k,default_value=v) for k,v in defaults.items()],OpaqueFunction(function=setup)])
