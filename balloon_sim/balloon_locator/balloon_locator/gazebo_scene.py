"""Reuse the bundled support aircraft/rig helpers, author one camera-only balloon scene."""
import json
import math
from pathlib import Path
from runpy import run_path
from string import Template
import xml.etree.ElementTree as ET


def make_world(share,px4_root,plugins,aero_plugins,profile,*,range_m=40.,height_m=10.,azimuth_deg=-90.,aircraft_model=None):
    if not (math.isfinite(range_m) and 2<range_m<=100 and math.isfinite(height_m) and 3<height_m<=18
            and math.isfinite(azimuth_deg) and -180<=azimuth_deg<=180):
        raise ValueError('invalid_balloon_experiment_geometry')
    share=Path(share);helpers=run_path(str(share/'launch/sim.launch.py'))
    target=ET.fromstring(helpers['static_target_model'](azimuth_deg,balloon=True))
    angle=math.radians(azimuth_deg)
    target.find('pose').text=f'{range_m*math.cos(angle)} {range_m*math.sin(angle)} {height_m} 0 0 0'
    parameters=dict(CAMERA='',CAMERA_MOUNT='gimbal',LIDAR='',LIDAR_MOUNT_MODE='payload',ARM='',
        TURRET_MOTION='false',TURRET_SURVEY='false',TURRET_YAW_AMPLITUDE=0,TURRET_PITCH_AMPLITUDE=0,
        TURRET_PERIOD=8,TURRET_SURVEY_ELEVATION=0,OCCLUDER='',DRONES=ET.tostring(target,encoding='unicode'),
        OBSERVER='',SCENARIO='static_flight',TRAJECTORY='nonlinear',RADIUS=range_m,SPEED=0,ALTITUDE=height_m,
        EMPTY_SECONDS=0,WAIT_BACKGROUND='false',SCENARIO_START_SECONDS=0)
    tree=ET.fromstring(Template((share/'worlds/monitor.world.in').read_text()).substitute(parameters))
    world=tree.find('world');world.remove(world.find("model[@name='fixed_turret']"))
    world.append(ET.fromstring(helpers['static_sensor_rig'](share,360,camera_only=True)))
    # Ground infrastructure is a neutral color, not a second orange candidate.
    material=world.find("model[@name='background_crate']/link/visual/material")
    for field in ('ambient','diffuse'):
        element=material.find(field)
        if element is None:element=ET.SubElement(material,field)
        element.text='.45 .45 .45 1'
    if aircraft_model is None:
        model,report=helpers['TAILSITTER_MODEL']['build_model'](share,px4_root,plugins,aero_plugins,profile,
            requested_speed=json.loads(Path(profile).read_text())['flight_parameters']['max_speed'])
    else:
        model=aircraft_model
    world.append(model)
    collision=world.find("model[@name='static_target']/link/collision")
    ET.SubElement(ET.SubElement(ET.SubElement(collision,'surface'),'contact'),'collide_without_contact').text='true'
    return ET.tostring(tree,encoding='unicode')
