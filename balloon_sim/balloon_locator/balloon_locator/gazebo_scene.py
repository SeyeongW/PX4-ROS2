"""Reuse the bundled support aircraft/rig helpers, author one camera-only balloon scene."""
import json
import math
from pathlib import Path
from runpy import run_path
from string import Template
import xml.etree.ElementTree as ET


def make_world(share,px4_root,plugins,aero_plugins,profile,*,range_m=10.,height_m=4.,azimuth_deg=0.,aircraft_model=None):
    if not (math.isfinite(range_m) and 2<range_m<=500 and math.isfinite(height_m) and 3<height_m<=200
            and math.isfinite(azimuth_deg) and -180<=azimuth_deg<=180):
        raise ValueError('invalid_balloon_experiment_geometry')
    share=Path(share);helpers=run_path(str(share/'launch/sim.launch.py'))
    target=ET.fromstring(helpers['static_target_model'](azimuth_deg,balloon=True))
    angle=math.radians(azimuth_deg)
    target.find('pose').text=f'{range_m*math.cos(angle)} {range_m*math.sin(angle)} {height_m} 0 0 0'
    for sphere in target.iter('sphere'):
        r=sphere.find('radius')
        if r is not None:r.text='0.15'
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
    turret=world.find("model[@name='fixed_turret']")
    if turret is not None:
        for limit in turret.iter('limit'):
            lo=limit.find('lower');hi=limit.find('upper')
            if lo is not None and hi is not None:
                try:
                    if abs(float(lo.text))<2.:lo.text=str(-math.pi/2);hi.text=str(math.pi/2)
                except ValueError:pass
        for clip in turret.iter('clip'):
            far_el=clip.find('far')
            if far_el is not None:far_el.text='500'
    collision=world.find("model[@name='static_target']/link/collision")
    ET.SubElement(ET.SubElement(ET.SubElement(collision,'surface'),'contact'),'collide_without_contact').text='true'
    return ET.tostring(tree,encoding='unicode')
