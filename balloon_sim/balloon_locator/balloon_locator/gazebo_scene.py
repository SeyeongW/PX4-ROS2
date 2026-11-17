"""Reuse the bundled support aircraft/rig helpers, author one camera-only balloon scene."""
import json
import math
from pathlib import Path
from runpy import run_path
from string import Template
import xml.etree.ElementTree as ET


def _find_x500_mesh_dir(px4_root=None) -> Path | None:
    """Locate PX4-Autopilot's x500_base mesh directory if available on the host."""
    candidates = []
    if px4_root is not None:
        candidates.append(Path(px4_root).resolve() / 'Tools/simulation/gz/models/x500_base/meshes')
    candidates.append(Path.home() / 'PX4-Autopilot/Tools/simulation/gz/models/x500_base/meshes')
    for cand in candidates:
        if (cand / 'NXP-HGD-CF.dae').is_file():
            return cand
    return None


def _attach_target_drone_airframe(link: ET.Element, px4_root=None) -> None:
    """Replace the balloon marker with a PX4 x500 quadrotor target (0.64m tip-to-tip span, zero balloon)."""
    for vis in list(link.findall('visual')):
        if vis.get('name') == 'orange_marker':
            link.remove(vis)
    for col in list(link.findall('collision')):
        geom = col.find('geometry')
        if geom is not None:
            geom.clear()
            box = ET.SubElement(geom, 'box')
            ET.SubElement(box, 'size').text = '0.64 0.64 0.24'

    carbon = '0.12 0.14 0.17 1'
    metal = '0.22 0.25 0.30 1'
    rotor_color = '0.07 0.09 0.11 1'
    accent_red = '0.78 0.16 0.12 1'

    mesh_dir = _find_x500_mesh_dir(px4_root)
    if mesh_dir is not None:
        frame_uri = (mesh_dir / 'NXP-HGD-CF.dae').resolve().as_uri()
        link.append(ET.fromstring(
            f'<visual name="x500_frame_mesh"><pose>0 0 0.025 0 0 3.141592654</pose>'
            f'<geometry><mesh><uri>{frame_uri}</uri><scale>1 1 1</scale></mesh></geometry></visual>'
        ))
        bell_path = mesh_dir / '5010Bell.dae'
        prop_ccw_path = mesh_dir / '1345_prop_ccw.stl'
        prop_cw_path = mesh_dir / '1345_prop_cw.stl'
        motor_xy = ((0.174, -0.174, prop_ccw_path), (-0.174, 0.174, prop_ccw_path),
                    (0.174, 0.174, prop_cw_path), (-0.174, -0.174, prop_cw_path))
        for idx, (mx, my, prop_path) in enumerate(motor_xy):
            if bell_path.is_file():
                bell_uri = bell_path.resolve().as_uri()
                link.append(ET.fromstring(
                    f'<visual name="x500_bell_mesh_{idx}"><pose>{mx} {my} 0.032 0 0 0</pose>'
                    f'<geometry><mesh><uri>{bell_uri}</uri><scale>1 1 1</scale></mesh></geometry></visual>'
                ))
            if prop_path.is_file():
                prop_uri = prop_path.resolve().as_uri()
                link.append(ET.fromstring(
                    f'<visual name="x500_prop_mesh_{idx}"><pose>{mx} {my} 0.060 0 0 0</pose>'
                    f'<geometry><mesh><uri>{prop_uri}</uri><scale>0.8461538 0.8461538 0.8461538</scale></mesh></geometry>'
                    f'<material><ambient>{rotor_color}</ambient><diffuse>{rotor_color}</diffuse></material></visual>'
                ))

    # Solid x500 structural silhouette (motor centers at +-0.174m, rotor radius 0.146m -> exact 0.640m span)
    parts = [
        ('drone_deck', '0 0 0.015 0 0 0', '<box><size>0.22 0.16 0.055</size></box>', carbon),
        ('drone_top_plate', '0 0 0.050 0 0 0', '<box><size>0.18 0.12 0.018</size></box>', accent_red),
        ('drone_arm_p45', f'0 0 0.012 0 0 {math.pi/4:.6f}', '<box><size>0.52 0.038 0.028</size></box>', carbon),
        ('drone_arm_m45', f'0 0 0.012 0 0 {-math.pi/4:.6f}', '<box><size>0.52 0.038 0.028</size></box>', carbon),
        ('drone_skid_l', '0 0.115 -0.095 0 0 0', '<box><size>0.30 0.020 0.018</size></box>', carbon),
        ('drone_skid_r', '0 -0.115 -0.095 0 0 0', '<box><size>0.30 0.020 0.018</size></box>', carbon),
        ('drone_leg_fl', '0.09 0.115 -0.045 0 0 0', '<cylinder><radius>0.009</radius><length>0.10</length></cylinder>', carbon),
        ('drone_leg_fr', '0.09 -0.115 -0.045 0 0 0', '<cylinder><radius>0.009</radius><length>0.10</length></cylinder>', carbon),
        ('drone_leg_rl', '-0.09 0.115 -0.045 0 0 0', '<cylinder><radius>0.009</radius><length>0.10</length></cylinder>', carbon),
        ('drone_leg_rr', '-0.09 -0.115 -0.045 0 0 0', '<cylinder><radius>0.009</radius><length>0.10</length></cylinder>', carbon),
    ]
    for idx, (mx, my) in enumerate(((0.174, 0.174), (0.174, -0.174), (-0.174, 0.174), (-0.174, -0.174))):
        parts.append((f'drone_motor_{idx}', f'{mx} {my} 0.032 0 0 0',
                      '<cylinder><radius>0.024</radius><length>0.042</length></cylinder>', metal))
        parts.append((f'drone_blade_{idx}', f'{mx} {my} 0.060 0 0 {idx * math.pi / 4:.6f}',
                      '<box><size>0.292 0.032 0.010</size></box>', rotor_color))
        parts.append((f'drone_rotor_disk_{idx}', f'{mx} {my} 0.060 0 0 0',
                      '<cylinder><radius>0.146</radius><length>0.022</length></cylinder>', rotor_color))
    for name, pose, geom, color in parts:
        vis = ET.fromstring(
            f'<visual name="{name}"><pose>{pose}</pose><geometry>{geom}</geometry>'
            f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>'
        )
        link.append(vis)


def make_world(share,px4_root,plugins,aero_plugins,profile,*,range_m=40.,height_m=10.,azimuth_deg=-90.,aircraft_model=None,target_type='drone'):
    if not (math.isfinite(range_m) and 2<range_m<=500 and math.isfinite(height_m) and 3<height_m<=200
            and math.isfinite(azimuth_deg) and -180<=azimuth_deg<=180):
        raise ValueError('invalid_balloon_experiment_geometry')
    if target_type not in ('drone', 'x500', 'balloon'):
        raise ValueError('invalid_target_type')
    share=Path(share);helpers=run_path(str(share/'launch/sim.launch.py'))
    target=ET.fromstring(helpers['static_target_model'](azimuth_deg,balloon=True))
    angle=math.radians(azimuth_deg)
    target.find('pose').text=f'{range_m*math.cos(angle)} {range_m*math.sin(angle)} {height_m} 0 0 0'
    for sphere in target.iter('sphere'):
        r=sphere.find('radius')
        if r is not None:r.text='0.15'
    if target_type in ('drone', 'x500'):
        _attach_target_drone_airframe(target.find('link'), px4_root=px4_root)
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
