"""Exposure-matched capture and camera/gimbal offline evidence regressions."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


def load_tool(name):
    path=Path(__file__).resolve().parents[3]/'tools'/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


recorder=load_tool('run_static_flight_trial');audit=load_tool('evaluate_gimbal_alignment')


class GimbalEvidenceTests(unittest.TestCase):
    def image(self,stamp):
        return SimpleNamespace(header=SimpleNamespace(frame_id='jol/camera_optical',stamp=SimpleNamespace(
            sec=int(stamp),nanosec=round((stamp-int(stamp))*1e9))),width=640,height=480,encoding='bgr8',
            step=1920,data=np.zeros((480,640,3),np.uint8).tobytes())

    def detection(self,stamp,center_x):
        return dict(stamp=stamp,frame_id='jol/camera_optical',width=640,height=480,
                    detections=[dict(bbox=[center_x-8,232.,16.,16.],confidence=.9)])

    def test_capture_rejects_wrong_exposure_and_records_centered_actual_frames(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);capture=recorder.CameraEvidence(output)
            capture.detection(self.detection(1.,138.))
            capture.image('raw',self.image(.9))
            self.assertFalse((output/'camera_initial_raw.png').exists())
            for kind in ('raw','annotated'):capture.image(kind,self.image(1.))
            capture.detection(self.detection(1.1,320.))
            for kind in ('raw','annotated'):capture.image(kind,self.image(1.1))
            manifest=json.loads((output/'camera_capture.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['saved']['initial_raw']['stamp'],1.)
            self.assertEqual(manifest['saved']['centered_raw']['stamp'],1.1)

    def test_alignment_uses_recorded_pixel_center_with_continuous_box_fallback(self):
        info=dict(k=[500.,0.,320.,0.,500.,240.,0.,0.,1.],d=[0.]*5)
        packet=self.detection(1.,320.)
        packet['detections'][0]['center_px']=[319.5,239.5]
        measured=audit.bearing_offset(packet,info)
        np.testing.assert_equal(measured['center_error_px'],[-.5,-.5])
        self.assertGreater(measured['elevation_offset_deg'],0.)
        packet['detections'][0].pop('center_px')
        np.testing.assert_equal(audit.bearing_offset(packet,info)['center_error_px'],[0.,0.])

    def test_offaxis_alignment_and_world_stability_are_measured_not_truth_control(self):
        with tempfile.TemporaryDirectory() as name:
            trial=Path(name);capture=recorder.CameraEvidence(trial)
            detections=[self.detection(1.,138.),*[self.detection(t,320.) for t in (1.1,1.2,1.3,2.2,2.3,2.4)]]
            for packet in detections:
                capture.detection(packet)
                for kind in ('raw','annotated'):capture.image(kind,self.image(packet['stamp']))
            joint=[dict(stamp=p['stamp'],name=['gimbal_yaw_joint','gimbal_pitch_joint'],position=[0. if i==0 else math.radians(20.),-.15]) for i,p in enumerate(detections)]
            info=[dict(stamp=1.,frame_id='jol/camera_optical',width=640,height=480,k=[500.,0.,320.,0.,500.,240.,0.,0.,1.],d=[0.]*5)]
            estimates=[dict(stamp=t,valid=True,frame_id='world',position=[37.+.003*i,13.,8.],
                evidence=dict(camera_tf_stamp=t,tf_support={'jol/gimbal_yaw':{}})) for i,t in enumerate((2.2,2.3,2.4))]
            transforms=[dict(stamp=t,child_frame_id=child) for t in (1.,1.1) for child in ('jol/gimbal_yaw','jol/gimbal_pitch')]
            static=[dict(child_frame_id=child) for child in ('jol/camera_optical','jol/lidar')]
            for filename,rows in (('detections_trace.jsonl',detections),('gimbal_state_trace.jsonl',joint),
                ('gimbal_command_trace.jsonl',joint),('camera_info_trace.jsonl',info),('estimate_trace.jsonl',estimates),
                ('tf_trace.jsonl',[dict(transforms=transforms)]),('tf_static_trace.jsonl',[dict(transforms=static)])):
                (trial/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result=audit.evaluate(trial)
            self.assertTrue(result['initial_offaxis_observed'] and result['image_centering_observed'])
            self.assertTrue(result['physical_gimbal_rotation_observed'])
            self.assertAlmostEqual(result['measured_gimbal_yaw_change_deg'],20.)
            self.assertLess(result['world_coordinate_stability']['rms_deviation_m'],.01)
            self.assertTrue(all(edge['supported'] for edge in result['captured_exposure_tf_support'].values()))
            self.assertFalse(result['ground_truth_used_by_control'])
            self.assertTrue((trial/'gimbal_alignment.png').is_file())
            (trial/'camera_initial_annotated.png').unlink()
            manifest=json.loads((trial/'camera_capture.json').read_text());manifest['saved'].pop('initial_annotated');manifest['complete']=False
            (trial/'camera_capture.json').write_text(json.dumps(manifest))
            partial=audit.evaluate(trial)
            self.assertFalse(partial['image_capture_complete'])
            self.assertTrue(partial['minimum_image_evidence_complete'])
            self.assertEqual(partial['missing_optional_images'],['initial_annotated'])
            self.assertEqual(partial['missing_required_images'],[])


if __name__=='__main__':unittest.main()
