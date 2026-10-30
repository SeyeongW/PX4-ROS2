"""공통 표적 입력 계약과 이격 경로의 실패 조건을 확인한다."""
import copy
import unittest

import numpy as np

from jol_perception.observation_node import prepare_observation, observation_inputs, _track_snapshot
from jol_perception.pointing_core import forecast
from jol_perception.tracking_core import Config, Track


class ObservationTests(unittest.TestCase):
    def inputs(self):
        track = dict(id=1, position=[18., 0., 3.], velocity=[0., 0., 0.],
                     state_covariance=(np.eye(6)*.01).tolist(), confirmed=True,
                     prediction=dict(acceleration_std=.1), last_seen=10.)
        return dict(status=dict(stamp=10., source_stamp=10., frame_id='world',
                         command_mode='follow', target_id=1, track_position=track['position'],
                         prediction=dict(forecast(track, 10., 10., .5), target_id=1, frame_id='world'),
                         turret_prediction=dict(forecast(track, 10., 10., 0.), target_id=1, frame_id='world')),
                    tracks=dict(stamp=10., frame_id='world', tracks=[track]),
                    own=dict(stamp=10., frame_id='world', position=[0., 0., 0.], velocity=[0., 0., 0.]),
                    scene=dict(frame_id='world', bounds=dict(min=[-30., -30., 0.], max=[30., 30., 15.]),
                               obstacles=[]), now=10.)

    def test_shared_target_protection_and_execution_boundary(self):
        inputs = self.inputs()
        plan = prepare_observation(**inputs)
        self.assertEqual(plan['execution_policy'], 'preview_only_requires_live_monitor')
        self.assertFalse(plan['dynamic_collision_checked'])
        self.assertTrue(plan['static_collision_checked'])
        self.assertEqual(plan['target_id'], 1)
        variance = inputs['status']['prediction']['position_covariance'][0][0]
        self.assertTrue(np.allclose(plan['goal'], [18.-10.-.75-.1-3*np.sqrt(variance), 0., 3.]))
        self.assertEqual(plan['stamp'], plan['planned_at'])
        self.assertFalse(plan['execution_allowed'])
        self.assertIn('not_independent_estimator', plan['own_pose_validation'])
        points = np.array([s['position'] for s in plan['samples']])
        self.assertGreater(np.linalg.norm(points-[18, 0, 3], axis=1).min(), 10.)
        self.assertEqual(prepare_observation(**inputs, fixed_scene=True)['execution_policy'],
                         'isolated_fixed_scene_sitl')

    def test_malformed_stale_or_inconsistent_packets_are_rejected(self):
        changes = [
            (('scene',), []), (('status', 'prediction'), None),
            (('scene', 'bounds'), []), (('tracks', 'tracks'), [None]),
            (('status', 'frame_id'), 'camera'), (('own', 'frame_id'), 'map'),
            (('status', 'source_stamp'), 9.9), (('status', 'track_position'), [19., 0., 3.]),
            (('status', 'prediction', 'target_id'), True),
            (('status', 'turret_prediction', 'horizon_s'), 1.),
            (('status', 'prediction', 'stamp'), 11.),
            (('own', 'position'), [False, False, False]),
            (('own', 'velocity'), [0., .2, 0.]), (('own',), None),
            (('tracks', 'stamp'), 10**1000), (('own', 'stamp'), 9.4),
            (('status', 'command_mode'), 'hold'),
        ]
        for path, value in changes:
            with self.subTest(path=path):
                inputs = self.inputs()
                target = inputs
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises((ValueError, TypeError)):
                    prepare_observation(**inputs)
        with self.assertRaises(ValueError):
            prepare_observation(**self.inputs(), fixed_scene='false')

    def test_all_obstacle_tracks_and_observation_times_are_validated(self):
        for change in ('duplicate', 'future_observation', 'unconfirmed', 'bad_covariance', 'moving_fixed'):
            with self.subTest(change=change):
                inputs = self.inputs()
                track = inputs['tracks']['tracks'][0]
                if change == 'duplicate':
                    inputs['tracks']['tracks'].append(copy.deepcopy(track))
                elif change == 'future_observation':
                    track['last_seen'] = 10.1
                elif change == 'unconfirmed':
                    track['confirmed'] = False
                elif change == 'bad_covariance':
                    track['state_covariance'][0][0] = -1.
                else:
                    track['velocity'][0] = 1.
                with self.assertRaises(ValueError):
                    prepare_observation(**inputs, fixed_scene=True)

    def test_entire_curve_and_other_tracks_are_checked(self):
        for change in ('static_obstacle', 'other_track', 'geofence', 'fused_current', 'fixed_position'):
            with self.subTest(change=change):
                inputs = self.inputs()
                if change == 'static_obstacle':
                    inputs['scene']['obstacles'] = [dict(min=[-.1, -.1, .4], max=[.1, .1, .6])]
                elif change == 'other_track':
                    track = copy.deepcopy(inputs['tracks']['tracks'][0])
                    track.update(id=2, position=[0., 0., 1.])
                    inputs['tracks']['tracks'].append(track)
                elif change == 'geofence':
                    inputs['scene']['bounds']['max'][0] = 6.
                elif change == 'fused_current':
                    # 융합 현재점이 원점 가까이에 있으면 보호 공간이 이륙을 차단해야 한다.
                    inputs['status']['turret_prediction']['position'] = [9., 0., 3.]
                else:
                    inputs['status']['prediction']['position'] = [19., 0., 3.]
                with self.assertRaises(ValueError):
                    prepare_observation(**inputs, fixed_scene=(change == 'fixed_position'))

    def test_current_and_forecast_uncertainty_both_expand_protected_volume(self):
        baseline = prepare_observation(**self.inputs())
        for field in ('prediction', 'turret_prediction'):
            inputs = self.inputs()
            inputs['status'][field]['position_covariance'] = (np.eye(3)*.25).tolist()
            plan = prepare_observation(**inputs)
            self.assertLess(plan['goal'][0], baseline['goal'][0])
            self.assertAlmostEqual(plan['obstacles'][-1]['min'][0], 18.-10.-1.5)
        inputs = self.inputs()
        inputs['status']['prediction']['position_covariance'] = (np.eye(3)*9.).tolist()
        with self.assertRaises(ValueError):
            prepare_observation(**inputs)  # 커진 불확실성으로 이륙점이 보호 공간에 들어간다.

    def test_rejects_invalid_estimator_metadata_and_source_covariance(self):
        for path, value in [
            (('own', 'pose_valid'), False), (('own', 'velocity_valid'), 1),
            (('own', 'pose_covariance'), (np.eye(6)*-1.).tolist()),
            (('status', 'prediction', 'position_covariance'), [[True]*3]*3),
            (('status', 'prediction', 'position_covariance'), [[.1, 0., 0.], [0., True, 0.], [0., 0., .1]]),
            (('status', 'prediction', 'position_covariance'), [[float('nan')]*3]*3),
            (('status', 'turret_prediction', 'position_covariance'), (np.eye(3)*-1.).tolist()),
            (('status', 'prediction', 'state_stamp'), 10.1),
            (('status', 'prediction', 'propagation_s'), .4),
            (('own', 'position'), ['0', 0., 0.]),
        ]:
            with self.subTest(path=path):
                inputs = self.inputs()
                target = inputs
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    prepare_observation(**inputs)

    def test_snapshot_tracks_geometry_inventory_covariance_and_pose_but_not_stamp(self):
        inputs = self.inputs()
        reference = observation_inputs(**inputs)['snapshot']
        shifted = copy.deepcopy(inputs)
        shifted['own']['stamp'] = 9.9
        self.assertEqual(observation_inputs(**shifted)['snapshot'], reference)
        for change in ('own_position', 'current_covariance', 'inventory', 'scene'):
            changed = copy.deepcopy(inputs)
            if change == 'own_position':
                changed['own']['position'][0] = .01
            elif change == 'current_covariance':
                changed['status']['turret_prediction']['position_covariance'][0][0] += .01
            elif change == 'inventory':
                other = copy.deepcopy(changed['tracks']['tracks'][0])
                other.update(id=2, position=[-10., -10., 3.])
                changed['tracks']['tracks'].append(other)
            else:
                changed['scene']['bounds']['max'][0] += 1
            self.assertNotEqual(observation_inputs(**changed)['snapshot'], reference)
        changed = copy.deepcopy(inputs['tracks'])
        changed['tracks'][0]['velocity'][0] = .1
        self.assertNotEqual(_track_snapshot(changed), _track_snapshot(inputs['tracks']))

    def test_latest_inventory_cannot_hide_invalid_observation_time(self):
        for last_seen in (10.1, 8.9, float('nan'), True):
            packet = self.inputs()['tracks']
            packet['tracks'][0]['last_seen'] = last_seen
            with self.assertRaises(ValueError):
                _track_snapshot(packet, 10.)
        packet = self.inputs()['tracks']
        packet['tracks'][0]['last_seen'] = 9.2
        with self.assertRaises(ValueError):
            _track_snapshot(packet, 10.3)

    def test_actual_track_packets_ignore_only_forecast_timestamp(self):
        config = Config(acceleration_std=.1, low_speed_acceleration_std=.1)
        track = Track(id=1, state=np.array([18., 0., 3., 0., 0., 0.]),
                      covariance=np.eye(6)*.01, size=np.ones(3), born=9., last_seen=10.,
                      hits=3, is_confirmed=True, process_acceleration_std=.1)
        inputs = []
        for stamp in (10., 10.1):
            data = self.inputs()
            track.last_seen = stamp
            item = track.as_dict(stamp, config=config)
            data.update(now=stamp)
            data['tracks'].update(stamp=stamp, tracks=[item])
            data['own']['stamp'] = stamp
            data['status'].update(stamp=stamp, source_stamp=stamp)
            for key, horizon in (('prediction', .5), ('turret_prediction', 0.)):
                data['status'][key] = dict(forecast(item, stamp, stamp, horizon),
                                           target_id=1, frame_id='world')
            inputs.append(data)
        first, second = inputs
        self.assertNotEqual(first['tracks']['tracks'][0]['prediction']['stamp'],
                            second['tracks']['tracks'][0]['prediction']['stamp'])
        self.assertEqual(_track_snapshot(first['tracks']), _track_snapshot(second['tracks']))
        self.assertEqual(observation_inputs(**first)['snapshot'], observation_inputs(**second)['snapshot'])
        # 시각 이외의 예측 의미가 달라지면 이전 미리보기와 같다고 보지 않는다.
        reference = _track_snapshot(second['tracks'])
        for key, value in [('model', 'different_model'), ('acceleration_std', .2),
                           ('process_noise_model', 'different_q'),
                           ('acceleration_spectral_density_m2_s3', .004),
                           ('horizon_s', 1.), ('position', [18.1, 0., 3.]),
                           ('position_covariance', (np.eye(3)*.02).tolist())]:
            with self.subTest(field=key):
                changed = copy.deepcopy(second['tracks'])
                changed['tracks'][0]['prediction'][key] = value
                self.assertNotEqual(_track_snapshot(changed), reference)
        with self.assertRaises(ValueError):
            _track_snapshot(second['tracks'], 10.7)
        # 비교 과정이 원본 메시지의 시각을 지우지 않아야 한다.
        self.assertEqual(second['tracks']['tracks'][0]['prediction']['stamp'], 10.6)


if __name__ == '__main__':
    unittest.main()
