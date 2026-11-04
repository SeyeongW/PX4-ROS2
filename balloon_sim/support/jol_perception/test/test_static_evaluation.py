"""Offline truth surface semantics and planned/actual/NED evaluation checks."""
import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jol_perception.static_flight_core import build_static_plan, to_ned


source = Path(__file__).resolve().parents[3]/'tools/evaluate_static_flight.py'
spec = importlib.util.spec_from_file_location('evaluate_static_flight', source)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class StaticEvaluationTests(unittest.TestCase):
    def test_return_records_with_unobserved_internal_setpoint_keep_unknown_error(self):
        plan=build_static_plan([-2.,0.,.1],[37.,13.,8.])
        records=[dict(stamp=2.,own_stamp=1.99,phase='autopilot_landing',
                      position=[-2.,0.,.1],velocity=[0.,0.,0.],return_stage='landing')]
        events=[dict(stamp=1.,phase='takeoff'),dict(stamp=1.9,phase='autopilot_landing')]
        result=evaluation.flight_metrics(plan,records,events,dict(phase='aborted',arrived=False))
        self.assertEqual(result['telemetry_tracking_error_by_phase_diagnostic']['autopilot_landing']['samples'],0)
        self.assertFalse(result['arrived'])

    def gate_fixture(self):
        plan = dict(home=[0.,0.,.1],goal=[30.,0.,8.],target_world=[10.,0.,8.],
            gate=dict(center_world=[10.,0.,8.],normal_world=[1.,0.,0.],
                      right_world=[0.,1.,0.],up_world=[0.,0.,1.],
                      width_m=1.2,height_m=1.2,body_radius_m=.16))
        def physical(stamp,position):
            return dict(stamp=stamp,position=position,velocity=[0.,0.,0.],
                        source='gazebo_p3d_ground_truth_evaluation_only')
        return plan,physical

    def test_gate_success_requires_independent_forward_crossing_and_hold_after_it(self):
        plan,physical = self.gate_fixture()
        events = [dict(stamp=1.,phase='takeoff'),dict(stamp=3.,phase='arrived'),
                  dict(stamp=2.05,checkpoint=dict(outcome='passed'))]
        rows = [physical(.9,[.12,0.,.1]),physical(1.,[.12,0.,.1]),
                physical(2.,[9.5,.2,8.1]),physical(2.1,[10.5,.4,8.3]),
                physical(3.,[30.,0.,8.])]
        result = evaluation.physical_checkpoint_metrics(plan,rows,events)
        self.assertTrue(result['physical_gate_pass_verified'])
        self.assertTrue(result['physical_hold_behind_gate_verified'])
        self.assertTrue(result['checkpoint_success_and_hold_verified'])
        self.assertAlmostEqual(result['physical_gate_crossing_stamp'],2.05)
        self.assertAlmostEqual(result['physical_gate_lateral_offset_m'],.3)
        self.assertAlmostEqual(result['physical_gate_vertical_offset_m'],.2)
        self.assertAlmostEqual(result['takeoff_home_EKF_minus_P3D_world'][0],-.12)
        self.assertFalse(result['return_to_takeoff_requested'])
        self.assertIsNone(result['physical_return_landing_verified'])
        moving_hold = rows[:-1]+[dict(rows[-1],velocity=[1.,0.,0.])]
        moving = evaluation.physical_checkpoint_metrics(plan,moving_hold,events)
        self.assertTrue(moving['physical_gate_pass_verified'])
        self.assertFalse(moving['physical_hold_behind_gate_verified'])
        self.assertFalse(moving['checkpoint_success_and_hold_verified'])
        # A controller-reported pass cannot hide a physical rectangle miss.
        missed = [dict(row,position=[row['position'][0],.6,row['position'][2]])
                  if 2.<=row['stamp']<=2.1 else row for row in rows]
        result = evaluation.physical_checkpoint_metrics(plan,missed,events)
        self.assertFalse(result['physical_gate_pass_verified'])
        self.assertFalse(result['checkpoint_success_and_hold_verified'])
        self.assertEqual(result['physical_gate_check_reason'],'forward_rectangle_missed')
        missing = evaluation.physical_checkpoint_metrics(plan,[],events)
        self.assertIsNone(missing['physical_gate_pass_verified'])
        self.assertIsNone(missing['checkpoint_success_and_hold_verified'])

    def test_gate_post_abort_old_session_nearby_and_large_gap_cannot_fake_pass(self):
        plan,physical = self.gate_fixture()
        events = [dict(stamp=.5,phase='takeoff'),dict(stamp=.7,phase='arrived'),
                  dict(stamp=1.,phase='takeoff'),dict(stamp=2.05,phase='autopilot_landing')]
        rows = [physical(.6,[9.5,0.,8.]),physical(.7,[10.5,0.,8.]),
                physical(1.,[0.,0.,.1]),physical(2.,[9.5,0.,8.]),
                physical(2.1,[10.5,0.,8.])]
        result = evaluation.physical_checkpoint_metrics(plan,rows,events)
        self.assertFalse(result['physical_gate_pass_verified'])
        self.assertFalse(result['checkpoint_success_and_hold_verified'])
        self.assertFalse(result['physical_gate_crossings'])
        # Even with fresh XYZ values at either end, an unrecorded long gap
        # cannot prove the aircraft actually intersected the rectangle.
        events = [dict(stamp=1.,phase='takeoff')]
        large_gap = [physical(1.,[9.5,0.,8.]),physical(2.,[10.5,0.,8.])]
        self.assertFalse(evaluation.physical_checkpoint_metrics(plan,large_gap,events)['physical_gate_pass_verified'])
        nearby = [physical(1.,[9.9,0.,8.]),physical(1.1,[9.99,0.,8.])]
        self.assertFalse(evaluation.physical_checkpoint_metrics(plan,nearby,events)['physical_gate_pass_verified'])
        touched = [physical(1.,[9.5,0.,8.]),physical(1.05,[10.,0.,8.])]
        self.assertFalse(evaluation.physical_checkpoint_metrics(plan,touched,events)['physical_gate_pass_verified'])
        through = touched+[physical(1.1,[10.5,0.,8.])]
        self.assertTrue(evaluation.physical_checkpoint_metrics(plan,through,events)['physical_gate_pass_verified'])
        started_on_plane = touched[1:]+[physical(1.1,[10.5,0.,8.])]
        self.assertFalse(evaluation.physical_checkpoint_metrics(plan,started_on_plane,events)['physical_gate_pass_verified'])
        # EKF rows cannot substitute for independent physical truth.
        disguised = [dict(row,source='mavros_px4_ekf_feedback_not_physical_truth') for row in large_gap]
        self.assertIsNone(evaluation.physical_checkpoint_metrics(plan,disguised,events)['physical_gate_pass_verified'])

    def test_return_requires_physical_home_and_fresh_actual_landed_disarmed_feedback(self):
        plan,physical = self.gate_fixture()
        events = [dict(stamp=1.,phase='takeoff'),dict(stamp=2.,phase='autopilot_landing'),
                  dict(stamp=2.,action='return_to_takeoff',success=True),dict(stamp=4.,phase='aborted')]
        rows = [physical(.9,[.12,0.,.1]),physical(1.,[.12,0.,.1]),
                physical(2.,[5.,0.,8.]),physical(4.,[.20,.10,.10])]
        final = dict(stamp=4.,position=[0.,0.,.1],flight_backend='mavros_px4',
                     fcu_state_stamp=3.95,armed=False,landed=True,landed_state=1,
                     return_stage='complete',return_home_reached=True)
        result = evaluation.physical_checkpoint_metrics(plan,rows,events,[final])
        self.assertTrue(result['physical_return_home_verified'])
        self.assertTrue(result['physical_return_landing_verified'])
        self.assertTrue(result['return_landed_disarmed_MAVROS_verified'])
        self.assertAlmostEqual(result['physical_return_home_horizontal_error_m'],np.hypot(.2,.1))
        self.assertAlmostEqual(result['physical_return_launch_P3D_horizontal_error_m'],np.hypot(.08,.1))
        # Backend completion and disarmed state at the target do not prove a
        # return to the recorded takeoff XY.
        away = rows[:-1]+[physical(4.,[5.,0.,.1])]
        wrong = evaluation.physical_checkpoint_metrics(plan,away,events,[final])
        self.assertFalse(wrong['physical_return_home_verified'])
        self.assertFalse(wrong['physical_return_landing_verified'])
        for change in (dict(armed=True),dict(landed=False),dict(landed_state=0),dict(fcu_state_stamp=1.)):
            unproved = evaluation.physical_checkpoint_metrics(plan,rows,events,[dict(final,**change)])
            self.assertFalse(unproved['return_landed_disarmed_MAVROS_verified'])
            self.assertFalse(unproved['physical_return_landing_verified'])
        no_truth = evaluation.physical_checkpoint_metrics(plan,[],events,[final])
        self.assertIsNone(no_truth['physical_return_home_verified'])
        self.assertIsNone(no_truth['physical_return_landing_verified'])

    def test_physical_arrival_is_independent_of_ekf_and_missing_truth_stays_unverified(self):
        plan = dict(goal=[10.,0.,8.],target_world=[10.,0.,8.])
        events = [dict(stamp=1.,phase='takeoff'),dict(stamp=2.,phase='arrived')]
        physical = [dict(stamp=t,position=[10.6,0.,8.],source='gazebo_p3d_ground_truth_evaluation_only') for t in (1.,2.,3.)]
        ekf = [dict(stamp=t,position=[10.,0.,8.]) for t in (1.,3.)]
        result = evaluation.physical_arrival_metrics(plan,physical,events,ekf)
        self.assertTrue(result['arrival_event_observed'])
        self.assertFalse(result['physical_arrival_verified'])
        self.assertAlmostEqual(result['physical_goal_error_at_arrival_m'],.6)
        self.assertAlmostEqual(result['ekf_goal_error_at_arrival_m'],0.)
        self.assertAlmostEqual(result['ekf_vs_physical_error_at_arrival_m'],.6)
        self.assertAlmostEqual(result['ekf_vs_physical_preabort_position_error']['rmse_m'],.6)
        close = [dict(row,position=[10.2,0.,8.]) for row in physical]
        self.assertTrue(evaluation.physical_arrival_metrics(plan,close,events)['physical_arrival_verified'])
        missing = evaluation.physical_arrival_metrics(plan,[],events,ekf)
        self.assertIsNone(missing['physical_arrival_verified'])
        self.assertEqual(missing['physical_arrival_check_reason'],'independent_P3D_missing')
        uncovered = evaluation.physical_arrival_metrics(plan,[physical[0]],events,ekf)
        self.assertIsNone(uncovered['physical_arrival_verified'])
        self.assertEqual(uncovered['physical_arrival_check_reason'],'arrival_time_outside_fresh_P3D_coverage')
        gap=evaluation.physical_arrival_metrics(plan,[physical[0],physical[-1]],events,ekf)
        self.assertIsNone(gap['physical_arrival_verified'])
        self.assertEqual(gap['physical_arrival_check_reason'],'arrival_time_outside_fresh_P3D_coverage')

    def test_hold_requires_full_fresh_window_and_checks_drift_after_arrival(self):
        plan,physical=self.gate_fixture()
        rows=[physical(3.+index*.1,[30.1,0.,8.]) for index in range(51)]
        good=evaluation.physical_hold_window_metrics(plan,rows,3.,8.,5.)
        self.assertTrue(good['physical_hold_window_verified'])
        drift=[dict(row,position=[30.29,0.,8.]) if 5.<row['stamp']<6. else row for row in rows]
        bad=evaluation.physical_hold_window_metrics(plan,drift,3.,8.,5.)
        self.assertFalse(bad['physical_hold_window_verified'])
        self.assertAlmostEqual(bad['physical_hold_window_max_position_error_m'],.29)
        self.assertIsNone(evaluation.physical_hold_window_metrics(plan,rows[:-10],3.,8.,5.)['physical_hold_window_verified'])
        gap=[row for row in rows if not 4.<row['stamp']<5.]
        self.assertIsNone(evaluation.physical_hold_window_metrics(plan,gap,3.,8.,5.)['physical_hold_window_verified'])
        moving=[dict(row,velocity=[.3,0.,0.]) if row['stamp']==5. else row for row in rows]
        self.assertFalse(evaluation.physical_hold_window_metrics(plan,moving,3.,8.,5.)['physical_hold_window_verified'])
        not_arrived=evaluation.physical_hold_window_metrics(plan,rows,None,8.,5.)
        self.assertIsNone(not_arrived['physical_hold_window_start_stamp'])
        self.assertIsNone(not_arrived['physical_hold_window_end_stamp'])
        self.assertIsNone(not_arrived['physical_hold_window_max_position_error_m'])
        self.assertIsNone(not_arrived['physical_hold_window_max_speed_m_s'])
        self.assertEqual(not_arrived['physical_hold_window_samples'],0)

    def test_motor_control_evidence_is_retained_after_failed_mission_returns_home(self):
        plan=build_static_plan([-5.,-6.,.1],[39.4,0.,8.])
        events=[dict(stamp=1.,phase='takeoff'),dict(stamp=2.,phase='spline_tracking'),
                dict(stamp=3.,phase='autopilot_landing'),dict(stamp=4.,phase='aborted',reason='tracking_error_exceeded')]
        positions=[plan['home'],[-1.,-6.,3.1],plan['home']]
        physics=[dict(stamp=stamp,position=p,velocity=[0.,0.,0.],source='gazebo_p3d_ground_truth_evaluation_only')
                 for stamp,p in zip((1.,2.,4.),positions)]
        with tempfile.TemporaryDirectory() as directory:
            trial=Path(directory);(trial/'mission').mkdir()
            for name,packet in [('plan.json',plan),('events.json',events),('report.json',dict(flight_backend='mavros_px4',arrived=False))]:
                (trial/'mission'/name).write_text(json.dumps(packet))
            files={'physics_trace.jsonl':physics,
                'mavros_state_trace.jsonl':[dict(stamp=1.,armed=True,mode='OFFBOARD'),dict(stamp=4.,armed=False,mode='AUTO.LAND')],
                'mavros_pva_trace.jsonl':[dict(stamp=1.+index*.02) for index in range(10)]}
            for name,rows in files.items():(trial/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            plugins=''.join('<plugin filename="libgazebo_motor_model.so" name="motor%d"/>'%index for index in range(4))
            (trial/'scene.world').write_text('<sdf><world><model name="tailsitter_20cm_px4">'+plugins+
                '<plugin filename="libgazebo_mavlink_interface.so" name="mavlink"/></model></world></sdf>')
            result=evaluation.evaluate(trial,plots=False)
        flight=result['flight']
        self.assertTrue(flight['mavros_controller_path_verified'])
        self.assertTrue(flight['mavros_evidence']['independently_measured_motion'])
        self.assertAlmostEqual(flight['mavros_evidence']['maximum_physical_displacement_from_launch_m'],5.)
        self.assertFalse(flight['physical_arrival_verified'])
        self.assertNotIn('MAVROS_PX4_motor_control_path_not_fully_recorded',result['unverified'])

    def test_nearest_physical_goal_before_abort_does_not_include_automatic_landing_or_old_session(self):
        plan = dict(goal=[10.,0.,8.],target_world=[10.,0.,8.])
        events = [dict(stamp=0.,phase='takeoff'),dict(stamp=.5,phase='arrived'),
                  dict(stamp=1.,phase='takeoff'),dict(stamp=2.,phase='autopilot_landing'),dict(stamp=3.,phase='aborted')]
        physical = [dict(stamp=t,position=[x,0.,8.],source='gazebo_p3d_ground_truth_evaluation_only')
                    for t,x in ((.5,10.),(1.,0.),(1.5,6.),(2.,8.),(2.5,10.))]
        result = evaluation.physical_arrival_metrics(plan,physical,events)
        self.assertFalse(result['arrival_event_observed'])
        self.assertFalse(result['physical_arrival_verified'])
        self.assertEqual(result['physical_arrival_check_reason'],'arrival_event_missing')
        self.assertAlmostEqual(result['minimum_physical_target_distance_before_abort_m'],2.)
        self.assertAlmostEqual(result['last_physical_goal_distance_before_abort_m'],2.)
        self.assertEqual(result['physical_preabort_samples'],3)
        self.assertEqual(result['minimum_physical_target_distance_stamp'],2.)

    def test_moving_c2_join_compares_both_sides_and_whole_path_rest_only(self):
        import copy
        plan = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],controller_profile='tailsitter20cm_reference',
            continuous_takeoff=True,takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
            max_acceleration=3.5,max_jerk=12.,body_radius=.12,target_is_virtual_waypoint=True,standoff=0.)
        joins = evaluation.plan_join_checks(plan)
        self.assertTrue(joins['c2_join_passed'])
        self.assertAlmostEqual(joins['max_join_speed_m_s'],1.)
        self.assertLess(joins['endpoint_velocity_acceleration_max'],1e-10)
        broken = copy.deepcopy(plan)
        # Change the curve entry speed while preserving monotone timing and a=0.
        broken['segments'][1]['time_control_points'][1] += .01
        broken['segments'][1]['time_control_points'][2] += .02
        failed = evaluation.plan_join_checks(broken)
        self.assertFalse(failed['c2_join_passed'])
        self.assertGreater(failed['max_velocity_jump_m_s'],.01)

    def test_recorded_progress_clock_is_interpolated_at_physics_time_and_exported_separately(self):
        from jol_perception.observation_path import sample_plan
        plan = build_static_plan([-2.,0.,.1],[36.99,13.52,8.],controller_profile='tailsitter20cm_reference',
            continuous_takeoff=True,takeoff_join_speed=1.,takeoff_min_duration=4.,max_speed=9.,
            max_acceleration=3.5,max_jerk=12.,body_radius=.12,target_is_virtual_waypoint=True,standoff=0.)
        plan['reference_clock_policy'] = 'measured_EKF_path_progress'
        launch = 100.
        events = [dict(stamp=launch,phase='takeoff'),dict(stamp=101.2,phase='spline_tracking',plan_time_s=4.)]
        references = [(100.,0.),(100.2,.8),(101.,3.5),(101.4,4.5),(102.,5.5)]
        telemetry = []
        for stamp,clock in references:
            sample = sample_plan(plan,clock)
            telemetry.append(dict(stamp=stamp,own_stamp=stamp,reference_time_s=clock,
                phase='takeoff' if clock<4. else 'spline_tracking',position=sample['position'],
                velocity=sample['velocity'],desired_position=sample['position'],desired_velocity=sample['velocity'],
                desired_acceleration=sample['acceleration'],tracking_error_m=0.))
        physical_clocks = [(100.1,.4),(100.6,2.15),(101.1,3.75),(101.7,5.)]
        physics = [dict(stamp=stamp,source='gazebo_p3d_ground_truth_evaluation_only',
                        **{field:sample_plan(plan,clock)[field] for field in ('position','velocity')})
                   for stamp,clock in physical_clocks]
        actual,source = evaluation.acquisition_motion_records(plan,physics,events,telemetry)
        self.assertEqual(source,'gazebo_p3d_ground_truth_evaluation_only')
        np.testing.assert_allclose([row['trajectory_time_s'] for row in actual],np.array(physical_clocks)[:,1])
        self.assertTrue(all(row['tracking_error_m']<1e-10 for row in actual))
        self.assertGreater(max(row['nominal_clock_tracking_error_m_diagnostic'] for row in actual),1.)
        metrics = evaluation.flight_metrics(plan,telemetry,events,{},physics)
        self.assertEqual(metrics['recorded_reference_clock_missing_samples'],0)
        self.assertEqual(metrics['tracking_reference_method'],'recorded_command_reference_time_interpolated_at_acquisition_stamp')
        self.assertIsNone(evaluation.trajectory_time(plan,events,launch,99.9,telemetry))
        self.assertIsNone(evaluation.trajectory_time(plan,events,launch,100.5,[]))
        missing = evaluation.acquisition_motion_records(plan,physics,events,[])[0]
        self.assertTrue(all(row['trajectory_time_s'] is None and 'tracking_error_m' not in row for row in missing))
        exporter_spec = importlib.util.spec_from_file_location('static_export_progress_test',
            Path(__file__).resolve().parents[3]/'tools/export_static_flight.py')
        exporter = importlib.util.module_from_spec(exporter_spec);exporter_spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)/'trial';mission=trial/'mission';mission.mkdir(parents=True)
            for name,value in [('plan.json',plan),('events.json',events),('report.json',dict(phase='spline_tracking'))]:
                (mission/name).write_text(json.dumps(value))
            for path,rows in [(mission/'telemetry.jsonl',telemetry),(trial/'physics_trace.jsonl',physics)]:
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            output = Path(directory)/'export'
            with patch.dict(sys.modules,{'evaluate_static_flight':evaluation}),patch.object(exporter,'plot'):
                metadata = exporter.export(trial,output)
            with (output/'recorded_actual_path.csv').open(encoding='utf-8-sig') as stream:csv_actual=list(csv.DictReader(stream))
            with (output/'planned_path.csv').open(encoding='utf-8-sig') as stream:csv_plan=list(csv.DictReader(stream))
        np.testing.assert_allclose([float(row['trajectory_time_s']) for row in csv_actual],np.array(physical_clocks)[:,1])
        np.testing.assert_allclose([float(row['time_s']) for row in csv_actual],np.array(physical_clocks)[:,0]-launch)
        self.assertTrue(all(float(row['tracking_error_m'])<1e-10 for row in csv_actual))
        self.assertAlmostEqual(float(csv_plan[-1]['time_s']),plan['duration_s'])
        self.assertIn('recorded command reference clock',metadata['time_policy']['recorded_actual_csv'])
        self.assertTrue(metadata['target_is_virtual_waypoint'])

    def test_export_time_curves_draw_recorded_takeoff_hold_without_altering_raw_plan_csv(self):
        from jol_perception.observation_path import sample_plan
        from matplotlib.figure import Figure
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        launch = 100.; first = plan['segments'][0]['duration_s']; hold = 5.
        midpoint = (plan['duration_s']-first)/2
        clocks = [(0.,0.), (first-.1,first-.1), (first+1.,first), (first+4.,first),
                  (first+hold,first), (first+hold+midpoint,first+midpoint)]
        events = [dict(stamp=launch, phase='takeoff'),
                  dict(stamp=launch+first+hold, phase='spline_tracking', plan_time_s=first,
                       takeoff_settle_extra_s=hold)]
        physics, telemetry = [], []
        for elapsed, trajectory in clocks:
            sample = sample_plan(plan, trajectory)
            stamp = launch+elapsed
            phase = 'takeoff' if elapsed < first+hold else 'spline_tracking'
            physics.append(dict(stamp=stamp, position=sample['position'], velocity=sample['velocity'],
                                source='gazebo_p3d_ground_truth_evaluation_only'))
            telemetry.append(dict(stamp=stamp, own_stamp=stamp, phase=phase,
                position=sample['position'], velocity=sample['velocity'], desired_position=sample['position'],
                desired_velocity=sample['velocity'], desired_acceleration=sample['acceleration'], tracking_error_m=0.))
        export_spec = importlib.util.spec_from_file_location('static_export_clock_test',
            Path(__file__).resolve().parents[3]/'tools/export_static_flight.py')
        exporter = importlib.util.module_from_spec(export_spec); export_spec.loader.exec_module(exporter)
        figures = {}
        def capture(figure, path, *args, **kwargs):
            figures[Path(path).name] = figure
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)/'trial'; mission = trial/'mission'; mission.mkdir(parents=True)
            for filename, packet in [('plan.json',plan), ('events.json',events), ('report.json',dict(phase='spline_tracking'))]:
                (mission/filename).write_text(json.dumps(packet))
            for path, rows in [(mission/'telemetry.jsonl',telemetry), (trial/'physics_trace.jsonl',physics)]:
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            output = Path(directory)/'export'
            with patch.dict(sys.modules, {'evaluate_static_flight':evaluation}), \
                    patch.object(Figure, 'savefig', autospec=True, side_effect=capture):
                metadata = exporter.export(trial, output)
            with (output/'recorded_actual_path.csv').open(encoding='utf-8-sig') as stream:
                actual = list(csv.DictReader(stream))
            with (output/'planned_path.csv').open(encoding='utf-8-sig') as stream:
                planned = list(csv.DictReader(stream))
        elapsed, trajectory = np.array(clocks).T
        np.testing.assert_allclose([float(row['time_s']) for row in actual], elapsed)
        np.testing.assert_allclose([float(row['trajectory_time_s']) for row in actual], trajectory)
        self.assertEqual(len(planned), len(plan['samples']))
        self.assertAlmostEqual(float(planned[-1]['time_s']), plan['duration_s'])
        positions = figures['position_xyz.png'].axes
        for axis, component in zip(positions, range(3)):
            line = axis.lines[0]
            np.testing.assert_allclose(line.get_xdata(), elapsed)
            np.testing.assert_allclose(line.get_ydata(), [row['desired_position'][component] for row in telemetry])
            self.assertIn('recorded handoff', line.get_label())
        for axis, field in zip(figures['velocity_acceleration.png'].axes, ('velocity','acceleration')):
            np.testing.assert_allclose(axis.lines[0].get_xdata(), elapsed)
            np.testing.assert_allclose(axis.lines[0].get_ydata(),
                                       [np.linalg.norm(row['desired_'+field]) for row in telemetry], atol=1e-12)
        self.assertEqual(positions[-1].get_xlabel(), 'Time since takeoff start [s]')
        self.assertIn('Original trajectory time', metadata['time_policy']['planned_csv'])

    def test_measured_takeoff_hold_uses_shifted_curve_clock_for_physical_tracking_error(self):
        from jol_perception.observation_path import sample_plan
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        launch = 100.; first = plan['segments'][0]['duration_s']; hold = 5.
        transfer = launch+first+hold
        midpoint = (plan['duration_s']-first)/2
        events = [dict(stamp=launch, phase='takeoff'),
                  dict(stamp=transfer, phase='spline_tracking', plan_time_s=first,
                       takeoff_settle_extra_s=hold)]
        physics = []
        for elapsed, trajectory in [(0.,0.), (first-.1,first-.1), (first+1.,first),
                                     (first+4.,first), (first+hold,first),
                                     (first+hold+midpoint,first+midpoint)]:
            sample = sample_plan(plan, trajectory)
            stamp = launch+elapsed
            self.assertAlmostEqual(evaluation.trajectory_time(plan, events, launch, stamp), trajectory)
            physics.append(dict(stamp=stamp, position=sample['position'], velocity=sample['velocity'],
                                source='gazebo_p3d_read_only_world_pose'))
        actual, source = evaluation.acquisition_motion_records(plan, physics, events, [])
        self.assertEqual(source, 'gazebo_p3d_read_only_world_pose')
        self.assertTrue(all(row['tracking_error_m'] < 1e-10 for row in actual))
        naive = sample_plan(plan, physics[-1]['stamp']-launch)
        self.assertGreater(np.linalg.norm(np.array(naive['position'])-physics[-1]['position']), .5)
        metrics = evaluation.flight_metrics(plan, [], events, {}, physics)
        self.assertAlmostEqual(metrics['tracking_error_by_phase']['spline_tracking']['rmse_m'], 0.)
        self.assertAlmostEqual(metrics['actual_height_at_spline_transition_m'], 3.)
        self.assertEqual(metrics['takeoff_settle_extra_s'], hold)

    def test_p3d_physics_overrides_ekf_trace_and_keeps_home_bias_and_missing_mavros_evidence_explicit(self):
        from jol_perception.observation_path import sample_plan
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        launch = 100.; first = plan['segments'][0]['duration_s']
        offset = np.array([0.,0.,.2])
        events = [dict(stamp=launch, phase='takeoff'),
                  dict(stamp=launch+first, phase='spline_tracking', plan_time_s=first),
                  dict(stamp=launch+plan['duration_s'], phase='arrived')]
        telemetry, physics, controller = [], [], []
        for elapsed in (0., first, plan['duration_s']):
            sample = sample_plan(plan, elapsed); stamp = launch+elapsed
            phase = 'takeoff' if elapsed == 0. else 'arrived' if elapsed == plan['duration_s'] else 'spline_tracking'
            telemetry.append(dict(stamp=stamp, own_stamp=stamp, phase=phase,
                position=sample['position'], velocity=sample['velocity'],
                desired_position=sample['position'], desired_velocity=sample['velocity'],
                desired_acceleration=sample['acceleration'], tracking_error_m=0.))
            physics.append(dict(stamp=stamp, position=(np.array(sample['position'])+offset).tolist(),
                                velocity=sample['velocity'], source='gazebo_p3d_read_only_world_pose'))
            controller.append(dict(stamp=stamp, position=sample['position'], velocity=sample['velocity'],
                                   armed=True, mode='OFFBOARD', source='mavros_px4_ekf'))
        report = dict(phase='arrived', arrived=True, flight_backend='mavros_px4', own_pose_source='mavros_px4_ekf')
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory); mission = trial/'mission'; mission.mkdir()
            for name, packet in [('plan.json',plan), ('events.json',events), ('report.json',report)]:
                (mission/name).write_text(json.dumps(packet))
            for path, rows in [(mission/'telemetry.jsonl',telemetry), (trial/'physics_trace.jsonl',physics),
                               (trial/'controller_trace.jsonl',controller)]:
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = evaluation.evaluate(trial, plots=False)
            flight = result['flight']
        self.assertEqual(flight['actual_motion_source'], 'gazebo_p3d_read_only_world_pose')
        self.assertAlmostEqual(flight['physical_vertical_rise_at_spline_transition_m'], 3.)
        self.assertAlmostEqual(flight['actual_height_at_spline_transition_m'], 3.2)
        self.assertAlmostEqual(flight['home_registration_error_m'], .2)
        self.assertAlmostEqual(flight['ekf_world_position_error']['rmse_m'], .2)
        self.assertAlmostEqual(flight['tracking_error_by_phase']['spline_tracking']['rmse_m'], .2)
        self.assertAlmostEqual(flight['telemetry_tracking_error_by_phase_diagnostic']['spline_tracking']['rmse_m'], 0.)
        # EKF metadata alone is not an observation of the MAVROS control chain.
        evidence = flight['mavros_evidence']
        self.assertFalse(evidence['armed_offboard_observed'])
        self.assertEqual(evidence['state_messages'], 0)
        self.assertEqual(evidence['position_target_messages'], 0)
        self.assertIsNone(evidence['median_position_target_rate_hz'])
        self.assertIsNone(evidence['final_mode'])
        self.assertIsNone(evidence['final_armed'])
        self.assertFalse(flight['mavros_controller_path_verified'])
        self.assertIn('MAVROS_PX4_motor_control_path_not_fully_recorded', result['unverified'])

    def test_missing_independent_physics_cannot_report_zero_ekf_localization_error(self):
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        events = [dict(stamp=10., phase='takeoff')]
        controller = [dict(stamp=10.+index, position=[-5., -6., .1+index], velocity=[0.,0.,1.],
                           source='mavros_px4_ekf', armed=True, mode='OFFBOARD') for index in (0,1)]
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory); mission = trial/'mission'; mission.mkdir()
            for name, packet in [('plan.json',plan), ('events.json',events),
                                 ('report.json',dict(flight_backend='mavros_px4'))]:
                (mission/name).write_text(json.dumps(packet))
            (trial/'controller_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in controller))
            result = evaluation.evaluate(trial, plots=False)
            flight = result['flight']
        self.assertEqual(flight['actual_motion_source'], 'mavros_px4_ekf_feedback_not_physical_truth')
        self.assertEqual(flight['ekf_world_position_error']['samples'], 0)
        self.assertNotIn('rmse_m', flight['ekf_world_position_error'])
        self.assertFalse(flight['mavros_controller_path_verified'])
        self.assertIn('independent_physical_truth_missing_EKF_is_not_ground_truth', result['unverified'])

    def test_repeated_session_uses_latest_takeoff_and_does_not_inherit_arrival(self):
        from jol_perception.observation_path import sample_plan
        plan = build_static_plan([10., 3., .1], [39.4, 0., 8.])
        launch = 200.
        takeoff_duration = plan['segments'][0]['duration_s']
        old_events = [dict(stamp=1., phase='takeoff'), dict(stamp=8., phase='spline_tracking'),
                      dict(stamp=95., phase='arrived'), dict(stamp=96., action='abort', success=True),
                      dict(stamp=110., action='reset', success=True)]
        events = old_events + [dict(stamp=launch, phase='takeoff'),
            dict(stamp=launch+takeoff_duration, phase='spline_tracking'),
            dict(stamp=launch+takeoff_duration+.1, phase='abort_braking'),
            dict(stamp=launch+takeoff_duration+1., phase='aborted', reason='controller_failure')]
        old = dict(stamp=8., phase='spline_tracking', position=[-5., -6., 20.],
                   velocity=[15., 0., 0.], tracking_error_m=100.)
        current = []
        for elapsed in (0., takeoff_duration/2, takeoff_duration, takeoff_duration+.05):
            row = sample_plan(plan, elapsed)
            current.append(dict(stamp=launch+elapsed, own_stamp=launch+elapsed, phase=row['phase'],
                position=row['position'], velocity=row['velocity'], desired_position=row['position'],
                desired_velocity=row['velocity'], desired_acceleration=row['acceleration'], tracking_error_m=0.))
        physics = [old]+current
        metrics = evaluation.flight_metrics(plan, [old]+current, events, dict(arrived=True), physics)
        self.assertEqual(metrics['launch_stamp'], launch)
        self.assertEqual(metrics['actual_samples'], len(current))
        self.assertAlmostEqual(metrics['actual_height_at_spline_transition_m'], 3.)
        self.assertAlmostEqual(metrics['tracking_error_by_phase']['spline_tracking']['max_m'], 0.)
        self.assertEqual(metrics['telemetry_tracking_error_by_phase_diagnostic']['spline_tracking']['max_m'], 0.)
        self.assertFalse(metrics['arrived'])
        self.assertFalse(metrics['controller_guard_checks']['abort_accepted'])
        # Old unstamped telemetry cannot downgrade a fresh, acquisition-stamped run.
        latest, source = evaluation.acquisition_motion_records(plan, [], events, [old]+current)
        self.assertEqual(source, 'mission_own_odometry_acquisition_stamp')
        self.assertEqual(len(latest), len(current))
        legacy, source = evaluation.acquisition_motion_records(plan, [], events,
            [{key: value for key, value in row.items() if key != 'own_stamp'} for row in [old]+current])
        self.assertTrue(source.startswith('legacy_'))
        self.assertEqual(len(legacy), len(current))
        # Exported actual and command CSV use the same latest flight and zero.
        export_spec = importlib.util.spec_from_file_location('static_export_under_test',
            Path(__file__).resolve().parents[3]/'tools/export_static_flight.py')
        exporter = importlib.util.module_from_spec(export_spec)
        export_spec.loader.exec_module(exporter)
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)/'trial'; mission = trial/'mission'; mission.mkdir(parents=True)
            for filename, value in (('plan.json', plan), ('events.json', events), ('report.json', dict(arrived=True))):
                (mission/filename).write_text(json.dumps(value))
            (mission/'telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in [old]+current))
            (trial/'controller_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in physics))
            output = Path(directory)/'export'
            with patch.dict(sys.modules, {'evaluate_static_flight': evaluation}), patch.object(exporter, 'plot'):
                metadata = exporter.export(trial, output)
            with (output/'recorded_actual_path.csv').open(encoding='utf-8-sig') as stream:
                actual_csv = list(csv.DictReader(stream))
            with (output/'mission_telemetry.csv').open(encoding='utf-8-sig') as stream:
                telemetry_csv = list(csv.DictReader(stream))
            self.assertEqual(len(actual_csv), len(current))
            self.assertEqual(len(telemetry_csv), len(current))
            self.assertEqual(float(actual_csv[0]['time_s']), 0.)
            self.assertEqual(float(actual_csv[0]['acquisition_stamp']), launch)
            self.assertEqual(metadata['flight_launch_stamp'], launch)
            self.assertFalse(metadata['report']['arrived'])

    def test_command_timestamp_jitter_does_not_create_physical_acceleration_spike(self):
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        events = [dict(stamp=1., phase='takeoff')]
        physics = [dict(stamp=stamp, position=[-5., -6., .4], velocity=[0., 0., speed])
                   for stamp, speed in ((1., 0.), (1.1, .04), (1.2, .08))]
        telemetry = [dict(row, stamp=stamp, phase='takeoff', tracking_error_m=0.)
                     for row, stamp in zip(physics, (1., 1.01, 1.2))]
        result = evaluation.flight_metrics(plan, telemetry, events, dict(arrived=False), physics)
        self.assertAlmostEqual(result['actual_motion_by_phase']['takeoff']['max_acceleration_fd_m_s2'], .4)
        self.assertAlmostEqual(result['telemetry_motion_by_phase_diagnostic']['takeoff']['max_acceleration_fd_m_s2'], 4.)
        self.assertTrue(result['free_flight_motion_limits_passed'])
        self.assertEqual(result['actual_motion_source'], 'gazebo_controller_physics_stamp_and_world_velocity')
        stamped = [dict(row, own_stamp=physical['stamp']) for row, physical in zip(telemetry, physics)]
        own_result = evaluation.flight_metrics(plan, stamped, events, dict(arrived=False))
        self.assertAlmostEqual(own_result['actual_motion_by_phase']['takeoff']['max_acceleration_fd_m_s2'], .4)
        self.assertEqual(own_result['actual_motion_source'], 'mission_own_odometry_acquisition_stamp')

    def test_actual_jerk_and_abort_boundary_can_fail_despite_passing_each_phase_acceleration(self):
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        events = [dict(stamp=1.,phase='takeoff'), dict(stamp=1.06,phase='abort_braking')]
        physics = [dict(stamp=stamp,position=[-5.,-6.,1.],velocity=[speed,0.,0.])
            for stamp,speed in ((1.,0.),(1.02,.008),(1.04,.016),(1.06,.008),(1.08,0.))]
        metrics=evaluation.flight_metrics(plan,[],events,{},physics)
        self.assertTrue(metrics['nominal_free_flight_motion_limits_passed'])
        self.assertTrue(metrics['whole_free_flight_motion']['within_acceleration_limit'])
        self.assertFalse(metrics['free_flight_motion_limits_passed'])
        self.assertAlmostEqual(metrics['whole_free_flight_motion']['max_jerk_fd_m_s3'],40.)
        self.assertEqual(metrics['whole_free_flight_motion']['phase_boundary_acceleration_samples'],1)
        self.assertEqual(metrics['whole_free_flight_motion']['phase_boundary_jerk_samples'],2)
        self.assertIn('abort_braking',metrics['whole_free_flight_motion']['jerk_peak']['phases'])

    def test_near_ground_exclusion_never_bridges_an_omitted_sample(self):
        records=[dict(stamp=t,phase='abort_landing',position=[0.,0.,z],velocity=[v,0.,0.])
            for t,z,v in ((1.,1.,0.),(1.02,.1,10.),(1.04,1.,0.),(1.06,1.,.008))]
        metrics=evaluation.motion_summary(records,1.5,.6,1.2,minimum_height=.3,include_cross_phase=True)
        self.assertEqual(metrics['acceleration_samples'],1)
        self.assertEqual(metrics['jerk_samples'],0)
        self.assertAlmostEqual(metrics['max_acceleration_fd_m_s2'],.4)
        self.assertEqual(metrics['excluded_near_ground_samples'],1)

    def test_surface_comparison_does_not_mislabel_center_offset_as_sensor_error(self):
        estimate = dict(valid=True, stamp=1., position=[39.4, 0., 8.], target_id=1,
            nearest_distance_m=39.4, evidence=dict(camera_origin_world=[0., 0., 8.], camera_ray_world=[1., 0., 0.]))
        truth = dict(stamp=1., frame_id='world', lidar_origin=[0., 0., 8.],
                     objects=[dict(id=1, position=[40., 0., 8.], size=[1.2]*3)])
        metrics, _ = evaluation.static_sensor_metrics([estimate], [truth])
        self.assertAlmostEqual(metrics['ray_surface_error']['rmse_m'], 0.)
        self.assertAlmostEqual(metrics['box_surface_error']['rmse_m'], 0.)
        self.assertAlmostEqual(metrics['center_difference']['rmse_m'], .6)
        self.assertIsNone(evaluation.ray_box_entry([0., 0., 20.], [1., 0., 0.], [40., 0., 8.], [1.2]*3))

    def test_end_to_end_synthetic_trace_plan_join_height_and_ned_roundtrip(self):
        plan = build_static_plan([-5., -6., .1], [39.4, 0., 8.])
        alternate = build_static_plan([-5., -6., .1], [39.4, 0., 8.], path_kind='right_angle')
        yaw = .37
        controller = dict(controller_ready=True, setpoint_frame='jol/flight_ned',
                          origin_world=plan['home'], north_yaw_world_rad=yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        plan['ned_reference'] = dict(origin_world=plan['home'], north_yaw_world_rad=yaw,
                                    rotation_world_to_ned=[[c,s,0.],[s,-c,0.],[0.,0.,-1.]])
        duration = plan['segments'][0]['duration_s']
        records = [dict(stamp=10.+row['time_s'], phase='takeoff' if row['time_s'] < duration else 'spline_tracking',
            position=row['position'], velocity=row['velocity'], desired_position=row['position'],
            desired_velocity=row['velocity'], desired_acceleration=row['acceleration'], tracking_error_m=0.,
            command_ned=to_ned(row['position'], controller).tolist()) for row in plan['samples']]
        records[-1]['phase'] = 'arrived'
        events = [dict(stamp=10.+duration, phase='spline_tracking'),
                  dict(stamp=9., action='launch', success=False, message='launch_not_ready_or_duplicate_launch')]
        report = dict(arrived=True, phase='arrived', target_source='camera_bbox_lidar',
                      own_pose_source='ideal_gazebo_self_odometry', native_setpoint_frame='jol/flight_ned')
        metrics = evaluation.flight_metrics(plan, records, events, report)
        self.assertTrue(metrics['join_checks']['c2_join_passed'])
        self.assertTrue(metrics['planned_3m_takeoff_passed'])
        self.assertAlmostEqual(metrics['actual_height_at_spline_transition_m'], 3.)
        self.assertTrue(metrics['ned_command_roundtrip_passed'])
        landing_position=list(plan['goal']);landing_position[2]=.1
        landed=dict(records[-1],stamp=records[-1]['stamp']+1.,phase='abort_landing',
            position=landing_position,desired_position=landing_position,
            command_ned=to_ned(landing_position,controller).tolist())
        after_landing=evaluation.flight_metrics(plan,records+[landed],events,report)
        self.assertAlmostEqual(after_landing['endpoint_goal_error_m'],0.)
        self.assertAlmostEqual(after_landing['endpoint_target_surface_distance_m'],1.)
        self.assertGreater(after_landing['last_record_to_goal_error_m'],7.)
        comparison = evaluation.plan_comparison(plan, alternate)
        self.assertTrue(comparison['same_goal'] and comparison['same_limits'])
        with tempfile.TemporaryDirectory() as name:
            trial = Path(name); (trial/'mission').mkdir()
            for filename, value in (('plan.json',plan),('alternate_plan.json',alternate),('events.json',events),('report.json',report)):
                (trial/'mission'/filename).write_text(json.dumps(value))
            (trial/'mission/telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            result = evaluation.evaluate(trial, plots=False)
            self.assertTrue(result['flight']['arrived'])
            self.assertIn('collision_contacts_not_recorded', result['unverified'])
            self.assertTrue((trial/'validation.json').is_file())


if __name__ == '__main__':
    unittest.main()
