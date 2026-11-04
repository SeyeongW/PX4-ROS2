"""Proposal limits and interruption handling, without ROS or hardware."""
import math
import unittest

import numpy as np

from jol_perception.real_tracking import JOINT_NAMES, ProposalTracker


def target(stamp, error=(.8, -.6), selection=1):
    x, y = error
    return dict(stamp=stamp, width=100, height=100, valid=True, source='manual_roi_csrt',
                bbox=[50+50*x-5, 50+50*y-5, 10, 10], error_normalized=[x, y], selection_id=selection)


class RealTrackingTests(unittest.TestCase):
    def feed(self, core, now, error=(.8, -.6), q=(0., -.4, .3, 0.), selection=1, wall=None):
        wall = now if wall is None else wall
        self.assertTrue(core.receive_joints(now, JOINT_NAMES, q, now, wall))
        self.assertTrue(core.receive_target(target(now, error, selection), now, wall))
        return core.sample(now, wall)

    def test_requires_real_feedback_and_never_invents_positions(self):
        core = ProposalTracker()
        core.receive_target(target(10.), 10., 10.)
        status = core.sample(10., 10.)
        self.assertEqual(status['proposed_positions'], [])
        self.assertIsNone(status['measured_positions'])
        self.assertFalse(status['hardware_writes'])
        self.assertEqual(status['mapping'], 'arm_camera_axes_uncalibrated')
        self.assertEqual(status['camera_mount'], 'arm_user_reported')

    def test_slew_filter_and_fixed_neutral_bound_without_cumulative_drift(self):
        core = ProposalTracker()
        previous = np.array(self.feed(core, 10.)['proposed_positions'])
        neutral = previous.copy()
        for step in range(1, 201):
            status = self.feed(core, 10.+step*.05)
            q = np.array(status['proposed_positions'])
            self.assertLessEqual(max(abs(q-previous)), math.radians(2.)*.05+1e-12)
            self.assertLessEqual(max(abs(q-neutral)), math.radians(5.)+1e-12)
            previous = q
        self.assertTrue(np.allclose(q[[0, 3]], np.radians([4., -3.]), atol=1e-5))
        self.assertTrue(np.allclose(q[1:3], neutral[1:3]))
        # Even a new ROI after physical repositioning keeps the original neutral.
        status = self.feed(core, 20.1, q=(.02, -.4, .3, .01), selection=2)
        self.assertTrue(np.allclose(status['proposed_positions'], [.02, -.4, .3, .01]))
        self.assertTrue(np.allclose(status['neutral_positions'], neutral))

    def test_stale_gap_clears_filter_and_starts_from_latest_measurement(self):
        core = ProposalTracker()
        for step in range(21):
            self.feed(core, 10.+step*.05)
        self.assertGreater(core.proposed[0], 0.)
        status = core.sample(11.6, 11.6)
        self.assertEqual(status['proposed_positions'], [])
        self.assertIsNone(status['measured_positions'])
        self.assertTrue(np.array_equal(core.filtered, [0., 0.]))
        # Fresh joints alone cannot resurrect the prior target/proposal.
        core.receive_joints(11.7, JOINT_NAMES, [-.01, -.4, .3, .02], 11.7, 11.7)
        self.assertEqual(core.sample(11.7, 11.7)['proposed_positions'], [])
        status = self.feed(core, 11.8, error=(-.8, .6), q=(-.01, -.4, .3, .02))
        self.assertTrue(np.allclose(status['proposed_positions'], [-.01, -.4, .3, .02]))
        self.assertTrue(np.array_equal(status['filtered_error'], [0., 0.]))

    def test_wall_freshness_clear_target_and_deadband(self):
        core = ProposalTracker()
        self.feed(core, 10., error=(.03, -.03))
        status = self.feed(core, 10.05, error=(.03, -.03))
        self.assertTrue(np.allclose(status['proposed_positions'], [0., -.4, .3, 0.]))
        self.assertEqual(core.sample(10.05, 10.6)['proposed_positions'], [])
        self.feed(core, 10.7)
        self.assertTrue(core.receive_target(dict(stamp=10.71, valid=False, source='manual_roi_csrt'), 10.71, 10.71))
        self.assertEqual(core.sample(10.71, 10.71)['state'], 'no_target')
        self.assertIsNone(core.proposed)

    def test_rejects_future_bad_geometry_invalid_feedback_and_unknown_sources(self):
        for bad in [dict(target(10.), stamp=10.1), dict(target(10.), source='neural_selection'),
                    dict(target(10.), error_normalized=[float('nan'), 0.]),
                    dict(target(10.), bbox=[-1, 0, 10, 10]), dict(target(10.), error_normalized=[0., 0.])]:
            core = ProposalTracker()
            self.feed(core, 9.9)
            self.assertFalse(core.receive_target(bad, 10., 10.))
            self.assertEqual(core.sample(10., 10.)['proposed_positions'], [])
        core = ProposalTracker()
        self.feed(core, 10.)
        # Actual reclined arm q3=-1.7319 is outside the model limit: never
        # silently clip the measured pose into a motion proposal.
        for q in ([4., 0., 0., 0.], [0., 0., -1.7319, 0.]):
            self.assertFalse(core.receive_joints(10.1, JOINT_NAMES, q, 10.1, 10.1))
            status = core.sample(10.1, 10.1)
            self.assertIsNone(status['measured_positions'])
            self.assertEqual(status['proposed_positions'], [])


if __name__ == '__main__':
    unittest.main()
