"""Shared runtime types for the single precision-landing ROS node.

These definitions live outside the ROS facade so the target, mission and
command roles can depend on one direction-neutral phase contract without
importing the large node module or creating another ROS node.
"""

from enum import Enum


class LandingPhase(Enum):
    """Externally visible phases of the production landing mission."""

    WAITING = "waiting_for_px4_and_sensors"
    READY = "ready_inactive"
    PRESTREAM = "offboard_setpoint_prestream"
    ARMING = "arming_and_entering_offboard"
    TAKEOFF = "vertical_takeoff"
    APPROACH = "trailer_approach"
    MARKER_TRACK_DOWN = "marker_track_down"
    PRECISION_ALIGN = "precision_align"
    PRECISION_DESCENT = "precision_descent"
    FINAL_APPROACH = "final_approach"
    TOUCHDOWN_CONFIRM = "touchdown_confirm"
    LANDED = "landed"
    ABORT_CLIMB = "abort_climb"
    FAILSAFE_HOLD = "failsafe_hold"


ACQUISITION_GUIDANCE_CONTROLLER_TYPE = "aruco_acquisition_guidance"
