---
name: feedback_gentle_control
description: Keep control gains LOW and motion gentle — high P gain / aggressive accel is unacceptable for real hardware
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8420beaf-12bc-4574-81f6-c789a51152bb
  modified: 2026-07-23T05:44:21.975Z
---

Control must stay **gentle**: never raise position/P gains to buy tracking
performance, and never let the vehicle command abrupt acceleration or
deceleration. Stated by the user on 2026-07-23 while tuning the moving-platform
landing.

**Why:** on real hardware an aggressive command is not just uncomfortable — a
sudden accel/decel pitches the airframe hard and can be catastrophic (loss of
control, marker leaving the camera FOV, structural/battery sag). SITL hides this
because there is no real airframe to upset.

**How to apply:**
- Prefer feed-forward (position+velocity+acceleration references from the MPC
  plan) over high-gain feedback. Smooth dense references let a LOW P gain work.
- Keep the jerk limit on: `j_max*dt` = accel change per step, measured knee
  0.2 m/s² (see [[project_landing_mpc]]).
- To cut overshoot, lower the approach speed (`v_max`) rather than raising
  position weights (`w_xy`) — raising the weight is the "high P gain" move and
  is what makes it snap.
- The PD fallback (`kp_fb`) must stay velocity-saturated and low-gain too.
