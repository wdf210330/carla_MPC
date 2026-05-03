# MPC Obstacle-Avoidance Tuning History

This file records the parameter-tuning path for the `two_obstacles` experiment.
It is meant to preserve what we changed, why we changed it, and what happened in
the logs after each adjustment.

## Baseline Goal

- Keep the original MPC structure.
- Avoid obstacles using the soft obstacle field only.
- Do not use reference-path offsetting.
- Make the behavior generic enough to handle multiple obstacles, not just one special case.

## Key Files

- `C:\Users\Administrator\Desktop\carla_MPC\carla_MPC-main2\src\mcp_controller.py`
- `C:\Users\Administrator\Desktop\carla_MPC\carla_MPC-main2\src\x_v2x_agent.py`
- `C:\Users\Administrator\Desktop\carla_MPC\carla_MPC-main2\two_obstacles\test_main_two_obstacles.py`
- `C:\Users\Administrator\Desktop\carla_MPC\carla_MPC-main2\two_obstacles\logs\test_main_two_obstacles.log`

## Diagnostic Conventions

- `obs*_ahead_m`: obstacle position in ego longitudinal coordinates.
- `obs*_lat_m`: obstacle position in ego lateral coordinates.
- `speed_mode`: speed policy branch selected in the test script.
- `actual_vx`: true CARLA longitudinal speed.
- `pred_vx`: predicted speed from the optimizer.

## Timeline

### 1. Initial setup

- Built `two_obstacles/test_main_two_obstacles.py`.
- Removed reference-path obstacle offsetting and used pure soft-field MPC.
- Added dedicated logging under `two_obstacles/logs`.

Observed:

- Vehicle often stopped near obstacles.
- It was unclear whether the stop came from collision, solver freeze, or local-minimum behavior.

### 2. Added detailed runtime logging

- Added `pred_*` and `actual_*` states.
- Added obstacle distances and ego-frame obstacle geometry.
- Added collision sensor logging.

Observed:

- Some earlier “mystery stops” were not actual collisions.
- The car often stopped with the obstacle still in front, indicating a local minimum.

### 3. Fixed hard implementation issues

- Fixed the soft-obstacle distance bug (`d3` using the wrong circle center).
- Fixed the sign of `obstacle_ahead` in the soft-field logic.
- Fixed yaw-wrap issues in logging/control interpretation.
- Added behind-waypoint pruning in `x_v2x_agent.py`.

Observed:

- Large erroneous steering jumps were reduced.
- Predictor behavior became more stable.

### 4. Side selection and rear release

- Added preferred passing-side locking in `x_v2x_agent.py`.
- Extended obstacle data with a preferred-side flag.
- Updated `mcp_controller.py` so side cost can follow a fixed passing side.
- Reduced influence from obstacles that have already moved behind the ego vehicle.

Observed:

- The second obstacle no longer randomly switched passing side.
- Reverse-looking predicted trajectories after passing were reduced.

### 5. Generic speed-policy iterations

- Introduced obstacle-aware `speed_mode` logic in the test script.
- Evolved from simple distance-based slowing to a combined:
  - longitudinal obstacle position
  - lateral clearance
  - pass/release state

Observed:

- A purely distance-based low-speed policy caused the vehicle to “crawl and stop”.
- Letting speed recover too early caused poor line recovery or unsafe behavior.

### 6. Soft-field parameter phases

We tried multiple parameter directions:

- Stronger repulsion / stronger side bias:
  - better obstacle commitment
  - but often led to conservative stopping

- Weaker repulsion / weaker side bias:
  - reduced early hesitation
  - but sometimes caused scraping or late avoidance

- Earlier `front_cost_dist`:
  - more preparation room
  - but sometimes introduced early “pre-bias”

- Later `front_cost_dist`:
  - cleaner entry
  - but sometimes too late for the second obstacle

### 7. Current pass-window logic attempt

Current intent:

- Once the ego vehicle has already built enough lateral clearance on the selected passing side,
  reduce the soft-field’s tendency to stop beside the obstacle.
- Keep a moderate forward target speed during the side-by-side phase.

Current notable logic:

- `pass_setup`
- `pass_window`
- `pass_commit`
- softened `rep_potential` and `close_potential` during side-by-side committed passing

## Latest Run Summary

Source:

- `C:\Users\Administrator\Desktop\carla_MPC\carla_MPC-main2\two_obstacles\logs\test_main_two_obstacles.log`

What happened near the first obstacle:

- Around `step 251-256`, the car had already built lateral clearance:
  - `obs0_lat_m` grew to about `3.0m`
  - `speed_mode` moved into `pass_setup` / `pass_window`
- Around `step 267-278`, it entered `pass_commit`:
  - `obs0_ahead_m` dropped from about `1.37m` to negative
  - `obs0_lat_m` stayed around `2.90m -> 3.00m`
- Around `step 279+`, the vehicle effectively stopped:
  - `actual_vx` collapsed to almost `0`
  - `speed_mode` remained `pass_commit`
  - `obs0_ahead_m` stayed slightly negative (about `-0.236m`)
  - `obs0_lat_m` remained around `3.00m`

Interpretation:

- The vehicle is no longer stopping because it failed to create lateral offset.
- It is stopping after the pass has almost completed.
- This suggests a residual local minimum in the side-by-side / just-passed geometry.
- The remaining problem is likely a mismatch between:
  - rear-release timing
  - pass-window speed policy
  - near-field cost still being too resistant to forward completion

## Update After The Next Run

New observation from the latest log:

- The stop still occurs at the first obstacle, but the geometry is now even more clearly “already passed enough”.
- Around `step 277-279`:
  - `obs0_ahead_m` moves from about `-0.0289m` to `-0.0352m`
  - `obs0_lat_m` stays around `3.055m`
  - `speed_mode` is already `pass_commit`
  - `actual_vx` collapses to nearly `0`

This confirms:

- The current failure is no longer a side-clearance failure.
- It is specifically a “post-nose pass completion” failure.
- The controller still keeps too much obstacle cost active even after the obstacle is slightly behind the ego reference point.

### New adjustment direction

We therefore changed the logic to release faster after the obstacle just passes the ego:

- `back_release_dist` reduced from `2.5` to `1.2`
- stronger attenuation of:
  - `rep_potential`
  - `close_potential`
  - `side_cost`
  - `brake_near_cost`
  once:
  - preferred pass side is established
  - lateral clearance is sufficient
  - `obstacle_ahead` is already slightly negative

We also changed the speed policy:

- `pass_commit`: `22 -> 24 km/h`
- new `release_finish` state when:
  - `nearest_ahead < -0.3`
  - `nearest_lat > 2.8`
  - target speed `30 km/h`

Purpose:

- Stop treating the just-passed obstacle as if it still needs strong close-range protection.
- Give the ego car one clean “finish the pass and leave” phase instead of staying stuck in `pass_commit`.

## Current Working Hypothesis

The most recent stop is best described as:

- “The vehicle nearly completes the pass, but the optimizer still finds a stationary solution cheaper than finishing the forward motion.”

This is not the same failure mode as:

- early wrong-side steering
- sidewalk obstacle confusion
- direct collision due to insufficient lateral room

## Suggested Next Focus

If we continue tuning, the next changes should focus on the pass-completion phase only:

1. Make rear release happen sooner after the obstacle is clearly behind.
2. Raise forward preference slightly during `pass_commit`.
3. Further reduce close-range resistance only after:
   - preferred side is locked
   - lateral clearance is already sufficient
   - obstacle is no longer clearly in front

## Current Parameter Snapshot

At the time of this note, the most relevant soft-field values are:

- `safe_dist = 2.30`
- `k_u = 780.0`
- `k_close = 1800.0`
- `front_cost_dist = 17.0`
- `back_release_dist = 2.5`
- `side_clearance = 2.75`
- `k_side = 300.0`
- `v_min_near = 2.0`
- `k_vnear = 72.0`
- `k_brake_near = 72.0`

And the key test-script speed states are:

- `prepare_ahead = 24 km/h`
- `mid_tight = 18 km/h`
- `approach_tight = 16 km/h`
- `pass_setup = 20 km/h`
- `pass_window = 20 km/h`
- `pass_commit = 22 km/h`
- `release_hold = 22 or 26 km/h`

## Notes For Future Edits

- Keep this file updated whenever:
  - a soft-field parameter changes
  - speed-policy thresholds change
  - a new failure mode is discovered
  - a previous hypothesis is disproven by logs

## Latest Addendum

After another log review, two coupled symptoms stood out:

1. Before the main avoidance turn, the ego vehicle still makes a small steering move in the opposite direction.
2. Once the ego and obstacle become nearly side-by-side, speed drops too much and the vehicle can still stop.

Working interpretation:

- The early small reverse bend is likely caused by tracking influence still competing with obstacle-side commitment.
- The parallel slowdown is still a post-commit local minimum, not a failure to create lateral clearance.

Latest changes made:

- In `x_v2x_agent.py`
  - `front_cost_dist: 16 -> 18`
  - `release_dist: 4 -> 3`
  - added `lock_lat = 0.35`
  - added `force_lock_ahead = 10.0`
  - result:
    - pass side is locked earlier
    - but not from tiny lateral noise far away

- In `mcp_controller.py`
  - made `pass_progress` saturate earlier
  - made `just_passed_gate` activate earlier
  - further weakened:
    - `rep_potential`
    - `close_potential`
    - `side_cost`
    - `brake_near_cost`
  - added an extra overall obstacle-cost scale-down once the obstacle is just behind the ego nose

Intent:

- Reduce the initial opposite-direction nibble.
- Make the controller release the passed obstacle faster so it finishes the maneuver instead of parking beside it.

## Cost-Matrix Tuning Stage

After the next log review, a new pattern became clear:

- In `pass_window`, `pass_commit`, and even `release_finish`, the target speed was already raised.
- But the optimizer still chose strong negative acceleration.
- That means the issue is not only speed-policy logic; the cost matrices themselves still make “slowing down / stopping” cheaper than “finish the pass”.

New strategy:

- Instead of using one fixed `Q / R / Rd` profile for the whole run, switch weights by maneuver phase.

Profiles introduced in `test_main_two_obstacles.py`:

- `base`
  - normal tracking behavior
- `avoid`
  - used in `pass_setup`, `pass_window`, `pass_commit`
  - lower centerline snap-back
  - higher forward-speed tracking
  - lower control and steering-rate penalties
- `release`
  - used in `release_hold`, `release_finish`
  - even stronger forward-speed preference
  - even lower steering-rate penalty

Purpose:

- Let the controller behave differently when:
  - approaching an obstacle
  - sliding beside it
  - leaving it behind
- Specifically reduce the tendency to brake hard while already side-by-side with the obstacle.

### More aggressive matrix revision

After another run, the log still showed the same pattern:

- `cost_mode` had already switched to `avoid`
- `speed_mode` stayed in `pass_window` / `pass_commit`
- but the optimizer still drove `a_opt` negative and parked near parallel geometry

So the obstacle-phase matrices were made more aggressive:

- `avoid`
  - lower position / yaw tracking weights
  - much higher speed-tracking weight
  - much lower input penalty
  - much lower steering-rate penalty

- `release`
  - even higher speed-tracking weight
  - even lower control and control-rate penalties

Intent:

- Make “keep going forward and finish the pass” significantly cheaper than
  “slow down and settle beside the obstacle”.

## Higher-Tracking / Lower-Control Revision

After the next run, the behavior was still close to the previous aggressive version.

The next revision therefore followed the explicit rule:

- increase tracking weights
- decrease control penalties

Applied direction:

- `base`
  - larger `Q`
  - smaller `R`
  - smaller `Rd`

- `avoid`
  - significantly larger tracking weights than before
  - significantly smaller input penalties
  - significantly smaller steering-rate penalties

- `release`
  - strongest forward/tracking preference among all three modes
  - weakest control and control-rate penalties among all three modes

Purpose:

- Make the optimizer care more about achieving the intended forward maneuver
  and less about keeping acceleration/steering effort artificially conservative.

## Passed-Waypoint Deletion Fix

A new issue was identified: already-passed waypoints could still remain in the queue long enough to be used again in the same control cycle, which could make the vehicle try to re-track old path points.

Fix applied in `x_v2x_agent.py`:

- remove passed waypoints before rebuilding the local reference waypoint array
- compute waypoint-ahead / behind status in the same right-handed frame used by the controller
- delete points that are clearly behind the ego vehicle
- also delete very-close points once they are no longer meaningfully ahead

Purpose:

- prevent the controller from reusing path points that the ego vehicle has already passed
- reduce the tendency to snap back toward already-completed parts of the route

## Monotonic Target-Index Fix

Waypoint deletion alone was not enough. The local reference builder could still choose a nearest path point that effectively jumped backward, especially near obstacles and curves.

Fix applied in `x_v2x_agent.py`:

- added `self._last_target_ind`
- clamped the newly selected local target index so it cannot move backward
- refined local target selection within a short forward window
- strongly penalized candidate points that are already behind the ego vehicle
- reset the stored target index when replanning the route

Purpose:

- stop the local reference from snapping back to already-passed path points
- prevent the predicted trajectory from appearing to chase the route in reverse
