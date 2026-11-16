# The Viola Control panel

`scripts/isaac_ik_mirror.py` starts Isaac Sim with the Viola on a table and a window named
**Viola Control**. Run it from the repo root.

```bash
$ISAACSIM_PATH/python.sh scripts/isaac_ik_mirror.py --plan
```

Without `--plan` the panel starts without cuMotion. The sliders and the IK target work. *Plan to
target*, *Above* and *Pick all* answer that they need `--plan`.

## Who drives the arm

There is one simulated arm and four possible sources. The script picks one every frame, with no
restart.

| Priority | Source | How you drive it |
|---|---|---|
| 1 | `LEADER` | Packets on udp/5599 from a leader arm, the arm you move by hand. It always wins while packets arrive |
| 2 | `PLAN` | A cuMotion planned move, while it runs |
| 3 | `JOINTS` | The sliders and the pose buttons |
| 3 | `IK` | Pink moves the gripper to the target cube |

The **Source** dropdown picks between the sliders, IK and planning. Whatever is not driving is kept
in sync. In slider mode the cube rides on the gripper, and in IK mode the sliders follow the arm, so
switching never makes the arm jump.

Slider and button moves are rate limited by `--jog-rate`, 40 units per second by default. That is
between 32 and 49 degrees per second on the base, shoulder, elbow and wrist pitch.

Each slider's -100..100 becomes a joint angle through a calibration file, the way the physical arm
converts it. See [joint-map.md](joint-map.md).

## Dragging the target cube

In the Stage panel click `World`, then `target`. Move the mouse over the viewport, press `W`, and
drag one of the arrows. Press `E` to rotate the cube.

Don't select the cube by clicking it in the viewport. It sits between the fingers, so the click
picks a gripper link, and dragging a link does nothing because physics puts it back.

The cube is a marker with no physics. Nothing in the sim pushes it or picks it up. It is 30 mm on
each side. `--target-size` changes the drawn size only. The grasp is always computed for a 30 mm cube.

## Buttons

| Button | What it does |
|---|---|
| *Sync<-robot* | The sliders take the sim arm's current joints |
| *Reach* | Moves to `--reach-pose`, a pose in the middle of the workspace |
| *Connect* | Moves to the connect pose, the pose the physical arm takes when its driver connects |
| *Rest* | Moves to `--rest-pose`, the folded parking pose |
| *Probe* | Prints the position of the gripper frame and of the fingertip, and how far the gripper's lowest point is above the table. It also appends a line to `outputs/probe_log.csv` |
| *Plan to target* | Plans a collision-free move to the cube and follows it. See [planning.md](planning.md) |
| *align pads* (tick box) | Ticked: the gripper arrives with its pads square to the cube. Unticked: position only, which reaches further |
| *Above* | Plans to the pre-grasp point, `--pre-grasp` back from the grasp along the approach |
| *Descend* | Opens the gripper, then moves in a straight line down the approach to the grasp |
| *Close* | Closes the gripper to `--grip-close` and waits for it |
| *Lift* | Backs out along the approach by `--lift-height`, and no further than the pre-grasp point |
| *Pick all* | Runs Above, Descend, Close and Lift in order |

If any step of a pick fails, the pick stops there and the arm holds its position. The terminal says
why, in a line that contains `pick aborted`.

*Descend* only works from the approach line that *Above* planned. If you press it from somewhere
else, it tells you to press *Above* first.

When the cube is less than 45 mm above the table and *align pads* is ticked, *Plan to target* goes to
the point above the cube, the same place *Above* goes. Press *Descend* to go down from there.

## Picking a cube

1. Start with `--plan` and press *Reach*.
2. Drag the cube to a spot on the table.
3. Press *Pick all*, or the four buttons one at a time.
4. Watch the terminal for lines starting with `[ik] pick:`.

The gripper goes through the whole pick, and the cube stays on the table, because it is only a marker.
It shows the sim where a real cube would be. With a physical arm following the sim, the real gripper
picks up a real cube placed at that spot.

The approach is straight down where the arm can reach the cube that way. Otherwise it tilts toward
the base, up to `--max-tilt` (60 degrees), and `grasp_frame.py` picks the smallest tilt that works.
With the default table settings, a cube on the table 16 to 24 cm out gets a tilt of 13 to 15 degrees.

## Keyboard

With the viewport focused: `O` opens the gripper, `C` closes it, and `,` and `.` nudge it by 5 units
toward closed or open.

## Flags

All flags are optional. This table has the main ones. `--help` lists them all, most with the reason
for the default.

| Flag | Effect |
|---|---|
| `--plan` | Loads cuMotion and enables the plan and pick buttons |
| `--start-mode joints\|ik\|plan` | The Source the panel starts in. Default: `plan` with `--plan`, else `joints` |
| `--no-align` | Plans and IK go to the position only. Same as unticking *align pads* |
| `--orientation-cost 0.3` | IK mode holds the pad alignment more firmly. The default 0.05 follows the position first. 1.0 twists the arm near its joint limits |
| `--plan-vel 0.7 --plan-acc 1.5` | Speed and acceleration of planned moves, in rad/s and rad/s² |
| `--table-z Z` | Height of the sim table. It sets the ground, the planner's floor and where a cube rests. Default 0.007, explained in [joint-map.md](joint-map.md) |
| `--table-clearance M` | The gripper's lowest point stays this far above the table at the grasp. Default 0.013 |
| `--grasp-depth 0.005` / `--pre-grasp 0.08` | How far the gripper stays back from the cube centre at the grasp, and how much further back *Above* stops |
| `--grip-close 40` | Gripper value for *Close*. In the sim, 40 leaves a 29.9 mm gap for a 30 mm cube |
| `--joint-map calib\|urdf` | How -100..100 maps onto joint angles. Default `calib` |
| `--follower-calib PATH` | Calibration file for `--joint-map calib`. Default: the example in `config/` |
| `--scene FILE.usd --no-ground` | Loads a scene USD under the robot. Its PhysX colliders become planner obstacles |
| `--udp-host`, `--udp-port` | Where leader packets come in. Default 127.0.0.1:5599, this machine only |
| `--echo-host`, `--echo-port` | Where the sim arm's joints are sent. Default 127.0.0.1:5600 |
| `--headless` with `--demo-jog`, `--demo-orbit`, `--demo-plan` or `--demo-pick` | Scripted runs with no window |

## What the terminal shows

At start:

```
[ik] joint map: calib. -100..100 per arm joint, deg (URDF limits, the sim's stops):
[ik] cuMotion ready: ... 1 obstacle prims tracked, safety 0.02 m, vel 0.7 rad/s, acc 1.5 rad/s^2
[ik] start pose set (connect pose)
[ik] source -> JOINTS
```

For each plan:

```
[ik] plan: path 2 waypoints, 3.3 s, planned in 180 ms [pose, pads aligned, tilt 25 deg]
[ik] source -> PLAN
[ik] plan trajectory finished; settling
[ik] plan done: ee=[...] goal=[...] err=4.5 mm
```

A refused plan prints `no path to [x, y, z]` with the reason, and the arm holds still.
