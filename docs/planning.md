# Planning with cuMotion

Collision-free moves come from NVIDIA's cuMotion, through the `isaacsim.robot_motion.cumotion`
extension in Isaac Sim 6.0.1. This page covers what cuMotion needs from this repo and how the panel
uses it.

## The robot description (XRDF)

cuMotion plans with collision spheres instead of meshes. `scripts/make_viola_xrdf.py` generates them
from the URDF and the STL files and writes `assets/viola/urdf/viola.xrdf`, next to `viola.urdf`.
cuMotion loads both from the same folder.

```bash
$ISAACSIM_PATH/python.sh scripts/make_viola_xrdf.py
```

The spheres are fitted to the mesh surface. Every mesh vertex ends up at least 2 mm inside a sphere.
The script then checks a second, denser sample of each surface and prints how much of it the spheres
cover. The shipped file has 119 spheres and covers every checked point.

| Link | Spheres | Largest radius |
|---|---|---|
| base_link | 15 | 35 mm |
| link1 | 5 | 25 mm |
| link2 | 17 | 25 mm |
| link3 | 8 | 24 mm |
| link4 | 7 | 22 mm |
| link5 | 6 | 24 mm |
| link6 (gripper body) | 20 | 22 mm |
| link7_left, link7_right (fingers) | 21, 20 | 12 mm |

The fingertip spheres reach 4 to 5 mm past the real fingertips. That margin decides how close to the
table the planner lets the gripper go.

Four things the generator does on purpose:

- **Only one finger joint is listed.** `joint7_right` mimics `joint7_left`, and cuMotion refuses a
  default position for a mimic joint.
- **The base only counts for self-collision.** As a world-collision body it would sit inside the
  table obstacle, and every pose would count as a collision.
- **Some link pairs are ignored for self-collision.** Each link ignores the next two links along the
  arm. Neighbouring links always touch at their joint, and `link2` and `link4` overlap in the folded
  poses the physical arm really takes.
- **Acceleration and jerk limits are chosen, not measured.** 5 rad/s² and 500 rad/s³ per joint.
  Change them with `--accel` and `--jerk`.

## What counts as an obstacle

Every PhysX collider that is on the stage when the script starts, except the robot, the target cube
and the default ground plane. In practice that is a scene loaded with `--scene`. The list is built
once, so a cube you add in the editor later is not seen. On top of that the
planner gets a floor at `--plan-floor`, which defaults to the table height `--table-z`. Without that
floor, cuMotion will swing the gripper under the table. `--plan-floor nan` turns it off.

Two limits come from cuMotion itself. Box obstacles must be scaled uniformly, and no obstacle may sit
under a scaled parent prim. So a scene has to have its scale baked in before loading, and `--plan`
refuses `--scene-scale` values other than 1.

## How a plan is made

1. The goal is the target cube's position, or with *align pads* a full pose with the pads square to
   the cube. Among the four symmetric ways to grip a cube, the one closest to the gripper's current
   heading is used, so the wrist turns least.
2. For a pose goal, `pick_planner.py` first looks for an exact IK solution with every joint at least
   3 degrees inside its limit. Only then is the goal handed to the planner. If cuMotion's pose
   planning fails, the plan falls back to a joint-space plan to that IK solution.
3. cuMotion's JtRRT planner searches for a collision-free path.
4. cuMotion doesn't check self-collision during the search. `pick_planner.py` checks the finished
   path and refuses it if the arm would hit itself.
5. The path becomes a minimal-time trajectory at `--plan-vel` and `--plan-acc`, and the sim arm
   follows it. Then control goes back to the sliders, which now sit at the final pose.

## Planner settings

The names are from cuMotion's `motion_planner.h` in
[nvidia-isaac/cumotion](https://github.com/nvidia-isaac/cumotion). NVIDIA's tutorial is
[Graph-Based Motion Planner](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/cumotion/tutorial_graph_planner.html).

| Flag | cuMotion parameter | Default here | Why |
|---|---|---|---|
| `--plan-limits` | `task_space_limits` | x and y ±0.5 m, z 0 to 0.6 m | cuMotion's default is ±1 m on every axis, and half of that is under the table |
| `--plan-zone-tol` | `translation_target_zone_tolerance` | 0.02 | cuMotion's default of 0.05 left paths 25 to 48 mm short on this arm |
| `--plan-step` | `step_size` | 0.03 | Step length in joint space |
| `--plan-iterations` | `max_iterations` | 10000 | cuMotion's default. 100 000 took 61 s and still found no path to a blocked goal |
| `--plan-safety` | obstacle safety tolerance | 0.02 m | Clearance kept from every obstacle |

## When it says "no path"

A refusal prints `no path to [x, y, z]` with a reason, and the arm holds still. The reasons come from
`scripts/pick_planner.py`, except the last row, which the panel prints itself.

| Reason starts with | What it means | What to do |
|---|---|---|
| `start outside the joint limits` | A joint reads past its limit | Press *Sync<-robot*, or slide it back inside |
| `start in collision` | The arm already touches an obstacle. Typical right after *Descend* | Press *Lift* first |
| `no IK for the ...: out of reach` | No arm pose reaches the goal | Move the cube closer or lower, rotate it, or untick *align pads* |
| `... too close to the floor` | Every solution puts the arm inside the padded table | Raise the target, or check `--table-z` |
| `no IK for the ... clear of the arm itself` or `path self-collides` | Only self-colliding solutions exist | Move or rotate the cube |
| `... deg from a joint limit` | Even the fallback ends less than 3 degrees from a limit | Move the cube |
| `no path found from here although ... has a clear IK solution` | The goal is fine and the way is blocked | Move the target or the obstacle |
| `already at the goal` | Nothing to do | |
| `plan requested but cuMotion off - restart with --plan` | Isaac was started without `--plan` | Restart with `--plan` |

**A plan can end a few millimetres from the goal.** cuMotion plans into a zone around the target and
then assumes a straight joint-space move to the IK solution. In the runs listed in the README the arm
ended 4.5 mm and 6.7 mm from the goal. Finish with the sliders, or switch to *IK: follow the cube*
for the last stretch.

## Self-tests

**`scripts/plan_selftest.py`** needs no Isaac window and no robot. It uses the standalone cuMotion
package inside the Isaac Sim folder, and cuMotion uses the GPU. It builds the planner the way the
panel does. The defaults are read straight from `isaac_ik_mirror.py`, so the two can't drift apart.
Then it runs 19 cases through `pick_planner.py` and `grasp_frame.py`.

```bash
$ISAACSIM_PATH/kit/python/bin/python3 scripts/plan_selftest.py
```

It prints one `PASS` or `FAIL` line per case and exits with status 1 if any case fails. `--only c`
runs the cases whose name starts with `c`. The cases cover:

- plans to cubes in mid-air and on a grid across the table (cases a, b, c1, c2)
- a cube too close to the base, which must be refused with the right reason (c3)
- the joint-space fallback (c4)
- the approach tilt staying as small as it can (d1, d2)
- refusing paths where the arm hits itself, without refusing normal ones (e1 to e4)
- three "no path" reasons being reported: out of reach, too close to the floor, and a start in collision (f)
- the gripper keeping its clearance above the table during Descend and Lift (g1 to g3)
- starting exactly on a joint limit (h), keeping the joint margin (i), and a cube near the table
  whose pre-grasp is out of reach (j)

**`scripts/cumotion_plan_test.py`** runs inside headless Isaac Sim. It stacks three 8 cm cubes into a
24 cm high obstacle between the start pose and the target, plans past it, and has the simulated arm
follow the trajectory. It prints the final error and the smallest distance between the end effector
and the stack. It only prints its results, so read the two `RESULT` lines. With `--gui` it opens a window so you can
watch the move. The move takes about 3 seconds. The window closes by itself a minute later, or when you press
`Ctrl+C` in the terminal.

```bash
$ISAACSIM_PATH/python.sh scripts/cumotion_plan_test.py
```
