"""Headless self-test: cuMotion plans a collision-free move for the Viola around a box.

    $ISAACSIM_PATH/python.sh scripts/cumotion_plan_test.py          # headless, prints the result
    $ISAACSIM_PATH/python.sh scripts/cumotion_plan_test.py --gui    # with a window, to watch the move

What it proves: the generated XRDF (assets/viola/urdf/viola.xrdf) loads in cuMotion, the
world binding picks up PhysX colliders from the stage (the same colliders any scene USD
with PhysX collision carries), JtRRT finds a path from the connect pose to a translation
target with a box on the straight line, the path is turned into a minimal-time trajectory,
and the simulated arm follows it. The end-effector track is logged; the minimum distance to
the obstacle centre must exceed the box half-size, and the final EE error is printed.

API path (verified in the 6.0.1 install):
  mg.SceneQuery().get_prims_in_aabb -> mg.WorldBinding(CumotionWorldInterface) ->
  load_cumotion_robot(dir, urdf, xrdf) -> GraphBasedMotionPlanner.plan_to_translation_target
  -> mg.Path.to_minimal_time_joint_trajectory -> mg.TrajectoryFollower
"""

import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--target", default="0.38,0.0,0.12", help="EE (link6) translation target, metres, world frame")
p.add_argument("--box", default="0.24,0.0,0.12", help="obstacle box centre; '' for no obstacle")
p.add_argument("--box-dims", default="0.08,0.08,0.24", help="obstacle box x,y,z size, m; built from uniform cubes "
                                                          "(cuMotion OBBs must be uniformly scaled)")
p.add_argument("--safety", type=float, default=0.01, help="cuMotion safety tolerance around obstacles, m")
p.add_argument("--max-vel", type=float, default=1.0, help="rad/s per joint for the trajectory")
p.add_argument("--max-acc", type=float, default=2.0, help="rad/s^2 per joint for the trajectory")
p.add_argument("--start", default="0,-1.57,0.942,0,0.471,0", help="start joints, radians (default: connect pose)")
p.add_argument("--floor", type=float, default=0.0, help="z of the table top as a 2 m cube obstacle; nan disables")
p.add_argument("--gui", action="store_true", help="open a window to watch the move; it closes by itself a minute after the move, or on Ctrl+C")
args = p.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.experimental.motion_generation")
enable_extension("isaacsim.robot_motion.cumotion")

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import isaacsim.robot_motion.experimental.motion_generation as mg  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.experimental.objects import Cube  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, GraphBasedMotionPlanner, load_cumotion_robot  # noqa: E402
from omni.physx.scripts import utils as physx_utils  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD = os.path.join(ROOT, "assets/viola/usd/viola/viola.usda")
URDF_DIR = os.path.join(ROOT, "assets/viola/urdf")
ROBOT_PATH = "/World/Viola"
TOOL = "link6"
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# connect pose in radians (normalized 0,-100,60,0,30,0 -> URDF limits, the old urdf joint map), matches viola.xrdf defaults
Q_CONNECT = [float(v) for v in args.start.split(",")]
assert len(Q_CONNECT) == 6


def find_prim(stage, root, name):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return prim
    return None


def world_pos(prim):
    t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(USD, ROBOT_PATH)
stage = world.stage
art_path = next(str(q.GetPath()) for q in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if q.HasAPI(UsdPhysics.ArticulationRootAPI))
arm = Articulation(art_path)

box_center = None
if args.box:
    box_center = np.array([float(v) for v in args.box.split(",")])
    box_dims = np.array([float(v) for v in args.box_dims.split(",")])
    cell = float(box_dims.min())
    counts = np.maximum(1, np.round(box_dims / cell).astype(int))
    box_dims = counts * cell  # actual extent after tiling
    k = 0
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            for iz in range(counts[2]):
                c = box_center - box_dims / 2.0 + (np.array([ix, iy, iz]) + 0.5) * cell
                Cube(paths=f"/World/obstacle_{k}", sizes=cell, positions=[c.tolist()])
                physx_utils.setStaticCollider(stage.GetPrimAtPath(f"/World/obstacle_{k}"))
                k += 1
    print(f"[cumotion] obstacle: {k} cubes of {cell} m, extent {box_dims.round(3).tolist()} at {box_center.tolist()}")


def box_distance(pt):
    # distance from a point to the obstacle's axis-aligned box surface (0 inside)
    d = np.maximum(np.abs(pt - box_center) - box_dims / 2.0, 0.0)
    return float(np.linalg.norm(d))

import math  # noqa: E402
if not math.isnan(args.floor):
    Cube(paths="/World/plan_floor", sizes=2.0, positions=[[0.0, 0.0, args.floor - 1.0]])
    physx_utils.setStaticCollider(stage.GetPrimAtPath("/World/plan_floor"))
    print(f"[cumotion] floor obstacle: top at z={args.floor}")
world.reset()
names = list(arm.dof_names)
arm_idx = [names.index(j) for j in ARM]
q0 = np.zeros(len(names), dtype=np.float32)
for i, j in zip(arm_idx, Q_CONNECT):
    q0[i] = j
arm.set_dof_positions(q0[None, :])
arm.set_dof_position_targets(q0[None, :])
for _ in range(30):
    world.step(render=args.gui)
link6 = find_prim(stage, ROBOT_PATH, TOOL)
ee0 = world_pos(link6)
print(f"[cumotion] start q={np.round(Q_CONNECT, 3).tolist()} ee={ee0.round(3).tolist()}")

# --- world binding: every PhysX collider on the stage except the robot becomes an obstacle ------
robot_pos, robot_ori = arm.get_world_poses()
objects = mg.SceneQuery().get_prims_in_aabb(
    search_box_origin=robot_pos.numpy()[0],
    search_box_minimum=[-5.0, -5.0, -5.0],
    search_box_maximum=[5.0, 5.0, 5.0],
    tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
    exclude_prim_paths=[ROBOT_PATH, "/World/defaultGroundPlane"],
)
strategy = mg.ObstacleStrategy()
strategy.set_default_configuration(Cube, mg.ObstacleConfiguration("obb", args.safety))
binding = mg.WorldBinding(
    world_interface=CumotionWorldInterface(),
    obstacle_strategy=strategy,
    tracked_prims=objects,
    tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
)
binding.initialize()
wi = binding.get_world_interface()
wi.update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
binding.synchronize_transforms()
print(f"[cumotion] obstacles tracked: {[str(getattr(o, 'prim_path', o)) for o in objects]}")

# --- robot + planner ------------------------------------------------------------------------
robot = load_cumotion_robot(directory=URDF_DIR, urdf_filename="viola.urdf", xrdf_filename="viola.xrdf")
print(f"[cumotion] robot loaded: cspace={robot.controlled_joint_names} tool_frames={robot.robot_description.tool_frame_names()}")
planner = GraphBasedMotionPlanner(cumotion_robot=robot, cumotion_world_interface=wi, tool_frame=TOOL)
import cumotion as _cumotion  # noqa: E402
pcfg = planner.get_graph_planner_config()
print("[cumotion] params set:", pcfg.set_param("task_space_limits", [_cumotion.MotionPlannerConfig.Limit(-0.5, 0.5), _cumotion.MotionPlannerConfig.Limit(-0.5, 0.5), _cumotion.MotionPlannerConfig.Limit(0.0, 0.6)]),
      pcfg.set_param("task_space_planning_params/translation_target_zone_tolerance", _cumotion.MotionPlannerConfig.ParamValue(0.02)),
      pcfg.set_param("step_size", _cumotion.MotionPlannerConfig.ParamValue(0.03)))
target = [float(v) for v in args.target.split(",")]
import time as _time  # noqa: E402
_t0 = _time.perf_counter()
path = planner.plan_to_translation_target(Q_CONNECT, target)
print(f"[cumotion] planning took {(_time.perf_counter() - _t0) * 1000:.0f} ms")
if path is None:
    print("[cumotion] NO PATH FOUND")
    simulation_app.close()
    raise SystemExit(1)
wps = path.get_waypoints().numpy()
print(f"[cumotion] path found: {len(wps)} waypoints")

traj = path.to_minimal_time_joint_trajectory(
    max_velocities=np.full(6, args.max_vel), max_accelerations=np.full(6, args.max_acc),
    robot_joint_space=names, active_joints=ARM)
print(f"[cumotion] trajectory duration {traj.duration:.2f} s")

follower = mg.TrajectoryFollower()
follower.set_trajectory(traj)


def est():
    return mg.RobotState(joints=mg.JointState.from_name(
        robot_joint_space=names, positions=(names, arm.get_dof_positions()), velocities=(names, arm.get_dof_velocities())))


t = 0.0
dt = 1.0 / 60.0
assert follower.reset(est(), None, t)
min_d = 1e9
track = []
n_steps = int((traj.duration + 1.0) / dt)
for k in range(n_steps):
    t += dt
    desired = follower.forward(est(), None, t)
    if desired is not None and desired.joints is not None and desired.joints.positions is not None:
        arm.set_dof_position_targets(desired.joints.positions, dof_indices=desired.joints.position_indices)
    world.step(render=args.gui)
    ee = world_pos(link6)
    track.append(ee)
    if box_center is not None:
        min_d = min(min_d, box_distance(ee))
    if k % 30 == 0:
        q = arm.get_dof_positions().numpy().reshape(-1)[arm_idx]
        print(f"[cumotion] t={t:4.2f} q={np.round(q, 2).tolist()} ee={ee.round(3).tolist()}")

ee_end = world_pos(link6)
err = float(np.linalg.norm(ee_end - np.array(target))) * 1000
print(f"[cumotion] RESULT final ee={ee_end.round(4).tolist()} target={target} err={err:.1f} mm")
if box_center is not None:
    straight = min(box_distance(ee0 + a * (np.array(target) - ee0)) for a in np.linspace(0, 1, 200))
    print(f"[cumotion] RESULT min EE distance to box surface {min_d*1000:.1f} mm along the followed path; "
          f"the straight start->target line would come within {straight*1000:.1f} mm")
if args.gui:
    # Keep the window up long enough to look around: shutting it the moment the move ends looks like a crash.
    # The hold is bounded, so a forgotten window does not keep the GPU busy.
    HOLD_S = 60
    print(f"[cumotion] done. The window closes by itself in {HOLD_S} s. To close it sooner, press Ctrl+C in this terminal.")
    t_end = _time.perf_counter() + HOLD_S
    try:
        while simulation_app.is_running() and _time.perf_counter() < t_end:
            world.step(render=True)
    except KeyboardInterrupt:
        pass
simulation_app.close()
