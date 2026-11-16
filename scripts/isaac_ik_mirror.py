"""Isaac Sim Viola driven by the physical leader, per-joint sliders, or a draggable IK target.

    $ISAACSIM_PATH/python.sh scripts/isaac_ik_mirror.py                    # GUI, with control panel
    $ISAACSIM_PATH/python.sh scripts/isaac_ik_mirror.py --headless --demo-jog     # self-test: joints
    $ISAACSIM_PATH/python.sh scripts/isaac_ik_mirror.py --headless --demo-orbit   # self-test: IK

Sources for one articulation. Priority every frame, no restart (a planned move, PLAN, ranks between 1 and 2):
    1. LEADER  leader-arm packets on udp/5599 always win (sent by a bridge for the physical arms, not in this repo)
    2. JOINTS  the "Viola Control" panel sliders, one per joint - the MoveIt joint-panel analog
    3. IK      Pink differential IK drives link6 to the /World/target cube (drag it)
Panel picks JOINTS or IK; the leader overrides either while it is talking.

Whichever source is NOT active is kept in sync (sliders follow the arm, the
target cube rides on the end effector), so switching never jumps. Slider and
button moves, and every IK step, are rate limited (--jog-rate) so the real arm follows smoothly.

IK is position-dominant (--orientation-cost 0.05 with 'align pads'; without, position
only) and regularises the null space toward --reach-pose, so dragging keeps a natural,
predictable posture. Pink steps from its own last command, not from the lagging measured
joints, and the pads-square rotation it follows turns smoothly (IK_ROT_TAU, IK_ROT_RATE).
Note: this is differential IK - the equivalent of MoveIt *Servo*. Planned,
collision-free moves (MoveIt's planner) are cuMotion with the Viola's XRDF (--plan, below).

Sim state is echoed on udp/5600 in LeRobot normalized units, so a physical arm can
follow in every mode (README, 'Connecting a physical arm').

The sim starts in the vendor connect pose (what the real Viola takes on connect).

Keyboard (GUI):  O gripper->100 (open)   C gripper->0 (closed)   , / . nudge -5/+5

Panel dropdown: 'Sliders' (target rides on the gripper) / 'IK: follow the cube' (Pink follows the
cube) / 'Plan: place the cube, then Plan' (arm holds, cube free: drag it, then press Plan). --plan starts there.
Pick sequence (panel 'Plan & pick'): Above = plan to a pre-grasp pose back along the approach;
Descend = open the gripper, then a straight-line Pink servo down the approach to the grasp pose
(link6 --grasp-depth back from the cube centre, further where the gripper would come within
--table-clearance of the table); Close = gripper to --grip-close; Lift = back out along the
approach to the pre-grasp. Descend and Lift servo only along the approach line of Above's grasp
(Lift without one: at most 5 cm straight up, the gripper near vertical) at Pink orientation cost
--servo-orientation-cost, and are done within 4 mm and --servo-tol-deg of the goal; any failure
aborts the pick with the arm holding its last command. 'Pick all' chains them (it needs --plan).
Approach is top-down when reachable, else tilted toward the base along a cube axis
(grasp_frame.py); Above picks the tilt by exact IK and Descend/Lift reuse its grasp.
--table-z moves the sim table (ground, planner floor, cube rest height); Probe logs the fingertip.

Planned moves (--plan, docs/planning.md): 'Plan to target' asks cuMotion (JtRRT, XRDF from
scripts/make_viola_xrdf.py) for a collision-free path from the current pose to /World/target's
position (with 'align pads', to the aligned pose; a cube within 45 mm of the table, to its
pre-grasp as Above), around every PhysX collider on the stage (a --scene bench, boxes), then
follows the minimal-time trajectory as source PLAN (below LEADER, above JOINTS/IK). cuMotion
does not check self-collision, so a path that makes the arm hit itself is refused. The goal
selection and checks live in pick_planner.py; scripts/plan_selftest.py runs them offline.

Gripper sense (both measured 2026-09-19): real Viola normalized 0 = closed, 100 = open
(by hand). USD joint7 lower (-0.03 m) = fingers 6 mm apart = closed, upper (0 m)
= 66 mm = open (headless, finger origins 0 -> 60 mm). So normalized 0 -> lower, 100 ->
upper, and sim and real agree. --invert-gripper flips ONLY the sim; the echoed value
round-trips unchanged, so it never changes what the real arm does.

Arm joint map (--joint-map, scripts/joint_map.py): calib (default) does what the real arm does, through the follower's LeRobot calibration to servo degrees (= URDF degrees, vendor ROS 2
driver); urdf spreads -100..100 over the URDF limits (the map until 2026-10-02). Measured 2026-10-01/02 with a phone
on the forearm: per step calib within 1.1 deg, urdf only within 4.6 (docs/joint-map.md).
"""

import argparse
import functools
import json
import math
import os
import socket
import time

import joint_map  # scripts/joint_map.py: stdlib only, so it loads before Isaac Sim does

print = functools.partial(print, flush=True)

p = argparse.ArgumentParser()
p.add_argument("--udp-host", default="127.0.0.1",
               help="address the leader port listens on. 127.0.0.1 (default): packets from this machine only. 0.0.0.0: from "
                    "the network too; whoever can reach the port then drives the sim arm, and a physical arm that follows it")
p.add_argument("--udp-port", type=int, default=5599, help="UDP port for leader packets, JSON {\"q\": [7 normalized values]}")
p.add_argument("--echo-host", default="127.0.0.1", help="where the sim arm's joints are sent every frame")
p.add_argument("--echo-port", type=int, default=5600, help="UDP port the sim arm's joints are sent to")
p.add_argument("--headless", action="store_true")
p.add_argument("--leader-timeout", type=float, default=0.3)
p.add_argument("--invert-gripper", action="store_true",
               help="flip the SIM gripper sense only (default: 0 = closed = joint lower, 100 = open = upper, matching the real arm). Never changes the value sent to the real arm.")
p.add_argument("--joint-map", choices=["urdf", "calib"], default="calib",
               help="how -100..100 maps onto the sim's arm joints. calib (default): through the follower's LeRobot "
                    "calibration to servo degrees, as the real arm moves (scripts/joint_map.py; phone check 2026-10-01/02: "
                    "per step within 1.1 deg). urdf: over the URDF limits, the map until 2026-10-02 (within only 4.6 deg "
                    "per step; the cause of the real gripper stopping too high)")
p.add_argument("--follower-calib", default=None,
               help="calibration JSON for --joint-map calib (default: config/viola_calibration_example.json; with your own arm, its LeRobot calibration file)")
p.add_argument("--start-mode", choices=["joints", "ik", "plan"], default=None,
               help="joints: sliders drive, target rides on the gripper. ik: Pink follows the target as you drag. "
                    "plan: arm holds, target is free to place, then press Plan. Default: plan with --plan, else joints.")
p.add_argument("--scene", default=None,
               help="USD of a scene with PhysX colliders (a scanned bench, for example), referenced at /World/scene")
p.add_argument("--scene-offset", default="0,0,0", help="x,y,z metres applied to /World/scene")
p.add_argument("--scene-rpy", default="0,0,0", help="roll,pitch,yaw degrees applied to /World/scene")
p.add_argument("--scene-scale", type=float, default=1.0, help="uniform scale applied to /World/scene (1 = scan already in metres)")
p.add_argument("--no-ground", action="store_true", help="skip the default ground plane (scan has its own floor)")
p.add_argument("--plan", action="store_true",
               help="enable cuMotion: 'Plan to target' button plans a collision-free move to /World/target "
                    "around every PhysX collider on the stage (scene, boxes). Needs assets/viola/urdf/viola.xrdf.")
p.add_argument("--plan-vel", type=float, default=0.7,
               help="rad/s per joint for planned trajectories, in wall seconds as the real arm moves (0.7 = 40 deg/s: "
                    "~50 units/s shoulder, 42 elbow, 39 wrist pitch, 33 base, 22 on the rolls)")
p.add_argument("--plan-acc", type=float, default=1.5, help="rad/s^2 per joint for planned trajectories, wall clock")
p.add_argument("--plan-safety", type=float, default=0.02, help="m of clearance cuMotion keeps from obstacles")
p.add_argument("--plan-iterations", type=int, default=10000, help="JtRRT max_iterations (cuMotion default 10000)")
p.add_argument("--plan-zone-tol", type=float, default=0.02,
               help="task_space_planning_params/translation_target_zone_tolerance (m). cuMotion header: keep it on the "
                    "order of the tool motion of one step_size; default 0.05 left the path 25-48 mm short on the Viola")
p.add_argument("--plan-step", type=float, default=0.03, help="JtRRT step_size in c-space (rad L2), cuMotion default 0.05")
p.add_argument("--plan-limits", default="-0.5,0.5,-0.5,0.5,0.0,0.6",
               help="task_space_limits xlo,xhi,ylo,yhi,zlo,zhi (m): the box JtRRT samples task-space targets in. "
                    "cuMotion default is +-1 m on every axis, i.e. half of it is under the table.")
p.add_argument("--xrdf", default="viola.xrdf", help="XRDF filename next to the URDF (scripts/make_viola_xrdf.py)")
p.add_argument("--plan-floor", type=float, default=None,
               help="z of the table/floor the planner must stay above (a 2 m cube obstacle whose top is here); "
                    "nan disables it")
p.add_argument("--demo-plan", action="store_true", help="self-test: plan to --demo-plan-target 2 s after start")
p.add_argument("--demo-plan-target", default="0.30,0.05,0.10")
p.add_argument("--no-align", action="store_true",
               help="do not align the finger pads with the target cube: plan/IK to position only (orientation free)")
p.add_argument("--table-z", type=float, default=0.007,
               help="z of the sim table top: ground plane, planner floor and cube rest height. 0.007 = the real arm's sag at "
                    "the table: with --joint-map calib the real gripper sits ~7 mm below the sim's at a touch at x 0.15 (Probe "
                    "at --table-z 0; 8-9 mm by ruler at x 0.16/0.22, ~16 mm at "
                    "x 0.30), the servos see 0.4-4.5 mm of it. Raising the sim table by it makes sim and real touch together "
                    "(2026-10-03: Probe lowest gripper point -0.4 mm while the real fingers touch). 0 = the URDF's own table")
p.add_argument("--max-tilt", type=float, default=60.0, help="deg from vertical the approach may tilt toward the base when top-down is out of reach")
p.add_argument("--grasp-depth", type=float, default=0.005,
               help="m the finger rail (link6 origin) stays back from the cube centre along the approach, at least; near the "
                    "table the pick backs off further to keep --table-clearance (the fingertips are 18 mm past the rail)")
p.add_argument("--table-clearance", type=float, default=0.013,
               help="m the gripper's lowest point (URDF meshes) stays above --table-z at the pick grasp: the grasp backs off "
                    "along the approach until it does, further the more the approach tilts. It counts from the SIM table, so "
                    "with the default --table-z 0.007 it is the real gripper's height above the real table at x 0.15-0.22. "
                    "0.013 picked 30 mm cubes on the rig (2026-10-03). At x 0.30 the real arm sags ~9 mm more: check before "
                    "picking that far out")
p.add_argument("--pre-grasp", type=float, default=0.08,
               help="m back along the approach from the grasp at --grasp-depth: the pre-grasp Above plans to. It stays "
                    "there when --table-clearance backs the grasp off, and Descend is that much shorter")
p.add_argument("--descend-time", type=float, default=2.0, help="s for the straight-line Descend")
p.add_argument("--lift-height", type=float, default=0.08,
               help="m Lift backs out along the approach, no further than the pre-grasp; without a grasp from Above, "
                    "straight up and at most 0.05 (grasp_frame.lift_goal)")
p.add_argument("--lift-time", type=float, default=2.0)
p.add_argument("--open-time", type=float, default=3.0,
               help="s the gripper gets to open to 100 before Descend (the sliders ramp it at --jog-rate); the pick aborts "
                    "if the sim gripper has not opened by then")
p.add_argument("--servo-orientation-cost", type=float, default=0.3,
               help="Pink orientation cost while Descend and Lift servo along the approach line; --orientation-cost is "
                    "restored after. Offline 2026-09-30 over the table grid: 0.05 ends them up to 26 deg off the grasp "
                    "rotation, 0.3 within 3.1; 1.0 started from Rest drives joint3 into its limit")
p.add_argument("--servo-tol-deg", type=float, default=4.0,
               help="deg: Descend and Lift are done within this of the goal rotation and 4 mm of its position; otherwise "
                    "the pick aborts and the arm holds. 0.3 ends them within 3.1 deg offline")
p.add_argument("--grip-close", type=float, default=40.0,
               help="gripper value for Close (0 closed, 100 open): in the sim 40 leaves a 29.9 mm pad gap, a 0.1 mm squeeze "
                    "on a 30 mm cube (URDF finger meshes: 5.9/20.9/29.9/65.9 mm at 0/25/40/100). The real gap at 40 is "
                    "unmeasured: measure it with calipers before any tighter close")
p.add_argument("--close-time", type=float, default=2.0, help="s to wait for the gripper before Lift")
p.add_argument("--demo-pick", action="store_true", help="self-test: cube on the table 22 cm out, then Above/Descend/Close/Lift")
p.add_argument("--target-size", type=float, default=0.03,
               help="edge length in metres of the draggable IK target cube /World/target")
p.add_argument("--jog-rate", type=float, default=40.0,
               help="max slider-following speed, normalized units/s (40 = 32-49 deg/s on the shoulder, elbow, wrist pitch "
                    "and base, ~70 on the rolls)")
p.add_argument("--reach-pose", default="0,-50,50,0,30,0,50",
               help="7 normalized values. Posture the IK regularises toward and the 'Reach' "
                    "button target. Starting guess - tune it to a comfortable mid-workspace pose.")
p.add_argument("--rest-pose", default="0,-100,97.2,1.0,0.6,0,50",
               help="7 normalized values. The 'Rest' button target: the folded parking posture. "
                    "Default recorded 2026-09-19 from the bridge log with the real Viola parked there "
                    "(real read 0/-100/97/0.5/3/0/49.5; wrist shows the known +2 droop).")
p.add_argument("--position-cost", type=float, default=5.0)
p.add_argument("--orientation-cost", type=float, default=0.05,
               help="0.05 = position-dominant dragging with a natural posture (recommended for "
                    "hand dragging). 0.3 locks the tool angle harder; 1.0 contorts near limits. "
                    "With 'align pads' only: without, IK follows the position alone (cost 0).")
p.add_argument("--posture-cost", type=float, default=2e-2)
p.add_argument("--demo-jog", action="store_true", help="self-test: script the sliders")
p.add_argument("--demo-orbit", action="store_true", help="self-test: orbit the IK target")
p.add_argument("--demo-radius", type=float, default=0.05)
p.add_argument("--demo-period", type=float, default=8.0)
args = p.parse_args()
if args.start_mode is None:
    args.start_mode = "plan" if (args.plan or args.demo_plan or args.demo_pick) else "joints"
if args.plan_floor is None:
    args.plan_floor = args.table_z
# Bad input stops here, before Isaac Sim boots: inside Kit a crash comes 15 s later and still exits 0.
if args.joint_map == "calib":
    joint_map.check_calib(p, args.follower_calib)
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    rx.bind((args.udp_host, args.udp_port))
except (OSError, OverflowError) as e:   # OverflowError: a port outside 0-65535
    p.error(f"cannot listen on udp {args.udp_host}:{args.udp_port} ({e}). If the port is in use, stop the other "
            "program or pass --udp-port")
rx.setblocking(False)
MODES = ["joints", "ik", "plan"]
CUMOTION_OFF = "cuMotion off - restart with --plan"
SERVO_MM = 4.0      # Descend/Lift done: the COMMANDED link6 (FK of the drive targets) within this of the goal
                    # (and --servo-tol-deg),
SERVO_MEAS_MM = 8.0  # and the measured link6 within this: the sim's gravity sag alone is 4.4-5.0 mm at x ~0.30,
                     # 6.5-6.9 mm at r 0.36-0.38 (Isaac logs, 'plan done')
GRIP_OPEN = 95.0    # sim gripper reading that counts as open for Descend (100 = 66 mm between the pads)
PINK_LEAD = 0.08    # rad: IK mode's command leads the measured joints by at most this per arm joint (Pink steps from
                    # last_targets clipped to it). Following lag 0.037-0.053 rad uncapped, offline 2026-09-30; a
                    # larger lead only presses a held-back arm (cube on the table: 0.35 rad = 15.8 Nm on joint2)
IK_ROT_TAU = 0.15   # s: IK mode's pads-square rotation eases toward grasp_for's with this time constant,
IK_ROT_RATE = 90.0  # deg/s: and turns no faster than this

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.robot_motion.experimental.motion_generation")
enable_extension("isaacsim.robot_motion.pink")
if any(f in __import__("sys").argv for f in ("--plan", "--demo-plan", "--demo-pick")):
    enable_extension("isaacsim.robot_motion.cumotion")

import numpy as np  # noqa: E402
import isaacsim.robot_motion.experimental.motion_generation as mg  # noqa: E402
from grasp_frame import (LINE_DEG, LINE_MM, ArmReach, R_to_quat_wxyz, ToolShape, lift_goal, line_offset,  # noqa: E402
                         link_pose, pick_poses, quat_wxyz_to_R, rot_deg, rotate_toward)  # (scripts/grasp_frame.py)
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.experimental.objects import Cube, Mesh  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot_motion.pink import PinkIKController, load_pink_robot  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # as import_viola.py:15, so a checkout loads its own assets
USD = os.path.join(REPO, "assets", "viola", "usd", "viola", "viola.usda")
URDF = os.path.join(REPO, "assets", "viola", "urdf", "viola.urdf")
ROBOT_PATH = "/World/Viola"
SCENE_PATH = "/World/scene"
TOOL_FRAME = "link6"
ARM_MAP = [(0, "joint1"), (1, "joint2"), (2, "joint3"), (3, "joint4"), (4, "joint5"), (5, "joint6")]
GRIPPER_JOINTS = ["joint7_left", "joint7_right"]
MOTOR_LABELS = ["Motor_0 base", "Motor_1 shoulder", "Motor_2 elbow", "Motor_3 wrist roll",
                "Motor_4 wrist pitch", "Motor_5 wrist rot", "gripper 0=closed 100=open"]
# vendor connect pose (lerobot_robot_viola/starai_viola.py:205-216), normalized units
CONNECT_POSE = [0.0, -100.0, 60.0, 0.0, 30.0, 0.0, 50.0]
REACH_POSE = [float(v) for v in args.reach_pose.split(",")]
assert len(REACH_POSE) == 7, "--reach-pose needs 7 values"
REST_POSE = [float(v) for v in args.rest_pose.split(",")]
assert len(REST_POSE) == 7, "--rest-pose needs 7 values"


def find_prim(stage, root, name):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return prim
    return None


def find_articulation_root(stage, root):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return root


def world_pose_wxyz(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    im = q.GetImaginary()
    return (np.array([[t[0], t[1], t[2]]], dtype=np.float32),
            np.array([[q.GetReal(), im[0], im[1], im[2]]], dtype=np.float32))


def rate_limit(target, current, max_step):
    return [c + max(-max_step, min(max_step, t - c)) for t, c in zip(target, current)]


def leader_packet(data):
    """The 7 normalized values of a leader packet, {"q": [6 arm joints -100..100, gripper 0..100]}, clamped to
    those ranges; None for anything else. The port is open to the network, so nothing from it is trusted."""
    try:
        q = json.loads(data)["q"]
        # type() and not isinstance: True is an int in Python, and a string of digits would iterate like a list
        ok = isinstance(q, list) and len(q) == 7 and all(type(v) in (int, float) and math.isfinite(v) for v in q)
    except (ValueError, KeyError, TypeError, OverflowError):   # OverflowError: an integer too large for a float
        return None
    if not ok:
        return None
    return [min(max(float(v), -100.0), 100.0) for v in q[:6]] + [min(max(float(q[6]), 0.0), 100.0)]


def main():
    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    if not args.no_ground:
        try:
            world.scene.add_default_ground_plane(z_position=args.table_z)
        except TypeError:
            world.scene.add_default_ground_plane()
            if args.table_z != 0.0:
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/defaultGroundPlane")).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, args.table_z))
        print(f"[ik] table top (ground plane) at z={args.table_z:+.3f} m")
    add_reference_to_stage(USD, ROBOT_PATH)
    stage = world.stage
    scene_prim = None
    if args.scene:
        scene_path = os.path.abspath(args.scene)
        add_reference_to_stage(scene_path, SCENE_PATH)
        scene_prim = stage.GetPrimAtPath(SCENE_PATH)
        xf = UsdGeom.Xformable(scene_prim)
        xf.ClearXformOpOrder()
        off = [float(v) for v in args.scene_offset.split(",")]
        rpy = [float(v) for v in args.scene_rpy.split(",")]
        xf.AddTranslateOp().Set(Gf.Vec3d(*off))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
        xf.AddScaleOp().Set(Gf.Vec3f(args.scene_scale, args.scene_scale, args.scene_scale))
        n_col = sum(1 for q in Usd.PrimRange(scene_prim) if q.HasAPI(UsdPhysics.CollisionAPI))
        print(f"[ik] scene {scene_path} at {SCENE_PATH} offset={off} rpy={rpy} scale={args.scene_scale} colliders={n_col}")
    art_path = find_articulation_root(stage, ROBOT_PATH)
    arm = Articulation(art_path)
    target = Cube(paths="/World/target", sizes=args.target_size, positions=[[0.3, 0.0, 0.3]])
    world.reset()

    names = list(arm.dof_names)
    pink_robot = load_pink_robot(URDF)
    lo = {n: float(v) for n, v in zip(pink_robot.controlled_joint_names, pink_robot.model.lowerPositionLimit)}
    hi = {n: float(v) for n, v in zip(pink_robot.controlled_joint_names, pink_robot.model.upperPositionLimit)}
    lower = np.array([lo[n] for n in names], dtype=np.float32)
    upper = np.array([hi[n] for n in names], dtype=np.float32)
    arm_idx = [(m, names.index(j)) for m, j in ARM_MAP]
    grip_idx = [names.index(j) for j in GRIPPER_JOINTS]
    print(f"[ik] articulation {art_path}  dofs={names}")
    # arm joints: the angles at normalized -100 / +100 (--joint-map); the gripper keeps the URDF range
    map_lo, map_hi = lower.copy(), upper.copy()
    if args.joint_map == "calib":
        for (mi, di), (a, b) in zip(arm_idx, joint_map.calib_ends(args.follower_calib)):
            map_lo[di], map_hi[di] = a, b
    print(f"[ik] joint map: {args.joint_map}. -100..100 per arm joint, deg (URDF limits, the sim's stops):")
    for mi, di in arm_idx:
        print(f"       {MOTOR_LABELS[mi]:18s} {math.degrees(map_lo[di]):+7.1f}..{math.degrees(map_hi[di]):+6.1f}   "
              f"(URDF {math.degrees(lower[di]):+7.1f}..{math.degrees(upper[di]):+6.1f})")
    if args.joint_map == "calib":
        # The sim's stops become the overlap of the URDF limits and the real arm's range (shoulder -88.1..+70.9 vs
        # +-90, base -122.1..+123.0 vs +-130.1): a sim joint past the real range echoes |n| > 100, the bridge drops
        # the whole sample and the real arm freezes (2026-10-03, shoulder at -102: 624 ticks). Every arm target
        # below is clipped to these, and cuMotion goals keep the joint margin inside them (pick_planner stops).
        for mi, di in arm_idx:
            a, b = sorted((float(map_lo[di]), float(map_hi[di])))
            if a > lower[di] or b < upper[di]:
                lower[di], upper[di] = max(lower[di], a), min(upper[di], b)
                print(f"       {MOTOR_LABELS[mi]}: sim stops narrowed to the real range "
                      f"{math.degrees(lower[di]):+.1f}..{math.degrees(upper[di]):+.1f} deg")

    def clip_arm(t):
        """Arm joint targets inside the sim's stops (the real arm's range under calib)."""
        for _, di in arm_idx:
            t[di] = min(max(t[di], lower[di]), upper[di])
        return t

    controller = PinkIKController(
        pink_robot=pink_robot, robot_joint_space=names, robot_site_space=[TOOL_FRAME],
        tool_frame=TOOL_FRAME, position_cost=args.position_cost,
        orientation_cost=args.orientation_cost, posture_cost=args.posture_cost,
        solver="osqp", dt=1.0 / 60.0,
    )
    frame_task = controller.get_frame_task()     # Descend/Lift raise its orientation cost while they servo

    def normalized_to_targets(q7):
        t = np.zeros(len(names), dtype=np.float32)
        for mi, di in arm_idx:
            u = (q7[mi] + 100.0) / 200.0
            # calib reaches past the URDF stops on some joints (elbow -100..91, wrist pitch -109..95, rolls
            # +-176..180 vs the URDF's +-90/+-130): clip, as PhysX would; with urdf the clip never acts
            t[di] = min(max(map_lo[di] + u * (map_hi[di] - map_lo[di]), lower[di]), upper[di])
        u = q7[6] / 100.0  # 0 = closed = lower, 100 = open = upper; matches real (measured 2026-09-19)
        if args.invert_gripper:
            u = 1.0 - u
        for di in grip_idx:
            t[di] = lower[di] + u * (upper[di] - lower[di])
        return t

    def to_normalized(q):
        out = []
        for _, j in ARM_MAP:
            i = names.index(j)
            out.append(float(((float(q[i]) - map_lo[i]) / (map_hi[i] - map_lo[i])) * 200.0 - 100.0))
        i = grip_idx[0]
        u = (float(q[i]) - lower[i]) / (upper[i] - lower[i])
        if args.invert_gripper:
            u = 1.0 - u
        out.append(float(u * 100.0))
        return out

    # mg.JointState.from_name requires 1-D float warp arrays (numpy is rejected)
    import warp as wp
    reach_targets = normalized_to_targets(REACH_POSE)
    reach_state = mg.JointState.from_name(
        robot_joint_space=names,
        positions=(names, wp.from_numpy(np.asarray(reach_targets, dtype=np.float32), dtype=wp.float32)))

    arm_reach = ArmReach(URDF)
    print(f"[ik] reach table from the URDF: {len(arm_reach.wr)} shoulder/elbow samples, wrist pitch range "
          f"{[round(math.degrees(v)) for v in arm_reach.pitch_range]} deg")
    tool = ToolShape(URDF)    # the gripper's mesh points in link6: the pick grasp keeps them --table-clearance up

    # --- cuMotion planner (optional): obstacles = every PhysX collider except robot/target/ground ----
    planner = follower = pick = None
    if args.plan or args.demo_plan or args.demo_pick:
        if not math.isnan(args.plan_floor):
            # the bench: cuMotion only sees PhysX colliders, so give it the floor as a big cube
            from omni.physx.scripts import utils as physx_utils
            Cube(paths="/World/plan_floor", sizes=2.0, positions=[[0.0, 0.0, args.plan_floor - 1.0]])
            physx_utils.setStaticCollider(stage.GetPrimAtPath("/World/plan_floor"))
            UsdGeom.Imageable(stage.GetPrimAtPath("/World/plan_floor")).MakeInvisible()
        if args.scene and args.scene_scale != 1.0:
            raise SystemExit("[ik] --plan needs --scene-scale 1: cuMotion's world binding rejects scaled ancestor "
                             "prims. Bake the scale into the scene USD before loading it")
        from isaacsim.robot_motion.cumotion import CumotionWorldInterface, GraphBasedMotionPlanner, load_cumotion_robot
        robot_pos, robot_ori = arm.get_world_poses()
        obstacles = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos.numpy()[0], search_box_minimum=[-5.0, -5.0, -5.0],
            search_box_maximum=[5.0, 5.0, 5.0], tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[ROBOT_PATH, "/World/target", "/World/defaultGroundPlane"])
        strategy = mg.ObstacleStrategy()
        strategy.set_default_configuration(Cube, mg.ObstacleConfiguration("obb", args.plan_safety))
        strategy.set_default_configuration(Mesh, mg.ObstacleConfiguration("triangulated_mesh", args.plan_safety))  # sphere decomposition; "mesh" is unimplemented in CumotionWorldInterface
        binding = mg.WorldBinding(world_interface=CumotionWorldInterface(), obstacle_strategy=strategy,
                                  tracked_prims=obstacles, tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION)
        binding.initialize()
        binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
        binding.synchronize_transforms()
        cumotion_robot = load_cumotion_robot(directory=os.path.dirname(URDF), urdf_filename=os.path.basename(URDF),
                                             xrdf_filename=args.xrdf)
        planner = GraphBasedMotionPlanner(cumotion_robot=cumotion_robot,
                                          cumotion_world_interface=binding.get_world_interface(), tool_frame=TOOL_FRAME)
        follower = mg.TrajectoryFollower()
        import pick_planner  # scripts/pick_planner.py: goal selection shared with scripts/plan_selftest.py
        pcfg = planner.get_graph_planner_config()
        lim = [float(v) for v in args.plan_limits.split(",")]
        lim[4] = min(lim[4], args.table_z)   # sampling box must include a lowered table
        ok_it, ok_lim, ok_zone, ok_step = pick_planner.configure(pcfg, args.plan_iterations, lim, args.plan_zone_tol,
                                                                 args.plan_step)
        print(f"[ik] planner params: max_iterations={args.plan_iterations} (set={ok_it}) task_space_limits={lim} (set={ok_lim}) "
              f"zone_tol={args.plan_zone_tol} (set={ok_zone}) step_size={args.plan_step} (set={ok_step})")
        pick = pick_planner.PickPlanner(cumotion_robot.robot_description, binding.get_world_interface().world_view, pcfg,
                                        table_z=args.table_z, max_tilt=args.max_tilt, grasp_depth=args.grasp_depth,
                                        pre_grasp=args.pre_grasp, tool=tool, clearance=args.table_clearance,
                                        stops=[(float(lower[di]), float(upper[di])) for _, di in arm_idx])
        plan_arm_joints = [j for _, j in ARM_MAP]
        plan_arm_idx = [names.index(j) for j in plan_arm_joints]
        print(f"[ik] cuMotion ready: {os.path.join(os.path.dirname(URDF), args.xrdf)}, {len(obstacles)} obstacle prims tracked, safety {args.plan_safety} m, "
              f"vel {args.plan_vel} rad/s, acc {args.plan_acc} rad/s^2, max_iterations {args.plan_iterations}")

    # --- shared control state (panel, keyboard and demos all write here) -------
    state = {"mode": args.start_mode, "slider": list(CONNECT_POSE), "sync_ui": None,
             "plan_request": False, "plan_active": False, "align": not args.no_align,
             "servo": None, "pick_auto": False, "pick_next": None, "pick_phase": None, "plan_override": None,
             "close_wait": None}

    # --- control panel (GUI only) --------------------------------------------------
    if not args.headless:
        try:
            import omni.ui as ui

            STYLE = {"Button": {"padding": 2, "margin": 1}, "Button.Label": {"font_size": 12},
                     "Label": {"font_size": 12}, "ComboBox": {"font_size": 12}, "CollapsableFrame": {"font_size": 12}}
            SHORT = ["M0 base", "M1 shoulder", "M2 elbow", "M3 roll", "M4 pitch", "M5 rot", "grip 0cl 100op"]
            win = ui.Window("Viola Control", width=340, height=470)
            models = []
            with win.frame:
                with ui.VStack(spacing=2, style=STYLE):
                    with ui.HStack(height=20, spacing=4):
                        ui.Label("Source", width=48)
                        combo = ui.ComboBox(MODES.index(state["mode"]), "Sliders", "IK: follow the cube",
                                            "Plan: place the cube, then Plan")

                    def on_mode(m, *_):
                        state["mode"] = MODES[max(0, min(2, m.get_item_value_model().as_int))]
                    combo.model.add_item_changed_fn(on_mode)
                    state["push_mode"] = lambda mode: combo.model.get_item_value_model().set_value(MODES.index(mode))
                    status = ui.Label("", height=16)
                    with ui.CollapsableFrame("Joints", collapsed=False):
                        with ui.VStack(spacing=1):
                            for i, label in enumerate(SHORT):
                                with ui.HStack(height=18):
                                    ui.Label(label, width=96)
                                    lo_i, hi_i = (0.0, 100.0) if i == 6 else (-100.0, 100.0)
                                    sl = ui.FloatSlider(min=lo_i, max=hi_i, step=0.5, height=16)
                                    sl.model.set_value(state["slider"][i])

                                    def on_slide(m, i=i):
                                        state["slider"][i] = float(m.as_float)
                                    sl.model.add_value_changed_fn(on_slide)
                                    models.append(sl.model)
                    with ui.CollapsableFrame("Poses", collapsed=False):
                        with ui.HStack(height=20, spacing=3):
                            ui.Button("Sync<-robot", clicked_fn=lambda: state.__setitem__("sync_ui", "robot"))
                            ui.Button("Reach", clicked_fn=lambda: state.__setitem__("sync_ui", "reach"))
                            ui.Button("Connect", clicked_fn=lambda: state.__setitem__("sync_ui", "connect"))
                            ui.Button("Rest", clicked_fn=lambda: state.__setitem__("sync_ui", "rest"))
                            ui.Button("Probe", clicked_fn=lambda: state.__setitem__("probe", True))
                    with ui.CollapsableFrame("Plan & pick", collapsed=False):
                        with ui.VStack(spacing=2):
                            with ui.HStack(height=20, spacing=3):
                                ui.Button("Plan to target" if planner else "Plan (needs --plan)",
                                          clicked_fn=lambda: state.__setitem__("plan_request", True))
                                align_cb = ui.CheckBox(width=18)
                                align_cb.model.set_value(state["align"])
                                align_cb.model.add_value_changed_fn(lambda m: state.__setitem__("align", bool(m.as_bool)))
                                ui.Label("align pads", width=70)
                            with ui.HStack(height=20, spacing=3):
                                for text, ph in (("Above", "above"), ("Descend", "descend"), ("Close", "close"), ("Lift", "lift")):
                                    ui.Button(text, clicked_fn=lambda ph=ph: (state.__setitem__("pick_auto", False),
                                                                             state.__setitem__("pick_next", ph)))
                                ui.Button("Pick all", clicked_fn=lambda: (state.__setitem__("pick_auto", True),
                                                                          state.__setitem__("pick_next", "above")))
                            plan_status = ui.Label("" if planner else CUMOTION_OFF, height=16)
                    ui.Label("arm -100..100  gripper 0=closed 100=open  keys O/C , .", height=14)

            def push_sliders(vals):
                for m, v in zip(models, vals):
                    m.set_value(float(v))
            state["push_sliders"] = push_sliders
            state["status"] = status
            state["plan_status"] = plan_status
            print("[ik] control panel open: Window 'Viola Control'")
        except Exception as e:  # pragma: no cover
            print(f"[ik] panel unavailable: {e}")

        try:
            import carb
            import omni.appwindow

            def on_key(event, *_):
                if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                    return True
                k = event.input
                g = state["slider"][6]
                if k == carb.input.KeyboardInput.O:
                    g = 100.0  # open (measured 2026-09-19: real 100 = open)
                elif k == carb.input.KeyboardInput.C:
                    g = 0.0  # closed
                elif k == carb.input.KeyboardInput.COMMA:
                    g = max(0.0, g - 5.0)
                elif k == carb.input.KeyboardInput.PERIOD:
                    g = min(100.0, g + 5.0)
                else:
                    return True
                state["slider"][6] = g
                if "push_sliders" in state:
                    state["push_sliders"](state["slider"])
                print(f"[ik] gripper -> {g:.0f}")
                return True

            kb = omni.appwindow.get_default_app_window().get_keyboard()
            state["kb_sub"] = carb.input.acquire_input_interface().subscribe_to_keyboard_events(kb, on_key)
        except Exception as e:  # pragma: no cover
            print(f"[ik] keyboard unavailable: {e}")

    # --- sockets ----------------------------------------------------------------
    echo = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    echo_dest = (args.echo_host, args.echo_port)
    print(f"[ik] leader in udp {args.udp_host}:{args.udp_port}, echo out udp {args.echo_host}:{args.echo_port}")

    link6 = find_prim(stage, ROBOT_PATH, TOOL_FRAME)
    if link6 is None:
        raise SystemExit(f"[ik] prim '{TOOL_FRAME}' not found under {ROBOT_PATH}")

    def grasp_for(cube_pos, cube_quat_wxyz):
        """The pads-square grasp IK mode turns toward, every frame, by the fast ArmReach table (grasp_frame.pick_poses);
        Above and Plan to target use pick_planner's exact IK search instead. The tilt is the grasp's alone: with the
        pre-grasp's reach too it jumped 8-30 deg from one frame to the next 22 times in two recorded IK drags of
        2026-09-30, without it 11-12 deg 3 times (offline replay)."""
        _, q6 = world_pose_wxyz(link6)
        return pick_poses(cube_pos, quat_wxyz_to_R(np.asarray(cube_quat_wxyz).reshape(-1)), quat_wxyz_to_R(q6[0]),
                          table_z=args.table_z, max_tilt_deg=args.max_tilt, grasp_depth=args.grasp_depth,
                          pre_grasp=0.0, arm=arm_reach)

    def cube_pose():
        tp, to = target.get_world_poses()
        return tp.numpy()[0].astype(float), to.numpy()[0].astype(float)

    def planned_grasp(p_cube, q_cube):
        """The grasp the last Above planned to, while the cube has not moved since (1 mm, ~1 deg)."""
        c = state.get("grasp")
        if c and np.linalg.norm(p_cube - c[0]) < 1e-3 and abs(float(np.dot(q_cube, c[1]))) > 0.99996:
            return c[2]
        return None

    def hold_command():
        """JOINTS holds the last drive targets. Holding the measured joints instead lets gravity pull the arm down by
        its sag once more: 1.5-5.4 mm at the gripper over the table grasps, PD + gravity model of the drives
        (offline 2026-09-30); a servo that started from the measured pose dipped by as much."""
        cmd = to_normalized(last_targets)
        cmd[6] = state["slider"][6]
        state["slider"][:6] = cmd[:6]
        state["hold"] = cmd
        if "push_sliders" in state:
            state["push_sliders"](state["slider"])

    def end_servo():
        """A Descend/Lift servo ends, done or not: hold its last command, Pink back to --orientation-cost, Plan mode."""
        hold_command()
        state["servo"] = None
        frame_task.set_orientation_cost(float(args.orientation_cost))
        state["mode"] = "plan"
        if "push_mode" in state:
            state["push_mode"]("plan")

    def abort_pick(why):
        """Any failure aborts the pick where it is, the arm holding its last command. Only a running Descend/Lift
        servo hands the panel back to Plan mode; a failed button in IK or Joints mode leaves the mode alone."""
        print(f"[ik] pick aborted: {why}")
        if "plan_status" in state:
            state["plan_status"].text = f"pick aborted, {why}"
        if state.get("servo"):
            end_servo()
        else:
            hold_command()
        for k in ("pick_next", "pick_phase", "grip_wait", "close_wait"):
            state[k] = None
        state["pick_auto"] = False

    def start_pick_phase(phase, sim_time):
        """Above: plan to the pre-grasp pose. Descend: open the gripper, then servo down the approach line to the
        grasp. Close: gripper, then wait. Lift: servo back out along the approach (grasp_frame.lift_goal)."""
        p_cube, q_cube = cube_pose()
        state["pick_phase"] = phase
        if phase == "above":
            if planner is None:
                # Descend from wherever the arm is would be an unplanned straight line that nothing checks
                # against the table or the scene; from Rest it passes through an off-centre cube (offline, 2026-09-30)
                msg = f"{CUMOTION_OFF}; pick stopped"
                print(f"[ik] pick: {msg}")
                if "plan_status" in state:
                    state["plan_status"].text = msg
                state["pick_phase"] = None
                state["pick_auto"] = False
                return
            state["plan_override"] = "above"     # the plan branch computes the grasp after syncing the scene
            state["plan_request"] = True
            return
        if phase == "close":
            state["slider"][6] = float(args.grip_close)
            if "push_sliders" in state:
                state["push_sliders"](state["slider"])
            # done when the sim gripper reads --grip-close, or has stopped on the cube after its command got there:
            # from Descend's 100 the sliders take (100 - 40) / --jog-rate 40 = 1.5 s wall to reach 40
            g = to_normalized(arm.get_dof_positions().numpy().reshape(-1))[6]
            travel = max(0.0, g - float(args.grip_close)) / max(float(args.jog_rate), 1e-3)
            state["close_wait"] = {"t0": sim_time, "travel": travel, "g_ref": g, "t_ref": sim_time}
            print(f"[ik] pick: close, gripper {g:.0f} -> {args.grip_close:.0f} (~{travel:.1f} s at --jog-rate)")
            return
        g = planned_grasp(p_cube, q_cube)
        p6, q6 = world_pose_wxyz(link6)
        p_now, R_now = p6[0].astype(float), quat_wxyz_to_R(q6[0])
        if phase == "descend":
            # only down the approach line of Above's grasp: its pre-grasp was planned clear of the floor and of the
            # arm itself, the grasp keeps --table-clearance, and the line runs between them at a fixed rotation
            if g is None:
                abort_pick("Descend needs the grasp from Above for the cube where it is now: press Above")
                return
            mm, deg = line_offset(g, p_now, R_now)
            if mm > LINE_MM or deg > LINE_DEG:
                abort_pick(f"Descend starts on the approach line, link6 is {mm:.0f} mm / {deg:.0f} deg off it: press Above")
                return
            grip = to_normalized(arm.get_dof_positions().numpy().reshape(-1))[6]
            if grip < GRIP_OPEN:
                state["slider"][6] = 100.0
                if "push_sliders" in state:
                    state["push_sliders"](state["slider"])
                state["grip_wait"] = sim_time + args.open_time
                print(f"[ik] pick: opening the gripper ({grip:.0f} -> 100) before Descend")
                return
            p1, R1, T, nxt = g.p_grasp, g.R, args.descend_time, "close"
            floor = args.table_clearance / 2          # the goal keeps --table-clearance: stop well before the table
        else:
            p1, R1, why = lift_goal(p_now, R_now, args.lift_height, g)
            if p1 is None:
                abort_pick(f"Lift: {why}")
                return
            T, nxt = args.lift_time, None
            floor = min(args.table_clearance / 2, tool.height(R_now, p_now, args.table_z) - 0.003)   # it only rises
        # the line starts where the arm is told to be (its last drive targets), not where it sags to: starting from
        # the measured pose re-anchors Pink there and the arm drops by its sag (up to 4.6 mm in the offline model)
        _, p_cmd = link_pose(URDF, [float(last_targets[di]) for _, di in arm_idx])
        state["servo"] = {"name": phase, "p0": p_cmd, "p1": p1, "R": R1, "q": R_to_quat_wxyz(R1), "t0": sim_time,
                          "T": T, "next": nxt, "floor": floor}
        how = "down the approach" if phase == "descend" else ("back out along the approach" if g else "straight up")
        print(f"[ik] pick: {phase} {np.linalg.norm(p1 - p_cmd) * 1000:.0f} mm {how} to {p1.round(3).tolist()}, the "
              f"gripper {tool.height(R1, p1, args.table_z) * 1000:.0f} mm above the table there")

    active = None
    last_leader = -1e9
    last_bad_packet = -1e9
    leader_q = None
    need_reset = True                 # Pink (re)starts from the measured joints
    ik_R = None                       # IK mode's pads-square target rotation, eased (IK_ROT_TAU, IK_ROT_RATE)
    posed = False
    frames = 0
    # the loop clock, in WALL seconds (sum of dt_wall), not 1/60 s per Kit frame: Kit runs this loop unthrottled at
    # 100-117 Hz, and on a 1/60 s clock the planned moves, the servos and the gripper waits ran 1.3-1.8x fast on the
    # real arm, which follows the echo in wall time. It times the TrajectoryFollower (trajectory_follower.py:98
    # t - start_time), the Descend/Lift servo, the pick waits and the demos; Pink ignores t (pink_ik_controller.py:218)
    sim_time = 0.0
    demo_center = None
    live = True
    cmd_norm = list(CONNECT_POSE)     # last normalized command actually applied (7)
    last_targets = None
    last_wall = time.time()
    CLOSE_TOL = 3.0           # Close is done within 3 units of --grip-close ...
    CLOSE_STILL = 0.5         # ... or once the gripper moved under 0.5 units (0.33 mm)
    CLOSE_STILL_S = 0.3       # for 0.3 s after its command got there: stopped on the cube
    ECHO_LO, ECHO_HI, ECHO_SNAP = [-100.0] * 6 + [0.0], [100.0] * 6 + [100.0], 0.5   # the range the arm-side bridge accepts

    while simulation_app.is_running():
        now = time.time()
        dt_wall = min(max(now - last_wall, 1e-3), 0.1)
        last_wall = now

        latest = None
        while True:
            try:
                latest = rx.recv(4096)
            except BlockingIOError:
                break
        if latest is not None:
            q = leader_packet(latest)
            if q is not None:
                leader_q, last_leader = q, now
            elif now - last_bad_packet > 1.0:   # say so, at most once a second
                last_bad_packet = now
                print(f"[ik] ignored a packet on udp/{args.udp_port}: expected JSON {{\"q\": [7 numbers]}}")

        if not world.is_playing() or not arm.is_physics_tensor_entity_valid():
            if live:
                print("[ik] physics not live - holding")
                live = False
            world.step(render=not args.headless)
            continue
        if not live:
            live = True
            need_reset = True
            print("[ik] physics back - resuming")

        if not posed:
            q0 = normalized_to_targets(CONNECT_POSE)
            arm.set_dof_positions(q0)
            arm.set_dof_position_targets(q0)
            last_targets = q0.copy()
            world.step(render=False)
            pos, _ = world_pose_wxyz(link6)
            target.set_world_poses(positions=pos)   # position only: the cube stays upright
            demo_center = pos.copy()
            posed = True
            print(f"[ik] start pose set (connect pose); target parked at EE {pos[0].round(3).tolist()}")
            continue

        if state.get("plan_settle") and sim_time >= state["plan_settle"]:
            state["plan_settle"] = None
            pos, _ = world_pose_wxyz(link6)
            err = float(np.linalg.norm(pos[0] - np.array(state["plan_goal"]))) * 1000
            print(f"[ik] plan done: ee={pos[0].round(4).tolist()} goal={np.round(state['plan_goal'], 4).tolist()} err={err:.1f} mm")
            if "plan_status" in state:
                state["plan_status"].text = f"done, err {err:.1f} mm"
            if state.get("pick_phase") == "above":
                state["pick_phase"] = None
                if state.get("pick_auto"):
                    state["pick_next"] = "descend"

        # panel buttons
        if state["sync_ui"]:
            what = state.pop("sync_ui")
            if what == "robot":
                state["slider"] = to_normalized(arm.get_dof_positions().numpy().reshape(-1))
            elif what == "reach":
                state["slider"] = list(REACH_POSE)
            elif what == "connect":
                state["slider"] = list(CONNECT_POSE)
            elif what == "rest":
                state["slider"] = list(REST_POSE)
            if "push_sliders" in state:
                state["push_sliders"](state["slider"])
            state["sync_ui"] = None

        # demos (headless self-tests)
        if args.demo_jog:
            if 2.0 <= sim_time < 8.0:
                state["slider"] = list(REACH_POSE)
            elif sim_time >= 8.0:
                state["slider"] = list(CONNECT_POSE)
        if args.demo_orbit:
            state["mode"] = "ik"
        if args.demo_plan and sim_time >= 2.0 and not state.get("demo_plan_placed"):
            tp = np.array([[float(v) for v in args.demo_plan_target.split(",")]], dtype=np.float32)
            target.set_world_poses(positions=tp)
            state["demo_plan_placed"] = True
            print(f"[ik] demo: target placed at {tp[0].round(3).tolist()}; pressing Plan in 1 s")
        if args.demo_plan and sim_time >= 3.0 and not state.get("demo_plan_sent"):
            tp_now, _ = target.get_world_poses()
            print(f"[ik] demo: target is at {tp_now.numpy()[0].round(3).tolist()} when Plan is pressed")
            state["plan_request"] = True
            state["demo_plan_sent"] = True
        if args.demo_pick and sim_time >= 2.0 and not state.get("demo_pick_placed"):
            tp = np.array([[0.22, 0.05, args.table_z + 0.015]], dtype=np.float32)
            target.set_world_poses(positions=tp, orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
            state["demo_pick_placed"] = True
            print(f"[ik] demo: cube on the table at {tp[0].round(3).tolist()}; Pick all in 1 s")
        if args.demo_pick and sim_time >= 3.0 and not state.get("demo_pick_sent"):
            state["pick_auto"] = True
            state["pick_next"] = "above"
            state["demo_pick_sent"] = True

        leader_live = (now - last_leader) < args.leader_timeout

        # --- pick sequence: Above (plan) -> Descend (open, servo) -> Close (wait) -> Lift (servo) ----
        if state.get("grip_wait"):
            g_open = to_normalized(arm.get_dof_positions().numpy().reshape(-1))[6]
            if g_open >= GRIP_OPEN:
                state["grip_wait"] = None
                print(f"[ik] pick: gripper open ({g_open:.0f})")
                state["pick_next"] = "descend"
            elif sim_time >= state["grip_wait"]:
                abort_pick(f"the gripper did not open in {args.open_time:.1f} s (it reads {g_open:.0f} of 100)")
        if (state.get("pick_next") and not state["plan_active"] and not state.get("servo") and not state.get("close_wait")
                and not state.get("grip_wait")):
            if leader_live:
                print("[ik] pick ignored: leader is live")
                state["pick_next"] = None
                state["pick_auto"] = False
            else:
                phase = state.pop("pick_next")
                try:
                    start_pick_phase(phase, sim_time)
                except Exception as e:  # a failed step must not stop the sim loop the real arm follows
                    abort_pick(f"{phase} failed: {type(e).__name__}: {e}")
        cw = state.get("close_wait")
        if cw:
            q_g = arm.get_dof_positions().numpy().reshape(-1)
            g_c = to_normalized(q_g)[6]
            if abs(g_c - cw["g_ref"]) > CLOSE_STILL:
                cw["g_ref"], cw["t_ref"] = g_c, sim_time
            reached = abs(g_c - float(args.grip_close)) <= CLOSE_TOL
            stopped = sim_time - cw["t0"] >= cw["travel"] and sim_time - cw["t_ref"] >= CLOSE_STILL_S
            if reached or stopped:
                state["close_wait"] = None
                print(f"[ik] pick: close done ({'at --grip-close' if reached else 'stopped, on the cube'}), gripper cmd "
                      f"{state['slider'][6]:.0f}, sim reads {g_c:.0f}, joint7 {float(q_g[grip_idx[0]])*1000:+.1f} mm")
                state["pick_phase"] = None
                if state.get("pick_auto"):
                    state["pick_next"] = "lift"
            elif sim_time - cw["t0"] >= cw["travel"] + args.close_time:
                abort_pick(f"the gripper did not close in {cw['travel'] + args.close_time:.1f} s "
                           f"(it reads {g_c:.0f}, --grip-close {args.grip_close:.0f})")
        if state.pop("probe", False):
            p6, q6 = world_pose_wxyz(link6)
            tip = p6[0].astype(float) + quat_wxyz_to_R(q6[0])[:, 2] * 0.0183
            low = tool.height(quat_wxyz_to_R(q6[0]), p6[0], args.table_z)
            print(f"[ik] probe: link6=({p6[0][0]:+.4f},{p6[0][1]:+.4f},{p6[0][2]:+.4f}) "
                  f"fingertip=({tip[0]:+.4f},{tip[1]:+.4f},{tip[2]:+.4f}) "
                  f"fingertip above sim table: {(tip[2]-args.table_z)*1000:.1f} mm, lowest gripper point {low * 1000:.1f} mm  "
                  f"(logged to outputs/probe_log.csv)")
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/probe_log.csv", "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f},"
                         f"{p6[0][0]:.4f},{p6[0][1]:.4f},{p6[0][2]:.4f}\n")

        # 'Plan to target': collision-free path from where the arm is to the target's position
        if state["plan_request"]:
            state["plan_request"] = False
            if planner is None:
                print(f"[ik] plan requested but {CUMOTION_OFF}")
                if "plan_status" in state:
                    state["plan_status"].text = CUMOTION_OFF
            elif leader_live:
                print("[ik] plan ignored: leader is live")
            else:
                if state["mode"] == "joints":
                    print("[ik] note: in Sliders mode the target rides on the gripper, so there is nothing to plan to; "
                          "pick 'Plan: place the cube, then Plan' in the dropdown, drag the cube, then press Plan to target")
                try:
                    binding.synchronize_transforms()  # scene may have been moved in the viewport
                    q_now = arm.get_dof_positions().numpy().reshape(-1)[plan_arm_idx].astype(np.float64)
                    p_cube, q_cube = cube_pose()
                    R_now = quat_wxyz_to_R(world_pose_wxyz(link6)[1][0])
                    if state.pop("plan_override", None) == "above":
                        plan = pick.plan_above(q_now, p_cube, quat_wxyz_to_R(q_cube), R_now)
                    else:
                        plan = pick.plan_to_target(q_now, p_cube, quat_wxyz_to_R(q_cube), R_now, align=state["align"])
                    g = plan.grasp                                  # set when the goal is the pick's pre-grasp
                    if g is not None and g.ok:
                        state["grasp"] = (p_cube, q_cube, g)     # Descend and Lift reuse it (start_pick_phase)
                        print(f"[ik] pick: above  cube={g.p_cube.round(3).tolist()} grasp={g.p_grasp.round(3).tolist()} "
                              f"above={g.p_above.round(3).tolist()} tilt={g.tilt:.0f} deg depth={g.depth * 1000:.0f} mm "
                              f"(gripper {tool.height(g.R, g.p_grasp, args.table_z) * 1000:.0f} mm above the table)")
                    goal, kind, ms = plan.goal, plan.kind, plan.ms
                    if plan.waypoints is None:
                        # the reason first: the panel line is one line wide (pick_planner.py says why: start in
                        # collision, no IK for the goal, goal too close to the floor, self-collision, blocked)
                        panel = f"no path: {plan.why}"
                        msg = f"no path to {np.round(goal, 3).tolist()} [{kind}] ({ms:.0f} ms): {plan.why}"
                        if state.get("pick_phase"):
                            msg += " -- pick aborted"
                            panel = "pick aborted, " + panel
                            state["pick_phase"] = None
                            state["pick_auto"] = False
                        state["plan_active"] = False
                    else:
                        path = mg.Path(np.array(plan.waypoints, dtype=np.float64))   # what GraphBasedMotionPlanner returns
                        traj = path.to_minimal_time_joint_trajectory(
                            max_velocities=np.full(6, args.plan_vel), max_accelerations=np.full(6, args.plan_acc),
                            robot_joint_space=names, active_joints=plan_arm_joints)
                        follower.set_trajectory(traj)
                        est0 = mg.RobotState(joints=mg.JointState.from_name(
                            robot_joint_space=names, positions=(names, arm.get_dof_positions()),
                            velocities=(names, arm.get_dof_velocities())))
                        follower.reset(est0, None, sim_time)
                        state["plan_active"] = True
                        state["plan_goal"] = goal
                        msg = panel = f"path {path.get_waypoints_count()} waypoints, {traj.duration:.1f} s, planned in {ms:.0f} ms [{kind}]"
                except Exception as e:  # a failed plan must not stop the sim loop the real arm follows
                    panel = f"no path: {type(e).__name__}: {e}"
                    msg = panel
                    if state.get("pick_phase"):
                        msg += " -- pick aborted"
                        panel = "pick aborted, " + panel
                        state["pick_phase"] = None
                        state["pick_auto"] = False
                    state["plan_active"] = False
                print(f"[ik] plan: {msg}")
                if "plan_status" in state:
                    state["plan_status"].text = panel

        src = "LEADER" if leader_live else ("PLAN" if state["plan_active"] else
                                             ("IK" if (state["mode"] == "ik" or state.get("servo")) else "JOINTS"))
        if src != active:
            print(f"[ik] source -> {src}")
            active = src
            need_reset = True
        if src != "JOINTS":
            state["hold"] = None     # hold_command() carries a servo's last command into the next JOINTS frame only

        if src == "LEADER":
            state["plan_active"] = False
            state["servo"] = None
            state["pick_next"] = None
            state["pick_auto"] = False
            state["pick_phase"] = None
            state["grip_wait"] = None
            state["close_wait"] = None
            cmd_norm = [float(v) for v in leader_q]
            t = normalized_to_targets(cmd_norm)
            arm.set_dof_position_targets(t)
            last_targets = t
            state["slider"] = list(cmd_norm)
            pos, _ = world_pose_wxyz(link6)
            target.set_world_poses(positions=pos)   # position only: the cube stays upright

        elif src == "PLAN":
            est = mg.RobotState(joints=mg.JointState.from_name(
                robot_joint_space=names, positions=(names, arm.get_dof_positions()),
                velocities=(names, arm.get_dof_velocities())))
            desired = follower.forward(est, None, sim_time)
            if desired is None or desired.joints is None or desired.joints.positions is None:
                state["plan_active"] = False
                state["mode"] = "plan"
                if "push_mode" in state:
                    state["push_mode"]("plan")
                # hand the FINAL WAYPOINT (last commanded targets) to the sliders, not the lagging
                # current position, so JOINTS mode finishes the move at the jog rate
                fin = to_normalized(last_targets)
                state["slider"][:6] = fin[:6]
                cmd_norm = to_normalized(arm.get_dof_positions().numpy().reshape(-1))
                cmd_norm[6] = state["slider"][6]
                if "push_sliders" in state:
                    state["push_sliders"](state["slider"])
                state["plan_settle"] = sim_time + 1.5  # report the error once the drives have caught up
                print("[ik] plan trajectory finished; settling")
                continue
            else:
                dj = desired.joints
                pos_arr = np.asarray(dj.positions.numpy() if hasattr(dj.positions, "numpy")
                                     else dj.positions, dtype=np.float32).reshape(-1)
                idx = np.asarray(dj.position_indices.numpy() if hasattr(dj.position_indices, "numpy")
                                 else dj.position_indices).reshape(-1).astype(int)
                t = last_targets.copy()
                for i, di in enumerate(idx):
                    t[di] = pos_arr[i]
                gt = normalized_to_targets(state["slider"])
                for di in grip_idx:
                    t[di] = gt[di]
                arm.set_dof_position_targets(clip_arm(t))
                last_targets = t
                if args.demo_plan and frames % 15 == 0:
                    pos, _ = world_pose_wxyz(link6)
                    print(f"[ik] PLAN t={sim_time:5.2f} ee=({pos[0][0]:+.3f},{pos[0][1]:+.3f},{pos[0][2]:+.3f})")
            cmd_norm = to_normalized(arm.get_dof_positions().numpy().reshape(-1))
            cmd_norm[6] = state["slider"][6]
            state["slider"][:6] = cmd_norm[:6]

        elif src == "JOINTS":
            if state.get("hold") is not None:
                cmd_norm = state.pop("hold")      # from the servo's last command, not from the sagging measured joints
            cmd_norm = rate_limit(state["slider"], cmd_norm, args.jog_rate * dt_wall)
            t = normalized_to_targets(cmd_norm)
            arm.set_dof_position_targets(t)
            last_targets = t
            if state["mode"] == "joints":
                # keep the target on the gripper so a switch to IK never jumps.
                # In "plan" mode the cube must stay where the user put it.
                pos, _ = world_pose_wxyz(link6)
                target.set_world_poses(positions=pos)   # position only: the cube stays upright

        else:  # IK
            if args.demo_orbit and demo_center is not None:
                a = 2 * math.pi * (sim_time / args.demo_period)
                pos = demo_center + np.array([[0.0, args.demo_radius * math.sin(a),
                                               args.demo_radius * (1 - math.cos(a)) * 0.5]], dtype=np.float32)
                target.set_world_poses(positions=pos)
            # Pink steps from its own last command, not from the measured joints: the shoulder drives settle ~20x slower
            # than the wrist's (import_viola.py:22-31), and restarting from the lagging arm every frame let the wrist
            # rolls take up the lag and counter-rotate (recorded IK drags 2026-09-30: 275-328 direction reversals
            # a minute in the echo, joint4 the most). Back to the measured joints on a source switch and after a
            # failed solve; never more than PINK_LEAD ahead of them (the arm held back). Clamped inside the
            # limits: PhysX read joint5 1-3 urad past +1.57 and Pink's check_limits failed (logged 2026-09-30).
            q_meas = arm.get_dof_positions().numpy().reshape(-1)
            q_pink = q_meas if need_reset else np.clip(last_targets, q_meas - PINK_LEAD, q_meas + PINK_LEAD)
            q_pink = np.clip(q_pink, lower + 1e-5, upper - 1e-5).astype(np.float32)
            # the gripper follows the slider at --jog-rate, as in JOINTS (Close after Descend travels 100 -> 40)
            grip_cmd = rate_limit([state["slider"][6]], [cmd_norm[6]], args.jog_rate * dt_wall)[0]
            est = mg.RobotState(joints=mg.JointState.from_name(
                robot_joint_space=names, positions=(names, wp.from_numpy(q_pink, dtype=wp.float32)),
                velocities=(names, arm.get_dof_velocities())))
            sv = state.get("servo")
            if sv:
                u = min(1.0, max(0.0, (sim_time - sv["t0"]) / max(sv["T"], 1e-3)))
                u = u * u * (3.0 - 2.0 * u)   # smoothstep: gentle start and stop
                p_i = (1.0 - u) * sv["p0"] + u * sv["p1"]
                tp = wp.array(np.asarray(p_i, dtype=np.float32).reshape(1, 3), dtype=wp.float32)
                to = wp.array(np.asarray(sv["q"], dtype=np.float32).reshape(1, 4), dtype=wp.float32)
                ik_R, ori_cost = None, args.servo_orientation_cost   # 0.05 leaves Descend/Lift up to 26 deg off square
            elif state["align"]:
                # the pads-square rotation turns smoothly from where the tool is toward grasp_for's, which steps: the
                # tilt by 1-12 deg, the pads 90 deg where the reach direction crosses the cube's diagonal
                tp, to = target.get_world_poses()
                if ik_R is None or need_reset:
                    ik_R = quat_wxyz_to_R(world_pose_wxyz(link6)[1][0])
                ik_R = rotate_toward(ik_R, grasp_for(tp.numpy()[0], to.numpy()[0]).R, frac=(1.0 / 60.0) / IK_ROT_TAU,
                                     max_deg=IK_ROT_RATE / 60.0)
                to = wp.array(R_to_quat_wxyz(ik_R).astype(np.float32).reshape(1, 4), dtype=wp.float32)
                ori_cost = args.orientation_cost
            else:
                # position only: the cube prim's own upright rotation as the target pulled the gripper to point up,
                # the wrist pitch to its limit (offline 2026-09-30)
                tp, to = target.get_world_poses()
                ik_R, ori_cost = None, 0.0
            if not sv:
                # link6 at a table cube's centre puts the fingertips ~3 mm under the table (pick_planner.py TABLE_BAND);
                # the bridge sends these joints and the real arm sags further (docs/joint-map.md). The target
                # rises until the gripper model is --table-clearance above --table-z; without align, at the tool's
                # current rotation.
                p_t = tp.numpy()[0].astype(np.float64)
                R_t = ik_R if ik_R is not None else quat_wxyz_to_R(world_pose_wxyz(link6)[1][0])
                rise = args.table_clearance - tool.height(R_t, p_t, args.table_z)
                if rise > 0:
                    p_t[2] += rise
                    tp = wp.array(p_t.astype(np.float32).reshape(1, 3), dtype=wp.float32)
            setp = mg.RobotState(
                sites=mg.SpatialState.from_name(spatial_space=[TOOL_FRAME],
                                                positions=([TOOL_FRAME], tp), orientations=([TOOL_FRAME], to)),
                joints=reach_state,   # posture regularisation toward the reach pose
            )
            if need_reset:
                need_reset = not controller.reset(est, setp, sim_time)
            if float(frame_task.cost[3]) != ori_cost:
                frame_task.set_orientation_cost(float(ori_cost))
            desired = controller.forward(est, setp, sim_time)
            if desired is not None and desired.joints.positions is not None:
                dj = desired.joints
                pos_arr = np.asarray(dj.positions.numpy() if hasattr(dj.positions, "numpy")
                                     else dj.positions, dtype=np.float32).reshape(-1)
                idx = np.asarray(dj.position_indices.numpy() if hasattr(dj.position_indices, "numpy")
                                 else dj.position_indices).reshape(-1).astype(int)
                t = q_pink.copy()
                for i, di in enumerate(idx):
                    t[di] = pos_arr[i]
                # the whole step scales so no arm joint moves faster than --jog-rate units/s nor --plan-vel rad/s of
                # wall time (0.63 rad/s on the +-1.57 joints, 0.7 on the +-2.27 ones): Pink and PhysX allow 1 rad/s
                # per sim-second, 0.97-1.45 rad/s wall at this loop's 58-87 Hz, and the re-pose on entering IK (align
                # on) reached 64 units/s (offline 2026-09-30). Uniform, so the path is kept.
                step = max(abs(float(t[di] - q_pink[di]))
                           / min(args.jog_rate * abs(float(map_hi[di] - map_lo[di])) / 200.0, args.plan_vel)
                           for _, di in arm_idx) / dt_wall
                if step > 1.0:
                    t = q_pink + (t - q_pink) / step
                gt = normalized_to_targets(list(state["slider"][:6]) + [grip_cmd])
                for di in grip_idx:
                    t[di] = gt[di]
                t = clip_arm(t)
                arm.set_dof_position_targets(t)
                last_targets = t
                grip_norm = grip_cmd
            else:
                need_reset = True           # the drives keep the last targets; Pink restarts from the measured joints
                grip_norm = cmd_norm[6]
            # keep sliders and cmd in sync with the arm so a switch to JOINTS is seamless
            cmd_norm = to_normalized(arm.get_dof_positions().numpy().reshape(-1))
            cmd_norm[6] = grip_norm
            state["slider"][:6] = cmd_norm[:6]
            if sv:
                pos_e, q_e = world_pose_wxyz(link6)
                p_e, R_e = pos_e[0].astype(float), quat_wxyz_to_R(q_e[0])
                err_s = float(np.linalg.norm(p_e - sv["p1"])) * 1000
                ang_s = rot_deg(R_e, sv["R"])
                # done is judged on the command: the measured link6 sags under gravity, >= 4 mm beyond x ~0.25-0.28
                R_c, p_c = link_pose(URDF, [float(last_targets[di]) for _, di in arm_idx])
                err_c, ang_c = float(np.linalg.norm(p_c - sv["p1"])) * 1000, rot_deg(R_c, sv["R"])
                low = tool.height(R_e, p_e, args.table_z)
                if low < sv["floor"]:
                    # the line keeps --table-clearance (Descend) or rises (Lift): the arm is not following it
                    abort_pick(f"{sv['name']}: the gripper came down to {low * 1000:.1f} mm above the table "
                               f"(the stop is {sv['floor'] * 1000:.1f} mm)")
                elif (sim_time - sv["t0"] >= sv["T"] and err_c < SERVO_MM and ang_c < args.servo_tol_deg
                      and err_s < SERVO_MEAS_MM):
                    end_servo()
                    print(f"[ik] pick: {sv['name']} done, commanded {err_c:.1f} mm / {ang_c:.1f} deg, measured "
                          f"{err_s:.1f} mm / {ang_s:.1f} deg from its goal")
                    if "plan_status" in state:
                        state["plan_status"].text = f"{sv['name']} done, {err_s:.1f} mm / {ang_s:.1f} deg"
                    state["pick_phase"] = None
                    if sv["next"] is None:
                        if state.get("pick_auto"):
                            print("[ik] pick: sequence complete")
                        state["pick_auto"] = False
                    elif state.get("pick_auto"):
                        state["pick_next"] = sv["next"]
                elif sim_time - sv["t0"] >= sv["T"] + 2.0:
                    abort_pick(f"{sv['name']} ended commanded {err_c:.1f} mm / {ang_c:.1f} deg, measured {err_s:.1f} mm "
                               f"from its goal (done is under {SERVO_MM:.0f} mm and {args.servo_tol_deg:.0f} deg "
                               f"commanded, {SERVO_MEAS_MM:.0f} mm measured)")

        qn = arm.get_dof_positions().numpy().reshape(-1)
        # PhysX reads a joint slightly past its limit (joint5 1-3 urad, logged 2026-09-30) and the arm-side bridge drops any
        # sample outside [-100, 100] / gripper [0, 100], freezing the real arm and then
        # catching up at its slew: pull readings within ECHO_SNAP of the range onto it, send larger excursions as they are
        q_echo = [v if not (lo - ECHO_SNAP <= v <= hi + ECHO_SNAP) else min(max(v, lo), hi)
                  for v, lo, hi in zip(to_normalized(qn), ECHO_LO, ECHO_HI)]
        if args.joint_map == "calib":
            # Under calib -100..100 IS the real arm's range: an arm joint past it (PhysX overshoot at a stop) goes out
            # as the end value, as the servo bus would clamp it (starai.py:194), instead of freezing the whole arm
            over = [(MOTOR_LABELS[i], v) for i, v in enumerate(q_echo[:6]) if abs(v) > 100.0 + 2.0]
            if over and sim_time - state.get("echo_warn", -9.0) > 2.0:
                state["echo_warn"] = sim_time
                print("[ik] WARNING sim past the real arm's range, sent as its end: "
                      + ", ".join(f"{n} {v:+.1f}" for n, v in over))
            q_echo[:6] = [min(max(v, -100.0), 100.0) for v in q_echo[:6]]
        echo.sendto(json.dumps({"t": frames, "q": q_echo}).encode(), echo_dest)

        frames += 1
        sim_time += dt_wall
        if frames % 120 == 0:
            ee, _ = world_pose_wxyz(link6)
            tp, _ = target.get_world_poses()
            err = float(np.linalg.norm(ee[0] - tp.numpy()[0])) * 1000
            qn7 = to_normalized(qn)
            msg = (f"[ik] {src:6s} q=" + " ".join(f"{v:+6.1f}" for v in qn7)
                   + f"  ee=({ee[0][0]:+.3f},{ee[0][1]:+.3f},{ee[0][2]:+.3f})  target-err={err:5.1f} mm")
            print(msg)
            if "status" in state:
                state["status"].text = f"active: {src}   |   leader {'LIVE' if leader_live else 'quiet'}"
            if "push_sliders" in state and src != "JOINTS":
                state["push_sliders"](state["slider"])

        world.step(render=not args.headless)

    if scene_prim is not None:
        # print where the scene ended up so a viewport adjustment can be copied into the flags
        m = UsdGeom.Xformable(scene_prim).GetLocalTransformation()
        t = m.ExtractTranslation()
        r = m.ExtractRotation().Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        print(f"[ik] scene transform on exit: --scene-offset {t[0]:.4f},{t[1]:.4f},{t[2]:.4f} "
              f"--scene-rpy {r[0]:.2f},{r[1]:.2f},{r[2]:.2f}")
    simulation_app.close()


if __name__ == "__main__":
    main()
