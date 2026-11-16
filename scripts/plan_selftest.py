"""Offline self-test of the Viola's cuMotion planning: no Isaac Sim / Kit, no robot (cuMotion uses the GPU for its world view).

    $ISAACSIM_PATH/kit/python/bin/python3 scripts/plan_selftest.py              # every case
    $ISAACSIM_PATH/kit/python/bin/python3 scripts/plan_selftest.py --only c     # cases whose name starts with c
    $ISAACSIM_PATH/kit/python/bin/python3 scripts/plan_selftest.py --joint-map urdf  # the map until 2026-10-02

Builds the planner the way isaac_ik_mirror.py --plan builds it (viola.urdf + viola.xrdf, the plan floor
as a 2 m cube whose top is --table-z, padded by --plan-safety, the same JtRRT parameters; the defaults
are read from isaac_ik_mirror.py itself) with the standalone cuMotion package that ships inside Isaac
Sim, then runs the acceptance cases through the goal code the Isaac script calls (pick_planner.py,
grasp_frame.py). One PASS/FAIL line per case; exit status 1 if any case fails.

Joint values below are the normalized units of the panel sliders, turned into radians by --joint-map as
isaac_ik_mirror.py does (urdf: -100..100 over each joint's URDF range; calib: the follower calibration).
"""

import argparse
import ast
import glob
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import joint_map  # scripts/joint_map.py (stdlib only), isaac_ik_mirror.py --joint-map

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_DIR = os.path.join(os.path.dirname(HERE), "assets", "viola", "urdf")


def isaac_defaults(flags):
    """Defaults of these isaac_ik_mirror.py flags, read from its argparse calls so the two cannot drift."""
    tree = ast.parse(open(os.path.join(HERE, "isaac_ik_mirror.py")).read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            flag = getattr(node.args[0], "value", None)
            for kw in node.keywords:
                if flag in flags and kw.arg == "default":
                    out[flag] = ast.literal_eval(kw.value)
    missing = set(flags) - set(out)
    if missing:
        raise SystemExit(f"isaac_ik_mirror.py has no literal default for {sorted(missing)}")
    return out


PLAN_FLAGS = ["--table-z", "--xrdf", "--plan-safety", "--plan-iterations", "--plan-zone-tol", "--plan-step",
              "--plan-limits", "--max-tilt", "--grasp-depth", "--pre-grasp", "--table-clearance", "--lift-height",
              "--joint-map"]
D = isaac_defaults(PLAN_FLAGS)
p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--isaac", default=os.environ.get("ISAACSIM_PATH") or "~/isaacsim",
               help="Isaac Sim 6.0.1 folder, for its cuMotion and numpy (default: $ISAACSIM_PATH, else ~/isaacsim)")
p.add_argument("--cumotion-path", default=None,
               help="directory that holds the cumotion package (default: <isaac>/exts/isaacsim.robot_motion.cumotion/pip_prebundle)")
p.add_argument("--only", default=None, help="run only the cases whose name starts with this")
for flag in PLAN_FLAGS:
    p.add_argument(flag, type=type(D[flag]), default=D[flag], help=f"as in isaac_ik_mirror.py (default {D[flag]})",
                   **({"choices": ["urdf", "calib"]} if flag == "--joint-map" else {}))
p.add_argument("--follower-calib", default=None, help="as in isaac_ik_mirror.py (default: config/viola_calibration_example.json)")
p.add_argument("--plan-floor", type=float, default=None, help="as in isaac_ik_mirror.py (default: --table-z; nan disables)")
args = p.parse_args()
args.isaac = os.path.expanduser(args.isaac)
if args.plan_floor is None:
    args.plan_floor = args.table_z
CUMOTION_DIR = args.cumotion_path or os.path.join(args.isaac, "exts", "isaacsim.robot_motion.cumotion", "pip_prebundle")
if not os.path.isdir(os.path.join(CUMOTION_DIR, "cumotion")):
    p.error(f"no cuMotion package in {CUMOTION_DIR}: set ISAACSIM_PATH to your Isaac Sim 6.0.1 folder, "
            "or pass --isaac / --cumotion-path")
if args.joint_map == "calib":
    joint_map.check_calib(p, args.follower_calib)

# Kit's python ships without numpy on its path; Isaac's own copy lives in the pip archive extension.
try:
    import numpy as np
except ImportError:
    sys.path[:0] = glob.glob(os.path.join(args.isaac, "extscache", "omni.kit.pip_archive-*", "pip_prebundle"))[:1]
    import numpy as np
sys.path.insert(0, CUMOTION_DIR)
import cumotion  # noqa: E402
import pick_planner  # noqa: E402  (scripts/pick_planner.py, the code isaac_ik_mirror.py --plan calls)
from grasp_frame import LIFT_UP_MAX, LINE_MM, ToolShape, lift_goal, line_offset, mesh_vertices  # noqa: E402

cumotion.set_log_level(cumotion.LogLevel.ERROR)
URDF = os.path.join(URDF_DIR, "viola.urdf")
XRDF = os.path.join(URDF_DIR, args.xrdf)
TOOL = pick_planner.TOOL_FRAME

# --- the planner, as isaac_ik_mirror.py --plan sets it up ------------------------------------------
# load_cumotion_robot() normalizes the URDF for urdfdom first; that is the identity on viola.urdf.
robot = cumotion.load_robot_from_file(XRDF, URDF)
kin = robot.kinematics()
world = cumotion.create_world()
if not math.isnan(args.plan_floor):
    # /World/plan_floor: a 2 m Cube with its top at the floor. CumotionWorldInterface turns a Cube into an
    # oriented box with side = 2 * half + safety (cumotion_world_interface.py:1038-1043): safety/2 per face.
    floor = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    floor.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.full(3, 2.0 + args.plan_safety))
    world.set_pose(world.add_obstacle(floor), cumotion.Pose3.from_translation(np.array([0.0, 0.0, args.plan_floor - 1.0])))
world_view = world.add_world_view()
world_view.update()
config = cumotion.create_default_motion_planner_config(robot_description=robot, tool_frame_name=TOOL, world_view=world_view)
limits = [float(v) for v in args.plan_limits.split(",")]
limits[4] = min(limits[4], args.table_z)
if not all(pick_planner.configure(config, args.plan_iterations, limits, args.plan_zone_tol, args.plan_step)):
    raise SystemExit("cuMotion rejected a planner parameter")
GRIPPER = ToolShape(URDF)                  # the gripper's mesh points, as isaac_ik_mirror.py passes them
LIMITS0 = [robot.kinematics().cspace_coord_limits(i) for i in range(robot.kinematics().num_cspace_coords())]
# --joint-map calib: the sim's stops are the overlap of the URDF limits and the real arm's range (isaac_ik_mirror.py)
STOPS = None if args.joint_map != "calib" else [
    (max(L.lower, min(e)), min(L.upper, max(e))) for L, e in zip(LIMITS0, joint_map.calib_ends(args.follower_calib))]
pick = pick_planner.PickPlanner(robot, world_view, config, table_z=args.table_z, max_tilt=args.max_tilt,
                                grasp_depth=args.grasp_depth, pre_grasp=args.pre_grasp, tool=GRIPPER,
                                clearance=args.table_clearance, stops=STOPS)

LIMITS = [kin.cspace_coord_limits(i) for i in range(kin.num_cspace_coords())]
REST = [0, -100, 97.9, 1.0, 0.9, 0]        # the folded parking pose, as read on the panel 2026-09-30
CONNECT = [0, -100, 60, 0, 30, 0]          # vendor connect pose (isaac_ik_mirror.py CONNECT_POSE)
REACH = [0, -50, 50, 0, 30, 0]             # isaac_ik_mirror.py --reach-pose
STARTS = {"Rest": REST, "Connect": CONNECT, "Reach": REACH}
UPRIGHT = np.eye(3)                        # cube rotation: upright, no yaw


ENDS = joint_map.calib_ends(args.follower_calib) if args.joint_map == "calib" else [(L.lower, L.upper) for L in LIMITS]
STOP_LIMITS = STOPS or [(L.lower, L.upper) for L in LIMITS]


def q_of(norm):
    """Panel units -> radians, as isaac_ik_mirror.py normalized_to_targets() (--joint-map, clipped to the sim's stops)."""
    return np.array([min(max(joint_map.to_rad(n, e), lo), hi) for n, e, (lo, hi) in zip(norm, ENDS, STOP_LIMITS)])


def link6(q):
    M = kin.pose(np.asarray(q, dtype=float), TOOL).matrix()
    return M[:3, :3], M[:3, 3]


def reached(plan):
    """How far the path's last waypoint leaves link6 from the goal: (mm, deg)."""
    R, p = link6(plan.waypoints[-1])
    mm = float(np.linalg.norm(p - np.asarray(plan.goal))) * 1000
    if plan.R is None:
        return mm, 0.0
    c = (np.trace(np.asarray(plan.R).T @ R) - 1.0) / 2.0
    return mm, math.degrees(math.acos(max(-1.0, min(1.0, c))))


def summary(plan):
    if plan.waypoints is None:
        return f"NO PATH [{plan.kind}] {plan.ms:.0f} ms: {plan.why or plan.code}"
    mm, deg = reached(plan)
    return f"path {len(plan.waypoints)} waypoints [{plan.kind}] {plan.ms:.0f} ms, ends {mm:.1f} mm / {deg:.1f} deg off"


def planned(plan):
    """A usable path: found, and it ends on the goal within 5 mm / 2 deg (the paths here end within 0.1 mm / 0.1 deg)."""
    if plan.waypoints is None:
        return False
    mm, deg = reached(plan)
    return mm < 5.0 and deg < 2.0


def plan_to_target(start, cube, align=True):
    q0 = q_of(start)
    return pick.plan_to_target(q0, cube, UPRIGHT, link6(q0)[0], align=align)


def plan_above(start, cube):
    q0 = q_of(start)
    return pick.plan_above(q0, cube, UPRIGHT, link6(q0)[0])


def on_table(x, y):
    return np.array([x, y, args.table_z + 0.015])   # a 30 mm cube resting on the table


# --- cases ------------------------------------------------------------------------------------------
def case_a():
    """(a) a mid-air cube: aligned Plan to target from just off the connect pose."""
    plan = plan_to_target([0, -100, 60.5, 0, 30.3, 0], np.array([0.185, 0.0, 0.226]))
    return planned(plan), summary(plan)


def rest_arc():
    """Cube centres 112.2 mm from link6 at Rest (the panel's target-err 2026-09-30), in front of the robot.
    A cube resting on the table is at least 113 mm away, so that cube was floating.
    The arc's centre stays the urdf-map Rest of that run (--joint-map calib puts link6 11 mm higher), so both
    maps plan to the same cubes."""
    _, p6 = link6(np.array([L.lower + (n + 100.0) / 200.0 * (L.upper - L.lower) for n, L in zip(REST, LIMITS)]))
    for y in (0.0, 0.03):
        for z in (0.02, 0.025, 0.03, 0.035, 0.04, 0.044, 0.05, 0.07, 0.10, 0.128):
            x = p6[0] + math.sqrt(0.1122 ** 2 - (p6[2] - z) ** 2 - (y - p6[1]) ** 2)
            yield np.array([x, y, z])


def case_b():
    """(b) aligned Plan to target along that arc from Rest; a cube near the table is planned to its pre-grasp."""
    bad, n = [], 0
    for cube in rest_arc():
        n += 1
        plan = plan_to_target(REST, cube)
        near_table = cube[2] - args.table_z < 0.045
        hover = plan.goal[2] - cube[2] > 0.05                  # the pre-grasp sits ~85 mm up the approach
        if not planned(plan) or (near_table and not hover):
            bad.append(f"({cube[0]:.3f},{cube[1]:.2f},{cube[2]:.3f}) {summary(plan)}")
    return not bad, f"{n - len(bad)}/{n} planned" + ("" if not bad else "; first failure " + bad[0])


TEAM_CUBES = [(0.15, 0.0), (0.20, 0.0), (0.22, 0.05), (0.25, 0.0), (0.28, -0.08), (0.30, 0.0), (0.18, 0.12), (0.33, 0.05)]


def case_c1():
    """(c) Above to 8 table cubes from Rest, Connect and Reach (24 plans)."""
    bad, ms = [], []
    for x, y in TEAM_CUBES:
        for name, start in STARTS.items():
            plan = plan_above(start, on_table(x, y))
            ms.append(plan.ms)
            if not planned(plan):
                bad.append(f"({x},{y}) from {name}: {summary(plan)}")
    n = len(TEAM_CUBES) * len(STARTS)
    return not bad, f"{n - len(bad)}/{n} planned, median {np.median(ms):.0f} ms" + ("" if not bad else "; first failure " + bad[0])


GRID = [(x, y) for x in np.round(np.arange(0.12, 0.3501, 0.02), 3)          # the table: 84 cube positions
        for y in np.round(np.arange(-0.15, 0.1501, 0.05), 3)]
GRID_GRASPS = {}                           # the Above grasps over GRID from Rest, filled by c2 or rest_grasp()


def rest_grasp(x, y):
    """The pick grasp Above computes for a cube resting at (x, y), the arm at Rest."""
    if (x, y) not in GRID_GRASPS:
        q0 = q_of(REST)
        GRID_GRASPS[(x, y)] = pick.grasp(on_table(x, y), UPRIGHT, link6(q0)[0], q0)
    return GRID_GRASPS[(x, y)]


def case_c2():
    """(c) Above over the table grid x 0.12..0.35, y -0.15..0.15 from Rest: plans, or says there is no pre-grasp."""
    counts, other, slowest = {"planned": 0, "no pre-grasp": 0}, [], 0.0
    for x, y in GRID:
        t0 = time.perf_counter()
        plan = plan_above(REST, on_table(x, y))
        slowest = max(slowest, time.perf_counter() - t0)
        GRID_GRASPS[(x, y)] = plan.grasp
        if planned(plan):
            counts["planned"] += 1
        elif plan.code in ("no_ik", "floor"):
            counts["no pre-grasp"] += 1
        else:
            other.append(f"({x},{y}): {summary(plan)}")
    ok = not other and slowest < 5.0
    return ok, (f"{counts['planned']} planned, {counts['no pre-grasp']} without a pre-grasp, {len(other)} other, "
                f"slowest {slowest:.1f} s" + ("" if not other else "; first other " + other[0]))


def case_c3():
    """(c) a cube too close to the base for any pre-grasp: fails fast with that reason. (x = 0.12 had none
    with the first sphere model; with the mesh-fitted one it plans at 18 deg, and x = 0.10 plans on a
    table lowered 17 mm, --table-z -0.017, the setup before 2026-10-03; measured 2026-09-30.)"""
    t0 = time.perf_counter()
    plan = plan_above(REST, on_table(0.08, 0.0))
    s = time.perf_counter() - t0
    return plan.waypoints is None and plan.code in ("no_ik", "floor") and s < 5.0, f"{summary(plan)} ({s:.1f} s)"


def case_c4():
    """(c) when JtRRT cannot finish a pose goal (forced here with a second planner config whose final
    orientation tolerance is 1e-12), Above still reaches the pre-grasp by the joint-space fallback."""
    stubborn = cumotion.create_default_motion_planner_config(robot_description=robot, tool_frame_name=TOOL,
                                                             world_view=world_view)
    pick_planner.configure(stubborn, args.plan_iterations, limits, args.plan_zone_tol, args.plan_step)
    stubborn.set_param("task_space_planning_params/orientation_target_final_tolerance",
                       cumotion.MotionPlannerConfig.ParamValue(1e-12))
    other = pick_planner.PickPlanner(robot, world_view, stubborn, table_z=args.table_z, max_tilt=args.max_tilt,
                                     grasp_depth=args.grasp_depth, pre_grasp=args.pre_grasp, tool=GRIPPER,
                                     clearance=args.table_clearance, stops=STOPS)
    q0 = q_of(REST)
    plan = other.plan_above(q0, on_table(0.22, 0.05), UPRIGHT, link6(q0)[0])
    return planned(plan) and "joint-space" in plan.kind, summary(plan)


def first_ik_tilt(p_cube, x_pads, depths, margin_deg=0.0):
    """Oracle: smallest tilt (deg) at which link6 has an exact IK solution, every joint margin_deg inside its
    limits, at each point depths[i] back from the cube centre along the approach (depths may be a function of
    the rotation), with pads along x_pads, tilting toward the base (grasp_frame.grasp_pose's geometry)."""
    r = np.array([p_cube[0], p_cube[1], 0.0])
    n1 = max((UPRIGHT[:, 0], -UPRIGHT[:, 0], UPRIGHT[:, 1], -UPRIGHT[:, 1]), key=lambda v: float(v @ r))
    m = math.radians(margin_deg)
    config = cumotion.IkConfig()
    for t in range(0, 91):
        a = np.array([0.0, 0.0, -math.cos(math.radians(t))]) + n1 * math.sin(math.radians(t))
        R = np.column_stack([x_pads, np.cross(a, x_pads), a])
        feasible = True
        for d in (depths(R) if callable(depths) else depths):
            ik = cumotion.solve_ik(kin, pick_planner.pose3(p_cube - a * d, R), TOOL, config)
            q = np.asarray(ik.cspace_position).reshape(-1)
            feasible &= bool(ik.success and all(L.lower + m <= v <= L.upper - m for v, L in zip(q, LIMITS)))
        if feasible:
            return t
    return None


def case_d1():
    """(d) table cubes where a squared top-down grasp is reachable: the pick tilts at most 4 deg more than
    exact IK needs for the grasp (at the depth that keeps the table clearance) and the pre-grasp (the 3 deg
    joint margin costs 1-3). Top-down reaches x = 0.245 with the gripper 15 mm clear on the URDF's own table (0.25 at
    10 mm, 0.26 at the bare 5 mm depth, offline 2026-09-30); with the default sim table 7 mm up (--table-z 0.007, the
    real arm's sag) the cubes sit 7 mm higher and the edge is x = 0.24."""
    rows, ok, q0 = [], True, q_of(REST)
    for x in (0.16, 0.20, 0.235, 0.24):
        g = pick.grasp(on_table(x, 0.0), UPRIGHT, link6(q0)[0], q0)

        def dep(R, g=g):
            return GRIPPER.depth(R, g.p_cube, args.grasp_depth, args.table_z + args.table_clearance)
        top_down = first_ik_tilt(g.p_cube, g.R[:, 0], lambda R: [dep(R)]) == 0
        t_min = first_ik_tilt(g.p_cube, g.R[:, 0], lambda R: [dep(R), args.grasp_depth + args.pre_grasp])
        ok &= top_down and t_min is not None and g.tilt <= t_min + 4
        rows.append(f"x={x}: {g.tilt:.0f} (min {t_min}{'' if top_down else ', top-down out of reach'})")
    return ok, "pick tilt deg " + ", ".join(rows)


def case_d2():
    """(d) cubes held 60 mm up, where link6 can sit at the centre top-down: aligned Plan to target keeps tilt 0
    where top-down clears the joint margin, else at most 4 deg more than exact IK needs. Exact top-down IK reaches
    x = 0.20 on the URDF's own table, x = 0.18 with the default sim table 7 mm up."""
    rows, ok = [], True
    for x in (0.16, 0.17, 0.18):
        cube = np.array([x, 0.0, args.table_z + 0.06])
        plan = plan_to_target(REST, cube)
        tilt = math.degrees(math.acos(min(1.0, -float(plan.R[2, 2]))))
        t_min = first_ik_tilt(cube, plan.R[:, 0], [0.0])
        square = first_ik_tilt(cube, plan.R[:, 0], [0.0], pick_planner.JOINT_MARGIN_DEG) == 0
        ok &= planned(plan) and t_min == 0 and tilt <= t_min + 4 and (tilt < 0.5 or not square)
        rows.append(f"x={x}: {tilt:.0f}{' (top-down in margin)' if square else ''}{'' if planned(plan) else ' NO PATH'}")
    return ok, "tilt deg " + ", ".join(rows)


def case_i():
    """(i) an aligned plan ends with every joint JOINT_MARGIN_DEG inside its limit: cuMotion's own pose IK left
    joint5 0.25 deg (Rest) / 0.88 deg (Connect) from its limit at (0.3, 0, 0.25) (measured offline 2026-09-30, with
    the old urdf joint map)."""
    rows, ok = [], True
    for start in ("Rest", "Connect"):
        for cube in ((0.3, 0.0, 0.25), (0.185, 0.0, 0.226)):
            plan = plan_to_target(STARTS[start], np.array(cube))
            m = pick.margin_deg(plan.waypoints[-1]) if plan.waypoints is not None else float("nan")
            ok &= planned(plan) and m >= pick_planner.JOINT_MARGIN_DEG - 1e-6
            rows.append(f"{start}->{cube}: {m:.2f}" + ("" if planned(plan) else f" {summary(plan)}"))
    return ok, "final joint margin deg " + ", ".join(rows)


def case_j():
    """(j) a cube near the table whose pre-grasp is out of reach: aligned Plan to target goes to the centre
    with the gripper model at least --table-clearance above the table (was no_ik, measured offline 2026-09-30)."""
    cube = np.array([0.15, 0.15, args.table_z + 0.04])
    plan = plan_to_target(REST, cube)
    if not planned(plan):
        return False, summary(plan)
    R, p = link6(plan.waypoints[-1])
    h = GRIPPER.height(R, p, args.table_z)
    return h >= args.table_clearance, f"{summary(plan)}, gripper {h * 1000:.1f} mm above the table"


def case_e1():
    """(e) Plan to target beside the base, align off: cuMotion's path ends with the gripper against link1
    (the meshes touch, offline 2026-09-30); the path is refused."""
    plan = plan_to_target(CONNECT, np.array([0.0, 0.08, args.table_z + 0.10]), align=False)
    return plan.waypoints is None and plan.code == "self", summary(plan)


def case_e2():
    """(e) ordinary translation plans from Rest, Connect and Reach are not refused."""
    bad = []
    for goal in ([0.25, 0.0, 0.15], [0.30, 0.05, 0.10], [0.22, 0.0, 0.10]):
        for name, start in STARTS.items():
            plan = plan_to_target(start, np.array(goal), align=False)
            if plan.waypoints is None:
                bad.append(f"{goal} from {name}: {summary(plan)}")
    return not bad, f"{9 - len(bad)}/9 planned" + ("" if not bad else "; first failure " + bad[0])


def case_e3():
    """(e) Rest overlaps link5-link1 in the sphere model (meshes 5 mm apart): a path may leave that contact
    (Rest -> Connect unfolds the elbow) but not deepen it (the elbow folded to its limit)."""
    rest, folded = q_of(REST), q_of([0, -100, 100, 1.0, 0.9, 0])
    leave = pick.self_collision(rest, [rest, q_of(CONNECT)])
    deepen = pick.self_collision(rest, [rest, folded])
    return leave is None and "deeper" in (deepen or ""), f"to Connect: {leave or 'clear'}; elbow to +100: {deepen}"


def case_e4():
    """(e) the start-overlap rule is per sphere pair: Rest -> panel (-39.4, -63.7, 98.1, 7.4, -14.3, -3.1) keeps the
    deepest link5-link1 overlap at 16.3 mm while another part of link5 moves into link1 (the URDF meshes intersect
    from 98% of the path, offline 2026-09-30, with the old urdf joint map)."""
    rest = q_of(REST)
    hit = pick.self_collision(rest, [rest, q_of([-39.4, -63.7, 98.1, 7.4, -14.3, -3.1])])
    return hit is not None, f"to the folded pose: {hit or 'clear'}"


def case_h():
    """(h) a start on a joint limit as PhysX reads it (float32; joint2 = float32(-1.57) = -1.5700000525 at Rest and
    Connect under the old urdf joint map; with calib joint2 there is -88.1 deg, so the case sets the limit itself) plans, since cuMotion refuses any start past its limits; a start 2 deg past is refused as such."""
    out, ok = [], True
    for name, start in (("Rest", REST), ("Connect", CONNECT)):
        q0 = q_of(start).astype(np.float32)
        q0[1] = np.float32(LIMITS[1].lower)
        plan = pick.plan_above(q0, on_table(0.22, 0.0), UPRIGHT, link6(q0)[0])
        ok &= planned(plan)
        out.append(f"{name}, joint2 {float(q0[1]):.10f}: {summary(plan)}")
    q0 = q_of(CONNECT)
    q0[1] = LIMITS[1].lower - math.radians(2.0)
    plan = pick.plan_to_target(q0, np.array([0.185, 0.0, 0.226]), UPRIGHT, link6(q0)[0])
    ok &= plan.waypoints is None and plan.code == "start" and "joint2" in plan.why
    out.append(f"2 deg past: {plan.why or plan.code}")
    return ok, "; ".join(out)


def case_f():
    """(f) no path says why: out of reach, too close to the floor, or the start already in collision
    (the arm at a table grasp after Descend, lowered to 3 mm clearance whatever --table-clearance is: fingertip
    spheres in the padded table)."""
    q0 = q_of(REST)
    g = pick.grasp(on_table(0.22, 0.0), UPRIGHT, link6(q0)[0], q0)
    low = g.p_grasp - np.array([0.0, 0.0, max(0.0, args.table_clearance - 0.003)])
    at_grasp = pick.ik(low, g.R, [g.q_above])
    plans = {"no_ik": plan_to_target(CONNECT, np.array([0.60, 0.0, 0.20]), align=False),
             "floor": plan_to_target(CONNECT, np.array([0.22, 0.0, args.table_z + 0.01]), align=False),
             "start": pick.plan_to_target(at_grasp, np.array([0.25, 0.0, 0.15]), UPRIGHT, g.R, align=False)}
    ok = all(p.waypoints is None and p.code == code and "min-dist" not in p.why for code, p in plans.items())
    return ok, "; ".join(f"{code}: {p.why or p.code}" for code, p in plans.items())


def case_g1():
    """(g) Descend's line, from the pre-grasp to the grasp at the grasp rotation (the servo in isaac_ik_mirror.py),
    keeps the gripper's lowest point (grasp_frame.ToolShape, the URDF meshes) at least --table-clearance above
    the table over the table grid from Rest; where the clearance set the depth, the gripper ends exactly there."""
    lows, depths, raised, bad = [], [], 0, []
    for x, y in GRID:
        g = rest_grasp(x, y)
        if not g.ok:
            continue
        h = min(GRIPPER.height(g.R, g.p_above + u * (g.p_grasp - g.p_above), args.table_z) for u in np.linspace(0, 1, 81))
        lows.append(h)
        depths.append(g.depth)
        if h < args.table_clearance - 1e-9:
            bad.append(f"({x},{y}) {h * 1000:.2f} mm above the table")
        if g.depth > args.grasp_depth + 1e-9:
            raised += 1
            if abs(h - args.table_clearance) > 1e-6:
                bad.append(f"({x},{y}) backed off to {g.depth * 1000:.1f} mm yet ends {h * 1000:.2f} mm up")
    if not lows:
        return False, "no grasp on the grid"
    return not bad, (f"{len(lows)} grasps, gripper {min(lows) * 1000:.1f}-{max(lows) * 1000:.1f} mm above the table "
                     f"(clearance {args.table_clearance * 1000:.0f}), {raised} backed off: depth {min(depths) * 1000:.1f}-"
                     f"{max(depths) * 1000:.1f} mm" + ("" if not bad else "; " + bad[0]))


def case_g2():
    """(g) the gripper model against cuMotion's reading of the URDF: at g1's pre-grasp solutions (pads level)
    the finger and link6 meshes placed by cuMotion's frames (fingers open, the XRDF default) have the model's
    lowest point; and at 3 mm clearance the model's depth on the pad side that lowers the fingertips is the
    offline fit 0.0065 + 0.017 sin(tilt) m (pinocchio and the STL meshes, 2026-09-30) within 0.6 mm, 0-30 deg."""
    links = {ln.get("name"): ln for ln in ET.parse(URDF).getroot().findall("link")}
    meshes = {f: mesh_vertices(links[f], URDF_DIR) for f in ("link6", "link7_left", "link7_right")}
    worst, n = 0.0, 0
    for x, y in GRID[::5]:
        g = rest_grasp(x, y)
        if not g.ok:
            continue
        low = min(float((V @ M[:3, :3].T + M[:3, 3])[:, 2].min())
                  for f, V in meshes.items() for M in [kin.pose(g.q_above, f).matrix()])
        R6, p6 = link6(g.q_above)
        worst, n = max(worst, abs(GRIPPER.height(R6, p6) - low)), n + 1
    cube, off = np.array([0.25, 0.0, args.table_z + 0.015]), 0.0
    for t in range(0, 31, 5):
        a = np.array([math.sin(math.radians(t)), 0.0, -math.cos(math.radians(t))])
        d = max(GRIPPER.depth(np.column_stack([x, np.cross(a, x), a]), cube, 0.0, args.table_z + 0.003)
                for x in (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])))
        off = max(off, abs(d - (0.0065 + 0.017 * math.sin(math.radians(t)))))
    return n > 0 and worst < 1e-6 and off < 0.0006, (f"model vs cuMotion frames at {n} poses: {worst * 1000:.4f} mm; "
                                                     f"depth vs the fit: {off * 1000:.2f} mm at most")


def case_g3():
    """(g) Lift (grasp_frame.lift_goal, as isaac_ik_mirror.py calls it after Close): from every grid grasp it backs out
    along the approach at the grasp rotation, rising all the way, the gripper at least --table-clearance up and no
    nearer the base than the pre-grasp, where it ends. Without Above's grasp: at most 5 cm straight up with the
    gripper near vertical; it refuses a tilted or upturned gripper, a start off the approach line, and a start at
    the pre-grasp (the goal would not be above it)."""
    bad, n = [], 0
    for x, y in GRID:
        g = rest_grasp(x, y)
        if not g.ok:
            continue
        n += 1
        p, R, why = lift_goal(g.p_grasp, g.R, args.lift_height, g)
        if p is None:
            bad.append(f"({x},{y}) refused: {why}")
            continue
        d, back = p - g.p_grasp, -g.R[:, 2]
        line = [g.p_grasp + u * d for u in np.linspace(0, 1, 81)]
        if (np.linalg.norm(np.cross(d, back)) > 1e-9 or d @ back <= 0 or d[2] < 0.001 or not np.array_equal(R, g.R)
                or np.linalg.norm(d) > np.linalg.norm(g.p_above - g.p_grasp) + 1e-12
                or min(GRIPPER.height(R, q, args.table_z) for q in line) < args.table_clearance - 1e-9
                or min(math.hypot(q[0], q[1]) for q in line) < math.hypot(g.p_above[0], g.p_above[1]) - 1e-9):
            bad.append(f"({x},{y}) lift {d.round(4).tolist()} not back out along the approach")

    def tool_at(tilt_deg):          # pads along y, approach tilted from straight down toward +x
        t = math.radians(tilt_deg)
        return np.column_stack([[0.0, 1.0, 0.0], [math.cos(t), 0.0, math.sin(t)], [math.sin(t), 0.0, -math.cos(t)]])
    p0 = np.array([0.2, 0.0, args.table_z + 0.1])
    g = next(rest_grasp(x, y) for x, y in GRID if rest_grasp(x, y).ok)
    side = np.cross(g.R[:, 2], [0.0, 0.0, 1.0])
    aside = g.p_grasp + side / np.linalg.norm(side) * 2 * LINE_MM / 1000       # twice the tolerance off the line
    p5 = lift_goal(p0, tool_at(5.0), args.lift_height)[0]
    refused = {"20 deg": lift_goal(p0, tool_at(20.0), args.lift_height)[0],
               "upturned": lift_goal(p0, tool_at(180.0), args.lift_height)[0],
               "off the line": lift_goal(aside, g.R, args.lift_height, g)[0],
               "at the pre-grasp": lift_goal(g.p_above, g.R, args.lift_height, g)[0]}
    if p5 is None or not np.allclose(p5 - p0, [0.0, 0.0, min(args.lift_height, LIFT_UP_MAX)]):
        bad.append(f"no grasp, 5 deg from vertical: {p5}")
    bad += [f"no refusal {k}" for k, v in refused.items() if v is not None]
    return not bad and n > 0, (f"{n} grasps back out to the pre-grasp ({args.lift_height * 1000:.0f} mm asked); without "
                               f"a grasp {min(args.lift_height, LIFT_UP_MAX) * 1000:.0f} mm straight up; refused "
                               f"{', '.join(k for k, v in refused.items() if v is None)} "
                               f"({line_offset(g, aside, g.R)[0]:.0f} mm off)" + ("" if not bad else "; " + bad[0]))


CASES = [("a  mid-air cube, aligned", case_a),
         ("b  arc of cubes from Rest, aligned", case_b),
         ("c1 Above, 8 cubes x 3 starts", case_c1),
         ("c2 Above, table grid from Rest", case_c2),
         ("c3 Above, no pre-grasp at x=0.08", case_c3),
         ("c4 Above, joint-space fallback", case_c4),
         ("d1 pick tilt stays minimal", case_d1),
         ("d2 top-down stays square", case_d2),
         ("e1 self-colliding path refused", case_e1),
         ("e2 normal plans not refused", case_e2),
         ("e3 start overlap may shrink, not grow", case_e3),
         ("e4 start overlap is per sphere pair", case_e4),
         ("f  no path says why", case_f),
         ("g1 Descend line keeps the clearance", case_g1),
         ("g2 gripper model vs cuMotion, offline fit", case_g2),
         ("g3 Lift backs out along the approach", case_g3),
         ("h  start on a joint limit, float32", case_h),
         ("i  aligned plan keeps the joint margin", case_i),
         ("j  near the table, pre-grasp out of reach", case_j)]

if __name__ == "__main__":
    print(f"cuMotion {cumotion.__file__}\n{XRDF}: table z {args.table_z:+.3f} m, plan floor {args.plan_floor:+.3f} m + "
          f"{args.plan_safety / 2 * 1000:.0f} mm pad, safety {args.plan_safety}, iterations {args.plan_iterations}, "
          f"zone {args.plan_zone_tol}, step {args.plan_step}, limits {limits}")
    failed = 0
    for name, fn in CASES:
        if args.only and not name.startswith(args.only):
            continue
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failure of that case, not of the run
            ok, detail = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}  ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"{'all cases passed' if not failed else f'{failed} case(s) failed'}")
    sys.exit(1 if failed else 0)
