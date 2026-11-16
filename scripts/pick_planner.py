"""cuMotion goals for the Viola's planned moves: 'Plan to target' and the pick's Above step.

Pure Python + numpy + the cuMotion package (no Kit, no warp), so the same code runs inside Isaac
Sim (isaac_ik_mirror.py --plan, on the objects the isaacsim.robot_motion.cumotion extension
builds) and offline (plan_selftest.py, the standalone cuMotion that ships with Isaac: no Isaac Sim /
Kit; cuMotion uses the GPU for its world view).

A pose goal is only handed to the planner once an exact IK solution for it exists: JtRRT can
only answer "no path" for a pose without one, after ~1 s of search. The approach tilt is the
smallest (1-degree steps, grasp_frame.grasp_pose) at which cuMotion's IK puts every joint at
least JOINT_MARGIN_DEG inside its limits, for the grasp and the pre-grasp, with the pre-grasp
(the planner's goal) clear of obstacles and of the arm itself. At every tilt the grasp backs off
along the approach until the gripper's lowest point (grasp_frame.ToolShape, the URDF meshes) is
`clearance` above the table, so Descend's goal and line stay that high. Every path is checked
for self-collision, which the planner ignores, and a request without a path says why: the start
in collision, no IK for the goal, the goal too close to the floor, self-collision, or blocked.

Frames: goals are in cuMotion's robot base frame, which is the world frame here: the Viola is
referenced at the world origin and viola.urdf roots it with world_to_base = identity (Pink IK in
isaac_ik_mirror.py makes the same assumption).
"""

import math
import time
from dataclasses import dataclass

import cumotion
import numpy as np

from grasp_frame import R_to_quat_wxyz, pick_poses

TOOL_FRAME = "link6"
# Every joint of an accepted IK solution stays this far inside its limit: the real wrist pitch
# settles ~2 deg low and reads in ~1 deg steps (measured on the arm, docs/joint-map.md). The first tilt with exact
# IK leaves joint5 0-2 deg from its limit; the margin costs 1-3 deg more tilt (8 table cubes,
# measured offline 2026-09-30).
JOINT_MARGIN_DEG = 3.0
# Plan to target puts link6 at the cube centre. A cube lower than this above the table cannot take
# it: the fingertips are 18.3 mm past link6 (3 mm into the table for a cube resting on it), their
# spheres 23 mm (model v2), the floor is padded safety/2 = 10 mm, and a tilted approach drops the
# finger corner further. Such a cube is planned to the pick's pre-grasp instead.
TABLE_BAND = 0.045
# cuMotion refuses a start even 5e-8 rad past a limit ('Invalid initial robot configuration', no path);
# PhysX reads float32 joints, float32(-1.57) = -1.5700000525 is joint2 at Rest/Connect with the old urdf
# joint map (calib: -88.1 deg, inside the limit), and a joint
# pressed on its limit reads 1-6 urad past (logged 2026-09-22). A start this close is clipped
# LIMIT_INSET inside; one further out is refused with its joint named.
LIMIT_TOL = math.radians(0.5)
LIMIT_INSET = 1e-5


def configure(config, max_iterations, task_space_limits, zone_tol, step_size):
    """Set the JtRRT parameters isaac_ik_mirror.py exposes as --plan-iterations/-limits/-zone-tol/-step.

    task_space_limits: xlo, xhi, ylo, yhi, zlo, zhi (m). Returns the four set_param results.
    """
    P = cumotion.MotionPlannerConfig
    lim = [float(v) for v in task_space_limits]
    return [config.set_param("max_iterations", P.ParamValue(int(max_iterations))),
            config.set_param("task_space_limits", [P.Limit(lim[0], lim[1]), P.Limit(lim[2], lim[3]),
                                                   P.Limit(lim[4], lim[5])]),
            config.set_param("task_space_planning_params/translation_target_zone_tolerance",
                             P.ParamValue(float(zone_tol))),
            config.set_param("step_size", P.ParamValue(float(step_size)))]


def pose3(p, R):
    w, x, y, z = R_to_quat_wxyz(R)
    return cumotion.Pose3(rotation=cumotion.Rotation3(w, x, y, z), translation=np.asarray(p, dtype=float))


@dataclass
class Plan:
    """One planning request: a c-space path to follow, or None and why not."""
    waypoints: list        # joint1..joint6 (rad) per waypoint; None = no path, the arm holds still
    kind: str              # what was planned to, for the log and the panel
    goal: list             # link6 goal position (m)
    R: np.ndarray          # link6 goal rotation, None for translation only
    ms: float              # time for the whole request
    why: str = ""          # empty on success
    code: str = ""         # empty on success, else start / no_ik / floor / self / no_path / here / limit
    grasp: object = None   # the pick's grasp_frame.Grasp when the goal is its pre-grasp


class PickPlanner:
    """Goal selection and planning on one cuMotion robot + world view + JtRRT config."""

    def __init__(self, robot_description, world_view, config, table_z=0.0, max_tilt=60.0,
                 grasp_depth=0.005, pre_grasp=0.08, tool=None, clearance=0.0, stops=None):
        self.robot = robot_description
        self.kin = robot_description.kinematics()
        self.world_view = world_view
        self.config = config
        self.inspector = cumotion.create_robot_world_inspector(robot_description, world_view)
        limits = [self.kin.cspace_coord_limits(i) for i in range(self.kin.num_cspace_coords())]
        margin = math.radians(JOINT_MARGIN_DEG)
        self.lim_lo = np.array([L.lower for L in limits])
        self.lim_hi = np.array([L.upper for L in limits])
        self.q_lo = np.array([L.lower for L in limits]) + margin
        self.q_hi = np.array([L.upper for L in limits]) - margin
        if stops is not None:
            # (lo, hi) rad per arm joint: the real arm's range where it is narrower than the URDF (--joint-map calib,
            # isaac_ik_mirror.py). IK goals keep the joint margin inside it, so the real arm can reach every goal.
            self.q_lo = np.maximum(self.q_lo, np.array([lo for lo, _ in stops]) + margin)
            self.q_hi = np.minimum(self.q_hi, np.array([hi for _, hi in stops]) - margin)
        self.table_z = table_z
        self.max_tilt = max_tilt
        self.grasp_depth = grasp_depth
        self.pre_grasp = pre_grasp
        self.tool = tool                  # grasp_frame.ToolShape: the pick grasp keeps the gripper clearance above the table
        self.clearance = clearance
        n = self.inspector.num_self_collision_spheres()
        self.sphere_frame = np.array([self.inspector.self_collision_sphere_frame_name(i) for i in range(n)])
        self.sphere_r = np.asarray(self.inspector.self_collision_sphere_radii(), dtype=float)
        self.empty_world = cumotion.create_world()            # no obstacles: tells "out of reach" from "blocked"
        self.empty_view = self.empty_world.add_world_view()
        self.empty_view.update()

    def start(self, q0):
        """(q0 clipped LIMIT_INSET inside the joint limits, None), or (None, why) for a joint more than
        LIMIT_TOL outside them."""
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        past = np.maximum(self.lim_lo - q0, q0 - self.lim_hi)
        if np.any(past > LIMIT_TOL):
            j = int(np.argmax(past))
            return None, (f"start outside the joint limits: joint{j + 1} reads {q0[j]:+.4f} rad, "
                          f"{math.degrees(past[j]):.1f} deg past [{self.lim_lo[j]:+.4f}, {self.lim_hi[j]:+.4f}]")
        return np.clip(q0, self.lim_lo + LIMIT_INSET, self.lim_hi - LIMIT_INSET), None

    def overlaps(self, q, pair):
        """Overlap (m) of every self-collision sphere pair between the two frames of pair at q, as
        {(sphere i, sphere j): overlap}; positive = the spheres intersect."""
        c = np.array([np.asarray(p, dtype=float).reshape(-1) for p in self.inspector.self_collision_sphere_positions(q)])
        ia, ib = np.flatnonzero(self.sphere_frame == pair[0]), np.flatnonzero(self.sphere_frame == pair[1])
        d = np.linalg.norm(c[ia][:, None, :] - c[ib][None, :, :], axis=2)
        o = self.sphere_r[ia][:, None] + self.sphere_r[ib][None, :] - d
        return {(int(ia[i]), int(ib[j])): float(o[i, j]) for i in range(len(ia)) for j in range(len(ib))}

    def self_collision(self, q0, waypoints, step=math.radians(1.0), grow=0.001):
        """Where the path first makes the arm hit itself, as text, or None.

        cuMotion's graph planner does not check self-collision (it planned into self-colliding goals
        12/12 on the Viola and 12/12 on NVIDIA's Franka; MotionPlannerConfig has no such parameter,
        offline 2026-09-30), so every path is checked here, sampled along its straight joint-space
        segments (the minimal-time trajectory follows them) with no joint moving more than `step`.
        A sphere pair that already overlaps at the start pose may stay overlapped while its overlap
        never grows more than `grow` above the least seen so far: the folded Rest pose overlaps
        link5-link1 by 16 mm in the sphere model while the meshes are 5 mm apart (model v2), so a path
        may leave that contact but not deepen it (16 mm measured with the old urdf joint map). The rule is per sphere pair: another sphere pair of
        the same links that starts to overlap is a hit (per link pair, Rest -> panel (-39.4, -63.7,
        98.1, 7.4, -14.3, -3.1) kept the deepest overlap at 16.3 mm while the meshes met, offline
        2026-09-30). Once clear, a pair counts like any other.
        """
        start_pairs = [tuple(sorted(p)) for p in self.inspector.frames_in_self_collision(q0)]
        allowed = {pair: {k: v for k, v in self.overlaps(q0, pair).items() if v > 0} for pair in start_pairs}
        samples = []
        w = [np.asarray(q, dtype=float).reshape(-1) for q in waypoints]
        for a, b in zip(w[:-1], w[1:]):
            n = max(1, math.ceil(float(np.max(np.abs(b - a))) / step))
            samples += [a + (b - a) * (i / n) for i in range(1, n + 1)]
        for k, q in enumerate(samples):
            pairs = {tuple(sorted(p)) for p in self.inspector.frames_in_self_collision(q)}
            for pair in [p for p in allowed if p not in pairs]:
                del allowed[pair]
            for pair in sorted(pairs):
                where = f"{pair[0]}-{pair[1]} at {100 * (k + 1) / len(samples):.0f}% of the path"
                if pair not in allowed:
                    return where
                now = {k: v for k, v in self.overlaps(q, pair).items() if v > 0}
                for sp, d in now.items():
                    if sp not in allowed[pair]:
                        return f"{where} (spheres {sp[0]}-{sp[1]} overlap {d * 1000:.0f} mm, clear at the start)"
                    if d > allowed[pair][sp] + grow:
                        return (f"{where} (overlap {allowed[pair][sp] * 1000:.0f} -> {d * 1000:.0f} mm, "
                                "deeper than at the start)")
                allowed[pair] = {sp: min(allowed[pair][sp], d) for sp, d in now.items()}
        return None

    def ik(self, p, R, seeds=()):
        """Joints that put link6 at (p, R) with every joint JOINT_MARGIN_DEG inside its limits, or None.

        solve_ik reports failure with an out-of-limit answer (|joint5| 95-106 deg on the pre-grasps
        the ArmReach tilt gave, measured 2026-09-30), so both its flag and the limits are checked.
        """
        config = cumotion.IkConfig()
        config.cspace_seeds = [np.asarray(s, dtype=float).reshape(-1) for s in seeds]
        r = cumotion.solve_ik(self.kin, pose3(p, R), TOOL_FRAME, config)
        q = np.asarray(r.cspace_position, dtype=float).reshape(-1)
        return q if r.success and np.all(q >= self.q_lo) and np.all(q <= self.q_hi) else None

    def _search(self, p_cube, R_cube, R_tool_now, q_now, depth, pre, name, clear=False):
        """Smallest-tilt grasp whose grasp pose (depth back from the centre; with clear, further while
        the gripper would be lower than the clearance above the table) and goal pose (pre further back)
        both have exact IK, the goal clear of obstacles and self-collision."""
        seeds = [] if q_now is None else [q_now]
        blocked = {}                                 # what stopped a reachable goal, at its first tilt
        found = {}
        self.world_view.update()

        def feasible(R, p_grasp, p_goal):
            q = self.ik(p_grasp, R, seeds)
            if q is not None and pre > 0:
                q = self.ik(p_goal, R, [q] + seeds)
            if q is None:
                return False
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))
            if self.inspector.in_collision_with_obstacle(q):
                blocked.setdefault("floor", (tilt, self.inspector.min_distance_to_obstacle(q)))
                return False
            if self.inspector.in_self_collision(q):
                blocked.setdefault("self", (tilt, self.inspector.frames_in_self_collision(q)))
                return False
            found["q"] = q
            return True

        g = pick_poses(p_cube, R_cube, R_tool_now, table_z=self.table_z, max_tilt_deg=self.max_tilt,
                       grasp_depth=depth, pre_grasp=pre, feasible=feasible,
                       tool=self.tool if clear else None, clearance=self.clearance)
        if g.ok:
            g.q_above = found["q"]
        elif "floor" in blocked:
            t, d = blocked["floor"]
            g.code, g.why = "floor", (f"{name} too close to the floor: the arm would be {-d * 1000:.0f} mm inside "
                                      f"the padded table/obstacles (tilt {t:.0f} deg)")
        elif "self" in blocked:
            t, pairs = blocked["self"]
            g.code, g.why = "self", f"no IK for the {name} clear of the arm itself: {pairs} (tilt {t:.0f} deg)"
        else:
            g.code, g.why = "no_ik", (f"no IK for the {name}: out of reach at every tilt 0-{self.max_tilt:.0f} deg "
                                      f"with each joint {JOINT_MARGIN_DEG:.0f} deg inside its limits")
        return g

    def grasp(self, p_cube, R_cube, R_tool_now, q_now=None):
        """Pick grasp: grasp and pre-grasp reachable, the pre-grasp clear (Above plans there), the gripper
        at the grasp at least the clearance above the table."""
        return self._search(p_cube, R_cube, R_tool_now, q_now, self.grasp_depth, self.pre_grasp, "pre-grasp",
                            clear=True)

    def plan_above(self, q0, p_cube, R_cube, R_tool_now):
        """Above: plan to the pre-grasp of the pick grasp for the cube at p_cube."""
        t0 = time.perf_counter()
        q0, why = self.start(q0)
        if why:
            return Plan(None, "pose, above the cube", [float(v) for v in p_cube], None, 0.0, why, "start")
        g = self.grasp(p_cube, R_cube, R_tool_now, q0)
        plan = self._plan(q0, g, g.p_above, f"pose, above the cube, tilt {g.tilt:.0f} deg")
        plan.ms = (time.perf_counter() - t0) * 1000
        return plan

    def plan_to_target(self, q0, p_target, R_cube, R_tool_now, align=True):
        """'Plan to target': link6 to the target position, pads aligned with the cube when align. With
        align, a cube within TABLE_BAND of the table is planned to its pick pre-grasp (as Above)."""
        t0 = time.perf_counter()
        q0, why = self.start(q0)
        p_target = np.asarray(p_target, dtype=float).reshape(-1)
        if why:
            plan = Plan(None, "pose" if align else "translation only", [float(v) for v in p_target], None, 0.0, why, "start")
        elif not align:
            plan = self._plan(q0, None, p_target, "translation only")
        elif p_target[2] - self.table_z < TABLE_BAND:
            g = self.grasp(p_target, R_cube, R_tool_now, q0)
            c = None
            if g.code == "no_ik":
                # the pre-grasp sits ~85 mm up the approach; where that is out of reach, the centre may not be
                # (Rest -> (0.15, 0.15, 0.04): centre top-down, measured offline). Taken only with the gripper
                # model at least the clearance above the table there; the path is checked against the padded floor.
                c = self._search(p_target, R_cube, R_tool_now, q0, 0.0, 0.0, "cube centre")
                if c.ok and self.tool is not None and self.tool.height(c.R, c.p_grasp, self.table_z) < self.clearance:
                    h = self.tool.height(c.R, c.p_grasp, self.table_z)
                    c.ok, c.why = False, (f"the gripper would be {h * 1000:.0f} mm above the table at the centre "
                                          f"(< {self.clearance * 1000:.0f} mm)")
            if c is not None and c.ok:
                plan = self._plan(q0, c, c.p_grasp, f"pose, pads aligned (near the table, pre-grasp out of reach), "
                                                    f"tilt {c.tilt:.0f} deg")
                plan.grasp = None
            else:
                plan = self._plan(q0, g, g.p_above, f"pose, above the cube (near the table), tilt {g.tilt:.0f} deg")
                if c is not None:
                    plan.why += f"; the cube is near the table and its centre has none either ({c.why}): move the cube"
        else:
            g = self._search(p_target, R_cube, R_tool_now, q0, 0.0, 0.0, "cube centre")
            plan = self._plan(q0, g, g.p_grasp, f"pose, pads aligned, tilt {g.tilt:.0f} deg")
            plan.grasp = None                            # link6 at the centre is not a pick grasp
            if plan.code == "floor":
                plan.why += "; Above stops at a pre-grasp higher up"
        plan.ms = (time.perf_counter() - t0) * 1000
        return plan

    def _plan(self, q0, g, p, kind):
        """Plan to link6 at p with g's rotation (translation only without g); no planning for a goal the
        search found unreachable. When JtRRT misses a pose goal, plan in joint space to the goal's IK
        solution from the search: still checked against the floor and the scene all the way."""
        R = None if g is None else g.R
        goal = [float(v) for v in p]
        name = "target" if g is None else ("pre-grasp" if np.linalg.norm(g.p_above - g.p_grasp) > 1e-9 else "cube centre")
        self.world_view.update()
        if self.inspector.in_collision_with_obstacle(q0):
            # the planner cannot leave a start inside an obstacle; typical after Descend: fingertip spheres in the padded table
            d = self.inspector.min_distance_to_obstacle(q0)
            return Plan(None, kind, goal, R, 0.0, f"start in collision: the arm is {-d * 1000:.0f} mm inside the padded "
                                                  "table/obstacles; lift it first", "start", g)
        if g is not None and not g.ok:
            return Plan(None, kind, goal, R, 0.0, g.why, g.code, g)
        planner = cumotion.create_motion_planner(config=self.config)
        if R is None:
            r = planner.plan_to_translation_target(q0, np.asarray(p, dtype=float))
        else:
            r = planner.plan_to_pose_target(q0, pose3(p, R))
            if not r.path_found:
                r = cumotion.create_motion_planner(config=self.config).plan_to_cspace_target(q0, g.q_above)
                kind += ", joint-space plan to its IK solution"
        if r.path_found and R is not None and self.margin_deg(r.path[-1]) < JOINT_MARGIN_DEG - 1e-6:
            # cuMotion's own pose IK ends up to 0.02 deg from a limit (joint5 0.25 deg at Rest -> (0.3, 0, 0.25),
            # measured offline); the bridge sends that to the real arm. The search's IK keeps the margin.
            r = cumotion.create_motion_planner(config=self.config).plan_to_cspace_target(q0, g.q_above)
            kind += ", joint-space plan to its IK solution (joint margin)"
            if r.path_found and self.margin_deg(r.path[-1]) < JOINT_MARGIN_DEG - 1e-6:
                return Plan(None, kind, goal, R, 0.0, f"the {name} path ends {self.margin_deg(r.path[-1]):.2f} deg from "
                            f"a joint limit (< {JOINT_MARGIN_DEG:.0f} deg)", "limit", g)
        if r.path_found:
            waypoints = []
            for w in r.path:   # mg.Path's trajectory refuses repeated waypoints: a joint-space plan to q0 returns [q0, q0]
                w = np.asarray(w, dtype=np.float64).reshape(-1)
                if not waypoints or not np.allclose(w, waypoints[-1], rtol=1e-6, atol=1e-10):
                    waypoints.append(w)
            if len(waypoints) < 2:
                return Plan(None, kind, goal, R, 0.0, "already at the goal, nothing to move", "here", g)
            hit = self.self_collision(q0, waypoints)
            if hit:
                return Plan(None, kind, goal, R, 0.0, f"path self-collides: {hit}", "self", g)
            return Plan(waypoints, kind, goal, R, 0.0, grasp=g)
        if g is None:
            try:
                code, why = self._why_unreachable(q0, p)
            except Exception as e:  # a failed diagnosis must not stop the sim loop the real arm follows
                code, why = "no_path", f"no path found ({type(e).__name__} while finding out why: {e})"
        else:
            code, why = "no_path", f"no path found from here although the {name} has a clear IK solution (obstacle in the way?)"
        return Plan(None, kind, goal, R, 0.0, why, code, g)

    def margin_deg(self, q):
        """Smallest distance of any joint of q to its limit (deg)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        return math.degrees(float(np.min(np.minimum(q - self.lim_lo, self.lim_hi - q))))

    def _why_unreachable(self, q0, p):
        """(code, text) for a translation target the planner found no path to: cuMotion's collision-free IK
        (position only) against the obstacles, then against an empty world."""
        C = cumotion.CollisionFreeIkSolver
        target = C.TaskSpaceTarget(C.TranslationConstraint.target(np.asarray(p, dtype=float)), C.OrientationConstraint.none())

        def solvable(world_view):
            config = cumotion.create_default_collision_free_ik_solver_config(
                robot_description=self.robot, tool_frame_name=TOOL_FRAME, world_view=world_view)
            return cumotion.create_collision_free_ik_solver(config).solve(target, [q0]).status() == C.Results.Status.SUCCESS

        if solvable(self.world_view):
            return "no_path", "no path found from here although the target has a clear IK solution (obstacle in the way?)"
        if solvable(self.empty_view):
            return "floor", "target too close to the floor: every IK solution puts the arm inside the padded table/obstacles"
        return "no_ik", "no IK for the target: out of reach"
