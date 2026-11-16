"""Tool-frame conventions of the Viola gripper and the 'align pads with the cube' rotation.

Pure numpy, no Isaac. Imported by isaac_ik_mirror.py; run directly for a self-check:

    $ISAACSIM_PATH/python.sh scripts/grasp_frame.py      (or any python with numpy)

Conventions (derived from the URDF, checked by the self-test below):
  link6 (tool frame) x  = finger travel axis: the pads face each other along +-x
  link6 (tool frame) +z = approach axis, from the wrist through the pads to the fingertips
  joint7_left/right origins sit 10.09 mm along link6 +y (viola.urdf), so the fingertip face spans
  link6 y 2.6..17.6 mm, 18.25 mm past the origin (ToolShape); grasps aim the link6 origin.

Aligning with a cube whose world rotation is R_cube (columns cx, cy, cz):
  z_tool = -cz            approach into the cube's top face (top-down when the cube is upright)
  x_tool = one of +-cx, +-cy   the cube is 4-fold symmetric about cz; pick the option closest
                               to the gripper's CURRENT x so the wrist turns as little as possible
  y_tool = z_tool x x_tool
"""

import math
import os
import struct
from dataclasses import dataclass

import numpy as np


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(R):
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def grasp_rotation(R_cube, R_tool_now):
    """World rotation the tool frame should take to grasp a cube with rotation R_cube.

    Approach along -cz (into the top face); pads along the cube face normal (among +-cx,
    +-cy) closest to the gripper's current x axis, so the wrist rotates the least.
    """
    R_cube = np.asarray(R_cube, dtype=float)
    cx, cy, cz = R_cube[:, 0], R_cube[:, 1], R_cube[:, 2]
    x_now = np.asarray(R_tool_now, dtype=float)[:, 0]
    candidates = [cx, -cx, cy, -cy]
    x_tool = max(candidates, key=lambda v: float(np.dot(v, x_now)))
    z_tool = -cz
    y_tool = np.cross(z_tool, x_tool)
    R = np.column_stack([x_tool, y_tool, z_tool])
    return R


# Arm geometry from the URDF (mm): shoulder pitch axis height, upper arm + elbow link reach,
# wrist-pitch-to-tool (forearm) length. The self-test below recomputes them from the URDF.
SHOULDER_Z = 0.0836
REACH_MAX = 0.3038
FOREARM = 0.1707


def grasp_pose(p_target, R_cube, R_tool_now, max_tilt_deg=60.0, reach_margin=0.97,
               arm=None, grasp_depth=0.0, pre_grasp=0.0, feasible=None, depth_for=None):
    """Tool rotation to grasp a cube at p_target, tilting the approach only as much as reach demands.

    The approach axis IS the forearm on this arm, so a top-down grasp needs the wrist FOREARM
    above the target; the shoulder-to-wrist chain only reaches REACH_MAX. When top-down puts the
    wrist out of reach, tilt the approach toward the base along the cube axis nearest the reach
    direction, in 1-degree steps, until the wrist is within reach. Pads stay on the cube's sides
    (tool x along the other horizontal cube axis).

    With `arm` (an ArmReach), feasibility is joint-limit aware and checked for BOTH the grasp pose
    (rail `grasp_depth` back from the target) and the pre-grasp pose (`pre_grasp` further back):
    the smallest tilt that makes both feasible wins; failing that, the smallest that makes the
    grasp feasible. Returns (R_tool, tilt_deg, reachable) where reachable means both.

    ArmReach is optimistic at the edge of reach: the tilt it accepts leaves the pre-grasp 3-7 deg
    past the wrist-pitch limit, with no exact IK (measured offline 2026-09-30, 312 table cubes).
    `feasible` replaces it with an exact test: feasible(R_tool, p_grasp, p_above) -> bool (IK in
    pick_planner.py). At each tilt it tries the pads on this face pair with the sign nearest the
    current x, then the other sign (joint4 and joint6 stop at +-130 deg, so a rolled wrist reaches
    only one), then the pads on the cube's other face pair, tilting along this one. The first it
    accepts wins: the smallest tilt, then the least wrist turn. If none up to max_tilt_deg does,
    reachable is False and R_tool is the max-tilt rotation.

    `depth_for(R_tool)` replaces the fixed grasp_depth where the depth depends on the rotation
    (pick_poses: the gripper kept a clearance above the table); None there = no depth works. The
    pre-grasp stays grasp_depth + pre_grasp back from the target whatever the depth: raising it with
    the grasp tilted the approach 1-5 deg more and lost 5 of the 82 table cubes with a plan (offline
    2026-09-30, 10 mm clearance).
    """
    p = np.asarray(p_target, dtype=float)
    R_cube = np.asarray(R_cube, dtype=float)
    # A cube on the table is upright: use world up, keep only the cube's yaw. (The target prim can
    # carry the gripper's orientation from being parked on it; that must not tilt the grasp.)
    up = np.array([0.0, 0.0, 1.0])
    r_h = np.array([p[0], p[1], 0.0])
    r_hat = r_h / np.linalg.norm(r_h) if np.linalg.norm(r_h) > 1e-6 else R_cube[:, 0]
    cands = []
    for v in (R_cube[:, 0], -R_cube[:, 0], R_cube[:, 1], -R_cube[:, 1]):
        h = v - np.dot(v, up) * up
        n = np.linalg.norm(h)
        if n > 1e-6:
            cands.append(h / n)
    n1 = max(cands, key=lambda v: float(np.dot(v, r_hat)))     # tilt direction: cube axis nearest the reach direction
    n2 = np.cross(up, n1)                                       # the other horizontal cube axis: pad normal
    shoulder = np.array([0.0, 0.0, SHOULDER_Z])
    reach = REACH_MAX * reach_margin
    x_now = np.asarray(R_tool_now, dtype=float)[:, 0]
    x_tool = n2 if np.dot(n2, x_now) >= np.dot(-n2, x_now) else -n2

    def frame(x, a):
        z_tool = a / np.linalg.norm(a)
        return np.column_stack([x, np.cross(z_tool, x), z_tool])

    def tool(a):
        return frame(x_tool, a)

    def depth(R):
        return grasp_depth if depth_for is None else depth_for(R)

    if feasible is not None:
        d_other = n2 if np.dot(n2, r_hat) >= 0 else -n2      # other face pair: tilt along n2, still away from the base
        x_other = n1 if np.dot(n1, x_now) >= np.dot(-n1, x_now) else -n1
        options = [(x_tool, n1), (-x_tool, n1), (x_other, d_other), (-x_other, d_other)]
        t = 0.0
        while t <= max_tilt_deg:
            th = math.radians(t)
            for x, d in options:
                a_t = -up * math.cos(th) + d * math.sin(th)
                R_t = frame(x, a_t)
                dep = depth(R_t)
                if dep is None:
                    continue
                if feasible(R_t, p - a_t * dep, (p - a_t * grasp_depth) - a_t * pre_grasp):
                    return R_t, t, True
            t += 1.0
        th = math.radians(max_tilt_deg)
        return tool(-up * math.cos(th) + n1 * math.sin(th)), max_tilt_deg, False

    def ok(point, a):
        wrist = point - FOREARM * a
        if arm is not None:
            return arm.feasible(wrist, a)
        return np.linalg.norm(wrist - shoulder) <= reach

    best_grasp_only = None
    tilt, reachable, a = 0.0, False, None
    t = 0.0
    while t <= max_tilt_deg:
        th = math.radians(t)
        a_t = -up * math.cos(th) + n1 * math.sin(th)          # approach: wrist -> target; tilting pulls the wrist back toward the base
        dep = depth(tool(a_t))
        if dep is None:
            t += 1.0
            continue
        p_grasp = p - a_t * dep
        p_above = (p - a_t * grasp_depth) - a_t * pre_grasp
        g_ok = ok(p_grasp, a_t)
        if g_ok and best_grasp_only is None:
            best_grasp_only = (t, a_t)
        if g_ok and (pre_grasp <= 0 or ok(p_above, a_t)):
            tilt, reachable, a = t, True, a_t
            break
        t += 1.0
    if not reachable:
        if best_grasp_only is not None:
            tilt, a = best_grasp_only
        else:
            tilt = max_tilt_deg
            th = math.radians(tilt)
            a = -up * math.cos(th) + n1 * math.sin(th)
    return tool(a), tilt, reachable


@dataclass
class Grasp:
    """Pick grasp for one cube (pick_poses). q_above, why and code are filled by the exact IK search."""
    R: np.ndarray          # tool rotation: x = pad normal, z = approach
    tilt: float            # approach tilt from vertical, deg
    ok: bool               # grasp and pre-grasp reachable at this tilt
    p_cube: np.ndarray     # cube centre, clamped onto the table
    p_grasp: np.ndarray    # link6 at the grasp
    p_above: np.ndarray    # link6 at the pre-grasp
    depth: float = 0.0     # link6 back from the cube centre along the approach at the grasp, m
    q_above: np.ndarray = None   # joint solution of the pre-grasp
    why: str = ""          # why not ok
    code: str = ""


def pick_poses(p_cube, R_cube, R_tool_now, table_z=0.0, half_size=0.015, max_tilt_deg=60.0,
               grasp_depth=0.005, pre_grasp=0.08, arm=None, feasible=None, tool=None, clearance=0.0):
    """Grasp of the pick sequence for a cube at p_cube (isaac_ik_mirror.py Above/Descend).

    The cube centre is clamped to at least half_size above the table (the sim target has no
    physics and can float or sink). Returns a Grasp: the clamped centre, link6 at the grasp
    (grasp_depth back from the centre along the approach) and link6 at the pre-grasp (pre_grasp
    further back), where Above plans to. arm / feasible: see grasp_pose.

    With `tool` (a ToolShape) the grasp backs off further along the approach wherever the gripper
    would come lower than `clearance` above the table, rotation by rotation during the tilt search:
    one corner of the fingertip face drops with the tilt (2.6-17.6 mm off link6 in y), so the depth
    grows with it, faster on one pad side than the other. The pre-grasp stays put, so Descend gets
    shorter by as much; a depth that would leave it under 10 mm counts as no grasp.
    """
    p_c = np.asarray(p_cube, dtype=float).reshape(-1).copy()
    p_c[2] = max(p_c[2], table_z + half_size)
    back = grasp_depth + pre_grasp          # the pre-grasp's distance from the centre

    def clear_depth(R):
        d = tool.depth(R, p_c, grasp_depth, table_z + clearance)
        return d if d is not None and d <= back - 0.01 else None
    depth_for = None if tool is None else clear_depth
    R, tilt, ok = grasp_pose(p_c, R_cube, R_tool_now, max_tilt_deg=max_tilt_deg, arm=arm,
                             grasp_depth=grasp_depth, pre_grasp=pre_grasp, feasible=feasible, depth_for=depth_for)
    depth = grasp_depth if depth_for is None else depth_for(R)
    if depth is None:                       # a failed search's max-tilt rotation, the gripper not clear of the table
        depth, ok = grasp_depth, False
    a = R[:, 2]
    return Grasp(R, tilt, ok, p_c, p_c - a * depth, (p_c - a * grasp_depth) - a * pre_grasp, depth)


def rot_deg(Ra, Rb):
    """Angle of the rotation between two rotation matrices, deg."""
    c = (float(np.trace(np.asarray(Ra, dtype=float).T @ np.asarray(Rb, dtype=float))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rotate_toward(R_from, R_to, frac=1.0, max_deg=180.0):
    """R_from turned toward R_to about the axis between them, by frac of the angle and at most max_deg. Called
    every step with frac = dt / tau and max_deg = rate * dt: a first-order low-pass with a rate limit on a rotation."""
    R_from, R_to = np.asarray(R_from, dtype=float), np.asarray(R_to, dtype=float)
    Rrel = R_to @ R_from.T
    ang = math.radians(rot_deg(R_from, R_to))
    if ang < 1e-9:
        return R_to.copy()
    if ang > math.pi - 1e-3:                  # near 180 deg the axis comes from the symmetric part
        B = (Rrel + np.eye(3)) / 2.0
        axis = B[:, int(np.argmax(np.diag(B)))]
    else:
        axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    return _axis_angle(axis, min(ang * float(frac), math.radians(max_deg))) @ R_from


# Descend and Lift (isaac_ik_mirror.py) servo only along the pick's approach line, the segment from the
# pre-grasp to the grasp: link6 within LINE_MM of it and within LINE_DEG of the grasp rotation.
LINE_MM = 10.0
LINE_DEG = 5.0
# Lift without the pick grasp goes straight up, at most LIFT_UP_MAX, and only with the approach within
# LIFT_VERTICAL_DEG of straight down: the straight-up line from the tilted table grasps leaves the
# workspace only above 6-7.5 cm (offline 2026-09-30).
LIFT_UP_MAX = 0.05
LIFT_VERTICAL_DEG = 10.0


def line_offset(grasp, p, R):
    """How far link6 at (p, R) is from the approach line of grasp (pre-grasp to grasp): (mm, deg)."""
    a, b = np.asarray(grasp.p_above, dtype=float), np.asarray(grasp.p_grasp, dtype=float)
    ab = b - a
    u = min(1.0, max(0.0, float(np.dot(np.asarray(p, dtype=float) - a, ab)) / max(float(np.dot(ab, ab)), 1e-12)))
    return float(np.linalg.norm(np.asarray(p, dtype=float) - (a + u * ab))) * 1000, rot_deg(R, grasp.R)


def lift_goal(p_now, R_now, lift_height, grasp=None):
    """Where Lift takes link6 from (p_now, R_now): (p, R, why); p is None when the arm must hold still.

    With the pick grasp (the one Above planned to): back out along its approach, lift_height but no
    further than its pre-grasp, at the grasp rotation, from on the approach line only. That is the
    Descend line reversed; it ends at the latest on the pose the planner checked against the floor
    and the arm itself, and comes no nearer the base than that pose. Without the grasp: straight up,
    at most LIFT_UP_MAX, holding R_now, only with the approach near vertical. Either way the goal is
    above p_now: Lift never drives toward the table.
    """
    p_now = np.asarray(p_now, dtype=float).reshape(-1)
    if grasp is not None:
        mm, deg = line_offset(grasp, p_now, R_now)
        if mm > LINE_MM or deg > LINE_DEG:
            return None, None, (f"the gripper is {mm:.0f} mm / {deg:.0f} deg off the pick's approach line "
                                f"(Lift backs out along it from within {LINE_MM:.0f} mm / {LINE_DEG:.0f} deg)")
        a = grasp.R[:, 2]
        s = min(float(lift_height), float(np.linalg.norm(grasp.p_above - grasp.p_grasp)))
        p, R = grasp.p_grasp - a * s, grasp.R
    else:
        R = np.asarray(R_now, dtype=float)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))
        if tilt > LIFT_VERTICAL_DEG:
            return None, None, (f"no grasp from Above for this cube, and the gripper is {tilt:.0f} deg from vertical "
                                f"(straight up needs {LIFT_VERTICAL_DEG:.0f} or less)")
        p = p_now + np.array([0.0, 0.0, min(float(lift_height), LIFT_UP_MAX)])
    if p[2] < p_now[2] + 0.001:
        return None, None, f"the lift goal is not above the gripper ({(p[2] - p_now[2]) * 1000:+.0f} mm)"
    return p, R, ""


# ----------------------------------------------------------------------------------
# Self-test: forward kinematics straight from the URDF, no simulator.
# ----------------------------------------------------------------------------------
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, th):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


_CHAINS = {}


def _chain(urdf_path):
    """Parse the arm chain once: list of (xyz, R_rpy, axis or None, child_link) plus joint limits."""
    if urdf_path in _CHAINS:
        return _CHAINS[urdf_path]
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    out, limits = [], {}
    for jn in ["world_to_base", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]:
        j = joints[jn]
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        axis = None
        if j.get("type") == "revolute":
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            lim = j.find("limit")
            limits[jn] = (float(lim.get("lower")), float(lim.get("upper")))
        out.append((xyz, _rpy(*rpy), axis, j.find("child").get("link")))
    _CHAINS[urdf_path] = (out, limits)
    return _CHAINS[urdf_path]


def _fk(urdf_path, q):
    """Return {link_name: (R, p)} for the arm chain at joint angles q (joint1..joint6)."""
    chain, _ = _chain(urdf_path)
    R, p = np.eye(3), np.zeros(3)
    frames = {"world": (R.copy(), p.copy())}
    qi = 0
    for xyz, Rrpy, axis, child in chain:
        p = p + R @ xyz
        R = R @ Rrpy
        if axis is not None:
            R = R @ _axis_angle(axis, q[qi])
            qi += 1
        frames[child] = (R.copy(), p.copy())
    return frames


def link_pose(urdf_path, q, link="link6"):
    """(R, p) of a link of the arm chain at joint angles q (joint1..joint6), from the URDF alone."""
    return _fk(urdf_path, q)[link]


class ArmReach:
    """Joint-limit-aware reachability of a wrist point + approach direction, from the URDF.

    Tabulates the elbow/wrist positions over the shoulder (joint2) and elbow (joint3) ranges at
    1 degree, with the base yaw at 0 (the arm's plane is the x-z plane; the base yaw supplies
    the rest). A wrist point W with approach a is feasible when some (q2, q3) puts the wrist
    within `tol` of W and the angle between the elbow->wrist link and a lies within the wrist
    pitch range (joint5) around its zero offset. Wrist roll (joint4) and tool roll (joint6)
    are assumed free within their +-130 degree ranges.
    """

    def __init__(self, urdf_path, step_deg=1.0):
        chain, limits = _chain(urdf_path)
        self.urdf = urdf_path
        lo2, hi2 = limits["joint2"]
        lo3, hi3 = limits["joint3"]
        self.pitch_range = limits["joint5"]
        q2s = np.arange(lo2, hi2 + 1e-9, math.radians(step_deg))
        q3s = np.arange(lo3, hi3 + 1e-9, math.radians(step_deg))
        wr, wz, d3r, d3z = [], [], [], []
        for q2 in q2s:
            for q3 in q3s:
                fr = _fk(urdf_path, [0.0, q2, q3, 0.0, 0.0, 0.0])
                pe, pw = fr["link3"][1], fr["link4"][1]      # elbow joint, wrist (joint4 = joint5 origin)
                d = pw - pe
                d /= np.linalg.norm(d)
                wr.append(math.hypot(pw[0], pw[1])); wz.append(pw[2])
                d3r.append(math.hypot(d[0], d[1]) * (1 if d[0] >= 0 else -1)); d3z.append(d[2])
        self.wr, self.wz = np.array(wr), np.array(wz)
        self.d3r, self.d3z = np.array(d3r), np.array(d3z)
        # wrist pitch zero offset: angle between the elbow->wrist link and the approach at q5 = 0
        fr0 = _fk(urdf_path, [0.0] * 6)
        d0 = fr0["link4"][1] - fr0["link3"][1]
        d0 /= np.linalg.norm(d0)
        self.pitch_zero = math.acos(float(np.clip(np.dot(d0, fr0["link6"][0][:, 2]), -1, 1)))

    def feasible(self, W, a, tol=0.008):
        W = np.asarray(W, dtype=float)
        r = math.hypot(W[0], W[1])
        rhat = np.array([W[0], W[1], 0.0]) / r if r > 1e-6 else np.array([1.0, 0.0, 0.0])
        m = (np.abs(self.wr - r) < tol) & (np.abs(self.wz - W[2]) < tol)
        if not m.any():
            return False
        a = np.asarray(a, dtype=float)
        ar, az = float(np.dot(a, rhat)), float(a[2])
        n = math.hypot(ar, az)
        if n < 1e-6:
            return True                                   # approach is sideways: the wrist roll does it
        cosang = (self.d3r[m] * ar + self.d3z[m] * az) / n
        ang = np.arccos(np.clip(cosang, -1, 1))
        lo, hi = self.pitch_range
        return bool(np.any((ang >= self.pitch_zero + lo - 0.02) & (ang <= self.pitch_zero + hi + 0.02)))


def _origin(el):
    """(xyz, R) of a URDF element's <origin>, identity when it has none."""
    o = el.find("origin")
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return xyz, _rpy(*rpy)


def mesh_vertices(link, urdf_dir):
    """Distinct vertices of a URDF link's collision mesh (else its visual mesh) in the link frame; None without one."""
    for tag in ("collision", "visual"):
        el = link.find(tag)
        mesh = None if el is None else el.find("geometry/mesh")
        if mesh is None:
            continue
        b = open(os.path.normpath(os.path.join(urdf_dir, mesh.get("filename"))), "rb").read()
        if b[:5].lower() == b"solid" and b"facet" in b[:400]:          # ASCII STL
            V = np.array([[float(v) for v in ln.split()[1:4]] for ln in b.decode(errors="ignore").splitlines()
                          if ln.strip().startswith("vertex")])
        else:
            n = struct.unpack("<I", b[80:84])[0]
            rec = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            V = rec["v"].reshape(-1, 3).astype(float)
        if mesh.get("scale"):
            V = V * np.array([float(v) for v in mesh.get("scale").split()])
        xyz, R = _origin(el)
        return np.unique(V, axis=0) @ R.T + xyz
    return None


class ToolShape:
    """The gripper as points in the tool frame, from the URDF: the tool link's mesh vertices plus those of
    every link on a fixed or prismatic joint off it (the fingers), at both ends of the travel. A finger
    slides in a straight line, so its lowest point over the travel is at one end: the set's lowest point
    is never above the real one at any opening, and equal to it when the travel axis is horizontal, as in
    every pick grasp (offline 2026-09-30: never above pinocchio's mesh FK over 300 random poses; equal to
    cuMotion's frames at the pick poses, plan_selftest.py g2). viola.urdf: link6 and both fingers, the
    fingertips 18.25 mm past link6.
    """

    def __init__(self, urdf_path, frame="link6"):
        import xml.etree.ElementTree as ET
        root = ET.parse(urdf_path).getroot()
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        links = {ln.get("name"): ln for ln in root.findall("link")}
        parts = [mesh_vertices(links[frame], urdf_dir)]
        for j in root.findall("joint"):
            if j.find("parent").get("link") != frame:
                continue
            V = mesh_vertices(links[j.find("child").get("link")], urdf_dir)
            if V is None:
                continue
            if j.get("type") not in ("fixed", "prismatic"):
                raise ValueError(f"ToolShape: {j.get('name')} off {frame} is {j.get('type')}; only fixed or prismatic")
            xyz, R = _origin(j)
            axis, ends = np.zeros(3), [0.0]
            if j.get("type") == "prismatic":
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                ends = [float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))]
            parts += [(V + axis * s) @ R.T + xyz for s in ends]
        self.points = np.vstack([P for P in parts if P is not None])

    def low(self, R):
        """World z of the gripper's lowest point relative to the tool frame origin, the tool at rotation R."""
        return float(np.min(self.points @ np.asarray(R, dtype=float)[2]))

    def height(self, R, p, table_z=0.0):
        """How far the gripper's lowest point is above table_z with the tool frame at (R, p), m."""
        return float(np.asarray(p, dtype=float).reshape(-1)[2]) + self.low(R) - table_z

    def depth(self, R, p_cube, depth, floor):
        """Grasp depth (tool frame back from the cube centre p_cube along the approach R[:, 2]) of at least
        `depth` that keeps the gripper's lowest point at or above z = floor; None when backing off along
        the approach cannot raise it (the approach not pointing down)."""
        R = np.asarray(R, dtype=float)
        down = -float(R[2, 2])                                   # rise of the tool frame per m backed off
        need = floor - float(np.asarray(p_cube, dtype=float).reshape(-1)[2]) - self.low(R)
        if down * depth >= need:
            return float(depth)
        return need / down if down > 1e-3 else None


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    urdf = os.path.join(here, "..", "assets", "viola", "urdf", "viola.urdf")
    q_connect = [0.0, -1.57, 0.942, 0.0, 0.471, 0.0]   # normalized 0,-100,60,0,30,0 under the old urdf joint map
    fr = _fk(urdf, q_connect)
    R6, p6 = fr["link6"]
    R5, p5 = fr["link5"]
    print(f"link6 origin at connect pose: {p6.round(4).tolist()}   (Isaac printed ee=(+0.120,+0.000,+0.224))")
    forearm = p6 - p5
    d = float(np.dot(R6[:, 2], forearm / np.linalg.norm(forearm)))
    print(f"dot(link6 +z, wrist->tool direction) = {d:+.3f}  -> +z points {'AWAY from' if d > 0 else 'TOWARD'} the wrist")
    print("   convention needed: +z = approach (away from the wrist). ", "OK" if d > 0 else "SIGN FLIP NEEDED")
    # joint7 slide axis in link6, from its origin rpy and axis
    import xml.etree.ElementTree as ET
    j7 = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}["joint7_left"]
    rpy7 = [float(v) for v in j7.find("origin").get("rpy").split()]
    ax7 = np.array([float(v) for v in j7.find("axis").get("xyz").split()])
    slide_in_link6 = _rpy(*rpy7) @ ax7
    print(f"finger slide axis in link6 = {slide_in_link6.round(3).tolist()}   (expect +-x)")
    # grasp rotation on an upright cube, gripper currently pointing +x
    R = grasp_rotation(np.eye(3), R6)
    print(f"grasp R for an upright cube: x_tool={R[:,0].round(2).tolist()} z_tool={R[:,2].round(2).tolist()} "
          f"det={np.linalg.det(R):+.3f} orthonormal={np.allclose(R.T @ R, np.eye(3))}")
    q = R_to_quat_wxyz(R)
    print(f"quat roundtrip ok: {np.allclose(quat_wxyz_to_R(q), R)}")
    # geometry constants vs the URDF
    import xml.etree.ElementTree as ET
    Jd = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    xyz = lambda n: np.array([float(v) for v in Jd[n].find("origin").get("xyz").split()])
    sz = xyz("joint1")[2] + xyz("joint2")[2]; rm = np.linalg.norm(xyz("joint3")) + np.linalg.norm(xyz("joint4")); fa = np.linalg.norm(xyz("joint6"))
    print(f"constants vs URDF: shoulder_z {SHOULDER_Z:.4f}/{sz:.4f}  reach {REACH_MAX:.4f}/{rm:.4f}  forearm {FOREARM:.4f}/{fa:.4f}  "
          f"{'OK' if abs(sz-SHOULDER_Z)<1e-3 and abs(rm-REACH_MAX)<1e-3 and abs(fa-FOREARM)<1e-3 else 'MISMATCH'}")
    import time as _t
    t0 = _t.perf_counter(); arm = ArmReach(urdf); dt = _t.perf_counter() - t0
    print(f"ArmReach table: {len(arm.wr)} (q2,q3) samples in {dt:.1f} s, wrist pitch zero offset {math.degrees(arm.pitch_zero):.1f} deg, range {[round(math.degrees(v)) for v in arm.pitch_range]}")
    for name, pt in (("cube 22 cm out, pre 8 cm", [0.22, 0.05, 0.015]), ("cube 25 cm out, pre 8 cm", [0.25, 0.0, 0.015]),
                     ("cube 30 cm out, pre 8 cm", [0.30, 0.0, 0.015]), ("cube 15 cm out, pre 8 cm", [0.15, 0.0, 0.015])):
        Rt, tilt, okb = grasp_pose(pt, np.eye(3), R6, arm=arm, grasp_depth=0.005, pre_grasp=0.08)
        print(f"grasp_pose+ArmReach {name:26s}: tilt {tilt:3.0f} deg  grasp+pregrasp feasible={okb}  approach={Rt[:,2].round(2).tolist()}")
    Rt2, tilt2, ok2 = grasp_pose([0.22, 0.05, 0.015], R6, R6)   # cube frame = the tilted tool frame (parked-on-gripper case)
    print(f"grasp_pose with a TILTED cube frame (parked on the gripper): tilt {tilt2:.0f} deg reachable={ok2} approach={Rt2[:,2].round(2).tolist()}  (must be 0 deg, straight down)")
    for name, pt in (("cube on table, 25 cm out", [0.25, 0.0, 0.015]), ("cube on table, 32 cm out", [0.32, 0.05, 0.015]),
                     ("mid-air no-path target", [0.147, -0.001, 0.317]), ("high and far", [0.30, 0.0, 0.25])):
        Rt, tilt, ok = grasp_pose(pt, np.eye(3), R6)
        print(f"grasp_pose {name:28s}: tilt {tilt:4.0f} deg  reachable={ok}  approach={Rt[:,2].round(2).tolist()} pads along={Rt[:,0].round(2).tolist()} "
              f"orthonormal={np.allclose(Rt.T @ Rt, np.eye(3), atol=1e-6)}")
    tool = ToolShape(urdf)
    down = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # approach straight down
    print(f"ToolShape: {len(tool.points)} gripper points, lowest {-tool.low(down) * 1000:.2f} mm below link6 pointing "
          f"straight down (fingertips)")
