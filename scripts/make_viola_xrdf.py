"""Generate a cuMotion XRDF for the Viola from its URDF and STL meshes.

    $ISAACSIM_PATH/python.sh scripts/make_viola_xrdf.py      (or any python with numpy)
    -> assets/viola/urdf/viola.xrdf   (next to viola.urdf; cuMotion loads both from one dir)

Collision spheres are FITTED, not guessed: for every link with a mesh, area-weighted surface
samples plus all vertices (link frame, visual origin applied) are split recursively along
their longest principal extent until each cluster's enclosing sphere + --pad is at most the
link's max radius (--finger-max-r on the fingers, --tool-max-r on link6, --max-r elsewhere),
then neighbouring clusters are merged while they still fit. Every vertex is therefore at least
--pad inside a sphere (printed per link as "vertex depth min"), and the script checks an
INDEPENDENT denser surface sample against the spheres (vertices alone miss the middle of long
faces: the old slicing left 1.8 % of link2's surface up to 12.8 mm outside every sphere,
measured 2026-09-30) and prints coverage and max miss per link. Over-approximation is intended
but bounded: the fingertip spheres reach 4-5 mm past the real tips (the old fit: 14 mm), which
is what sets how close to the table the planner may put the gripper.

Format copied from the shipped reference
$ISAACSIM_PATH/exts/isaacsim.robot_motion.cumotion/robot_configurations/franka/robot.xrdf
(format xrdf 2.0: default_joint_positions, cspace, tool_frames, world_collision,
self_collision.ignore, geometry.<name>.spheres).

Acceleration/jerk limits are CHOSEN conservative values (not measured): the servo firmware
takes ~130 ms to complete a 10-unit (~9 deg) step (measured on the arm), and a real arm
follows the sim through its bridge's slew guard anyway. Tune with --accel/--jerk.
"""

import argparse
import math
import os
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--urdf", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "assets", "viola", "urdf", "viola.urdf"))
p.add_argument("--out", default=None, help="default: <urdf dir>/viola.xrdf")
p.add_argument("--pad", type=float, default=0.002, help="radius padding over the enclosing radius, m (covers surface-sampling gaps)")
p.add_argument("--max-r", type=float, default=0.025, help="max sphere radius (pad included) on link1..link5, m")
p.add_argument("--tool-max-r", type=float, default=0.022, help="max sphere radius on link6 (gripper body), m")
p.add_argument("--finger-max-r", type=float, default=0.012, help="max sphere radius on link7_left/right, m")
p.add_argument("--base-max-r", type=float, default=0.035, help="max sphere radius on base_link (self-collision only), m")
p.add_argument("--samples", type=int, default=8000, help="area-weighted surface samples per link, fitted together with all vertices")
p.add_argument("--gripper-model", choices=["open", "sweep"], default="open",
               help="the planner locks joint7 at its XRDF default. open: default = fully open (the widest the real "
                    "fingers get; the fingertip reach along the approach does not depend on the opening). sweep: "
                    "default 50%%, finger spheres cover the whole joint7 travel (~2.7x the finger spheres)")
p.add_argument("--accel", type=float, default=5.0, help="rad/s^2 per cspace joint (chosen, not measured)")
p.add_argument("--jerk", type=float, default=500.0, help="rad/s^3 per cspace joint (chosen, not measured)")
p.add_argument("--tool-frame", default="link6", help="must match TOOL_FRAME in isaac_ik_mirror.py")
p.add_argument("--no-base-split", action="store_true", help="debug: keep base_link in world collision too")
args = p.parse_args()

urdf_path = os.path.abspath(args.urdf)
urdf_dir = os.path.dirname(urdf_path)
out = os.path.abspath(args.out or os.path.join(urdf_dir, "viola.xrdf"))
root = ET.parse(urdf_path).getroot()

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["joint7_left", "joint7_right"]
# vendor connect pose in LeRobot normalized units (isaac_ik_mirror.py CONNECT_POSE)
CONNECT_NORM = {"joint1": 0.0, "joint2": -100.0, "joint3": 60.0, "joint4": 0.0, "joint5": 30.0, "joint6": 0.0}
CONNECT_GRIP = 50.0


def load_stl(path):
    """Triangles, shape (n, 3, 3)."""
    b = open(path, "rb").read()
    if b[:5].lower() == b"solid" and b"facet" in b[:400]:
        V = np.array(re.findall(rb"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", b), dtype=float)
        return V.reshape(-1, 3, 3)
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(float)


def sample_surface(T, n, seed):
    """Area-weighted points on the triangles: vertices alone miss the middle of long faces."""
    rng = np.random.default_rng(seed)
    a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    idx = rng.choice(len(T), size=n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = T[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def rpy_matrix(r, p_, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def enclosing_sphere(P, iters=400):
    """Approximate minimum enclosing sphere (Badoiu-Clarkson); the radius is exact for the centre found."""
    c = (P.min(0) + P.max(0)) / 2
    for i in range(iters):
        far = P[np.argmax(np.einsum("ij,ij->i", P - c, P - c))]
        c = c + (far - c) / (i + 2)
    return c, float(np.sqrt(np.einsum("ij,ij->i", P - c, P - c).max()))


def fit_spheres(P, max_r, pad):
    """Split along the longest principal extent at its midpoint until every cluster's enclosing
    sphere + pad <= max_r, then greedily merge neighbouring clusters while the merged sphere still
    fits. Every point of P is inside its cluster's sphere, so coverage of P is exact by construction."""
    todo, clusters = [P], []
    while todo:
        Q = todo.pop()
        _, r = enclosing_sphere(Q)
        if r + pad <= max_r or len(Q) < 8:
            clusters.append(Q)
            continue
        X = Q - Q.mean(0)
        axes = np.linalg.svd(X[:: max(1, len(X) // 4000)], full_matrices=False)[2]
        s = X @ axes[int(np.argmax([np.ptp(X @ a) for a in axes]))]
        m = s <= (s.min() + s.max()) / 2
        todo += [Q[m], Q[~m]]
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        sph = [enclosing_sphere(Q, 150) for Q in clusters]
        C = np.array([c for c, _ in sph])
        for i in np.argsort([r for _, r in sph]):
            d = np.linalg.norm(C - C[i], axis=1)
            d[i] = np.inf
            for j in np.argsort(d)[:4]:
                U = np.vstack([clusters[i], clusters[j]])
                if enclosing_sphere(U, 150)[1] + pad <= max_r:
                    clusters = [q for k, q in enumerate(clusters) if k not in (i, j)] + [U]
                    merged = True
                    break
            if merged:
                break
    return [(c, r + pad) for c, r in (enclosing_sphere(Q) for Q in clusters)]


def coverage(P, spheres):
    """(covered, total, max distance outside the union in m) for an independent point set."""
    d = np.full(len(P), np.inf)
    for c, r in spheres:
        d = np.minimum(d, np.linalg.norm(P - c, axis=1) - r)
    return int((d <= 1e-9).sum()), len(P), float(max(d.max(), 0.0))


def min_depth(V, spheres):
    """Smallest depth of the points V inside the union (m): every vertex is at least this far inside a sphere."""
    d = np.full(len(V), -np.inf)
    for c, r in spheres:
        d = np.maximum(d, r - np.linalg.norm(V - c, axis=1))
    return float(d.min())


# --- joints ---------------------------------------------------------------------
joints = {j.get("name"): j for j in root.findall("joint")}
limits = {}
for name, j in joints.items():
    l = j.find("limit")
    if l is not None and l.get("lower") is not None:
        limits[name] = (float(l.get("lower")), float(l.get("upper")))


def norm_to_rad(name, v):
    # the old urdf joint map (until 2026-10-02): the XRDF default pose is the connect pose under it, ~2-4 deg
    # from the real connect pose (isaac_ik_mirror.py --joint-map calib); plans start from the sim's current joints
    lo, hi = limits[name]
    return lo + (v + 100.0) / 200.0 * (hi - lo)


# cuMotion (urdfdom) treats a <mimic> joint as driven by its parent c-space coordinate and
# rejects a default position for it (RuntimeError at load), so only non-mimic joints are listed.
mimic = {name for name, j in joints.items() if j.find("mimic") is not None}
default_pos = {jn: norm_to_rad(jn, CONNECT_NORM[jn]) for jn in ARM_JOINTS}
for jn in GRIPPER_JOINTS:
    if jn in mimic:
        continue
    lo, hi = limits[jn]
    # "open": planner fingers at the upper limit (0 m = 66 mm = open, measured 2026-09-19), the outermost
    # position the real fingers take; "sweep": the connect-pose 50 %, spheres cover the whole travel.
    default_pos[jn] = hi if args.gripper_model == "open" else lo + CONNECT_GRIP / 100.0 * (hi - lo)
DEFAULT_JOINTS = [jn for jn in ARM_JOINTS + GRIPPER_JOINTS if jn in default_pos]

# --- spheres per link -------------------------------------------------------------
MAX_R = {"base_link": args.base_max_r, args.tool_frame: args.tool_max_r,
         "link7_left": args.finger_max_r, "link7_right": args.finger_max_r}
# gripper sweep: the planner locks joint7 at default_pos (mimic partner follows with multiplier 1), but the
# real fingers can be anywhere in [lower, upper]; a point of the finger mesh then sits at V + axis * (g - g0)
# in the default-opening link frame, so fit the union over the whole travel.
SWEEP = {}
if args.gripper_model == "sweep":
    g0 = default_pos[[jn for jn in GRIPPER_JOINTS if jn not in mimic][0]]
    for jn in GRIPPER_JOINTS:
        j = joints[jn]
        ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        lo, hi = limits[jn] if jn in limits else limits[j.find("mimic").get("joint")]
        SWEEP[j.find("child").get("link")] = [ax * (g - g0) for g in np.linspace(lo, hi, 16)]

link_spheres = {}
report = []
for ln in root.findall("link"):
    vis = ln.find("visual")
    if vis is None:
        continue
    mesh = vis.find("geometry/mesh")
    if mesh is None:
        continue
    T = load_stl(os.path.normpath(os.path.join(urdf_dir, mesh.get("filename"))))
    o = vis.find("origin")
    if o is not None:
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        T = T @ rpy_matrix(*rpy).T + xyz
    name = ln.get("name")
    V = T.reshape(-1, 3)
    P = np.vstack([sample_surface(T, args.samples, seed=0), V])
    check = np.vstack([sample_surface(T, 8 * args.samples, seed=1), V])      # independent points for the check
    if name in SWEEP:
        P = np.vstack([P[:: len(SWEEP[name])] + d for d in SWEEP[name]] + [P + d for d in SWEEP[name][:: len(SWEEP[name]) - 1]])
        steps = SWEEP[name]
        mids = [(a + b) / 2 for a, b in zip(steps[:-1], steps[1:])]        # between the fitted positions
        check = np.vstack([check[:: 4] + d for d in steps + mids])
    sph = fit_spheres(P, MAX_R.get(name, args.max_r), args.pad)
    cov, n, out_mm = coverage(check, sph)
    if cov < n:   # a sampling gap wider than the pad: grow this link's spheres by the miss, then re-check
        sph = [(c, r + out_mm + 0.0005) for c, r in sph]
        cov, n, out_mm = coverage(check, sph)
    link_spheres[name] = sph
    report.append((name, len(sph), cov, n, max(r for _, r in sph), out_mm, min_depth(V, sph)))

# --- self-collision ignore: adjacent links + clusters with coincident joint origins -------
ignore = {
    "base_link": ["link1", "link2"],
    "link1": ["link2", "link3"],
    "link2": ["link3", "link4"],  # link4 overlaps link2 in the folded connect/rest poses (inspector, 2026-09-19)
    "link3": ["link4", "link5"],
    "link4": ["link5", "link6"],
    "link5": ["link6", "link7_left", "link7_right"],
    "link6": ["link7_left", "link7_right"],
    "link7_left": ["link7_right"],
}

# --- write ---------------------------------------------------------------------------
L = []
L.append("# Generated by scripts/make_viola_xrdf.py - do not hand-edit; re-run instead.")
L.append(f"# Spheres fitted to STL surface samples + vertices (max r arm {args.max_r} / link6 {args.tool_max_r} / fingers "
         f"{args.finger_max_r} / base {args.base_max_r} m, pad {args.pad} m, gripper model {args.gripper_model}); "
         f"limits chosen, not measured.")
L.append("format: xrdf")
L.append("format_version: 2.0")
L.append("")
L.append("default_joint_positions:")
for jn in DEFAULT_JOINTS:
    L.append(f"  {jn}: {default_pos[jn]:.4f}")
L.append("")
L.append("cspace:")
L.append("  joint_names:")
for jn in ARM_JOINTS:
    L.append(f'    - "{jn}"')
L.append(f"  acceleration_limits: [{', '.join(f'{args.accel}' for _ in ARM_JOINTS)}]")
L.append(f"  jerk_limits: [{', '.join(f'{args.jerk}' for _ in ARM_JOINTS)}]")
L.append("")
L.append(f'tool_frames: ["{args.tool_frame}"]')
L.append("")
# base_link never moves, so it must not be a WORLD collision body (it would sit "inside" a
# table/floor obstacle and every configuration would be in collision). It stays in
# self-collision only - the same split the shipped Franka XRDF uses for panda_link0.
L.append("world_collision:")
L.append('  geometry: "viola_collision_spheres"')
L.append("")
L.append("self_collision:")
L.append('  geometry: "viola_self_collision_spheres"' if not args.no_base_split else '  geometry: "viola_collision_spheres"')
L.append("  ignore:")
for a, bs in ignore.items():
    quoted = ", ".join('"' + b + '"' for b in bs)
    L.append(f"    {a}: [{quoted}]")
L.append("")
L.append("geometry:")
L.append("  viola_collision_spheres:")
L.append("    spheres:")
for link, sph in link_spheres.items():
    if link == "base_link" and not args.no_base_split:
        continue
    L.append(f"      {link}:")
    for c, r in sph:
        L.append(f"        - center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
        L.append(f"          radius: {r:.4f}")
L.append("  viola_base_spheres:")
L.append("    spheres:")
L.append("      base_link:")
for c, r in link_spheres["base_link"]:
    L.append(f"        - center: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
    L.append(f"          radius: {r:.4f}")
L.append("  viola_self_collision_spheres:")
L.append('    clone: ["viola_collision_spheres", "viola_base_spheres"]')
open(out, "w").write("\n".join(L) + "\n")

print(f"[xrdf] written {out}")
print(f"[xrdf] cspace {ARM_JOINTS} tool_frame={args.tool_frame}")
print(f"[xrdf] default_joint_positions (arm: connect pose; gripper: {args.gripper_model}, rad): "
      + " ".join(f"{jn}={default_pos[jn]:+.3f}" for jn in DEFAULT_JOINTS) + f"   (mimic joints skipped: {sorted(mimic)})")
print(f"[xrdf] {'link':12s} spheres  covered/checked      max r   max outside   vertex depth min (pad {args.pad*1000:.1f} mm)")
tot = 0
for name, k, cov, n, rmax, out_m, depth in report:
    tot += k
    flag = "" if cov == n else "   <-- NOT FULLY COVERED"
    flag += "" if depth >= args.pad - 1e-9 else "   <-- VERTEX INSIDE THE PAD"
    print(f"[xrdf] {name:12s} {k:7d}  {cov:7d}/{n:<7d}  {rmax:.3f}   {out_m*1000:.2f} mm       {depth*1000:.2f} mm{flag}")
print(f"[xrdf] total spheres: {tot}")
