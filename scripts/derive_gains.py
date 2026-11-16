"""Derive PhysX joint drive gains for the Viola from OEM servo datasheets.

Sources (all figures below are quoted, not estimated):
  RX8-U50H-M  "UART Bus Servo Actuator 50KG Series" datasheet EN V1.1.08
      https://files.seeedstudio.com/products/Fashionstar/Fashion%20Star_U50H_Datasheet_EN_V1.1.08.pdf
      reduction 387:1, stall 4.90 N.m, continuous 2.26 N.m, rated 0.88 N.m, 52 rpm, 83 g
  RA8-U25H-M  "UART Bus Servo Actuator 25KG Series" datasheet EN V1.1.08
      https://files.seeedstudio.com/products/Fashionstar/Fashion%20Star_U25(H)_Datasheet_EN_V1.1.08.pdf
      reduction 273:1, stall 2.45 N.m, continuous 0.88 N.m, rated 0.44 N.m, 63 g
  RA8-U26H-M  no public datasheet found (Seeed file server returns 403) -> gripper handled
      conservatively, NOT from spec.
  Servo counts per arm: RX8-U50H-M x2, RA8-U25H-M x4, RA8-U26H-M x1
      https://wiki.seeedstudio.com/lerobot_starai_arm/

Model:
    tau = k * (q_target - q) + d * (qdot_target - qdot)
    k [N.m/rad], d [N.m.s/rad]

k comes from the torque budget: the joint should be able to command its full
continuous torque at an acceptable steady-state tracking error.
    k = tau_continuous / theta_allowed

d comes from critical damping (zeta = 1, no overshoot) against the effective
inertia the joint actually sees:
    d = 2 * zeta * sqrt(k * J_eff)
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "viola", "urdf", "viola.urdf")

# --- OEM datasheet values -------------------------------------------------
SERVO = {
    "RX8-U50H-M": {"stall": 4.90, "cont": 2.26, "rated": 0.88, "ratio": 387, "mass": 0.083},
    "RA8-U25H-M": {"stall": 2.45, "cont": 0.88, "rated": 0.44, "ratio": 273, "mass": 0.063},
}

# INFERENCE (not from OEM docs): the two high-torque U50H units are assigned to the
# joints carrying the largest gravity moment - shoulder pitch (joint2) and elbow
# (joint3). Verify by querying servo IDs/models over the UART bus on the real arm.
JOINT_SERVO = {
    "joint1": "RA8-U25H-M",
    "joint2": "RX8-U50H-M",
    "joint3": "RX8-U50H-M",
    "joint4": "RA8-U25H-M",
    "joint5": "RA8-U25H-M",
    "joint6": "RA8-U25H-M",
}

THETA_ALLOWED = 0.05  # rad (2.9 deg) tracking error at full continuous torque
ZETA = 1.0            # critically damped


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def parse(urdf):
    root = ET.parse(urdf).getroot()
    links, joints = {}, []
    for l in root.iter("link"):
        i = l.find("inertial")
        if i is None:
            links[l.get("name")] = {"m": 0.0, "com": np.zeros(3)}
            continue
        o = i.find("origin")
        com = np.array([float(x) for x in o.get("xyz").split()]) if o is not None else np.zeros(3)
        links[l.get("name")] = {"m": float(i.find("mass").get("value")), "com": com}
    for j in root.iter("joint"):
        o, a = j.find("origin"), j.find("axis")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": np.array([float(x) for x in o.get("xyz").split()]) if o is not None else np.zeros(3),
            "rpy": np.array([float(x) for x in o.get("rpy").split()]) if o is not None and o.get("rpy") else np.zeros(3),
            "axis": np.array([float(x) for x in a.get("xyz").split()]) if a is not None else np.array([0, 0, 1.0]),
        })
    return links, joints


def forward_kinematics(links, joints):
    """World pose of every link frame at the URDF zero pose."""
    pose = {"world": (np.zeros(3), np.eye(3))}
    for j in joints:  # URDF joint list is parent-before-child in this file
        p_pos, p_R = pose[j["parent"]]
        R = p_R @ rpy_to_R(*j["rpy"])
        pos = p_pos + p_R @ j["xyz"]
        pose[j["child"]] = (pos, R)
    return pose


def main():
    links, joints = parse(URDF)
    pose = forward_kinematics(links, joints)

    # child links downstream of each joint
    children = {j["parent"]: [] for j in joints}
    for j in joints:
        children.setdefault(j["parent"], []).append(j["child"])

    def subtree(link):
        out = [link]
        for c in children.get(link, []):
            out += subtree(c)
        return out

    # ---- mass audit ------------------------------------------------------
    urdf_total = sum(v["m"] for v in links.values())
    servo_total = sum(SERVO[s]["mass"] for s in JOINT_SERVO.values())
    print("=" * 72)
    print("MASS AUDIT")
    print("=" * 72)
    print(f"URDF total link mass          : {urdf_total*1000:7.1f} g")
    print(f"Datasheet mass of 6 arm servos: {servo_total*1000:7.1f} g  (gripper servo unknown)")
    print(f"Structure left over           : {(urdf_total-servo_total)*1000:7.1f} g")
    print()
    for jn, sm in JOINT_SERVO.items():
        child = next(j["child"] for j in joints if j["name"] == jn)
        print(f"  {child:10s} URDF mass {links[child]['m']*1000:6.1f} g  vs  "
              f"{sm} alone {SERVO[sm]['mass']*1000:.0f} g"
              f"{'   <-- IMPOSSIBLE' if links[child]['m'] < SERVO[sm]['mass'] else ''}")

    # corrected masses: each link carries the servo that drives it
    corrected = {n: v["m"] for n, v in links.items()}
    for jn, sm in JOINT_SERVO.items():
        child = next(j["child"] for j in joints if j["name"] == jn)
        corrected[child] += SERVO[sm]["mass"]

    # ---- gains -----------------------------------------------------------
    print()
    print("=" * 72)
    print("GAIN DERIVATION")
    print("=" * 72)
    print(f"{'joint':9s} {'servo':12s} {'tau_c':>6s} {'k':>7s} {'J_eff':>9s} {'d':>7s}")
    print(f"{'':9s} {'':12s} {'N.m':>6s} {'N.m/rad':>7s} {'kg.m^2':>9s} {'N.m.s':>7s}")
    print("-" * 72)

    stiffness, damping, effort = {}, {}, {}
    for j in joints:
        jn = j["name"]
        if jn not in JOINT_SERVO:
            continue
        spec = SERVO[JOINT_SERVO[jn]]

        k = spec["cont"] / THETA_ALLOWED

        # effective inertia: downstream masses about this joint's axis
        origin, R = pose[j["child"]]
        axis_w = R @ (j["axis"] / np.linalg.norm(j["axis"]))
        J = 0.0
        for ln in subtree(j["child"]):
            m = corrected[ln]
            if m <= 0:
                continue
            lp, lR = pose[ln]
            com_w = lp + lR @ links[ln]["com"]
            r = com_w - origin
            r_perp = r - np.dot(r, axis_w) * axis_w  # distance from the rotation axis
            J += m * float(np.dot(r_perp, r_perp))

        d = 2 * ZETA * np.sqrt(k * J) if J > 0 else 0.0

        # float(): numpy 2 would print np.float64(...) in the dicts below
        stiffness[jn], damping[jn], effort[jn] = round(float(k), 1), round(float(d), 3), spec["cont"]
        print(f"{jn:9s} {JOINT_SERVO[jn]:12s} {spec['cont']:6.2f} {k:7.1f} {J:9.5f} {d:7.3f}")

    # gripper: no datasheet for RA8-U26H-M -> conservative, gentle-close values
    print("-" * 72)
    print("joint7_*  RA8-U26H-M   (no public datasheet) -> conservative gripper gains")
    print()
    print("stiffness =", stiffness)
    print("damping   =", damping)
    print("effort    =", effort)


if __name__ == "__main__":
    main()
