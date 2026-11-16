"""LeRobot normalized units (-100..100) <-> sim joint angles (rad) for the Viola's six arm joints.

Two maps, both linear in the normalized value, so each is a pair of end points per joint:
    urdf   -100..100 spread over the URDF limits (the sim's map until 2026-10-02)
    calib  what the real arm does (the sim's default since 2026-10-03): -100..100 -> the follower calibration's range_min..range_max counts
           (lerobot_motor_starai/starai.py:140 read, :192 write), counts -> servo degrees, 4096 counts = 360
           (starai.py:121, :212), servo degrees = URDF joint degrees (the vendor's ROS 2 driver,
           fashionstar-starai-arm-ros2/src/viola_controller/viola_controller/viola_controller.py:44-47)

Measured 2026-10-01/02 on the physical arm, held at reference poses, with a phone clinometer on the forearm
(6 holds, two runs; docs/joint-map.md):
per step the phone agreed with calib within 1.1 deg, with urdf only within 4.6 deg. The forearm tilt the phone
reads depends on the shoulder (real span 159 deg, URDF 180) and the elbow (191 vs 180).

    python3 scripts/joint_map.py      # print both maps per joint
"""

import json
import math
import os

MOTORS = [f"Motor_{i}" for i in range(6)]
# The calibration of one physical Viola, shipped as the example. With your own arm, pass your LeRobot
# calibration file instead (isaac_ik_mirror.py --follower-calib PATH).
FOLLOWER_CALIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config", "viola_calibration_example.json")
COUNTS_PER_TURN = 4096


def counts_to_rad(counts):
    return math.radians(counts * 360.0 / COUNTS_PER_TURN - 180.0)


def calib_ends(path=None):
    """Per arm joint (Motor_0..5), the joint angle in rad at normalized -100 and at +100 on the real arm."""
    cal = json.load(open(path or FOLLOWER_CALIB))
    ends = []
    for m in MOTORS:
        lo, hi = counts_to_rad(cal[m]["range_min"]), counts_to_rad(cal[m]["range_max"])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo != hi):   # a zero-width range would divide by zero later
            raise ValueError(f"{m}: range_min and range_max must be two different numbers")
        ends.append((hi, lo) if cal[m].get("drive_mode") else (lo, hi))  # drive_mode negates (starai.py:142, :193)
    return ends


def check_calib(parser, path=None):
    """Stop with a usage error when the calibration file cannot be used. Call it right after parse_args,
    so a wrong path fails in a second and not after Isaac Sim has booted."""
    try:
        calib_ends(path)
    except (OSError, KeyError, TypeError, ValueError) as e:   # missing file, missing motor, wrong shape, bad JSON
        parser.error(f"calibration file {path or FOLLOWER_CALIB}: {type(e).__name__}: {e}")


def to_rad(n, ends):
    return ends[0] + (n + 100.0) / 200.0 * (ends[1] - ends[0])


def to_norm(rad, ends):
    return (rad - ends[0]) / (ends[1] - ends[0]) * 200.0 - 100.0


if __name__ == "__main__":
    import re
    urdf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets/viola/urdf/viola.urdf")).read()
    print("joint    -100..+100 calib (deg)    -100..+100 urdf (deg)    calib deg per unit vs urdf")
    for i, (a, b) in enumerate(calib_ends()):
        blk = re.search(rf'<joint\s+name="joint{i + 1}".*?</joint>', urdf, re.S).group(0)
        lo, hi = (float(v) for v in re.search(r'lower="([^"]+)"\s+upper="([^"]+)"', blk).groups())
        print(f"Motor_{i}  {math.degrees(a):+7.1f}..{math.degrees(b):+6.1f}        {math.degrees(lo):+7.1f}..{math.degrees(hi):+6.1f}"
              f"          {math.degrees(b - a) / 200:.3f} vs {math.degrees(hi - lo) / 200:.3f}")
