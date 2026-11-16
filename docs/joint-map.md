# The joint map: making the sim arm match the physical arm

LeRobot reports and commands each joint as a number from -100 to 100. The sim needs joint angles.
This page is about the conversion between the two, how it was checked on the physical arm, and two
table defaults that came out of the same work.

Nothing here is needed to run the sim. It explains why the defaults are what they are, and what to
check when you connect an arm of your own.

## The problem

With the physical arm following the sim, planned moves and the pick sequence ran, but the real
gripper stopped too high to grasp the cube. Lowering the sim table didn't fix it. The error grew with
the joint angles, so it had to come from the joint mapping or the model.

## Cause: the -100..100 scale belongs to each arm

LeRobot maps -100..100 onto that arm's own calibration range, in servo counts
(`lerobot_motor_starai/starai.py`). 4096 counts are 360 servo degrees. The vendor's ROS 2 driver
sends URDF joint angles to the servos one to one in degrees
(`viola_controller.py` in the vendor repo).

The first version of the sim spread -100..100 over the URDF joint limits instead. Where the two
ranges differ, the same number means a different angle.

| Joint | Example calibration | URDF limits | The URDF map, for the same number |
|---|---|---|---|
| Motor_0 base | -122.1 to +123.0° | ±130.1° | turns 6 % too far |
| Motor_1 shoulder | -88.1 to +70.9° | ±90.0° | 13 % too far. 8.6° off at 0, 19° at +100 |
| Motor_2 elbow | -100.0 to +91.0° | ±90.0° | 6 % short |
| Motor_3 wrist roll | -180 to +180° | ±130.1° | 28 % short |
| Motor_4 wrist pitch | -109.1 to +94.9° | ±90.0° | 12 % short, 7° off at mid-range |
| Motor_5 wrist rotation | -176.0 to +176.0° | ±130.1° | 26 % short |

`scripts/joint_map.py` prints the two ranges from the calibration file and the URDF.

```bash
$ISAACSIM_PATH/python.sh scripts/joint_map.py
```

## How it was checked

The physical arm was held at reference poses, one joint moving at a time, at least 5 s per hold. A
phone clinometer lay on top of the forearm. Every link past a joint turns with that joint, so the
change in the phone's reading between two holds is the joint's real rotation. Only changes count, so
it doesn't matter exactly how the phone sits.

| Run | Phone, per step | Calibration map | URDF map |
|---|---|---|---|
| Shoulder +40 and elbow -40 units per step | 5°, 5° | 6.1°, 6.0° | 0.5°, 0.4° |
| Shoulder +30 units per step | 26°, 24° | 25.0°, 24.0° | 28.2°, 27.2° |

Per step, the phone agreed with the calibration map within 1.1° and with the URDF map only within
4.6°. Summed over a run the difference was 2.0° against 9.1°. That is 6 holds over two runs, measured
on 2026-10-01 and 2026-10-02.

## The fix

`--joint-map calib` is the default. It converts through the calibration file, as the physical arm
does. Sim targets are clipped to the range that both the URDF joint stops and the calibration allow.
`--joint-map urdf` keeps the old map for comparison, and `plan_selftest.py` takes the same flag.

The repo ships the calibration of one physical Viola as `config/viola_calibration_example.json`.
Its wrist-rotation range was shifted by 46 counts so that the middle of the range sits on the servo's
zero. With your own arm, pass your own file with `--follower-calib`.

## The table defaults

**The physical gripper sits lower than the sim's.** With the sim gripper held 40 mm above the table
on purpose, the real gripper sat lower:

| Distance from the base | Sim, lowest gripper point | Physical arm, by ruler | Physical arm lower by |
|---|---|---|---|
| 0.16 m | 38.9 mm | about 30 mm | about 9 mm |
| 0.22 m | 38.0 mm | about 30 mm | about 8 mm |
| 0.30 m | 36.1 mm | about 20 mm | about 16 mm |

The servos only see 0.4 to 4.5 mm of that. I didn't measure where the rest comes from. My guess is
play and flex that the encoders can't see. The vendor drawing also differs from the URDF meshes by
about 1 mm in base height and about 7 mm in wrist-to-fingertip length, and the model is not corrected
for either.

Two defaults absorb it:

- `--table-z 0.007` raises the sim table, and a cube on it, by 7 mm. That is the difference measured
  with the real fingers touching the table 15 cm from the base.
- `--table-clearance 0.013` is then the height of the real gripper above the real table at the grasp.

With both, the physical arm following the sim picked 30 mm cubes between 15 and 22 cm from the base
(2026-10-03). It is one constant for the whole table. At 30 cm the physical arm still sits about 9 mm
lower than the sim, so check the height there before picking that far out.

For a sim-only setup with the URDF's own table height, use `--table-z 0`.

## What the servos do that the sim doesn't

Measured on the physical arm on 2026-09-18:

- Position reads come in steps of about 1°.
- The servo firmware has a 1° deadband and no integral action. Every joint stops within about 1° of
  its target.
- Joints that carry weight settle off target by a constant amount. The wrist pitch settles about 2°
  off, the elbow about 0.5°.

The sim drives are stiff and have none of this. `pick_planner.py` keeps every joint of an accepted
solution 3° inside its limit for that reason.

Five of the six arm links are lighter in the vendor URDF than the servo that drives them. Joint
positions and planned paths don't depend on the masses. Contact forces do, so don't read grasp forces
off this model.

## Checking an arm of your own

After a servo swap, a new calibration or a different gripper:

1. **Joint map.** Hold the arm at two poses that differ in one joint. Put a phone clinometer on the
   forearm, in the same spot for both holds. Compare the change in its reading with the change
   `joint_map.py` predicts for that joint.
2. **Table height.** Start the sim with `--table-z 0` and let the arm follow it. Jog down until the
   real fingers touch the table, then press *Probe*. Use the "lowest gripper point" it prints, in
   metres, as `--table-z`.
