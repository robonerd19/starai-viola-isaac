"""Import the patched Viola URDF into USD.

Run with Isaac Sim's python:
    $ISAACSIM_PATH/python.sh scripts/import_viola.py
"""

import os
import shutil

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "assets", "viola", "urdf", "viola.urdf")
OUT = os.path.join(REPO, "assets", "viola", "usd")

os.makedirs(OUT, exist_ok=True)

# Gains derived from OEM servo datasheets - see scripts/derive_gains.py
STIFFNESS = {
    "joint1": 17.6, "joint2": 45.2, "joint3": 45.2,
    "joint4": 17.6, "joint5": 17.6, "joint6": 17.6,
    "joint7_.*": 200.0,   # prismatic: N/m, conservative (no RA8-U26H-M datasheet)
}
DAMPING = {
    "joint1": 1.109, "joint2": 2.307, "joint3": 1.446,
    "joint4": 0.059, "joint5": 0.357, "joint6": 0.049,
    "joint7_.*": 20.0,
}

config = URDFImporterConfig(
    urdf_path=URDF,
    usd_path=OUT,
    fix_base=True,
    merge_fixed_joints=False,
    # Isaac Sim 6.0.1 applies collision_type only together with collision_from_visuals, which stays off here:
    # the colliders come out as one convex hull per link.
    collision_type="Convex Decomposition",
    allow_self_collision=False,
    joint_drive_type="force",
    joint_target_type="position",
    override_joint_stiffness=STIFFNESS,
    override_joint_damping=DAMPING,
)

usd_file = URDFImporter(config).import_urdf()

# The importer never overwrites: with assets/viola/usd/viola in place it writes to viola_1. Move the new import
# over the old one, so the panel, which loads usd/viola, gets it.
new_dir, target = os.path.dirname(usd_file), os.path.join(OUT, "viola")
if new_dir != target:
    shutil.rmtree(target)
    os.replace(new_dir, target)
    usd_file = os.path.join(target, os.path.basename(usd_file))

# The importer writes its temp folders and the absolute output path into every layer's doc string.
# Clear them, so a re-import gives the same files on any machine.
from pxr import Sdf  # noqa: E402

for folder, _, names in os.walk(os.path.dirname(usd_file)):
    for name in names:
        if name.endswith((".usd", ".usda", ".usdc")):
            path = os.path.join(folder, name)
            layer = Sdf.Layer.FindOrOpen(path)
            layer.documentation = ""
            # Export to a new file, not Save: a binary layer saved in place keeps the old string in its string table.
            tmp = os.path.join(folder, "tmp_" + name)
            layer.Export(tmp)
            os.replace(tmp, path)
print(f"[import_viola] wrote {usd_file}")

simulation_app.close()
