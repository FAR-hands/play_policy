#!/usr/bin/env python3
"""Convert the Sharpa right-hand URDF -> USD for Isaac Lab.

Needs the Isaac Lab env (the URDF importer runs inside the Isaac Sim app).
Output: isaac/assets/sharpa_right/sharpa_right.usd

The release already ships this USD, so you only need this script after editing the
URDF. Re-converting with a different Isaac Sim version can change the convex-hull
colliders slightly -- compare `sim_params_hash` of a fresh take against the dataset's
before mixing data (see docs/TELEOP.md).

    OMNI_KIT_ACCEPT_EULA=YES python isaac/convert_sharpa_usd.py
    OMNI_KIT_ACCEPT_EULA=YES python isaac/convert_sharpa_usd.py --out-dir /tmp/sharpa_usd
"""
import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

_HERE = os.path.dirname(os.path.abspath(__file__))
_URDF = os.path.abspath(os.path.join(
    _HERE, "..", "assets", "urdf", "sharpa", "right_fixed.urdf"))
_OUT_DIR = os.path.join(_HERE, "assets", "sharpa_right")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", default=_URDF)
parser.add_argument("--out-dir", default=_OUT_DIR)
parser.add_argument("--name", default="sharpa_right.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- app live below ----
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main():
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(args.urdf)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=args.urdf,
        usd_dir=args.out_dir,
        usd_file_name=args.name,
        fix_base=True,                 # hand is mounted, not floating
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=False,
        # Position-controlled finger joints; gains are refined in the env actuators.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=3.0, damping=0.1),
        ),
    )
    conv = UrdfConverter(cfg)
    print(f"[convert] Sharpa USD written -> {conv.usd_path}")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:              # noqa: BLE001
        traceback.print_exc()
        code = 1
    # Kit can hang inside simulation_app.close(); the USD is on disk by now
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
