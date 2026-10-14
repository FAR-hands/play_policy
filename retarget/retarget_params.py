"""Shared retargeting parameters for the glove teleop scripts, backed by a YAML file.

One source of truth so the viewer (retarget_viz.py) and the Isaac teleop
(isaac/teleop_isaac.py) agree on master/finger size, DIP coupling, the abduction
fix, and the thumb-joint settings. retarget_viz.py can overwrite it (Ctrl-S or the
"Save params" button); other scripts read it for their argparse defaults.
"""
import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(_HERE, "retarget_params.yaml")

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
THUMB_JOINTS = ["CMC_AA", "CMC_FE", "MCP_FE", "IP"]

DEFAULTS = {
    "scaling": 0.9,                                   # master size multiplier
    "finger_scales": {"thumb": 1.0, "index": 1.0, "middle": 1.0,
                      "ring": 1.0, "pinky": 1.2},
    "dip_coupling": 0.66,
    "ab_correct_deg": 0.0,                            # abduction-lean fix (deg)
    "thumb_mode": "joint",                            # "shape" | "joint"
    "thumb_ema": 0.3,
    "thumb_joint_weight": 0.5,                        # joint-mode blend: 1=pure ergo, 0=pure pose
    "thumb_invert": {"CMC_AA": False, "CMC_FE": True,
                     "MCP_FE": False, "IP": False},
}


def _merge(base, override):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load(path=None):
    """Return the params dict (DEFAULTS merged with the YAML file if present)."""
    path = path or PARAMS_PATH
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, data)


def save(params, path=None):
    """Write the params dict to the YAML file; returns the path."""
    path = path or PARAMS_PATH
    with open(path, "w") as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)
    return path


def finger_scales_str(params):
    """The finger_scales as a "thumb,index,middle,ring,pinky" CLI string."""
    fs = params["finger_scales"]
    return ",".join(str(fs[f]) for f in FINGER_ORDER)
