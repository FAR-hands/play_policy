# Installation

Everything in this repository runs in **one conda environment** (called `dsp` below):
Isaac Sim 5.1 + Isaac Lab 2.3.2 (Python 3.11) plus a handful of pip packages. Training and
the offline quick-start check do not need Isaac; if you only want those, a plain
`python>=3.10` env with PyTorch and `requirements.txt` is enough (see the end of this page).

Tested on Ubuntu 22.04/24.04, one NVIDIA RTX 5090 (32 GB), driver 570+, CUDA 12.8 wheels.

## 1. Isaac Sim 5.1 + Isaac Lab 2.3.2

Follow the official pip installation; condensed:

```bash
conda create -n dsp python=3.11 -y
conda activate dsp
pip install --upgrade pip

# PyTorch first (CUDA 12.8 wheels; RTX 50xx GPUs need cu128)
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 5.1 (pip distribution)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab 2.3.2 (editable, from source)
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab-2.3.2
cd ~/IsaacLab-2.3.2 && ./isaaclab.sh --install none && cd -
```

The first Isaac launch asks you to accept the NVIDIA Omniverse EULA; all scripts here set
`OMNI_KIT_ACCEPT_EULA=YES`, which accepts it non-interactively. Booting a headless scene takes
~30–60 s the first time (shader/extension cache), less afterwards.

## 2. This repository's Python dependencies

```bash
cd dsp_release            # repository root
pip install -r requirements.txt
```

Nothing needs to be built or installed in editable mode; scripts are run from the repository
root (`python -m play_policy.train ...`, `python isaac/wm_mpc_eval_batch.py ...`).

## 3. Optional: Manus glove (only for collecting your own teleoperation data)

See [TELEOP.md](TELEOP.md): MANUS SDK download (not redistributable, so not included),
`libzmq3-dev`, udev rules, building the ZMQ glove server, calibration.

## 4. Sanity check

```bash
bash quickstart/download_checkpoints.sh          # paper checkpoints -> checkpoints/
python quickstart/check_inference.py             # ~1 min, no simulator
```

then the closed-loop simulator check described in [quickstart/README.md](../quickstart/README.md).

## Physics reproducibility

All physics settings live in `isaac/sim_params.py`; every recorded file stamps them as
`sim_params_json` and a 12-character `sim_params_hash`. Self-play collected with the released
code must stamp **`21c4c9c015e1`**, the hash of the released dataset — compare a smoke rollout
with `python isaac/inspect_npz.py <file>.npz` before mixing your data with ours. Isaac Sim /
PhysX versions other than 5.1 may change contact behaviour even when the hash matches.

## Training-only environment (no simulator)

```bash
conda create -n dsp-train python=3.11 -y && conda activate dsp-train
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or your CUDA/CPU build
pip install -r requirements.txt
```

This runs `play_policy.train`, `play_policy.train_worldmodel` and
`quickstart/check_inference.py`. Self-play collection, MPC evaluation and teleoperation need
the full `dsp` environment.
