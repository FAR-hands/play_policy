# Teleoperation: collecting human play with a Manus glove

Stage 1 of the pipeline needs a few minutes of **goal-free human play**: you wear a Manus glove,
the glove drives a simulated Sharpa right hand, and you push a 5 cm cube around in the palm
however you like. Those takes train the play policy (`python -m play_policy.train`).

```
Manus glove ──USB──> manus/server (C++, MANUS SDK) ──ZMQ :5555──> retarget/ (25-point skeleton
  -> 22 Sharpa joints, dex-retargeting vector optimizer) ──> isaac/teleop_isaac.py (Isaac Lab,
  palm-up hand + cube, 1/240 s of physics per frame) ──> data/teleop_takes/play_<time>_take<n>.npz
```

The released dataset already contains the human takes used in the paper
(`data/dsp_sim_dataset/teleop/`: 2 takes, 29,474 frames, recorded over 5.3 + 14.6 min of wall
clock), so you only need this page to collect **your own** play. Without a glove you can still run
the scene (`--source demo`) or replay a dataset take (`--source replay`), see the end of this page.

All commands below run from the repository root, in the Isaac Lab python env (see
[INSTALL.md](INSTALL.md)).

## 1. Hardware

| item | notes |
|---|---|
| Manus glove, right hand | tested with a Manus Quantum Metaglove (USB `3325:00ca`) and a Prime 3 Haptic XR (`3325:0425`) |
| the glove's **license dongle** | Metaglove: "Manus VR Sensor Dongle" (`3325:0049`); Prime 3: "Prime1 Dongle" (`1915:83fd`) |
| NVIDIA GPU | for Isaac Sim (tested on an RTX 5090) |

**The MANUS SDK license lives on the dongle, not on the glove.** With only the glove plugged in,
the server logs `No compatible license found` and streams messages whose skeleton and ergonomics
are EMPTY — the hand in the sim then never moves. Always plug in the glove *and* its dongle.

## 2. One-time setup

```bash
sudo apt install libzmq3-dev cmake pkg-config build-essential

# USB permissions for the glove + dongles (vendor 3325, and the 1915:83fd Nordic dongle)
sudo tee /etc/udev/rules.d/70-manus-hid.rules << 'EOF'
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="1915", ATTR{idProduct}=="83fd", MODE="0666", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**MANUS SDK.** It is proprietary and not part of this repository. Download the MANUS Core SDK for
Linux from MANUS and copy its `ManusSDK` folder to `manus/ManusSDK/` (so that
`manus/ManusSDK/include/ManusSDK.h` and `manus/ManusSDK/lib/libManusSDK_Integrated.so` exist), or
`export MANUS_SDK_DIR=/path/to/ManusSDK`. We used the SDK of MANUS Core 3.1.1; the exact file
hashes are in [manus/README.md](../manus/README.md).

Build and start the glove server (it builds on first run), then check the stream:

```bash
manus/server.sh                                             # owns the glove; leave it running
PYTHONPATH=manus python manus/examples/print_glove_data.py --side right --ergo
```

A healthy stream reports 25 skeleton nodes and 19 ergonomics values at ~118 Hz. `nodes=0` means
the license dongle is missing.

## 3. Calibrate the glove

With the server running:

```bash
manus/calibrate.sh --side right        # viser UI at http://localhost:8080
```

Follow the guided poses (Start, then Next after each pose; timed steps advance on their own).
The calibration is stored by the MANUS SDK; `Save` in the UI also writes it to a file you can
`Load` later. Re-calibrate whenever someone else wears the glove.

## 4. Check the retargeting (no simulator)

```bash
python retarget/retarget_viz.py        # live glove -> Sharpa hand, http://localhost:8080
```

The Manus skeleton (green) and the retargeted Sharpa hand are drawn side by side. Sliders tune
the master size, per-finger scales, DIP coupling, the abduction correction and the thumb mode;
**Ctrl-S** (terminal) or **Save params** writes them to `retarget/retarget_params.yaml`, which
the teleop reads for its defaults. The shipped YAML is the setting that was in place when the
paper's takes were recorded. The Sharpa finger has 4 DoF, so ~15° of fingertip-orientation residual is the
kinematic floor, not a tuning problem. Keep the parameters fixed within one collection.

## 5. Teleoperate and record

```bash
scripts/teleop.sh --no-sim-panel
```

`scripts/teleop.sh` starts `manus/server.sh` if nothing is publishing on port 5555 yet (and stops
it again on exit), then runs `isaac/teleop_isaac.py --source live`. Before booting Isaac, the
teleop waits up to 20 s (`--glove-timeout`) for a non-empty glove skeleton and exits with a
message saying which problem you have: no server, or a server without a license. Isaac then boots
headless (~30–60 s) and the 3D view appears at **http://localhost:8080** (`--viz-port` to change).
All extra arguments are passed to `teleop_isaac.py`.

| control | terminal key | viser button |
|---|---|---|
| start a take | `s` | Start rec (s) |
| end the take and save it | `e` | End rec (e) |
| reset the cube (new random orientation) | `r` | Reset (r) |
| quit (a running take is saved first) | Ctrl-C | |

- **Drops reset automatically**: when the cube centre falls below z = 0.15 m it respawns, and
  while recording ±1.5 s around every reset is cut from the take (`--reset-trim`), so drops are
  harmless — no need to stop the take.
- The cube spawns at a **uniform random SO(3) orientation** at startup and on every reset, so
  your play covers the whole rotation group rather than one start pose.
- Takes go to `data/teleop_takes/play_<YYYYmmdd_HHMMSS>_take<n>.npz` (`--out-dir DIR` to change,
  or `--record-path FILE` for an explicit file name).
- Unattended fixed-length takes: `--record --record-secs 120` starts a take at launch, saves it
  after 120 s and exits.

**What to play.** Free play: turn the cube about every axis, in both directions, pass it between
fingers, let it roll, catch near-drops. The play policy imitates the *distribution* of what you
do, and the world model can only predict what the autonomous play later visits, so variety
matters more than skill. The paper used 29,474 frames.

### Keep the data compatible

Everything that defines the simulator lives in `isaac/sim_params.py` and is stamped into every
take as `sim_params_json` + `sim_params_hash` (`python isaac/inspect_npz.py <take>` prints it on
the first line).

- **Physics.** Each control frame advances the simulation by 1/240 s (3 PhysX substeps of
  1/720 s), whatever the loop rate. With default flags a live take stamps
  `sim_params_hash 90d13aa1b16d`, exactly like the released teleop takes. The released self-play
  data stamps `21c4c9c015e1`: the physics fields are identical; the hash differs only in
  bookkeeping (the `source`/`cube_size` fields the teleop adds, and the cube's prim path in the
  multi-env collection scene).
- **Lock the physics knobs while collecting**: `--no-sim-panel` hides the viser "Simulator"
  sliders. (An untouched panel changes nothing, but a moved slider makes the take mix dynamics;
  such takes record `sim_param_changes > 0` — do not train on them.)
- **Loop rate.** Frames are labelled 60 Hz (`--hz`, default `sim_params.CONTROL_HZ`), but the rate
  the loop actually achieves sets how much of *your* motion lands in each frame, so it is part of
  a demo pool's identity. Every take stores the achieved rate as `hz_measured`. The paper's takes
  stamp `hz=60` but ran at ~26 Hz (median frame interval 37–39 ms), i.e. about 9x slow motion in
  sim time, which gives the operator time to react. To match their per-frame hand speed, record
  with `--hz 26` (this changes `control_hz` in the stamp, nothing else). The teleop prints the
  loop rate and per-frame work every 120 frames; if the machine cannot hold `--hz`, the loop
  runs slower and each frame carries more hand motion.
- Keep `--device` at its default (`cuda:0`, what the data was recorded on), and keep the default
  hand mount (`--hand-pos`, `--hand-rpy`), cube (`--cube-size`) and gains (`--stiffness`,
  `--damping`); they all change the data.

## 6. Inspect, replay, train

```bash
python isaac/inspect_npz.py data/teleop_takes/play_*_take1.npz   # stamp + every array
scripts/replay.sh --rollout data/teleop_takes                    # browser replay, no Isaac needed
scripts/replay.sh --sim --rollout data/teleop_takes/<take>.npz   # re-simulate with physics sliders
```

A take holds, per frame: `t` (s since the take started), `hand_joint_pos` (22, articulation
order, rad), `cmd_q22` (22 commanded joint targets = the action, `cmd_q22_names` order),
`cube_state` (13 = position xyz [m], quaternion **wxyz**, linear velocity, angular velocity, world
frame); plus the joint-name legends, `hand_root_pos`/`hand_root_quat`, `reset_times`, `hz`,
`hz_measured`, `source` and the simulator stamp.

A directory of takes is directly a training set (the loader reads every `*.npz` under the
directory, recursively):

```bash
python -m play_policy.train --data_dir data/teleop_takes ...       # see docs/PIPELINE.md
HUMAN_DATA=data/teleop_takes bash scripts/pipeline.sh               # the whole pipeline
```

To train on your takes plus the paper's, pass both directories:
`--data_dir data/teleop_takes data/dsp_sim_dataset/teleop`.

## 7. Without a glove

```bash
# the scene alone: fingers follow a slow open/close sinusoid
OMNI_KIT_ACCEPT_EULA=YES python isaac/teleop_isaac.py --source demo

# drive the hand with a dataset take's recorded joint commands (cmd_q22)
OMNI_KIT_ACCEPT_EULA=YES python isaac/teleop_isaac.py --source replay \
    --npz data/dsp_sim_dataset/teleop/<take>.npz
```

`--source replay` starts from the take's first frame (cube pose + hand joints) and re-simulates
its commands open loop, so the cube follows the recording for a fraction of a second and then
diverges (contact dynamics are chaotic). For a frame-exact view use `scripts/replay.sh`. It also
accepts raw glove logs that store `manus_node_pos`, which go through the retargeter instead.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `no right glove data on tcp://localhost:5555` | server not running or not linked to a glove: start `manus/server.sh`, check USB + udev rules |
| `the glove server ... publishes, but the skeleton is EMPTY` / server logs `No compatible license found` | license dongle missing: plug it in next to the glove, restart the server |
| CMake: `MANUS SDK not found in ...` | put the SDK in `manus/ManusSDK/` or set `MANUS_SDK_DIR` |
| server cannot open the glove | only one process may own it: stop MANUS Core or a second server |
| hand moves but fingers look wrong | re-run calibration (3), then re-check `retarget/retarget_viz.py` (4) |
| nothing at http://localhost:8080 | the page appears only after Isaac has booted (~30–60 s); another program may hold the port (`--viz-port`) |
| terminal stops echoing after a crash | `stty sane` |
| a killed teleop keeps running | Isaac ignores SIGTERM: `kill -9 <pid>` |
