# isaac/ — simulation: scene, teleoperation, self-play, MPC

Everything here that opens a scene needs Isaac Sim 5.1 + Isaac Lab 2.3.2 and
`OMNI_KIT_ACCEPT_EULA=YES`; the Isaac-free modules (`mpc_core.py`, `deploy_core.py`,
`sim_params.py` at import time, reports, coverage) run in any env with torch.

```
# scene + physics
sim_params.py          every physics knob (single source of truth), the sim_params_json/_hash stamp,
                       and the optional viser slider panel
convert_sharpa_usd.py  URDF (assets/urdf/sharpa) -> assets/sharpa_right/sharpa_right.usd (shipped)
make_rubik_texture.py  sticker textures for the cube (shipped in assets/)

# human play
teleop_isaac.py        Manus glove -> retargeting -> Sharpa hand in Isaac Lab; records takes
                       (sources: live / demo / replay)           launcher: scripts/teleop.sh

# self-play collection + MPC
wm_mpc_eval_batch.py   batched harness: N envs in lockstep in one process
                         --mode policy --record   autonomous play collection (roll_XXXX.npz)
                         --mode policy            goal-blind play-policy baseline
                         (default) --mode mpc     play policy + world model + MPC
mpc_core.py            Isaac-free planner (proposals, world-model rollout, scoring, elite archive);
                       also used by quickstart/check_inference.py
deploy_core.py         Isaac-free policy loading + observation / chunk helpers
wm_mpc_report.py       stats.json (SR@0.4/0.6/0.8 rad, reach@15/30, stuck trials) + mpc_report.png

# inspection
inspect_npz.py         print the physics stamp and every array of a recording
coverage_so3.py        SO(3) coverage of a dataset (46,656-bin Euler grid)
coverage_metrics.py    S^2 occupancy / transition coverage (used by coverage_so3.py)
viser_replay_server.py browser replay of takes / rollouts, no simulator   launcher: scripts/replay.sh
viser_sim_replay.py    re-simulate a recording with physics sliders        scripts/replay.sh --sim
```

Usage of each stage: [docs/PIPELINE.md](../docs/PIPELINE.md); teleoperation:
[docs/TELEOP.md](../docs/TELEOP.md); checkpoints and the benchmark: [quickstart/](../quickstart/README.md).

Simulated time: each control frame advances the physics by 1/240 s (3 substeps of 1/720 s);
frames are labelled 60 Hz. Isaac Lab ignores SIGTERM (stop a run with `kill -9 <pid>`), and the
scripts here exit with `os._exit` because Kit can hang inside `simulation_app.close()`.
