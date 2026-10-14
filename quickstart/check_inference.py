#!/usr/bin/env python3
"""Offline check of the inference pipeline: play policy -> world model -> MPC scoring.

No simulator needed (torch + numpy + scipy + matplotlib). For each of the 4 recorded states in
``quickstart/sample_states.npz`` (7 consecutive frames of hand joints + cube pose, i.e. the
world model's 3-frame history at stride 3, plus a goal orientation >= 90 deg away) it runs ONE
MPC replan through ``isaac/mpc_core.py`` -- the same code the Isaac harness runs -- with the
paper settings: 1024 play-policy proposals of 200 steps (stochastic DDPM sampling, eta 1.0,
init noise 1.1), each rolled 50 steps through the world model, scored by the closest predicted
approach to the goal.

It prints the replan time, the current vs best-predicted distance to the goal and the spread of
the proposals, checks the numbers against ``quickstart/reference.json`` (generated with the
released checkpoints), and saves a figure to ``quickstart_out/``.

  python quickstart/check_inference.py                    # GPU, fp32 (~1 min incl. model load)
  python quickstart/check_inference.py --amp bf16         # what the harness runs
  python quickstart/check_inference.py --device cpu       # no GPU: same check, ~1-5 min
  python quickstart/check_inference.py --device cpu --states 1   # just the first state

Why the tolerances are what they are: the proposals are diffusion SAMPLES, so their exact values
depend on the random stream. A different GPU, driver, torch version or the CPU generator gives a
different stream -- effectively a different seed. ``reference.json`` therefore stores, per state,
the result of 8 seeds, and the check accepts a result within max(1 deg, 4 x seed std) of the
reference mean on three statistics (best predicted approach, 10th-percentile and median proposal
score; the percentiles are the tight ones, std < 1 deg).
A wrong checkpoint, normalizer or joint ordering moves them by tens of degrees.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "isaac"))
from mpc_core import Planner, geo_deg  # noqa: E402

# the paper's MPC settings (scripts/pipeline.sh stage 4); the archive is empty on a first replan
PAPER = dict(n_samples=1024, plan_batch=256, sample_eta=1.0, init_noise_scale=1.1,
             reuse_elites=True, reuse_frac=0.5, reuse_hist=3, reuse_keep=512, reuse_joint_tol=0.5)
REF_SEEDS = 8


def build_inputs(planner, s, dev):
    """sample_states.npz -> (hand_art (E,22), cube_st (E,13), hist (E,HBUF,frame_dim), last_cmd)."""
    hj = torch.as_tensor(s["hand_joint_pos"], device=dev)            # (E,7,22) oldest -> newest
    cs = torch.as_tensor(s["cube_state"], device=dev)                # (E,7,13)
    E, F = hj.shape[:2]
    assert F >= planner.HBUF, f"need {planner.HBUF} frames of history, sample has {F}"
    hist = torch.stack([planner.wm_frame(hj[:, i], cs[:, i]) for i in range(F - planner.HBUF, F)], 1)
    # cmd_q22 is stored in the recording's legend; the planner wants the policy's
    cmd_names = [str(x) for x in s["cmd_q22_names"]]
    perm = [cmd_names.index(n) for n in planner.meta_p["cmd_q22_names"]]
    last_cmd = torch.as_tensor(s["cmd_q22"][:, perm], device=dev)
    return hj[:, -1], cs[:, -1], hist, last_cmd


def run(planner, inputs, goal_q, seed, dev):
    hand, cube, hist, last_cmd = inputs
    planner.reset_archive()
    torch.manual_seed(seed)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    ch, chosen, best_step, execute, geo_now, det = planner.plan(
        np.arange(len(goal_q)), hand, cube, goal_q, hist, 0, last_cmd, details=True)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    return time.time() - t0, ch, chosen, best_step, execute, geo_now, det


def stats_of(chosen, geo_now, det):
    cost = det["cost"].float().cpu().numpy()                          # (E,N) predicted min geo
    return {"geo_now": geo_now.cpu().numpy().tolist(),
            "best_pred": chosen.cpu().numpy().tolist(),
            "p10": np.percentile(cost, 10, axis=1).tolist(),
            "p50": np.median(cost, axis=1).tolist(),
            "p90": np.percentile(cost, 90, axis=1).tolist()}


def figure(path, s, det, chosen, geo_now, best_step, n_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    geo = det["geo"].float().cpu().numpy()                            # (E*N, WH)
    cost = det["cost"].float().cpu().numpy()                          # (E, N)
    E, N = cost.shape
    WH = geo.shape[1]
    fig, axes = plt.subplots(2, E, figsize=(4.2 * E, 7.0), squeeze=False)
    rng = np.random.default_rng(0)
    for e in range(E):
        g = geo[e * N:(e + 1) * N]
        order = np.argsort(cost[e])
        ax = axes[0, e]
        for i in rng.choice(N, size=min(64, N), replace=False):
            ax.plot(g[i], color="0.75", lw=0.6)
        for i in order[:16]:
            ax.plot(g[i], color="tab:blue", lw=0.9, alpha=0.8)
        ax.plot(g[order[0]], color="tab:red", lw=2.0, label=f"chosen (min {cost[e, order[0]]:.1f} deg)")
        ax.axhline(float(geo_now[e]), color="k", ls="--", lw=1.0, label=f"now ({float(geo_now[e]):.1f} deg)")
        ax.plot([], [], color="0.75", label="random proposals"); ax.plot([], [], color="tab:blue", label="top 16")
        ax.set(title=str(s["source"][e]).split("/")[-1], xlabel="world-model step", xlim=(0, WH - 1),
               ylabel="predicted distance to goal (deg)" if e == 0 else None)
        ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
        ax = axes[1, e]
        ax.hist(cost[e], bins=40, color="tab:blue", alpha=0.8)
        ax.axvline(float(geo_now[e]), color="k", ls="--", lw=1.0)
        ax.axvline(float(chosen[e]), color="tab:red", lw=1.5)
        ax.set(xlabel="closest predicted approach (deg)", ylabel=f"proposals (of {N})" if e == 0 else None)
    fig.suptitle(f"One MPC replan per recorded state: {n_samples} play-policy proposals scored by the world model",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", default=os.path.join(_ROOT, "checkpoints"))
    ap.add_argument("--policy_ckpt", default=None, help="default: <ckpt_dir>/dsp_sim_play_policy.pt")
    ap.add_argument("--wm_ckpt", default=None, help="default: <ckpt_dir>/dsp_sim_world_model.pt")
    ap.add_argument("--samples", default=os.path.join(_HERE, "sample_states.npz"))
    ap.add_argument("--reference", default=os.path.join(_HERE, "reference.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", choices=["off", "bf16"], default="off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", type=int, default=0, help="use only the first N sample states (0 = all 4)")
    ap.add_argument("--out", default="quickstart_out")
    ap.add_argument("--write_reference", action="store_true",
                    help="(maintainers) regenerate reference.json from REF_SEEDS seeds")
    args = ap.parse_args()
    dev = args.device
    pol = args.policy_ckpt or os.path.join(args.ckpt_dir, "dsp_sim_play_policy.pt")
    wmp = args.wm_ckpt or os.path.join(args.ckpt_dir, "dsp_sim_world_model.pt")
    for p in (pol, wmp):
        assert os.path.exists(p), f"missing {p} -- run quickstart/download_checkpoints.sh first"
    os.makedirs(args.out, exist_ok=True)

    s = dict(np.load(args.samples, allow_pickle=False))
    n_all = len(s["goal_quat"])
    keep = slice(0, args.states if 0 < args.states < n_all else n_all)
    for key in ("hand_joint_pos", "cube_state", "cmd_q22", "goal_quat", "frame", "source"):
        s[key] = s[key][keep]
    names = [str(x) for x in s["hand_joint_names"]]
    goal_q = np.asarray(s["goal_quat"], np.float64)
    E = len(goal_q)
    t0 = time.time()
    planner = Planner(pol, wmp, names, dev, num_envs=E, amp=args.amp, **PAPER)
    print(f"[load] {time.time() - t0:.1f}s  device={dev} amp={args.amp}\n{planner.describe()}")
    inputs = build_inputs(planner, s, dev)

    if dev.startswith("cuda"):          # warm-up (cuDNN autotune, allocator) on a throwaway seed
        run(planner, inputs, goal_q, 12345, dev)
    dt, ch, chosen, best_step, execute, geo_now, det = run(planner, inputs, goal_q, args.seed, dev)
    st = stats_of(chosen, geo_now, det)
    n_rows = E * PAPER["n_samples"]
    print(f"[replan] {E} states x {PAPER['n_samples']} proposals = {n_rows} rollouts in {dt * 1e3:.0f} ms "
          f"({dt * 1e3 / E:.0f} ms per state)")
    print(f"{'state':<40} {'now':>7} {'best':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'step':>5} exec")
    for e in range(E):
        print(f"{str(s['source'][e]):<40} {st['geo_now'][e]:7.1f} {st['best_pred'][e]:7.1f} "
              f"{st['p10'][e]:7.1f} {st['p50'][e]:7.1f} {st['p90'][e]:7.1f} {int(best_step[e]):5d} "
              f"{bool(execute[e])}")
    print("  now  = current distance to the goal (deg)        best = chosen plan's closest predicted approach")
    print("  p10/p50/p90 = spread of the 1024 proposals' closest predicted approach   step = WM step of it")
    sane = bool(np.all(np.isfinite(st["best_pred"])) and
                np.all(np.asarray(st["best_pred"]) <= np.asarray(st["p10"]) + 1e-4) and
                tuple(ch.shape[1:]) == (planner.H, 22))
    fig = os.path.join(args.out, "check_inference.png")
    figure(fig, s, det, chosen.cpu().numpy(), geo_now.cpu().numpy(), best_step, PAPER["n_samples"])
    print(f"[figure] {fig}")

    if args.write_reference:
        assert len(goal_q) == n_all, "--write_reference needs all sample states"
        per ={k: [] for k in ("best_pred", "p50", "p10", "p90")}
        for sd in range(REF_SEEDS):
            _, _, c, _, _, g, d = run(planner, inputs, goal_q, sd, dev)
            r = stats_of(c, g, d)
            for k in per:
                per[k].append(r[k])
        ref = {"settings": dict(PAPER, amp=args.amp, device=str(torch.cuda.get_device_name(0))
                                if dev.startswith("cuda") else "cpu", torch=torch.__version__,
                                seeds=REF_SEEDS),
               "sources": [str(x) for x in s["source"]], "geo_now": st["geo_now"],
               "mean": {k: np.mean(v, 0).tolist() for k, v in per.items()},
               "std": {k: np.std(v, 0).tolist() for k, v in per.items()},
               "seed0": {k: per[k][0] for k in per}, "replan_ms": dt * 1e3}
        with open(args.reference, "w") as f:
            json.dump(ref, f, indent=2)
        print(f"[reference] wrote {args.reference}")
        return 0

    ok = sane
    if not os.path.exists(args.reference):
        print("[check] no reference.json -- skipped the comparison")
    else:
        ref = json.load(open(args.reference))
        print(f"[check] vs reference ({ref['settings']['device']}, torch {ref['settings']['torch']}, "
              f"{ref['settings']['seeds']} seeds):")
        dg = np.abs(np.asarray(st["geo_now"]) - np.asarray(ref["geo_now"])[keep])
        ok &= bool(dg.max() < 1e-2)
        print(f"  current distance (deterministic input)   max |diff| {dg.max():.4f} deg   "
              f"{'ok' if dg.max() < 1e-2 else 'MISMATCH'}")
        for k, label in (("best_pred", "best predicted approach"), ("p10", "10th-percentile proposal score"),
                         ("p50", "median proposal score")):
            got, mu, sd = st[k], np.asarray(ref["mean"][k])[keep], np.asarray(ref["std"][k])[keep]
            tol = np.maximum(1.0, 4.0 * sd)
            dev_ = np.abs(got - mu)
            good = bool(np.all(dev_ <= tol))
            ok &= good
            print(f"  {label:<40} |diff| {np.array2string(dev_, precision=2)} deg  "
                  f"tol {np.array2string(tol, precision=2)}   {'ok' if good else 'MISMATCH'}")
        if args.seed == 0 and args.amp == ref["settings"]["amp"] and args.states in (0, len(ref["geo_now"])):
            # same seed + settings: on the reference's GPU/torch stack this is exact (info only)
            d0 = np.abs(np.asarray(st["best_pred"]) - np.asarray(ref["seed0"]["best_pred"])).max()
            print(f"  (info) seed-0 best predicted approach vs the reference's seed 0: max |diff| {d0:.4f} deg"
                  f"{'  -- bit-identical stack' if d0 < 1e-3 else ''}")
    print("[check] PASS -- policy, world model and MPC scoring behave as released" if ok else
          "[check] FAIL -- see the mismatches above (wrong checkpoint / install?)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
