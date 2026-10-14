"""Aggregate + visualize the world-model MPC experiment (isaac/wm_mpc_eval_batch.py).

Reads all shard_*.jsonl + traj_*.npz under a run dir, computes success/reduction
statistics, and writes a multi-panel figure + a stats.json.

  python isaac/wm_mpc_report.py runs/mpc

Headline metrics (e_i(t) = true geodesic from the cube to trial i's goal at control step t):
  SR@theta    = mean_i[ min_t e_i(t) <= theta ]  for theta = 0.4 / 0.6 / 0.8 rad (the paper's metric)
  reach@X     = mean_i[ min_t e_i(t) <  X deg ]  for X = 15 / 30
  min-geo     = mean_i min_t e_i(t)
Stuck trials: the cube can come to rest on the hand above the drop line where no finger reaches
it, so it is never re-spawned. A trial whose geodesic stays frozen (|de/dt| < 0.05 deg/step) for
>= 300 consecutive steps while > 30 deg from the goal is listed under ``stuck_trials``; redo it
with its init/goal pinned (--init_quats_npz / --goal_quats_npz) and a new --seed.
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(run_dir, "shard_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    recs.sort(key=lambda r: r["trial"])
    return recs


def stuck_trials(run_dir, recs, still_deg=0.05, min_steps=300, far_deg=30.0):
    """Trial ids whose geodesic is frozen for >= min_steps consecutive steps while > far_deg."""
    out = []
    for r in recs:
        p = os.path.join(run_dir, f"traj_{r['trial']:04d}.npz")
        if not os.path.exists(p):
            continue
        g = np.load(p)["geo"].astype(np.float64)
        frozen = (np.abs(np.diff(g)) < still_deg) & (g[1:] > far_deg)
        run = best = 0
        for f in frozen:
            run = run + 1 if f else 0
            best = max(best, run)
        if best >= min_steps:
            out.append(int(r["trial"]))
    return out


def best_so_far_curves(run_dir, recs):
    """(n_trials, n_steps) running-minimum geodesic-to-goal, one row per rollout.

    ``traj_XXXX.npz['geo']`` is the true per-step distance; the running min is what the
    controller has ACHIEVED by step t (the metric the success rate thresholds). Rollouts are
    truncated to the shortest length so the mean is over a constant trial count at every step."""
    curves = []
    for r in recs:
        p = os.path.join(run_dir, f"traj_{r['trial']:04d}.npz")
        if not os.path.exists(p):
            continue
        curves.append(np.minimum.accumulate(np.load(p)["geo"].astype(np.float64)))
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--success_deg", type=float, default=15.0)
    ap.add_argument("--label", default=None,
                    help="override the figure's title prefix. Needed for runs that have no "
                         "WM plan scores but are NOT the policy-only baseline (e.g. the oracle, "
                         "which is scored by the simulator)")
    args = ap.parse_args()
    recs = load(args.run_dir)
    n = len(recs)
    assert n, f"no results in {args.run_dir}"

    goal0 = np.array([r["goal_init_deg"] for r in recs])
    ming = np.array([r["min_geo_deg"] for r in recs])
    fing = np.array([r["final_geo_deg"] for r in recs])
    minat = np.array([r["min_at_step"] for r in recs])
    drops = np.array([r["drops"] for r in recs])
    pscore = np.array([r.get("mean_plan_score") or np.nan for r in recs])

    def sr(th):
        return float((ming < th).mean() * 100)
    stats = {
        "n_trials": n,
        "goal_init_deg_mean": float(goal0.mean()), "goal_init_deg_min": float(goal0.min()),
        "min_geo_mean": float(ming.mean()), "min_geo_median": float(np.median(ming)),
        "min_geo_p25": float(np.percentile(ming, 25)), "min_geo_p75": float(np.percentile(ming, 75)),
        "final_geo_mean": float(fing.mean()), "final_geo_median": float(np.median(fing)),
        "reduction_mean_deg": float((goal0 - ming).mean()),
        "success_rate_15": sr(15), "success_rate_10": sr(10),
        "success_rate_20": sr(20), "success_rate_30": sr(30), "success_rate_45": sr(45),
        "drops_mean": float(drops.mean()),
        "min_at_step_median": float(np.median(minat)),
    }
    for th in (0.4, 0.6, 0.8):   # the paper's metric: SR@theta = mean_i[min_t e_i(t) <= theta]
        stats[f"sr_{th:.1f}rad"] = float((ming <= np.degrees(th)).mean() * 100)
    stats["stuck_trials"] = stuck_trials(args.run_dir, recs)
    with open(os.path.join(args.run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9))
    # (1) min-geo histogram
    ax = fig.add_subplot(2, 3, 1)
    ax.hist(ming, bins=24, range=(0, 150), color="tab:blue", alpha=0.8)
    for th, c in [(15, "tab:green"), (30, "tab:orange")]:
        ax.axvline(th, ls="--", color=c, label=f"{th}deg ({sr(th):.0f}%)")
    ax.set(title="best approach to goal (min geodesic)", xlabel="min geo to goal (deg)", ylabel="trials")
    ax.set_xlim(0, 150)
    ax.legend(fontsize=8)
    # (2) reduction: init distance vs achieved min
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(goal0, ming, s=18, alpha=0.6, color="tab:blue")
    lim = [0, max(goal0.max(), ming.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="no progress")
    ax.set(title="goal difficulty vs best approach", xlabel="init goal distance (deg)",
           ylabel="min geo achieved (deg)", xlim=lim, ylim=lim)
    ax.legend(fontsize=8)
    # (3) min vs final (does it hold?)
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(ming, fing, s=18, alpha=0.6, color="tab:purple")
    lim2 = [0, max(ming.max(), fing.max()) * 1.05]
    ax.plot(lim2, lim2, "k--", lw=0.8, label="holds at best")
    ax.set(title="best approach vs final (drift)", xlabel="min geo (deg)", ylabel="final geo (deg)",
           xlim=lim2, ylim=lim2)
    ax.legend(fontsize=8)
    # (4) AVERAGE BEST-SO-FAR error over time: for every rollout the running minimum of the
    #     geodesic-to-goal, averaged across all trials -> "how close has the controller gotten
    #     by sim step t, on average". Monotone by construction; the curve's level and how fast
    #     it flattens are the two things that separate controllers.
    ax = fig.add_subplot(2, 3, 4)
    bsf = best_so_far_curves(args.run_dir, recs)
    if bsf is not None:
        steps = np.arange(bsf.shape[1])
        mean_c, med_c = bsf.mean(0), np.median(bsf, 0)
        ax.fill_between(steps, np.percentile(bsf, 25, axis=0), np.percentile(bsf, 75, axis=0),
                        color="tab:blue", alpha=0.18, lw=0, label="IQR")
        ax.plot(steps, mean_c, color="tab:blue", lw=1.8, label=f"mean (final {mean_c[-1]:.1f}deg)")
        ax.plot(steps, med_c, color="tab:cyan", lw=1.2, ls="--", label=f"median (final {med_c[-1]:.1f}deg)")
        ax.axhline(args.success_deg, ls=":", color="tab:green", lw=1.0,
                   label=f"{args.success_deg:.0f}deg success")
        ax.set(title=f"average best-so-far distance to goal ({len(bsf)} rollouts)",
               xlabel="sim step", ylabel="mean best-so-far geo to goal (deg)", ylim=(0, None))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        stats["bsf_mean_final"] = float(mean_c[-1])
        stats["bsf_mean_curve_ds20"] = mean_c[::20].round(3).tolist()   # downsampled, for overlays
        with open(os.path.join(args.run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    # (5) planning fidelity: WM-predicted min (plan score) vs achieved min. With no world model
    #     (the play-policy-only baseline) there is no plan score, so show example traces instead.
    ax = fig.add_subplot(2, 3, 5)
    m = ~np.isnan(pscore)
    if m.sum():
        ax.scatter(pscore[m], ming[m], s=18, alpha=0.6, color="tab:red")
        ax.set(title="WM planned score vs achieved min", xlabel="mean WM-predicted min (deg)",
               ylabel="actual min geo (deg)")
    else:
        order = np.argsort(ming)
        for name, i in (("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])):
            p = os.path.join(args.run_dir, f"traj_{recs[i]['trial']:04d}.npz")
            if os.path.exists(p):
                ax.plot(np.load(p)["geo"], lw=1.1,
                        label=f"{name} (min={ming[i]:.0f}, goal={goal0[i]:.0f})")
        ax.axhline(args.success_deg, ls="--", color="tab:green", lw=0.8)
        ax.set(title="distance to goal over time (examples)", xlabel="sim step",
               ylabel="geo to goal (deg)")
        ax.legend(fontsize=8)
    # (6) success-rate vs threshold
    ax = fig.add_subplot(2, 3, 6)
    ths = np.arange(5, 91, 5)
    ax.plot(ths, [ (ming < t).mean() * 100 for t in ths], "-o", color="tab:green", ms=3)
    ax.set(title="success rate vs threshold", xlabel="success threshold (deg)", ylabel="% trials", ylim=(0, 100))
    ax.grid(alpha=0.3)

    kind = args.label or ("World-model MPC" if (~np.isnan(pscore)).sum()
                          else "Play policy only (no world model)")
    fig.suptitle(f"{kind}: {n} trials, goal >= {goal0.min():.0f}deg  |  "
                 f"min-geo mean {ming.mean():.1f}deg  success@15={sr(15):.0f}% @30={sr(30):.0f}%", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.run_dir, "mpc_report.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)
    print(json.dumps({k: v for k, v in stats.items() if k != "bsf_mean_curve_ds20"}, indent=2))
    print(f"[summary] n={n}  SR@0.4/0.6/0.8 rad = {stats['sr_0.4rad']:.1f} / {stats['sr_0.6rad']:.1f} / "
          f"{stats['sr_0.8rad']:.1f} %  reach@15/30 = {sr(15):.1f} / {sr(30):.1f} %  "
          f"min-geo {ming.mean():.2f} deg  drops {drops.mean():.2f}/trial  "
          f"stuck {len(stats['stuck_trials'])}")


if __name__ == "__main__":
    main()
