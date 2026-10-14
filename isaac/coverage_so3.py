#!/usr/bin/env python
"""True-SO(3) coverage metrics for the teleop play dataset (finer than the S^2 proxy).

The old ``coverage_metrics.py`` bins only the cube's body +z axis on a 12x24 = 288 cell
S^2 grid (2 DoF, ignores spin). This script instead discretizes the FULL rotation on a
10-degree Euler grid: each of the 3 xyz Euler angles is mapped to [0,360), floor divided
by 10 -> integer in [0,36), and combined into a single bin id
    bin = i*36*36 + j*36 + k
giving a 36^3 = 46656 bin space. Coverage = distinct bins hit / 46656 * 100.

NOTE ON THE DENOMINATOR: Euler ('xyz') binning OVER-parameterizes SO(3). The middle
(pitch) angle physically spans only 180 degrees, not 360, and Euler cells are non-uniform
in volume with gimbal-lock degeneracies near pitch = +-90 deg. So 46656 is a *bin-space*
count, NOT 46656 equal-volume SO(3) cells. Treat the SO(3)% here as a relative,
resolution-matched read, not an absolute fraction of rotation space. (46656 is used
because it matches the requested 10-degree, 36^3 discretization exactly.)

Also renders per-bin visit-count visualizations (pairwise marginals + occupied-bin
histogram) for the POOLED occupancy and transition distributions.

  python isaac/coverage_so3.py '<dataset>/play/*.npz' --outdir coverage_out
  python isaac/coverage_so3.py '<dataset>/teleop/*.npz' '<dataset>/play/*.npz'   # pooled
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage_metrics as cm  # reuse _rel_R

NBINS_AXIS = 36          # 360 deg / 10 deg
NBINS = NBINS_AXIS ** 3  # 46656


# ---------------------------------------------------------------------------
# 46656-bin SO(3) (Euler 10-degree) binning
# ---------------------------------------------------------------------------

def _so3_idx(R):
    """Per-frame (i,j,k) Euler-grid indices in [0,36) for rotations R -> (n,3) int."""
    e = R.as_euler("xyz")                                   # (n,3) radians in [-pi,pi]
    deg = np.degrees(e) % 360.0                             # -> [0,360)
    return (deg // 10.0).astype(np.int64) % NBINS_AXIS      # -> [0,36) per axis, wrap


def _so3_ids(R):
    """Per-frame flat bin ids in [0,46656)."""
    idx = _so3_idx(R)
    return idx[:, 0] * NBINS_AXIS * NBINS_AXIS + idx[:, 1] * NBINS_AXIS + idx[:, 2]


def _hist3d(R):
    """Per-bin visit-count histogram over the 36x36x36 Euler cube (roll,pitch,yaw)."""
    idx = _so3_idx(R)
    h = np.zeros((NBINS_AXIS, NBINS_AXIS, NBINS_AXIS), np.int64)
    np.add.at(h, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    return h


def so3_occupancy(R):
    return len(set(_so3_ids(R).tolist())) / NBINS * 100.0


# ---------------------------------------------------------------------------
# t-gap-aware segmentation (union of t-gap / t<0 / pos jump / ang jump)
# mirrors play_policy.data.load_worldmodel_segments
# ---------------------------------------------------------------------------

def segment_bounds_teleop(pos, R, t, pos_jump=0.03, ang_jump=45.0, gap_factor=5.0, min_gap_s=0.5):
    """Start/end spans of continuous segments, splitting on the UNION of reset cues:
    teleport in cube pos (>pos_jump m) OR orientation (>ang_jump deg), t reset (dt<0),
    and trimmed-reset t gaps (dt > gap_factor * median positive dt AND > min_gap_s seconds).

    The absolute min_gap_s floor matters: online collections vary the per-frame wall-clock
    dt, and heavy replans (e.g. q75's 64-candidate sampling adds ~40-250ms pauses at every
    replan boundary) would otherwise clear the relative 5x-median threshold and be mistaken
    for resets -- over-segmenting q75 into ~76-frame stubs (8375 false breaks vs ~30 real).
    Real trimmed resets are multi-100ms+, so a 0.5s floor keeps them while dropping pauses."""
    n = len(pos)
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)             # (n-1,)
    q = R.as_quat()
    dang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))
    brk = (dpos > pos_jump) | (dang > ang_jump)
    if t is not None and len(t) == n:
        dt = np.diff(np.asarray(t, np.float64))
        brk |= (dt < 0)                                            # t reset
        pos_dt = dt[dt > 0]
        if pos_dt.size:
            brk |= (dt > gap_factor * float(np.median(pos_dt))) & (dt > min_gap_s)  # real gap
    starts = sorted(set([0] + [i + 1 for i in np.where(brk)[0]]))
    segs = []
    for a, b in zip(starts, starts[1:] + [n]):
        if b - a >= 2:
            segs.append((a, b))
    return segs


def so3_transition(R, pos, t, settle=30, min_len=60):
    """46656-bin coverage of start-relative rotations dR = R_ref^{-1} R_t per segment.
    Returns coverage, a pooled per-bin visit-count histogram, seg counts and net reorient."""
    segs = segment_bounds_teleop(pos, R, t)
    hist = np.zeros((NBINS_AXIS, NBINS_AXIS, NBINS_AXIS), np.int64)
    nets, used = [], 0
    for a, b in segs:
        ref_i = a + settle
        if b - ref_i < min_len:
            continue
        used += 1
        dR = R[ref_i].inv() * R[ref_i:b]
        hist += _hist3d(dR)
        nets.append(dR[-1].magnitude() * 180 / np.pi)
    cov = int((hist > 0).sum()) / NBINS * 100.0
    nets = np.asarray(nets) if nets else np.array([0.0])
    return {
        "transition_coverage": cov,
        "hist": hist,
        "n_segments": len(segs),
        "n_segments_used": used,
        "net_reorient_mean": float(nets.mean()),
        "net_reorient_median": float(np.median(nets)),
        "net_reorient_max": float(nets.max()),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _render(hist, metric, cov, out_path):
    """One PNG: 3 pairwise-marginal heatmaps (log color) + occupied-bin count histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    occ_bins = int((hist > 0).sum())
    total_visits = int(hist.sum())
    axis_names = ["roll x", "pitch y", "yaw z"]
    # (axis_x, axis_y, axis-summed-out) for roll x pitch, roll x yaw, pitch x yaw
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    extent = [0, 360, 0, 360]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"SO(3) {NBINS}-bin ({NBINS_AXIS}^3, 10deg Euler) {metric} distribution "
                 f"(POOLED)\ncoverage = {cov:.2f}%   occupied bins = {occ_bins} / {NBINS}   "
                 f"total visits = {total_visits}", fontsize=12)

    for ax, (ia, ib, isum) in zip(axes.flat[:3], pairs):
        m = hist.sum(axis=isum)                 # 2D marginal, indexed [axis_a, axis_b]
        # place the lower-index remaining axis on x, the higher on y for imshow (rows=y)
        img = m.T                               # rows -> axis_b (y), cols -> axis_a (x)
        im = ax.imshow(img, origin="lower", extent=extent, aspect="auto",
                       cmap="viridis", norm=LogNorm(vmin=1, vmax=max(img.max(), 2)))
        ax.set_xlabel(f"{axis_names[ia]} (deg)")
        ax.set_ylabel(f"{axis_names[ib]} (deg)")
        ax.set_title(f"{axis_names[ia]} x {axis_names[ib]}  (sum over {axis_names[isum]})")
        ax.set_xticks(range(0, 361, 90)); ax.set_yticks(range(0, 361, 90))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="visit count (log)")

    # 4th panel: histogram of per-occupied-bin visit counts (log y)
    ax = axes.flat[3]
    counts = hist[hist > 0].ravel()
    if counts.size:
        cmax = int(counts.max())
        bins = np.logspace(0, np.log10(max(cmax, 2)), 40)
        ax.hist(counts, bins=bins, color="#3b6ea5", edgecolor="k", linewidth=0.3)
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("visits per occupied bin")
    ax.set_ylabel("# occupied bins (log)")
    med = float(np.median(counts)) if counts.size else 0.0
    mx = int(counts.max()) if counts.size else 0
    ax.set_title(f"concentration of coverage (median={med:.0f}, max={mx} visits/bin)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _render_variance(hist, metric, cov, out_path):
    """One PNG visualizing the NON-UNIFORMITY of visitation (what the `cv` metric summarizes):
    3 pairwise marginals recolored as log2(visits / mean) on a diverging map (red = over-,
    blue = under-visited, gray = never visited) so uneven regions are visible, plus a Lorenz
    curve of visit inequality annotated with cv and the Gini coefficient. Lower cv / flatter
    Lorenz = more even coverage (e.g. q75 cv 1.45 vs plain-baseline 2.27)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = hist > 0
    counts = hist[occ].ravel().astype(float)
    occ_bins = int(occ.sum())
    mean_v = float(counts.mean()) if counts.size else 0.0
    cv = float(counts.std() / mean_v) if mean_v else 0.0
    c = np.sort(counts)
    n = c.size
    gini = (2 * np.sum(np.arange(1, n + 1) * c) / (n * c.sum()) - (n + 1) / n) if n and c.sum() else 0.0

    axis_names = ["roll x", "pitch y", "yaw z"]
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    extent = [0, 360, 0, 360]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e6e6e6")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"SO(3) visitation UNIFORMITY - {metric} (POOLED)\n"
                 f"cv = {cv:.2f}  (std/mean over occupied bins, LOWER = more even)    "
                 f"Gini = {gini:.2f}    occupied = {occ_bins} / {NBINS}", fontsize=12)

    for ax, (ia, ib, isum) in zip(axes.flat[:3], pairs):
        m = hist.sum(axis=isum).astype(float)
        mo = m > 0
        ratio = np.full(m.shape, np.nan)
        if mo.any():
            ratio[mo] = np.log2(m[mo] / m[mo].mean())
        vmax = max(float(np.nanmax(np.abs(ratio))) if mo.any() else 1.0, 0.5)
        im = ax.imshow(np.ma.masked_invalid(ratio.T), origin="lower", extent=extent,
                       aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xlabel(f"{axis_names[ia]} (deg)")
        ax.set_ylabel(f"{axis_names[ib]} (deg)")
        ax.set_title(f"{axis_names[ia]} x {axis_names[ib]}  (sum over {axis_names[isum]})")
        ax.set_xticks(range(0, 361, 90)); ax.set_yticks(range(0, 361, 90))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log2(visits / mean); gray = unvisited")

    ax = axes.flat[3]
    if n:
        lx = np.linspace(0, 1, n + 1)
        ly = np.concatenate([[0.0], np.cumsum(c) / c.sum()])
        ax.plot(lx, ly, color="#b2182b", lw=2, label="actual visitation")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect uniformity")
        ax.fill_between(lx, ly, lx, alpha=0.15, color="#b2182b")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("cumulative fraction of occupied bins (least -> most visited)")
    ax.set_ylabel("cumulative fraction of total visits")
    ax.set_title(f"Lorenz curve of visits  (cv={cv:.2f}, Gini={gini:.2f}, "
                 f"mean={mean_v:.0f}, max={int(counts.max()) if n else 0})")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------

def analyze(paths, outdir="coverage_out"):
    files = []
    for p in paths:
        files += sorted(glob.glob(p, recursive=True))
    if not files:
        raise SystemExit(f"no npz matched: {paths}")

    hdr = (f"{'source':34s}{'frames':>8s}{'SO3occ%':>9s}{'SO3trn%':>9s}"
           f"{'segs':>6s}{'used':>6s}{'net med/max(deg)':>18s}")
    print(f"SO(3) bins = {NBINS_AXIS}^3 = {NBINS} (10-deg Euler)")
    print(hdr)
    print("-" * len(hdr))

    allR, allpos, allt, t_off = [], [], [], 0.0
    for f in files:
        d = np.load(f, allow_pickle=True)
        R, pos = cm._rel_R(d)
        t = np.asarray(d["t"], np.float64) if "t" in d.files else None
        so3_occ = so3_occupancy(R)
        so3_tr = so3_transition(R, pos, t)
        tag = f.split("/")[-1]
        print(f"{tag[:34]:34s}{len(pos):>8d}{so3_occ:>9.2f}{so3_tr['transition_coverage']:>9.2f}"
              f"{so3_tr['n_segments']:>6d}{so3_tr['n_segments_used']:>6d}"
              f"{so3_tr['net_reorient_median']:>9.0f}/{so3_tr['net_reorient_max']:<8.0f}")
        allR.append(R.as_quat()); allpos.append(pos)
        if t is not None:
            allt.append(t + t_off); t_off += float(t.max() - t.min()) + 1e6
        else:
            allt.append(np.arange(len(pos), dtype=np.float64) + t_off); t_off += len(pos) + 1e6

    print("-" * len(hdr))
    R = Rot.from_quat(np.concatenate(allR))
    pos = np.concatenate(allpos)
    t = np.concatenate(allt)
    occ_hist = _hist3d(R)
    occ_cov = int((occ_hist > 0).sum()) / NBINS * 100.0
    tr = so3_transition(R, pos, t)
    print(f"{'POOLED':34s}{len(pos):>8d}{occ_cov:>9.2f}{tr['transition_coverage']:>9.2f}"
          f"{tr['n_segments']:>6d}{tr['n_segments_used']:>6d}"
          f"{tr['net_reorient_median']:>9.0f}/{tr['net_reorient_max']:<8.0f}")
    print()
    print(f"POOLED net reorient (deg): mean={tr['net_reorient_mean']:.1f} "
          f"median={tr['net_reorient_median']:.1f} max={tr['net_reorient_max']:.1f}")
    print(f"POOLED segments: {tr['n_segments']} total, {tr['n_segments_used']} used "
          f"(settle=30, min_len=60)")

    os.makedirs(outdir, exist_ok=True)
    occ_png = os.path.join(outdir, "coverage_so3_occ.png")
    trn_png = os.path.join(outdir, "coverage_so3_trn.png")
    var_png = os.path.join(outdir, "coverage_so3_var.png")
    _render(occ_hist, "OCCUPANCY (abs cube-in-hand)", occ_cov, occ_png)
    _render(tr["hist"], "TRANSITION (start-relative dR)", tr["transition_coverage"], trn_png)
    _render_variance(occ_hist, "OCCUPANCY (abs cube-in-hand)", occ_cov, var_png)
    print(f"\nwrote {occ_png}")
    print(f"wrote {trn_png}")
    print(f"wrote {var_png}")
    print("\nNOTE: 46656 is a 10-deg Euler BIN-SPACE count, not equal-volume SO(3) cells "
          "(mid angle spans 180deg, cells non-uniform, gimbal degeneracy).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="npz path(s)/glob(s) (quote globs; ** recurses)")
    ap.add_argument("--outdir", default="coverage_out", help="dir to write the PNGs")
    args = ap.parse_args()
    analyze(args.paths, outdir=args.outdir)


if __name__ == "__main__":
    main()
