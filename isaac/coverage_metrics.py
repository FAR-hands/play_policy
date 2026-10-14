#!/usr/bin/env python
"""Object-pose coverage metrics for policy rollouts / play datasets.

Two complementary numbers, both on a 12x24 = 288-cell equirectangular grid of a
unit direction on S^2 (the cube's body +z axis expressed in the HAND frame):

  occupancy_coverage  -- fraction of grid cells the cube ever occupies. With
      --randomize-orientation resets this is inflated by the re-spawns themselves:
      teleporting to N random orientations fills cells for free.

  transition_coverage -- the SAME grid, but binned on the START-RELATIVE rotation
      dR = R_ref^{-1} @ R_t inside each continuous (reset-free) segment, where R_ref
      is the pose a short settle after the segment begins. Every segment is measured
      relative to its own (possibly random) start, so random initialization maps to
      the identity and contributes nothing: if the policy never reorients the cube,
      dR stays at the north pole (1 cell) regardless of how many random resets there
      were. This isolates the reorientation the POLICY actually produces.

Segments are split at resets, detected as single-frame teleports (a large jump in
cube position OR orientation) plus episode-index changes.

Caveats (shared with the occupancy metric): this is an S^2 proxy, not true SO(3) --
it tracks one body axis (2 DoF), ignores spin about that axis, uses non-equal-area
bins, and does not fold in the cube's 24-fold symmetry. Good for RELATIVE reads.

  python isaac/coverage_metrics.py '<dataset>/play/*.npz'
"""
import argparse
import glob
import numpy as np
from scipy.spatial.transform import Rotation as Rot

NB = 12  # latitude bands; longitude = 2*NB


def _rel_R(d):
    """Cube orientation in the hand frame, as a Rotation (per frame)."""
    cs = d["cube_state"].astype(np.float64)
    hq = np.asarray(d["hand_root_quat"], float)          # wxyz
    Rm = Rot.from_quat([hq[1], hq[2], hq[3], hq[0]])
    q = cs[:, 3:7]                                        # wxyz
    Rc = Rot.from_quat(np.c_[q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    return Rm.inv() * Rc, cs[:, :3]


def _bins(R, nb=NB):
    """Set of (lat,lon) grid cells hit by the body +z axis of each rotation."""
    z = R.apply(np.array([0, 0, 1.0]))
    lat = np.arcsin(np.clip(z[:, 2], -1, 1))
    lon = np.arctan2(z[:, 1], z[:, 0])
    ib = np.clip(((lat + np.pi / 2) / np.pi * nb).astype(int), 0, nb - 1)
    jb = np.clip(((lon + np.pi) / (2 * np.pi) * (2 * nb)).astype(int), 0, 2 * nb - 1)
    return set(zip(ib.tolist(), jb.tolist()))


def occupancy_coverage(R, nb=NB):
    return len(_bins(R, nb)) / (nb * 2 * nb) * 100.0


def segment_bounds(pos, R, ep, pos_jump=0.03, ang_jump=45.0):
    """Start indices of continuous (reset-free) segments. A reset shows up as a
    single-frame teleport (cube re-spawned high + reoriented) or an episode change."""
    n = len(pos)
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)                 # (n-1,)
    q = R.as_quat()
    dang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))
    jump = (dpos > pos_jump) | (dang > ang_jump)
    if ep is not None:
        jump |= (np.diff(ep) != 0)
    starts = [0] + [i + 1 for i in np.where(jump)[0]]
    starts = sorted(set(starts))
    segs = []
    for a, b in zip(starts, starts[1:] + [n]):
        if b - a >= 2:
            segs.append((a, b))
    return segs


def transition_coverage(R, pos, ep, settle=30, min_len=60, nb=NB):
    """Coverage of start-relative rotations across continuous segments -> the policy's
    own reorientation, with random init factored out. Also returns per-segment net
    reorientation (deg) and the segment count."""
    segs = segment_bounds(pos, R, ep)
    cells = set()
    nets = []
    used = 0
    for a, b in segs:
        ref_i = a + settle
        if b - ref_i < min_len:                 # too short after settle to be meaningful
            continue
        used += 1
        Rref_inv = R[ref_i].inv()
        dR = Rref_inv * R[ref_i:b]              # start-relative rotations
        cells |= _bins(dR, nb)
        nets.append(dR[-1].magnitude() * 180 / np.pi)
    cov = len(cells) / (nb * 2 * nb) * 100.0
    nets = np.asarray(nets) if nets else np.array([0.0])
    return {
        "transition_coverage": cov,
        "n_segments": len(segs),
        "n_segments_used": used,
        "net_reorient_mean": float(nets.mean()),
        "net_reorient_median": float(np.median(nets)),
        "net_reorient_max": float(nets.max()),
    }


def analyze(paths, pooled=True, nb=NB):
    files = []
    for p in paths:
        files += sorted(glob.glob(p))
    if not files:
        raise SystemExit(f"no npz matched: {paths}")
    allR, allpos, allep, ep_off = [], [], [], 0
    print(f"grid {nb}x{2*nb} = {nb*2*nb} cells")
    print(f"{'source':26s}{'frames':>8s}{'occ%':>7s}{'trans%':>8s}{'segs':>6s}{'net°(med/max)':>16s}")
    print("-" * 71)
    for f in files:
        d = np.load(f, allow_pickle=True)
        R, pos = _rel_R(d)
        ep = d["episode"] if "episode" in d.files else np.zeros(len(pos), int)
        occ = occupancy_coverage(R, nb)
        tr = transition_coverage(R, pos, ep, nb=nb)
        tag = "/".join(f.split("/")[-3:]) if "/" in f else f
        print(f"{tag[:26]:26s}{len(pos):>8d}{occ:>7.1f}{tr['transition_coverage']:>8.1f}"
              f"{tr['n_segments']:>6d}{tr['net_reorient_median']:>8.0f}/{tr['net_reorient_max']:<7.0f}")
        allR.append(R.as_quat()); allpos.append(pos)
        allep.append(np.asarray(ep) + ep_off); ep_off += int(np.max(ep)) + 1
    if pooled and len(files) > 1:
        R = Rot.from_quat(np.concatenate(allR))
        pos = np.concatenate(allpos); ep = np.concatenate(allep)
        occ = occupancy_coverage(R, nb); tr = transition_coverage(R, pos, ep, nb=nb)
        print("-" * 71)
        print(f"{'POOLED':26s}{len(pos):>8d}{occ:>7.1f}{tr['transition_coverage']:>8.1f}"
              f"{tr['n_segments']:>6d}{tr['net_reorient_median']:>8.0f}/{tr['net_reorient_max']:<7.0f}")
        print(f"\nocc={occ:.1f}% (reset-inflated)  vs  transition={tr['transition_coverage']:.1f}% "
              f"(policy-only)   over {tr['n_segments_used']} usable segments, "
              f"mean net reorient {tr['net_reorient_mean']:.0f}°/segment")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="rollout .npz path(s) or glob(s)")
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--nb", type=int, default=NB, help="latitude bands (grid = nb x 2nb cells)")
    args = ap.parse_args()
    analyze(args.paths, pooled=not args.no_pooled, nb=args.nb)


if __name__ == "__main__":
    main()
