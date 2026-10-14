"""Isaac-free core of play-policy + world-model MPC (one replan, batched over envs).

Shared by the simulator harness (``wm_mpc_eval_batch.py``) and the offline quick-start check
(``quickstart/check_inference.py``), so both run exactly the same planning code:

  for the envs due to replan:
    1. sample ``n_samples`` action chunks PER ENV from the play policy        (one (k*N) batch)
       -- with the elite archive on, only (1-reuse_frac)*N are fresh and the rest are the
          best chunks of the last ``reuse_hist`` replans, time-aligned to the current step
    2. roll every chunk through the world model from the env's state history (same batch)
    3. score = min over the horizon of the predicted geodesic to that env's goal
    4. return each env's best chunk (and whether it beats the current pose -- the no-op gate)

Nothing leaves the GPU between (1) and (3): chunks are denormalized, re-ordered into the world
model's joint legend and re-normalized in torch.

Ordering contract: ``hand_art`` rows are joint positions in ``joint_names`` order (the
articulation order in Isaac, the ``hand_joint_names`` order in a recording); every remap to the
policy / world-model legends is by joint name. Cube state is ``[pos(3), quat_wxyz(4), ...]`` in
the hand-scene frame (env origin removed). Returned chunks are raw joint targets (rad) in the
policy's ``cmd_q22_names`` order.
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root -> play_policy
sys.path.insert(0, _HERE)                    # isaac/ -> deploy_core
from deploy_core import load_policy  # noqa: E402
from play_policy.data import Normalizer  # noqa: E402
from play_policy.model import load_world_model_ckpt  # noqa: E402

AMP_DTYPES = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def geo_deg(qa, qb):
    """Geodesic angle (deg) between wxyz quats (numpy); broadcasts. |dot| -> sign-agnostic."""
    d = np.clip(np.abs((np.asarray(qa) * np.asarray(qb)).sum(-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


def geo_deg_t(qa, qb):
    """Same, in torch (keeps scoring on the GPU)."""
    d = (qa * qb).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(d))


def rand_quat_wxyz(rng):
    """Uniform random rotation (Shoemake), wxyz."""
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    return np.array([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                     b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)], np.float64)


def name_perm(names, sub):
    """Indices into ``names`` of every name in ``sub`` (by name, so robust to ordering)."""
    names = list(names)
    return np.array([names.index(n) for n in sub], dtype=np.int64)


def load_world_model(path, device):
    """World-model checkpoint -> (model, state normalizer, action normalizer, meta, ckpt dict)."""
    sd = torch.load(path, map_location=device, weights_only=False)
    wm, _ = load_world_model_ckpt(sd, device=device)
    return wm, Normalizer.from_state_dict(sd["norm_state"]), \
        Normalizer.from_state_dict(sd["norm_action"]), sd["meta"], sd


def compile_denoiser(module, label):
    """torch.compile whichever denoiser the checkpoint built: .unet (U-Net), .dit (DiT1D) or
    .net (transformer). Hardcoding one of them crashes the others after Isaac has booted."""
    for attr in ("unet", "dit", "net"):
        if getattr(module, attr, None) is not None:
            setattr(module, attr, torch.compile(getattr(module, attr), dynamic=False))
            print(f"[compile] {label}: compiled .{attr}")
            return
    print(f"[compile] {label}: no known denoiser attribute -- left uncompiled")


class Planner:
    """Play policy proposes, world model scores. ``wm_ckpt=None`` = play policy only (one chunk
    per env per replan, executed blind: the goal-agnostic baseline)."""

    def __init__(self, policy_ckpt, wm_ckpt, joint_names, device="cuda", *, num_envs=1,
                 n_samples=1024, plan_batch=256, sample_eta=1.0, init_noise_scale=1.1,
                 amp="off", compile=False, compile_wm=False, time_weight=0.0, noop_gate=True, noop_margin=0.0,
                 reuse_elites=False, reuse_frac=0.5, reuse_hist=3, reuse_keep=0,
                 reuse_joint_tol=0.5):
        dev = self.device = device
        self.amp_dtype = AMP_DTYPES[amp] if str(device).startswith("cuda") else None
        self.n_samples, self.plan_batch = int(n_samples), int(plan_batch)
        self.sample_eta, self.init_noise_scale = float(sample_eta), float(init_noise_scale)
        self.time_weight = float(time_weight)
        self.noop_gate, self.noop_margin = bool(noop_gate), float(noop_margin)
        self.reuse_elites, self.reuse_joint_tol = bool(reuse_elites), float(reuse_joint_tol)

        # ---- models ----
        self.policy, ns_p, na_p, self.meta_p, self.cfg_p = load_policy(policy_ckpt, dev)
        self.H = H = int(self.cfg_p["horizon"])
        self.pol_cube_dims = int(self.cfg_p["cube_dims"])
        self.wm = self.sd_w = None
        self.n_hist, self.hist_stride, self.frame_dim, self.cube_dim = 1, 1, 29, 7
        self.WH = H
        if wm_ckpt is not None:
            self.wm, ns_w, na_w, meta_w, self.sd_w = load_world_model(wm_ckpt, dev)
            sd_w, wm = self.sd_w, self.wm
            # the world model may predict a SHORTER horizon than the policy's chunk: it then
            # scores only the first wm.horizon actions of each proposal -- the regime MPC
            # actually executes -- and avoids the far horizon where prediction error is worst.
            assert wm.horizon <= H, f"wm H={wm.horizon} exceeds policy H={H}"
            self.WH = wm.horizon
            rot_repr = str(sd_w.get("rot_repr", "quat"))
            assert rot_repr == "quat", f"world model rot_repr={rot_repr!r}: this planner scores quats"
            self.cube_dim = int(sd_w.get("cube_dim", wm.obj_dim))
            self.n_hist = int(sd_w.get("n_hist", 1)); self.hist_stride = int(sd_w.get("hist_stride", 1))
            self.frame_dim = int(sd_w.get("frame_dim", wm.state_dim))
            assert self.frame_dim * self.n_hist == wm.state_dim, \
                f"cond width mismatch: {self.n_hist}x{self.frame_dim} != {wm.state_dim}"
        # ``compile`` compiles the POLICY only -- the configuration behind the paper's numbers
        # (its world model ran eager). ``compile_wm`` also compiles the world model: faster,
        # but not bit-identical to eager, so first replans no longer reproduce the paper's.
        if compile:
            compile_denoiser(self.policy, "policy")
        if compile_wm and self.wm is not None:
            compile_denoiser(self.wm, "world model")

        # ---- joint-name remaps + normalizers as device tensors ----
        T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
        self.hp_pol_t = T(name_perm(joint_names, self.meta_p["hand_joint_names"]))
        self.act_col_t = T(name_perm(joint_names, self.meta_p["cmd_q22_names"]))
        self.nsp_c, self.nsp_s = T(ns_p.center), T(ns_p.scale)
        self.nap_c, self.nap_s = T(na_p.center), T(na_p.scale)
        if self.wm is not None:
            self.hp_wm_t = T(name_perm(joint_names, meta_w["hand_joint_names"]))
            self.cmd_pol2wm = T(name_perm(self.meta_p["cmd_q22_names"], meta_w["cmd_q22_names"]))
            self.nsw_c, self.nsw_s = T(ns_w.center), T(ns_w.scale)
            self.naw_c, self.naw_s = T(na_w.center), T(na_w.scale)
            self.obj_c = self.nsw_c[22:22 + self.cube_dim]; self.obj_s = self.nsw_s[22:22 + self.cube_dim]
        self.HBUF = (self.n_hist - 1) * self.hist_stride + 1          # frames of history to keep
        self.hist_off = T(np.arange(self.n_hist) * self.hist_stride)  # current-first

        # ---- elite archive (per env): deque of (chunks (K,H,22), created_t, valid_len (K,), score (K,)) ----
        # Chunks are ABSOLUTE joint targets, so a plan made at t0 is re-used at t simply by
        # shifting it by (t - t0) steps -- entry j of the stored chunk is the command for t0 + j.
        self.N_ARCH = int(round(self.n_samples * reuse_frac)) if self.reuse_elites else 0
        self.K_PER = reuse_keep if reuse_keep > 0 else self.N_ARCH     # elites stored per replan
        self.archive = [collections.deque(maxlen=reuse_hist) for _ in range(num_envs)]
        self.reuse_stat = [0, 0]      # [archived rows used, total rows scored]
        self.arch_won = [0, 0]        # [picks that came from the archive, picks]

    # ------------------------------------------------------------------ helpers
    @property
    def mode(self):
        return "policy" if self.wm is None else "mpc"

    def amp_ctx(self):
        return torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def describe(self):
        if self.wm is None:
            return (f"[policy-only] policy step={self.cfg_p['step']} H={self.H} "
                    f"(no world model, goal-agnostic baseline)")
        return (f"[mpc] policy step={self.cfg_p['step']} H={self.H}  wm step={self.sd_w['step']} "
                f"WH={self.WH} cond={self.n_hist}x{self.frame_dim}@stride{self.hist_stride}  "
                f"N={self.n_samples} reuse={'on' if self.reuse_elites else 'off'}")

    def wm_frame(self, hand_art, cube_st):
        """Per-frame WM state [hand(22) in WM legend, cube pos+quat] -> normalized (E, frame_dim)."""
        f = torch.cat([hand_art[:, self.hp_wm_t], cube_st[:, :7]], 1)
        return (f - self.nsw_c) / self.nsw_s

    def reset_archive(self, env_ids=None):
        """Plans do not survive a trial or a cube teleport."""
        for e in (range(len(self.archive)) if env_ids is None else env_ids):
            self.archive[e].clear()

    @staticmethod
    def _shift(ch, off):
        """Align a stored chunk so index 0 is the command for the CURRENT step.
        Tail is padded by repeating the last command; valid_len tracks the real part."""
        K_, Hh, D = ch.shape
        out = torch.empty_like(ch)
        n = Hh - off
        out[:, :n] = ch[:, off:]
        if n < Hh:
            out[:, n:] = ch[:, -1:].expand(K_, Hh - n, D)
        return out

    # ------------------------------------------------------------------ one replan
    @torch.no_grad()
    def plan(self, env_ids, hand_art, cube_st, goal_q, hist=None, t_now=0, last_cmd=None,
             details=False):
        """Batched proposal + scoring for the envs in ``env_ids``.

        hand_art (E,22) joint positions in ``joint_names`` order; cube_st (E,>=7); goal_q (E,4)
        numpy wxyz; hist (E, HBUF, frame_dim) normalized WM frames, oldest -> newest; last_cmd
        (E,22) last issued command in policy cmd order (elite-archive continuity gate).

        Returns (chunks (k,H,22) raw rad in policy cmd order, chosen_geo (k,), best_step (k,),
                 execute (k,) bool, geo_now (k,)) and, with ``details``, a dict holding every
                 proposal's predicted geodesic (k*N, WH), cube quats (k*N, WH, 4) and cost (k, N).
        """
        dev, H, WH = self.device, self.H, self.WH
        k = len(env_ids)
        N = 1 if self.wm is None else self.n_samples
        ei = torch.as_tensor(env_ids, device=dev)
        # -- policy conditioning (one row per env, repeated N times) --
        ps = torch.cat([hand_art[ei][:, self.hp_pol_t], cube_st[ei][:, :self.pol_cube_dims]], 1)
        pcond = ((ps - self.nsp_c) / self.nsp_s).float().repeat_interleave(N, 0)  # (k*N, 29)
        if self.wm is None:                                                    # 1 chunk/env
            with self.amp_ctx():
                ch = self.policy(pcond, eta=self.sample_eta, init_noise_scale=self.init_noise_scale)
            chunk_raw = (ch.float() * self.nap_s + self.nap_c)[:, :, :22]
            zeros = torch.zeros(k, device=dev)
            out = (chunk_raw, zeros, torch.zeros(k, dtype=torch.long, device=dev),
                   torch.ones(k, dtype=torch.bool, device=dev), zeros)
            return (*out, {}) if details else out
        # -- WM conditioning: history stack, current-first, per env --
        hidx = (self.HBUF - 1 - self.hist_off).clamp(min=0)                    # (n_hist,)
        wcond = hist[ei][:, hidx].reshape(k, -1).float().repeat_interleave(N, 0)   # (k*N, n_hist*frame)
        goal_t = torch.as_tensor(goal_q[env_ids], device=dev, dtype=torch.float32)  # (k,4)
        goal_flat = goal_t.repeat_interleave(N, 0)[:, None, :]                 # (k*N,1,4)
        chunks = torch.empty(k * N, H, 22, device=dev)
        geo = torch.empty(k * N, WH, device=dev)
        vlen_all = torch.full((k * N,), H, dtype=torch.long, device=dev)       # real steps per row
        pred_q_all = torch.empty(k * N, WH, 4, device=dev) if details else None

        # ---- gather re-usable elites, time-aligned to t_now, gated on ACTION continuity only ----
        arch_use = [None] * k                                                  # per env: (n_e,H,22)
        arch_vl = [None] * k
        if self.reuse_elites:
            for j, e_id in enumerate(env_ids):
                cs, vs, ss = [], [], []
                for (ch_a, t0, vl, sc) in self.archive[e_id]:
                    off = int(t_now - t0)
                    if off <= 0:
                        continue
                    newv = vl - off
                    ok = newv >= WH                                            # enough REAL steps left
                    if not bool(ok.any()):
                        continue
                    al = self._shift(ch_a, off)
                    # Action continuity: compare COMMAND to COMMAND. The implicit actuator sits
                    # 0.24-0.79 rad behind its target in steady state, so a command-vs-measured-
                    # position gate can never pass. Fresh proposals already jump 0.10-0.82 rad
                    # (median ~0.35) from the last command, so 0.5 puts re-used chunks on equal
                    # footing, not stricter.
                    dj = (al[:, 0, :] - last_cmd[e_id][None, :]).abs().amax(dim=1)
                    ok = ok & (dj <= self.reuse_joint_tol)
                    if bool(ok.any()):
                        cs.append(al[ok]); vs.append(newv[ok]); ss.append(sc[ok])
                if cs:
                    # pool every surviving candidate from ALL history rollouts, then rank the
                    # pool TOGETHER by WM score and keep the best N_ARCH -- not first-come by age.
                    c = torch.cat(cs, 0); v = torch.cat(vs, 0); sco = torch.cat(ss, 0)
                    if c.shape[0] > self.N_ARCH:
                        keep = sco.topk(self.N_ARCH, largest=False).indices
                        c, v = c[keep], v[keep]
                    arch_use[j] = c; arch_vl[j] = v

        # per-env fresh count = N minus whatever the archive supplied (total stays N)
        n_arch = [0 if arch_use[j] is None else arch_use[j].shape[0] for j in range(k)]
        n_fresh = [N - a for a in n_arch]
        if self.reuse_elites:
            cnt = torch.as_tensor(n_fresh, device=dev)
            pcond = ((ps - self.nsp_c) / self.nsp_s).float().repeat_interleave(cnt, dim=0)
        tot_fresh = int(sum(n_fresh))
        fresh = torch.empty(tot_fresh, H, 22, device=dev)
        for s in range(0, tot_fresh, self.plan_batch):
            e = min(s + self.plan_batch, tot_fresh)
            with self.amp_ctx():
                ch = self.policy(pcond[s:e], eta=self.sample_eta, init_noise_scale=self.init_noise_scale)
            fresh[s:e] = (ch.float() * self.nap_s + self.nap_c)[:, :, :22]
        self.reuse_stat[0] += int(sum(n_arch)); self.reuse_stat[1] += k * N
        p = 0
        for j in range(k):
            b, nf = j * N, n_fresh[j]
            chunks[b:b + nf] = fresh[p:p + nf]; p += nf
            if n_arch[j]:
                chunks[b + nf:b + N] = arch_use[j]
                vlen_all[b + nf:b + N] = arch_vl[j]

        # ---- score every row with the world model ----
        for s in range(0, k * N, self.plan_batch):
            e = min(s + self.plan_batch, k * N)
            craw = chunks[s:e]                                                 # rad, policy cmd order
            a_wm = ((craw[:, :WH, self.cmd_pol2wm] - self.naw_c) / self.naw_s).float()  # WM order, first WH
            with self.amp_ctx():
                pred = self.wm(wcond[s:e], a_wm)
            obj = pred.float()[:, :, :self.cube_dim] * self.obj_s + self.obj_c  # raw object trajectory
            q = torch.nn.functional.normalize(obj[:, :, 3:7], dim=-1)
            geo[s:e] = geo_deg_t(q, goal_flat[s:e])
            if pred_q_all is not None:
                pred_q_all[s:e] = q
        cost = geo + self.time_weight * torch.arange(geo.shape[1], device=dev, dtype=torch.float32)
        sel = cost.min(dim=1).values.view(k, N)                                # closest approach
        best_local = sel.argmin(dim=1)
        if self.reuse_elites:
            nf_t = torch.as_tensor(n_fresh, device=dev)
            self.arch_won[0] += int((best_local >= nf_t).sum()); self.arch_won[1] += k
        best = best_local + torch.arange(k, device=dev) * N                   # flat index of each best
        best_step = cost[best].argmin(dim=1)
        chosen_geo = geo[best].min(dim=1).values
        # ---- push this replan's elites (best-scoring chunks) into the archive ----
        if self.reuse_elites:
            top = sel.topk(min(self.K_PER, N), dim=1, largest=False).indices   # (k,K_PER)
            for j, e_id in enumerate(env_ids):
                rows = top[j] + j * N
                self.archive[e_id].append((chunks[rows].clone(), int(t_now),
                                           vlen_all[rows].clone(), sel[j][top[j]].clone()))
        cur_q = torch.nn.functional.normalize(cube_st[ei][:, 3:7].float(), dim=-1)
        geo_now = geo_deg_t(cur_q, goal_t)
        execute = (torch.ones(k, dtype=torch.bool, device=dev) if not self.noop_gate
                   else chosen_geo < geo_now - self.noop_margin)
        out = (chunks[best], chosen_geo, best_step, execute, geo_now)
        if details:
            return (*out, {"geo": geo, "pred_q": pred_q_all, "cost": sel, "chunks": chunks,
                           "best_local": best_local, "n_fresh": n_fresh})
        return out
