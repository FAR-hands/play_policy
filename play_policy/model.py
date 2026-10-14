"""Diffusion Transformer (DiT) play policy and action-conditioned world model.

Both models share one DDPM-train / DDIM-sample core and the same 1-D DiT denoiser
(Peebles & Xie 2023, adapted to trajectories: patchify + adaLN-Zero blocks):

* ``PlayDiTPolicy``    -- goal-free play policy. Diffuses an action chunk ``(H, 22)`` of joint
  commands conditioned on the current normalized state ``[hand(22), cube_pose(7)]``.
  Receding horizon: predict ``horizon`` actions, execute ``n_action_steps``, replan.
* ``PlayDPWorldModel`` -- world model. Diffuses the future cube-pose trajectory ``(H, 7)``
  conditioned on a stack of recent states (global conditioning) and the future action chunk
  (clean per-step input channels).

Everything operates in normalized space (~[-1, 1]); sampling returns the denoised sequence
clamped to [-clip, clip].
"""

from __future__ import annotations

import math
import os
import sys
import types

import torch
from torch import nn


class _SinusoidalTimeEmb(nn.Module):
    """Sinusoidal embedding of the (integer) diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(1, half - 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        return emb


class _ChunkDiffusionBase(nn.Module):
    """Shared diffusion core: DDPM noise-prediction loss + (stochastic) DDIM sampler.

    Subclasses build the denoiser and implement ``_denoise_pred(cond, x_t, t)``.
    ``schedule`` is "cosine" (squaredcos_cap_v2, used by both release models) or "linear".
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        n_action_steps: int,
        n_timesteps: int,
        n_sample_steps: int,
        clip: float = 1.0,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_timesteps = n_timesteps
        self.n_sample_steps = min(n_sample_steps, n_timesteps)
        self.clip = clip
        self.schedule = schedule
        if schedule == "cosine":  # squaredcos_cap_v2
            s = 0.008
            ts = torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps
            f = torch.cos((ts + s) / (1.0 + s) * math.pi / 2.0) ** 2
            betas = (1.0 - f[1:] / f[:-1]).clamp(max=0.999).to(torch.float32)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_timesteps)
        else:
            raise ValueError(f"unknown beta schedule: {schedule!r}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _denoise_pred(self, cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """DDPM noise-prediction MSE. ``state`` (B, state_dim), ``action_chunk`` (B, H, action_dim)."""
        b = state.shape[0]
        device = state.device
        x0 = action_chunk
        t = torch.randint(0, self.n_timesteps, (b,), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(b, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(b, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        eps_pred = self._denoise_pred(state, x_t, t)
        return ((eps_pred - noise) ** 2).mean()

    @torch.no_grad()
    def forward(
        self,
        state: torch.Tensor,
        x_init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        eta: float = 0.0,
        init_noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample a normalized action chunk (B, H, action_dim) via (stochastic) DDIM.

        ``eta`` = 0 (default) is deterministic DDIM -- the only stochasticity is the initial
        latent. ``eta`` > 0 injects fresh Gaussian noise at every reverse step (eta=1 -> full
        DDPM-ancestral variance), widening the sampled-trajectory distribution.
        ``init_noise_scale`` (>1) scales the initial latent for a little extra spread. Autonomous
        play collection and MPC proposals both use ``eta=1.0, init_noise_scale=1.1``.
        Pass ``x_init``/``generator`` to control/seed the initial noise (and, when eta>0, the
        per-step noise); otherwise the global torch RNG is used.
        """
        b = state.shape[0]
        device = state.device
        cond = state
        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(b, self.horizon, self.action_dim, device=device, generator=generator)
        if init_noise_scale != 1.0:
            x = x * init_noise_scale
        step_indices = (
            torch.linspace(self.n_timesteps - 1, 0, self.n_sample_steps, device=device).round().long()
        )
        for i, t in enumerate(step_indices):
            t_int = int(t.item())
            t_batch = torch.full((b,), t_int, device=device, dtype=torch.long)
            eps = self._denoise_pred(cond, x, t_batch)
            alpha_bar_t = self.alpha_bars[t_int]
            x0_hat = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            # clip the x0 estimate every step (reference Diffusion Policy's clip_sample=True)
            x0_hat = x0_hat.clamp(-self.clip, self.clip)
            if i + 1 < self.n_sample_steps:
                t_prev = int(step_indices[i + 1].item())
                alpha_bar_prev = self.alpha_bars[t_prev]
                if eta > 0.0:
                    # stochastic DDIM (Song et al. 2021): sigma sets the injected-noise level;
                    # eta=0 -> deterministic, eta=1 -> DDPM-ancestral variance.
                    sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) \
                                * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
                    dir_coef = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0))
                    z = torch.randn(x.shape, device=device, generator=generator)
                    x = torch.sqrt(alpha_bar_prev) * x0_hat + dir_coef * eps + sigma * z
                else:
                    x = torch.sqrt(alpha_bar_prev) * x0_hat + torch.sqrt(1.0 - alpha_bar_prev) * eps
            else:
                x = x0_hat
        return x.clamp(-self.clip, self.clip)


# ---------------------------------------------------------------------------
# DiT (Peebles & Xie 2023) adapted to 1-D trajectories
# ---------------------------------------------------------------------------

class _DiTBlock(nn.Module):
    """Pre-norm self-attention + MLP, both modulated by adaLN-Zero.

    Each block regresses (shift, scale, gate) for the attention and the MLP from the
    conditioning vector. The regressor is zero-initialised, so every block starts as the
    identity and the network begins training as a no-op -- the "Zero" in adaLN-Zero."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        h = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, h), nn.GELU(approximate="tanh"),
                                 nn.Linear(h, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # ``c`` (B, d) is one conditioning vector broadcast over the whole sequence
        sa, ba, ga, sm, bm, gm = (m[:, None] for m in self.ada(c).chunk(6, dim=-1))
        h = self.norm1(x) * (1 + sa) + ba
        x = x + ga * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + sm) + bm
        return x + gm * self.mlp(h)


class _DiTFinal(nn.Module):
    """adaLN-modulated LayerNorm + linear projection back to patch space (zero-init)."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(d_model, out_dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        for m in (self.lin, self.ada[1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s, b = self.ada(c).chunk(2, dim=-1)
        return self.lin(self.norm(x) * (1 + s[:, None]) + b[:, None])


class DiT1D(nn.Module):
    """Diffusion Transformer over a length-``horizon`` trajectory.

    TOKENIZER (DiT's "patchify"): the (H, input_dim) sequence is cut into non-overlapping
    patches of ``patch_size`` timesteps, each flattened to ``patch_size*input_dim`` and linearly
    embedded to ``d_model`` -- so H=200 with patch 4 becomes 50 tokens. A learned positional
    embedding is added, ``depth`` adaLN-Zero blocks run global self-attention over the tokens,
    and the final layer projects back to ``patch_size*out_dim`` which is un-patchified to
    (H, out_dim).

    Conditioning follows DiT: the diffusion-step embedding and the global conditioning vector
    (the state, or the state-history stack for the world model) are summed into one vector that
    modulates every block. The world model's per-step action conditioning arrives concatenated
    onto ``input_dim`` as clean channels."""

    def __init__(self, input_dim: int, global_cond_dim: int, out_dim: int, horizon: int,
                 patch_size: int = 4, d_model: int = 512, depth: int = 8, n_heads: int = 8,
                 dsed: int = 256, mlp_ratio: float = 4.0):
        super().__init__()
        if horizon % patch_size != 0:
            raise ValueError(f"horizon={horizon} must be divisible by patch_size={patch_size}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.patch_size = patch_size; self.horizon = horizon; self.out_dim = out_dim
        self.n_tokens = horizon // patch_size
        self.x_emb = nn.Linear(input_dim * patch_size, d_model)          # <- the tokenizer
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.t_emb = nn.Sequential(_SinusoidalTimeEmb(dsed), nn.Linear(dsed, d_model),
                                   nn.SiLU(), nn.Linear(d_model, d_model))
        self.c_emb = nn.Sequential(nn.Linear(global_cond_dim, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([_DiTBlock(d_model, n_heads, mlp_ratio)
                                     for _ in range(depth)])
        self.final = _DiTFinal(d_model, patch_size * out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        """``x`` (B, H, input_dim), ``t`` (B,) integer timesteps, ``global_cond`` (B, cond)."""
        b = x.shape[0]
        tok = self.x_emb(x.reshape(b, self.n_tokens, -1)) + self.pos      # patchify + embed
        c = self.t_emb(t) + self.c_emb(global_cond)                       # adaLN conditioning
        for blk in self.blocks:
            tok = blk(tok, c)
        return self.final(tok, c).reshape(b, self.horizon, self.out_dim)  # un-patchify


class PlayDiTPolicy(_ChunkDiffusionBase):
    """Diffusion Transformer play policy: the DiT1D denoiser on the action chunk, conditioned on
    the normalized state. Attention is global from the first block, which is what a long chunk
    (H=200 -> 50 tokens at patch 4) needs."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 22,
        horizon: int = 200,
        n_action_steps: int = 50,
        patch_size: int = 4,
        d_model: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dsed: int = 256,
        n_timesteps: int = 50,
        n_sample_steps: int = 10,
        schedule: str = "cosine",
        clip: float = 1.0,
    ):
        super().__init__(state_dim, action_dim, horizon, n_action_steps, n_timesteps,
                         n_sample_steps, clip=clip, schedule=schedule)
        self.dit = DiT1D(action_dim, state_dim, action_dim, horizon,
                         patch_size=patch_size, d_model=d_model, depth=depth,
                         n_heads=n_heads, dsed=dsed, mlp_ratio=mlp_ratio)

    def _denoise_pred(self, cond: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.dit(x_t, t, cond)


class PlayDPWorldModel(_ChunkDiffusionBase):
    """Action-conditioned diffusion WORLD MODEL: denoise the future cube-pose sequence
    conditioned on the recent states (global conditioning) and the future action sequence
    (concatenated as clean input channels).

    diffused x          = future cube pose          (B, H, obj_dim)     [pos3 + quat4]
    global cond         = state-history stack       (B, state_dim)      [n_hist x (22 hand + 7 cube)]
    per-step cond       = future actions            (B, H, act_dim)     [cmd_q22]

    The DiT sees ``[noisy_target(obj_dim) ; actions(act_dim)]`` channels at each step and predicts
    noise on the ``obj_dim`` target channels only; the action channels are never noised and
    never predicted. ``compute_loss``/``forward`` therefore take the action chunk as an extra
    argument (vs the policy's state-only conditioning).
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int = 22,
        obj_dim: int = 7,
        horizon: int = 50,
        dsed: int = 256,
        n_timesteps: int = 100,
        n_sample_steps: int = 10,
        schedule: str = "cosine",
        clip: float = 1.0,
        patch_size: int = 5,
        d_model: int = 512,
        depth: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        out_weight=None,
    ):
        # the diffused width (base's ``action_dim`` slot) is the object-state dim
        super().__init__(state_dim, obj_dim, horizon, 0, n_timesteps, n_sample_steps,
                         clip=clip, schedule=schedule)
        self.act_dim = act_dim
        self.obj_dim = obj_dim
        self.backbone = "dit"
        # per-output-channel weight on the eps-MSE (all ones = the paper recipe). Derived from
        # the constructor args, so it is not stored in the state_dict.
        w = torch.ones(obj_dim) if out_weight is None else torch.as_tensor(out_weight, dtype=torch.float32)
        assert w.numel() == obj_dim, f"out_weight has {w.numel()} entries, obj_dim={obj_dim}"
        self.register_buffer("out_w", w.view(1, 1, -1), persistent=False)
        self.patch_size = int(patch_size)
        self.n_tokens = horizon // int(patch_size)
        self.dit = DiT1D(obj_dim + act_dim, state_dim, obj_dim, horizon,
                         patch_size=patch_size, d_model=d_model, depth=depth,
                         n_heads=n_heads, dsed=dsed, mlp_ratio=mlp_ratio)

    @property
    def denoiser(self):
        return self.dit

    def _denoise_pred(self, state, x_t, t, act_chunk):
        xin = torch.cat([x_t, act_chunk], dim=-1)         # (B, H, obj_dim + act_dim)
        return self.dit(xin, t, state)                     # (B, H, obj_dim)

    def compute_loss(self, state, obj_chunk, act_chunk):
        """DDPM noise-prediction MSE on the object sequence, conditioned on actions.
        ``state`` (B, state_dim), ``obj_chunk`` (B, H, obj_dim), ``act_chunk`` (B, H, act_dim)."""
        b = state.shape[0]
        t = torch.randint(0, self.n_timesteps, (b,), device=state.device)
        noise = torch.randn_like(obj_chunk)
        sqrt_ab = self.sqrt_alpha_bars[t].view(b, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(b, 1, 1)
        x_t = sqrt_ab * obj_chunk + sqrt_1mab * noise
        eps_pred = self._denoise_pred(state, x_t, t, act_chunk)
        se = (eps_pred - noise) ** 2
        w = self.out_w.expand_as(se)
        return (se * w).sum() / w.sum()                    # channel-weighted mean

    @torch.no_grad()
    def forward(self, state, act_chunk, x_init=None, generator=None):
        """Sample a normalized future object-state sequence (B, H, obj_dim) via eta=0 DDIM,
        conditioned on the ``state`` history stack and the future ``act_chunk`` (B, H, act_dim)."""
        b = state.shape[0]
        device = state.device
        if x_init is not None:
            x = x_init.to(device)
        else:
            x = torch.randn(b, self.horizon, self.obj_dim, device=device, generator=generator)
        step_indices = torch.linspace(self.n_timesteps - 1, 0, self.n_sample_steps,
                                      device=device).round().long()
        for i, t in enumerate(step_indices):
            ti = int(t.item())
            tb = torch.full((b,), ti, device=device, dtype=torch.long)
            eps = self._denoise_pred(state, x, tb, act_chunk)
            ab = self.alpha_bars[ti]
            x0h = ((x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)).clamp(-self.clip, self.clip)
            if i + 1 < self.n_sample_steps:
                tp = int(step_indices[i + 1].item())
                abp = self.alpha_bars[tp]
                x = torch.sqrt(abp) * x0h + torch.sqrt(1.0 - abp) * eps
            else:
                x = x0h
        return x.clamp(-self.clip, self.clip)


# ---------------------------------------------------------------------------
# Builders + checkpoint loading
# ---------------------------------------------------------------------------

def build_policy(policy_type: str, state_dim: int, args, action_dim: int = 22) -> nn.Module:
    """Build the play policy from an argparse-style namespace (train.py's flags or a checkpoint)."""
    if policy_type != "dit":
        raise ValueError(f"policy_type={policy_type!r}: this release implements the DiT policy only")
    return PlayDiTPolicy(
        state_dim, action_dim=action_dim, horizon=args.H, n_action_steps=args.n_action_steps,
        patch_size=int(getattr(args, "patch_size", 4)),
        d_model=int(getattr(args, "d_model", 256)),
        depth=int(getattr(args, "depth", 6)),
        n_heads=int(getattr(args, "n_heads", 8)),
        mlp_ratio=float(getattr(args, "mlp_ratio", 4.0)),
        dsed=int(getattr(args, "dsed", 256)),
        n_timesteps=args.n_timesteps, n_sample_steps=args.n_sample_steps,
        schedule=getattr(args, "schedule", None) or "cosine",
    )


def build_world_model(state_dim: int, args, act_dim: int = 22, obj_dim: int = 7) -> nn.Module:
    """Build the DiT world model from an argparse-style namespace. ``state_dim`` is the full
    conditioning width (n_hist x per-frame state)."""
    return PlayDPWorldModel(
        state_dim, act_dim=act_dim, obj_dim=obj_dim, horizon=args.H,
        dsed=int(getattr(args, "dsed", 256)),
        n_timesteps=args.n_timesteps, n_sample_steps=args.n_sample_steps,
        schedule=getattr(args, "schedule", None) or "cosine",
        patch_size=int(getattr(args, "patch_size", 5)),
        d_model=int(getattr(args, "d_model", 512)),
        depth=int(getattr(args, "depth", 8)),
        n_heads=int(getattr(args, "n_heads", 8)),
        mlp_ratio=float(getattr(args, "mlp_ratio", 4.0)),
        out_weight=getattr(args, "out_weight", None) or None)


def _check_supported(ck: dict, fields: dict, what: str):
    """Fail loudly on a checkpoint feature this release cannot express (rather than silently
    building a differently-wired model)."""
    bad = {k: ck[k] for k, ok in fields.items() if k in ck and ck[k] not in ok}
    if bad:
        raise ValueError(f"{what} checkpoint uses features outside this release: {bad}")


def wm_args_from_ckpt(ck: dict):
    """Rebuild the ``args`` a world-model checkpoint was constructed with."""
    _check_supported(ck, {"backbone": ("dit",), "objective": ("ddpm",), "rot_repr": ("quat",),
                          "predict_hand": (False,), "hist_encoder": (False,),
                          "predict_drop": (False,), "predict_fresh": (False,),
                          "trunk_cond_dim": (0, None), "residual_out": (False,)}, "world-model")
    return types.SimpleNamespace(
        H=ck["horizon"], dsed=ck.get("dsed", 256), schedule=ck.get("schedule", "cosine"),
        n_timesteps=ck.get("n_timesteps", 100), n_sample_steps=ck.get("n_sample_steps", 10),
        patch_size=ck.get("patch_size", 5), d_model=ck.get("d_model", 512),
        depth=ck.get("depth", 8), n_heads=ck.get("n_heads", 8),
        mlp_ratio=ck.get("mlp_ratio", 4.0),
        n_hist=int(ck.get("n_hist", 1)), hist_stride=int(ck.get("hist_stride", 1)),
        frame_dim=int(ck.get("frame_dim", 0)),
        out_weight=(ck.get("out_weight") or None))


# derived constant buffers that some checkpoints carry in their state_dict; rebuilt from the
# constructor args here, so they are dropped before the strict load
_DERIVED_BUFFERS = ("out_w", "cube_sel", "rot_sel")


def compile_denoiser(model, mode: str | None = None, verbose: bool = True):
    """torch.compile the ``dit`` denoiser in place; returns the model.

    Compiled kernels are fast but NOT bit-identical to eager, so anything that needs
    reproducibility against a stored result should stay eager. Recompiles once per new batch
    shape, so keep the batch size fixed across replans.
    """
    mode = mode or os.environ.get("WM_TORCH_COMPILE_MODE") or "max-autotune"
    try:
        model.dit = torch.compile(model.dit, mode=mode)
        if verbose:
            print(f"[compile] {type(model).__name__}.dit -> torch.compile(mode={mode!r})",
                  file=sys.stderr)
    except Exception as e:                                  # never let a compile failure be fatal
        if verbose:
            print(f"[compile] skipped ({type(e).__name__}: {e})", file=sys.stderr)
    return model


def load_world_model_ckpt(ckpt, device="cpu", eval_mode: bool = True):
    """``ckpt`` (path or already-loaded dict) -> (world model on ``device``, the ckpt dict).

    Loads STRICTLY: a checkpoint whose architecture this build cannot express fails here.
    ``ckpt["policy"]`` holds the EMA weights (the ones to deploy). Set ``WM_TORCH_COMPILE=1`` to
    torch.compile the denoiser after loading.
    """
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if isinstance(ckpt, (str, bytes, os.PathLike)) \
        else ckpt
    a = wm_args_from_ckpt(ck)
    n_out = len(ck.get("out_cols", []) or []) or int(ck.get("cube_dim", ck["obj_dim"]))
    wm = build_world_model(int(ck["state_dim"]), a, act_dim=int(ck["act_dim"]), obj_dim=n_out)
    sd = {k: v for k, v in ck["policy"].items() if k not in _DERIVED_BUFFERS}
    wm.load_state_dict(sd, strict=True)
    wm.to(device)
    if eval_mode:
        wm.eval()
    if os.environ.get("WM_TORCH_COMPILE", "0") not in ("0", ""):
        compile_denoiser(wm)
    return wm, ck
