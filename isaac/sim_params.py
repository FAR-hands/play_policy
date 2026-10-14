"""The simulator parameters of a live Isaac Lab hand + cube scene: read them, stamp them
into recordings, and (optionally) put them on viser sliders.

Both halves live here because they are the same knowledge -- which physics knob is where,
and which of them PhysX will accept while the timeline runs:

  read_sim_params()  snapshot the physics a rollout was produced under. Every recorder
                     (teleop takes, self-play collection, MPC eval rollouts) stamps this
                     into its .npz, so a dataset can always answer "which simulator is this?"
                     -- the question that decides whether two datasets may be mixed, or
                     whether a world model still describes the current sim.
  SimParamPanel      the same knobs as browser sliders for a live scene, used by
                     teleop_isaac.py (glove play) and viser_sim_replay.py (rollout re-sim).

  LIVE -- written straight into PhysX, effective on the next step
    physics rate      ``physics_context.set_physics_dt`` (PhysX reads the scene dt on
                      every simulate() call, so a change lands immediately)
    substeps          purely the caller's loop count -- ``panel.substeps`` physics steps
                      per control frame at dt = 1/(rate*substeps): finer integration at
                      identical action timing
    hand gains        ``write_joint_{stiffness,damping,armature}_to_sim``; implicit
                      actuators never overwrite these per step

  USD -- only re-read when PhysX re-parses the stage, i.e. on a timeline bounce
    cube compliant-contact stiffness / damping   (PhysxMaterialAPI on the cube material)
    cube linear / angular damping                (PhysxRigidBodyAPI on the cube)

The bounce is ``sim.reset()`` (~0.3 s, no Isaac reboot). NEVER call ``sim.stop()`` yourself:
outside ``reset()`` IsaacLab's stop handler spins in ``while not is_playing(): render()``
waiting for a play button that a headless run has no way to press. The physical state is
snapshotted and written back across the bounce, so play continues where it left off (a
caller with a better anchor -- e.g. a recorded frame -- passes ``after_restart``).

Nothing is written to the sim until a slider actually moves: the panel seeds itself from
the live scene, so having it open changes no dynamics.

Isaac imports are deferred into the functions that need them, so the pure helpers
(``params_json`` / ``params_hash`` / ``summarize``) can be used from a plain env -- e.g.
isaac/inspect_npz.py reading a recording without a simulator.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque

# ===========================================================================================
# THE PROJECT'S SIMULATOR, IN ONE PLACE. Every scene builder (teleop_isaac, wm_mpc_eval_batch,
# viser_sim_replay) imports these, so the config the data is collected at, the policy is
# deployed at, and the world model is evaluated at can never drift apart.
#
# Compliant cube contact + 3 substeps: chaos measurements showed 2-3x lower short-horizon
# divergence than rigid contact at equal or better mobility, and substeps help once contact
# is compliant (the ~25-50 Hz contact mode is resolved at 720 Hz, not at 240 Hz). Data
# recorded under any other config is NOT interchangeable with the released data -- compare
# sim_params_hash (the released self-play data stamps 21c4c9c015e1) before mixing.
# ===========================================================================================
PHYS_BASE_HZ = 240.0            # sim time advanced per control frame = 1/240 s
SUBSTEPS = 3                    # physics steps per control frame
PHYSICS_DT = 1.0 / (PHYS_BASE_HZ * SUBSTEPS)          # 1/720 s
CUBE_CONTACT_STIFFNESS = 3000.0                       # compliant contact spring [N/m]
CUBE_CONTACT_DAMPING = 15.0                           # its damping [N.s/m] -- the value the released
                                                      # data was recorded at. (Critical at
                                                      # k=3000, m=0.08 is ~31; an A/B under a REAL-TIME
                                                      # frame config liked it, but that config is not this
                                                      # one -- re-measure before changing a physics knob.)
CUBE_LINEAR_DAMPING = 0.2
CUBE_ANGULAR_DAMPING = 0.2
HAND_STIFFNESS = 3.0
HAND_DAMPING = 0.1
CONTROL_HZ = 60.0               # control-loop rate every collector pins. NOT a physics knob: physics
                                # advances 1/240 s per frame whatever this is, but the loop rate sets
                                # how much HUMAN motion lands in each frame, so it is part of a demo
                                # pool's identity. The released teleop takes stamp 60 but their loop
                                # actually ran near 26 Hz; `--hz 26` reproduces that pacing (at the
                                # cost of a different stamp). Takes record `hz_measured`.


def cube_physics_material(**kw):
    """The cube's physics material at the project default (override any field by keyword)."""
    import isaaclab.sim as sim_utils
    return sim_utils.RigidBodyMaterialCfg(**{
        "static_friction": 1.2, "dynamic_friction": 1.0, "restitution": 0.0,
        "restitution_combine_mode": "min",
        "compliant_contact_stiffness": CUBE_CONTACT_STIFFNESS,
        "compliant_contact_damping": CUBE_CONTACT_DAMPING, **kw})


# Fallbacks for anything the live scene doesn't tell us (same values as above).
LAUNCH = dict(phys_hz=PHYS_BASE_HZ, substeps=SUBSTEPS,
              stiffness=HAND_STIFFNESS, damping=HAND_DAMPING, armature=0.0,
              cube_stiff=CUBE_CONTACT_STIFFNESS, cube_damp=CUBE_CONTACT_DAMPING,
              cube_lin_damp=CUBE_LINEAR_DAMPING, cube_ang_damp=CUBE_ANGULAR_DAMPING)

# key, label, min, max, step, int?, hint
SPECS = (
    ("phys_hz", "Physics rate [Hz]", 30, 1920, 30, True,
     "sim time one control frame advances = 1/rate"),
    ("substeps", "Substeps", 1, 8, 1, True,
     "split each frame into N finer physics steps (same action timing)"),
    ("stiffness", "Hand stiffness", 0.1, 50.0, 0.1, False, "implicit-actuator P gain"),
    ("damping", "Hand damping", 0.0, 5.0, 0.01, False, "implicit-actuator D gain"),
    ("armature", "Hand armature", 0.0, 0.05, 0.001, False,
     "added rotor inertia — conditions the contact solver"),
    ("cube_stiff", "Cube contact stiffness", 0.0, 20000.0, 50.0, False,
     "compliant (soft) contact spring [N/m]; 0 = rigid contact. ~2000 gives sub-mm "
     "penetration under load and a 25 Hz contact mode, well resolved at 240 Hz"),
    ("cube_damp", "Cube contact damping", 0.0, 200.0, 5.0, False,
     "damping of that spring [N·s/m]; critical is 2*sqrt(k*m) ≈ 25 at k=2000. "
     "Only acts when stiffness > 0; at 0 a compliant contact is a springboard"),
    ("cube_lin_damp", "Cube linear damping", 0.0, 5.0, 0.05, False, "velocity damping"),
    ("cube_ang_damp", "Cube angular damping", 0.0, 5.0, 0.05, False,
     "spin damping — brakes chaotic tumbling and useful reorientation alike"),
)
USD_KEYS = ("cube_stiff", "cube_damp", "cube_lin_damp", "cube_ang_damp")

# USD attributes that carry the cube's physics, as (key, on-material?, attribute)
_CUBE_ATTRS = (
    ("lin_damping", False, "physxRigidBody:linearDamping"),
    ("ang_damping", False, "physxRigidBody:angularDamping"),
    ("solver_pos_iters", False, "physxRigidBody:solverPositionIterationCount"),
    ("solver_vel_iters", False, "physxRigidBody:solverVelocityIterationCount"),
    ("max_depenetration_vel", False, "physxRigidBody:maxDepenetrationVelocity"),
    ("sleep_threshold", False, "physxRigidBody:sleepThreshold"),
    ("stabilization_threshold", False, "physxRigidBody:stabilizationThreshold"),
    ("contact_stiffness", True, "physxMaterial:compliantContactStiffness"),
    ("contact_damping", True, "physxMaterial:compliantContactDamping"),
    ("static_friction", True, "physics:staticFriction"),
    ("dynamic_friction", True, "physics:dynamicFriction"),
    ("restitution", True, "physics:restitution"),
)

_STEP = {key: step for key, _, _, _, step, _, _ in SPECS}


def _snap(key, value):
    """Clean float32 read-back noise (0.10000000149 -> 0.1) without moving a value that is
    genuinely off the slider's grid."""
    step = _STEP.get(key)
    if not step:
        return value
    grid = round(float(value) / step) * step
    return round(grid, 6) if abs(grid - float(value)) < 1e-4 else float(value)


def _try(fn, default=None):
    try:
        return fn()
    except Exception:                                   # noqa: BLE001 - provenance is best-effort
        return default


def resolve_cube_prim(stage, cube_prim="/World/Cube"):
    """Path of the cube's rigid body, tolerating the multi-env layout.

    Single-env scripts spawn ``/World/Cube``; anything built on InteractiveScene uses
    ``{ENV_REGEX_NS}/Cube`` -> ``/World/envs/env_0/Cube``. Falling back here keeps a
    recorder from silently stamping a rollout with no cube properties at all."""
    name = cube_prim.rsplit("/", 1)[-1]
    for path in (cube_prim, f"/World/envs/env_0/{name}", f"/World/{name}"):
        prim = _try(lambda p=path: stage.GetPrimAtPath(p))
        if prim is not None and _try(lambda pr=prim: pr.IsValid(), False):
            return path
    return None


def cube_material_prim(stage, cube_prim="/World/Cube"):
    """The cube's physics-material prim (CuboidCfg binds it under <prim>/geometry/material)."""
    from pxr import UsdPhysics
    cube_prim = resolve_cube_prim(stage, cube_prim) or cube_prim
    for p in (f"{cube_prim}/geometry/material", f"{cube_prim}/material"):
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            return prim
    root = stage.GetPrimAtPath(cube_prim)
    for child in (root.GetAllChildren() if root.IsValid() else []):
        for sub in child.GetAllChildren():
            if sub.HasAPI(UsdPhysics.MaterialAPI):
                return sub
    return None


# ---------------------------------------------------------------- provenance
def read_sim_params(sim, robot=None, cube=None, *, cube_prim="/World/Cube",
                    steps_per_frame=1, control_hz=None, extra=None):
    """Everything about the physics a recording was produced under, as a plain dict.

    Read from the RUNNING simulation (PhysX + USD), not from the cfg objects, so it stays
    truthful after gains or cube properties were retuned at runtime by SimParamPanel.
    ``steps_per_frame`` and ``control_hz`` can't be inferred -- the caller's loop owns them.
    Every field is best-effort: a lookup that fails is simply absent.
    """
    dt = _try(lambda: float(sim.get_physics_dt()))
    out = {"physics_dt": dt,
           "steps_per_frame": int(steps_per_frame),
           "physics_hz": round(1.0 / dt, 4) if dt else None,
           "sim_s_per_frame": round(dt * steps_per_frame, 8) if dt else None,
           "control_hz": float(control_hz) if control_hz is not None else None}
    if robot is not None:
        d = robot.data
        out["hand"] = {
            "stiffness": _try(lambda: round(float(d.joint_stiffness[0].mean()), 6)),
            "damping": _try(lambda: round(float(d.joint_damping[0].mean()), 6)),
            "armature": _try(lambda: round(float(d.joint_armature[0].mean()), 6)),
            # the means above describe the hand only if every joint shares the gains
            "uniform_gains": _try(lambda: bool(float(d.joint_stiffness[0].std()) < 1e-6
                                               and float(d.joint_damping[0].std()) < 1e-6)),
            "num_joints": _try(lambda: int(d.joint_pos.shape[1])),
        }
    stage = _try(lambda: sim.stage)
    cube_out = {}
    if stage is not None:
        path = resolve_cube_prim(stage, cube_prim)
        cube_out["prim"] = path or cube_prim
        prim = _try(lambda: stage.GetPrimAtPath(path)) if path else None
        mat = _try(lambda: cube_material_prim(stage, cube_prim))
        for key, on_material, attr in _CUBE_ATTRS:
            holder = mat if on_material else prim
            if holder is None or not _try(lambda h=holder: h.IsValid(), False):
                continue
            val = _try(lambda h=holder, a=attr: h.GetAttribute(a).Get())
            if val is not None:
                cube_out[key] = round(float(val), 6) if isinstance(val, float) else val
    if cube is not None:
        cube_out["mass"] = _try(lambda: round(float(cube.root_physx_view.get_masses()[0]), 6))
    out["cube"] = cube_out
    out["physx"] = {
        "gravity": _try(lambda: [round(float(g), 4) for g in sim.cfg.gravity]),
        "enable_stabilization": _try(lambda: bool(sim.cfg.physx.enable_stabilization)),
        "bounce_threshold_velocity": _try(lambda: float(sim.cfg.physx.bounce_threshold_velocity)),
    }
    out["device"] = _try(lambda: str(sim.device), "")
    if extra:
        out.update(extra)
    return out


def params_json(params):
    """Canonical JSON (sorted keys) -- the form stored in .npz and hashed."""
    return json.dumps(params, sort_keys=True, default=str)


def params_hash(params):
    """Short stable id of a physics config: equal hashes == same simulator."""
    return hashlib.sha1(params_json(params).encode()).hexdigest()[:12]


def summarize(params):
    """One-line human summary. Accepts the dict or the stored JSON string (no Isaac needed)."""
    p = json.loads(params) if isinstance(params, (str, bytes)) else dict(params or {})
    h = p.get("hand") or {}
    c = p.get("cube") or {}

    def g(d, k, fmt="{:g}"):
        v = d.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) else "?"
    return (f"{g(p, 'physics_hz')}Hz x{g(p, 'steps_per_frame')}"
            f" · hand k={g(h, 'stiffness')} d={g(h, 'damping')} arm={g(h, 'armature')}"
            f" · cube lin/ang {g(c, 'lin_damping')}/{g(c, 'ang_damping')}"
            f" contact {g(c, 'contact_stiffness')}/{g(c, 'contact_damping')}"
            f" mu {g(c, 'static_friction')}/{g(c, 'dynamic_friction')}"
            f" m={g(c, 'mass')}")


def stamp(data, sim, robot=None, cube=None, **kw):
    """Add ``sim_params_json`` + ``sim_params_hash`` to a dict headed for np.savez.

    One call is all a recorder needs:
        stamp(data, sim, robot, cube, steps_per_frame=1, control_hz=args.hz)
    """
    import numpy as np
    p = read_sim_params(sim, robot, cube, **kw)
    data["sim_params_json"] = np.array(params_json(p))
    data["sim_params_hash"] = np.array(params_hash(p))
    return p


class SimParamPanel:
    """Simulator parameters as viser sliders over a live ``(sim, robot, cube)``.

    ``srv=None`` builds no GUI -- the parameter/apply logic alone, which is what the
    headless self-test drives (and what a caller uses before its viser server is up).

    Call :meth:`poll` once per control frame from the SAME thread that steps the sim
    (viser callbacks only queue; nothing touches Isaac from the GUI thread), and step
    :attr:`substeps` times per frame.
    """

    def __init__(self, srv, sim, robot, cube, *, cube_prim="/World/Cube",
                 folder="Simulator", overrides=None, debounce=0.25,
                 after_live=None, after_restart=None, reset_label="Reset to launch values",
                 baseline_note="matches the launch configuration", verbose=True):
        self.srv, self.sim, self.robot, self.cube = srv, sim, robot, cube
        self.cube_prim = cube_prim
        self.debounce = float(debounce)
        self.after_live = after_live            # called after a live-knob apply
        self.after_restart = after_restart      # replaces the state restore after a bounce
        self.baseline_note = baseline_note
        self.verbose = verbose
        self.params = self._read_scene(overrides or {})
        self.launch = dict(self.params)
        self._applied = dict(self.params)
        self._cmds = deque()
        self._usd_at = 0.0                      # >0: a USD knob changed and is settling
        self._widgets = {}
        self._listeners = []
        self.n_changes = 0                      # knob edits since construction
        if srv is not None:
            self._build_gui(folder, reset_label)

    # ------------------------------------------------------------------ state
    @property
    def substeps(self):
        """Physics steps the caller should run per control frame."""
        return max(int(self.params["substeps"]), 1)

    @property
    def modified(self):
        return any(self.params[k] != self.launch[k] for k in self.params)

    def on_change(self, cb):
        """Register ``cb(params)`` for every applied knob edit (e.g. to flag a recording)."""
        self._listeners.append(cb)

    def _read_scene(self, overrides):
        """Seed from the running scene so an untouched panel is a no-op."""
        full = read_sim_params(self.sim, self.robot, self.cube, cube_prim=self.cube_prim)
        p = dict(LAUNCH)
        # dt alone can't be split into rate x substeps -- the caller's loop owns the split, so
        # take substeps from the override (or the project default) and derive the rate from it.
        if overrides.get("substeps") is not None:
            p["substeps"] = int(overrides["substeps"])
        if overrides.get("phys_hz") is None and full.get("physics_dt"):
            p["phys_hz"] = round(1.0 / (full["physics_dt"] * max(p["substeps"], 1)))
        for src, dst in (("stiffness", "stiffness"), ("damping", "damping"),
                         ("armature", "armature")):
            v = (full.get("hand") or {}).get(src)
            if v is not None:
                p[dst] = float(v)
        for src, dst in (("contact_stiffness", "cube_stiff"), ("contact_damping", "cube_damp"),
                         ("lin_damping", "cube_lin_damp"), ("ang_damping", "cube_ang_damp")):
            v = (full.get("cube") or {}).get(src)
            if v is not None:
                p[dst] = float(v)
        # explicit caller values (CLI flags) win; None means "leave whatever the scene has"
        for k, v in overrides.items():
            if v is not None and k in p:
                p[k] = int(v) if k == "substeps" else float(v)
        return {k: _snap(k, v) for k, v in p.items()}

    # ------------------------------------------------------------------- apply
    def set_phys_dt(self, dt):
        try:
            self.sim.get_physics_context().set_physics_dt(float(dt))
        except Exception:                       # older/newer API: whole-sim setter
            self.sim.set_simulation_dt(physics_dt=float(dt), rendering_dt=float(dt))
        self.sim.cfg.dt = float(dt)             # keep IsaacLab's own view in sync

    def apply_live(self):
        """Gains + physics dt: straight into PhysX, effective on the next step."""
        p = self.params
        self.robot.write_joint_stiffness_to_sim(float(p["stiffness"]))
        self.robot.write_joint_damping_to_sim(float(p["damping"]))
        self.robot.write_joint_armature_to_sim(float(p["armature"]))
        self.set_phys_dt(1.0 / (float(p["phys_hz"]) * self.substeps))

    def write_usd(self):
        """Cube material / body properties into USD (PhysX picks them up on the bounce)."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim import schemas
        from pxr import PhysxSchema
        p = self.params
        stage = self.sim.stage
        schemas.modify_rigid_body_properties(self.cube_prim, sim_utils.RigidBodyPropertiesCfg(
            linear_damping=float(p["cube_lin_damp"]),
            angular_damping=float(p["cube_ang_damp"])), stage=stage)
        prim = cube_material_prim(stage, self.cube_prim)
        if prim is None:
            print("[simparams] WARN: cube physics material prim not found -- contact knobs inert")
            return
        api = (PhysxSchema.PhysxMaterialAPI(prim) if prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
               else PhysxSchema.PhysxMaterialAPI.Apply(prim))
        api.CreateCompliantContactStiffnessAttr().Set(float(p["cube_stiff"]))
        if p["cube_stiff"] > 0.0:      # PhysX warns (and ignores it) when compliance is off
            api.CreateCompliantContactDampingAttr().Set(float(p["cube_damp"]))

    def restart(self):
        """Write USD, bounce the timeline, put the scene back where it was.

        ``sim.reset()`` is stop+play, which is the only thing that makes PhysX re-parse the
        stage -- and it also snaps assets back to their spawn pose, hence the snapshot."""
        t0 = time.time()
        jp = self.robot.data.joint_pos.clone()
        jv = self.robot.data.joint_vel.clone()
        cs = self.cube.data.root_state_w.clone()
        self.write_usd()
        self.sim.reset()
        self.apply_live()                       # gains/dt live in PhysX -> re-write them
        if self.after_restart is not None:
            self.after_restart()                # caller has a better anchor (e.g. a recording)
        else:
            self.robot.write_joint_state_to_sim(jp, jv)
            self.robot.reset()
            self.cube.write_root_pose_to_sim(cs[:, :7])
            self.cube.write_root_velocity_to_sim(cs[:, 7:13])
            self.cube.reset()
        if self.verbose:
            print(f"[simparams] cube properties applied, sim restarted ({time.time() - t0:.1f}s)")

    def poll(self):
        """Apply queued edits. Call once per control frame. True if the sim was restarted."""
        while self._cmds:
            self._cmds.popleft()
            self.apply_live()
            if any(self.params[k] != self._applied[k] for k in USD_KEYS):
                # a slider drag emits continuously and each cube change costs a timeline
                # bounce, so coalesce: restart once the value settles
                self._usd_at = time.time()
            elif self.after_live is not None:
                self.after_live()
            self._applied.update({k: v for k, v in self.params.items() if k not in USD_KEYS})
            self.n_changes += 1
            for cb in self._listeners:
                cb(self.params)
            self._refresh_md()
        if self._usd_at and time.time() - self._usd_at > self.debounce:
            self.restart()
            self._applied.update(self.params)
            self._usd_at = 0.0
            self._refresh_md()
            return True
        return False

    # --------------------------------------------------------------------- gui
    def summary(self):
        dt = 1.0 / (self.params["phys_hz"] * self.substeps)
        state = "**modified**" if self.modified else f"**{self.baseline_note}**"
        return (f"physics dt&nbsp;**{dt * 1e3:.3f}&nbsp;ms**&nbsp;({1 / dt:.0f}&nbsp;Hz)"
                f"&nbsp;·&nbsp;{self.substeps}&nbsp;step(s)&nbsp;per&nbsp;frame\n\n{state}")

    def _refresh_md(self):
        if self._widgets.get("md") is not None:
            self._widgets["md"].content = self.summary()

    def attach(self, srv, folder="Simulator", reset_label="Reset to launch values"):
        """Build the widgets after construction -- for callers whose viser server comes up
        later than the sim (the panel is fully usable headless until then)."""
        self.srv = srv
        self._build_gui(folder, reset_label)
        return self

    def _build_gui(self, folder, reset_label):
        gui = self.srv.gui
        with gui.add_folder(folder):
            self._widgets["md"] = gui.add_markdown(self.summary())
            for key, label, lo, hi, step, is_int, hint in SPECS:
                val = self.params[key]
                cast = int if is_int else float
                w = gui.add_slider(label, cast(lo), cast(hi), cast(step),
                                   cast(min(max(val, lo), hi)), hint=hint or None)
                self._widgets[key] = w
                w.on_update(self._on_update)
            btn = gui.add_button(reset_label)
            btn.on_click(self._on_reset)

    def _pull_widgets(self):
        for key, _, _, _, _, is_int, _ in SPECS:
            w = self._widgets.get(key)
            if w is not None:
                self.params[key] = int(w.value) if is_int else float(w.value)

    def _on_update(self, event=None):
        if event is not None and event.client is None:
            return                              # server-side .value set: not a user edit
        self._pull_widgets()
        self._refresh_md()                      # instant feedback; the sim follows in poll()
        self._cmds.append("params")

    def _on_reset(self, _event=None):
        for key, _, _, _, _, is_int, _ in SPECS:
            w = self._widgets.get(key)
            if w is not None:
                w.value = int(self.launch[key]) if is_int else float(self.launch[key])
        self._on_update()
