#!/usr/bin/env python3
"""Interactive Manus glove calibration with a live 3D hand view in viser.

The C++ manus_zmq_server owns the Manus device and runs the calibration step
sequence; this script subscribes to its status topic, renders the live hand so
you can see your pose, and drives the flow (start / next / abort / save) through
the server's control socket.

Usage:
    # Start the server first (it owns the SDK + device):
    #   ./server/build/manus_zmq_server

    # Then launch this UI and open the printed viser URL:
    python calibrate_hand.py --side right

    # Remote server:
    python calibrate_hand.py --side right --host 192.168.1.100
"""

import argparse
import time

import numpy as np
import viser

from manus_glove import ManusCalibration, ManusGlove
from manus_glove.client import CalibrationStatus, GloveData

# Colors per finger chain (RGB 0-255), matching visualize_hand.py.
CHAIN_COLORS = {
    "Hand": (200, 200, 200),
    "Thumb": (255, 100, 100),
    "Index": (100, 255, 100),
    "Middle": (100, 100, 255),
    "Ring": (255, 255, 100),
    "Pinky": (255, 100, 255),
}

SPHERE_RADIUS = 0.004
LINE_WIDTH = 3.0

_sphere_handles: dict[str, object] = {}


def _update_hand(server: viser.ViserServer, prefix: str, data: GloveData):
    """Draw/update spheres + bones for one hand (Manus x is negated for viser)."""
    nodes = data.raw_nodes
    if not nodes:
        return
    node_by_id = {n.node_id: n for n in nodes}

    def _to_vis(pos: np.ndarray) -> np.ndarray:
        p = pos.astype(np.float32).copy()
        p[0] = -p[0]
        return p

    for node in nodes:
        name = f"{prefix}/joints/{node.chain_type}_{node.joint_type}_{node.node_id}"
        pos = _to_vis(node.position)
        if name not in _sphere_handles:
            _sphere_handles[name] = server.scene.add_icosphere(
                name=name,
                radius=SPHERE_RADIUS,
                color=CHAIN_COLORS.get(node.chain_type, (180, 180, 180)),
                position=pos,
            )
        else:
            _sphere_handles[name].position = pos

    chain_segments: dict[str, list] = {}
    for node in nodes:
        if node.parent_node_id < 0:
            continue
        parent = node_by_id.get(node.parent_node_id)
        if parent is None:
            continue
        chain_segments.setdefault(node.chain_type, []).append(
            (_to_vis(parent.position), _to_vis(node.position))
        )

    for chain, segments in chain_segments.items():
        points = np.array(segments, dtype=np.float32)  # (N, 2, 3)
        color = CHAIN_COLORS.get(chain, (180, 180, 180))
        colors = np.tile(np.array(color, dtype=np.uint8), (points.shape[0], 2, 1))
        server.scene.add_line_segments(
            name=f"{prefix}/bones/{chain}",
            points=points,
            colors=colors,
            line_width=LINE_WIDTH,
        )


def _icon(name: str):
    """Return viser.Icon.<name> if available, else None (icons are optional)."""
    return getattr(viser.Icon, name, None)


def _progress_bar(fraction: float, width: int = 24) -> str:
    """Render a unicode text progress bar for a 0-1 fraction."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "▰" * filled + "▱" * (width - filled)


def _status_markdown(status: CalibrationStatus | None) -> str:
    """Build the markdown shown in the status panel."""
    if status is None:
        return "**Waiting for server...**\n\nIs `manus_zmq_server` running?"

    lines = [f"**State:** `{status.state}`"]

    if status.is_active and status.num_steps > 0:
        lines.append(f"**Step {status.step_index + 1} / {status.num_steps}**")
        if status.title:
            lines.append(f"### {status.title}")
        if status.description:
            lines.append(status.description)

        if status.manual:
            lines.append("\n_Hold the pose, then press **Next →**_")
            if status.step_time > 0:
                lines.append(f"(estimated ~{status.step_time:.0f}s)")
        elif status.step_time > 0:
            frac = 1.0 - (status.remaining / status.step_time)
            lines.append(f"\n`{_progress_bar(frac)}`  {status.remaining:0.1f}s")
    elif status.state == "done":
        lines.append("### ✅ Calibration complete")
    elif status.state == "aborted":
        lines.append("### ⚠️ Calibration aborted")
    elif status.state == "error":
        lines.append("### ❌ Error")

    if status.message:
        lines.append(f"\n_{status.message}_")
    return "\n\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Interactive Manus glove calibration")
    parser.add_argument("--host", default="localhost", help="ZMQ server host")
    parser.add_argument("--pub-port", type=int, default=5555, help="ZMQ PUB port")
    parser.add_argument("--control-port", type=int, default=5557, help="ZMQ control port")
    parser.add_argument(
        "--side", choices=["left", "right"], default="right",
        help="Which hand to calibrate / visualize",
    )
    parser.add_argument("--hz", type=float, default=60.0, help="UI update rate")
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser web port")
    args = parser.parse_args()

    server = viser.ViserServer(port=args.viser_port)
    server.scene.world_axes.visible = False
    server.scene.add_grid(name="/grid", width=1.0, height=1.0, cell_size=0.05)
    print(f"Viser running at http://localhost:{args.viser_port}")

    glove = ManusGlove(host=args.host, pub_port=args.pub_port, sides=[args.side])
    calib = ManusCalibration(
        host=args.host, pub_port=args.pub_port, control_port=args.control_port
    )
    glove.connect()
    calib.connect()
    print(f"Connected to {args.host} (data:{args.pub_port} control:{args.control_port})")

    # --- GUI panel ---
    server.gui.add_markdown(f"## Manus Calibration — {args.side.capitalize()} hand")
    side_dropdown = server.gui.add_dropdown(
        "Hand", options=("left", "right"), initial_value=args.side
    )
    status_md = server.gui.add_markdown(_status_markdown(None))

    start_btn = server.gui.add_button("▶ Start", icon=_icon("PLAYER_PLAY"))
    next_btn = server.gui.add_button("Next →", icon=_icon("ARROW_RIGHT"))
    abort_btn = server.gui.add_button("■ Abort", icon=_icon("PLAYER_STOP"), color="red")

    with server.gui.add_folder("Save / Load (server-side path)"):
        path_text = server.gui.add_text("Path", initial_value="calibration.bin")
        save_btn = server.gui.add_button("💾 Save", icon=_icon("DEVICE_FLOPPY"))
        load_btn = server.gui.add_button("📂 Load", icon=_icon("FOLDER_OPEN"))

    start_btn.on_click(lambda _: calib.start(side_dropdown.value))
    next_btn.on_click(lambda _: calib.next())
    abort_btn.on_click(lambda _: calib.abort())
    save_btn.on_click(lambda _: calib.save(path_text.value, side_dropdown.value))
    load_btn.on_click(lambda _: calib.load(path_text.value, side_dropdown.value))

    interval = 1.0 / args.hz
    try:
        while True:
            data = glove.get_data(args.side)
            if data is not None:
                _update_hand(server, prefix=f"/{args.side}_hand", data=data)

            status = calib.get_status()
            status_md.content = _status_markdown(status)

            # Only enable Next during a manual (continuous) step.
            next_btn.disabled = not (status is not None and status.manual)
            # Disable Start while a sequence is already running.
            start_btn.disabled = status is not None and status.is_active

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        calib.disconnect()
        glove.disconnect()


if __name__ == "__main__":
    main()
