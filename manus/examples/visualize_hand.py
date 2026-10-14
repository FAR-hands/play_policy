#!/usr/bin/env python3
"""Visualize Manus glove hand keypoints in 3D using viser.

Usage:
    # Default: both hands, connect to localhost
    python visualize_hand.py

    # Right hand only, remote server
    python visualize_hand.py --side right --host 192.168.1.100
"""

import argparse
import time

import numpy as np
import viser

from manus_glove import ManusGlove
from manus_glove.client import GloveData

# Colors per finger chain (RGB 0-255)
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
LABEL_Y_OFFSET = 0.15  # how far above the hand root to place the label
HAND_X_OFFSET = 0.15  # x-axis offset between left and right hands

SIDE_OFFSETS = {
    "left": np.array([HAND_X_OFFSET, 0.0, 0.0], dtype=np.float32),
    "right": np.array([-HAND_X_OFFSET, 0.0, 0.0], dtype=np.float32),
}

# Cache for sphere handles so we can update position without re-creating
_sphere_handles: dict[str, object] = {}


def _color_for_chain(chain_type: str) -> tuple:
    return CHAIN_COLORS.get(chain_type, (180, 180, 180))


def _update_hand(
    server: viser.ViserServer,
    prefix: str,
    data: GloveData,
    offset: np.ndarray,
):
    """Update sphere and line visualizations for one hand."""
    nodes = data.raw_nodes
    if not nodes:
        return

    # Build lookup: node_id -> node
    node_by_id = {n.node_id: n for n in nodes}

    # Negate x to convert from Manus SDK coordinate convention to viser
    def _to_vis(pos: np.ndarray) -> np.ndarray:
        p = pos.astype(np.float32).copy()
        p[0] = -p[0]
        return p + offset

    # Draw/update spheres at each joint
    for node in nodes:
        pos = _to_vis(node.position)
        name = f"{prefix}/joints/{node.chain_type}_{node.joint_type}_{node.node_id}"
        if name not in _sphere_handles:
            color = _color_for_chain(node.chain_type)
            _sphere_handles[name] = server.scene.add_icosphere(
                name=name,
                radius=SPHERE_RADIUS,
                color=color,
                position=pos,
            )
        else:
            _sphere_handles[name].position = pos

    # Draw bones (lines from each node to its parent)
    # Group line segments by chain for efficient rendering
    chain_segments: dict[str, list] = {}
    for node in nodes:
        if node.parent_node_id < 0:
            continue
        parent = node_by_id.get(node.parent_node_id)
        if parent is None:
            continue
        chain = node.chain_type
        if chain not in chain_segments:
            chain_segments[chain] = []
        chain_segments[chain].append(
            (_to_vis(parent.position), _to_vis(node.position))
        )

    for chain, segments in chain_segments.items():
        if not segments:
            continue
        points = np.array(segments, dtype=np.float32)  # (N, 2, 3)
        color = _color_for_chain(chain)
        color_arr = np.tile(
            np.array(color, dtype=np.uint8), (points.shape[0], 2, 1)
        )  # (N, 2, 3)
        server.scene.add_line_segments(
            name=f"{prefix}/bones/{chain}",
            points=points,
            colors=color_arr,
            line_width=LINE_WIDTH,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize Manus glove hand keypoints")
    parser.add_argument("--host", default="localhost", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    parser.add_argument(
        "--side", choices=["left", "right", "both"], default="both",
        help="Which hand(s) to visualize",
    )
    parser.add_argument("--hz", type=float, default=60.0, help="Visualization update rate")
    parser.add_argument("--viser-port", type=int, default=8080, help="Viser web server port")
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    server = viser.ViserServer(port=args.viser_port)
    server.scene.world_axes.visible = False
    print(f"Viser running at http://localhost:{args.viser_port}")

    # Add a ground grid
    server.scene.add_grid(
        name="/grid",
        width=1.0,
        height=1.0,
        cell_size=0.05,
        position=(0.0, 0.0, 0.0),
    )

    # Add labels above each hand
    for side in sides:
        offset = SIDE_OFFSETS.get(side, np.zeros(3, dtype=np.float32))
        label_pos = offset + np.array([0.0, LABEL_Y_OFFSET, 0.0], dtype=np.float32)
        server.scene.add_label(
            name=f"/{side}_hand/label",
            text=f"{side.capitalize()} Hand",
            position=label_pos,
        )

    glove = ManusGlove(host=args.host, pub_port=args.port, sides=sides)
    glove.connect()
    print(f"Connected to Manus server at {args.host}:{args.port}, waiting for data...")

    interval = 1.0 / args.hz

    # Manus SDK reports left/right swapped relative to visual expectation
    VISUAL_SIDE = {"left": "left", "right": "right"}

    try:
        while True:
            for side in sides:
                data = glove.get_data(side)
                if data is None:
                    continue
                vis_side = VISUAL_SIDE.get(side, side)
                offset = SIDE_OFFSETS.get(vis_side, np.zeros(3, dtype=np.float32))
                _update_hand(server, prefix=f"/{vis_side}_hand", data=data, offset=offset)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        glove.disconnect()


if __name__ == "__main__":
    main()
