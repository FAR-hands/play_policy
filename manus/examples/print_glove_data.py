#!/usr/bin/env python3
"""Print live Manus glove data from the ZMQ server.

Usage:
    # Default: connect to localhost:5555, print both hands
    python print_glove_data.py

    # Connect to remote server, right hand only
    python print_glove_data.py --host 192.168.1.100 --side right

    # Print fingertip positions only
    python print_glove_data.py --tips
"""

import argparse
import time

from manus_glove import ManusGlove


def main():
    parser = argparse.ArgumentParser(description="Print Manus glove data")
    parser.add_argument("--host", default="localhost", help="ZMQ server host")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--tips", action="store_true", help="Print only fingertip positions")
    parser.add_argument("--ergo", action="store_true", help="Print ergonomics data")
    parser.add_argument("--hz", type=float, default=30.0, help="Print rate (Hz)")
    args = parser.parse_args()

    sides = ["left", "right"] if args.side == "both" else [args.side]

    with ManusGlove(host=args.host, pub_port=args.port, sides=sides) as glove:
        print(f"Connected to {args.host}:{args.port}, waiting for data...")
        interval = 1.0 / args.hz

        while True:
            for side in sides:
                data = glove.get_data(side)
                if data is None:
                    continue

                if args.tips:
                    tips = data.get_fingertip_positions()
                    print(f"[{side.upper()}] Fingertips:")
                    for finger, pos in tips.items():
                        print(f"  {finger}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                elif args.ergo:
                    ergo = data.get_ergonomics_dict()
                    print(f"[{side.upper()}] Ergonomics:")
                    for name, val in ergo.items():
                        print(f"  {name}: {val:.3f}")
                else:
                    print(
                        f"[{side.upper()}] glove_id={data.glove_id} "
                        f"nodes={len(data.raw_nodes)} "
                        f"ergo={len(data.ergonomics)} "
                        f"t={data.timestamp:.3f}"
                    )

            time.sleep(interval)


if __name__ == "__main__":
    main()
