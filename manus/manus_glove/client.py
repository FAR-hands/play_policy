"""ZMQ client for reading Manus glove data from the manus_zmq_server."""

import json
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import zmq


@dataclass
class RawNode:
    """A single skeleton node from the Manus glove."""

    node_id: int
    parent_node_id: int
    joint_type: str
    chain_type: str
    position: np.ndarray  # (3,) xyz
    orientation: np.ndarray  # (4,) xyzw quaternion


@dataclass
class Ergonomics:
    """A single ergonomics measurement."""

    type: str
    value: float


@dataclass
class GloveData:
    """Complete data snapshot for one glove."""

    glove_id: int
    side: str  # "Left" or "Right"
    timestamp: float
    raw_nodes: List[RawNode] = field(default_factory=list)
    ergonomics: List[Ergonomics] = field(default_factory=list)
    raw_sensor_orientation: Optional[np.ndarray] = None  # (4,) xyzw
    raw_sensors: Optional[List[dict]] = None

    def get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        """Return {finger_name: position} for TIP nodes only."""
        tips = {}
        for node in self.raw_nodes:
            if node.joint_type == "TIP":
                tips[node.chain_type] = node.position
        return tips

    def get_node_positions(self) -> np.ndarray:
        """Return (N, 3) array of all node positions."""
        return np.array([n.position for n in self.raw_nodes])

    def get_node_orientations(self) -> np.ndarray:
        """Return (N, 4) array of all node orientations (xyzw)."""
        return np.array([n.orientation for n in self.raw_nodes])

    def get_ergonomics_dict(self) -> Dict[str, float]:
        """Return {type_name: value} dict for all ergonomics data."""
        return {e.type: e.value for e in self.ergonomics}


def _parse_glove_data(msg: dict) -> GloveData:
    """Parse a JSON message dict into a GloveData dataclass."""
    nodes = []
    for n in msg.get("raw_nodes", []):
        nodes.append(RawNode(
            node_id=n["node_id"],
            parent_node_id=n.get("parent_node_id", -1),
            joint_type=n.get("joint_type", ""),
            chain_type=n.get("chain_type", ""),
            position=np.array(n["position"], dtype=np.float64),
            orientation=np.array(n["orientation"], dtype=np.float64),
        ))

    ergonomics = []
    for e in msg.get("ergonomics", []):
        ergonomics.append(Ergonomics(type=e["type"], value=e["value"]))

    raw_sensor_orientation = None
    raw_sensors = None
    if "raw_sensors" in msg:
        rs = msg["raw_sensors"]
        raw_sensor_orientation = np.array(rs["orientation"], dtype=np.float64)
        raw_sensors = rs.get("sensors", [])

    return GloveData(
        glove_id=msg["glove_id"],
        side=msg["side"],
        timestamp=msg["timestamp"],
        raw_nodes=nodes,
        ergonomics=ergonomics,
        raw_sensor_orientation=raw_sensor_orientation,
        raw_sensors=raw_sensors,
    )


@dataclass
class CalibrationStatus:
    """Snapshot of the server-side glove calibration state machine."""

    state: str  # idle|preparing|running_step|waiting_user|finishing|done|aborted|error
    side: str
    glove_id: int
    step_index: int
    num_steps: int
    title: str
    description: str
    step_time: float  # nominal step duration (seconds)
    remaining: float  # countdown for timed steps (seconds)
    manual: bool  # True => continuous step, waiting for next()
    message: str

    @property
    def is_active(self) -> bool:
        """True while a calibration sequence is running."""
        return self.state in ("preparing", "running_step", "waiting_user", "finishing")

    @property
    def is_terminal(self) -> bool:
        """True once a sequence has ended (done/aborted/error)."""
        return self.state in ("done", "aborted", "error")


def _parse_calibration_status(msg: dict) -> CalibrationStatus:
    return CalibrationStatus(
        state=msg.get("state", "idle"),
        side=msg.get("side", ""),
        glove_id=msg.get("glove_id", 0),
        step_index=msg.get("step_index", 0),
        num_steps=msg.get("num_steps", 0),
        title=msg.get("title", ""),
        description=msg.get("description", ""),
        step_time=msg.get("step_time", 0.0),
        remaining=msg.get("remaining", 0.0),
        manual=msg.get("manual", False),
        message=msg.get("message", ""),
    )


class ManusCalibration:
    """ZMQ client for driving and monitoring Manus glove calibration.

    Subscribes to the ``manus_calibration`` status topic published by the
    manus_zmq_server and sends control commands (start/next/abort/save/load)
    to the server's control PULL socket.

    The actual SDK calibration calls run inside the C++ server (the single
    owner of the Manus device connection); this client only orchestrates and
    visualizes them.

    Args:
        host: Hostname or IP of the machine running manus_zmq_server.
        pub_port: ZMQ PUB port on the server (default 5555).
        control_port: ZMQ control port on the server (default 5557).

    Example::

        from manus_glove import ManusCalibration

        with ManusCalibration() as calib:
            calib.start("right")
            while True:
                status = calib.get_status()
                if status and status.is_terminal:
                    print(status.message)
                    break
    """

    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5555,
        control_port: int = 5557,
    ):
        self._host = host
        self._pub_port = pub_port
        self._control_port = control_port

        self._context: Optional[zmq.Context] = None
        self._sub_socket: Optional[zmq.Socket] = None
        self._control_socket: Optional[zmq.Socket] = None

        self._lock = threading.Lock()
        self._status: Optional[CalibrationStatus] = None
        self._callback: Optional[Callable[[CalibrationStatus], None]] = None

        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self):
        """Connect to the server and start receiving calibration status."""
        self._context = zmq.Context()

        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.connect(f"tcp://{self._host}:{self._pub_port}")
        self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "manus_calibration")

        self._control_socket = self._context.socket(zmq.PUSH)
        self._control_socket.connect(f"tcp://{self._host}:{self._control_port}")

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def disconnect(self):
        """Stop receiving and close all sockets."""
        self._running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._sub_socket is not None:
            self._sub_socket.close()
            self._sub_socket = None
        if self._control_socket is not None:
            self._control_socket.close()
            self._control_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def get_status(self) -> Optional[CalibrationStatus]:
        """Return the latest calibration status, or None if none received yet."""
        with self._lock:
            return self._status

    def set_callback(self, callback: Optional[Callable[[CalibrationStatus], None]]):
        """Set a callback invoked on every new status message (from recv thread)."""
        self._callback = callback

    def _send(self, msg: dict):
        if self._control_socket is not None:
            self._control_socket.send_string(json.dumps(msg))

    def start(self, side: str = "right"):
        """Begin the calibration sequence for the given hand side."""
        self._send({"cmd": "start", "side": side.lower()})

    def next(self):
        """Advance a continuous (manual) step once the pose is held."""
        self._send({"cmd": "next"})

    def abort(self):
        """Abort the in-progress calibration sequence."""
        self._send({"cmd": "abort"})

    def save(self, path: str, side: str = "right"):
        """Ask the server to save the side's active calibration to ``path``.

        Note: ``path`` is resolved on the server machine's filesystem.
        """
        self._send({"cmd": "save", "side": side.lower(), "path": path})

    def load(self, path: str, side: str = "right"):
        """Ask the server to load a calibration blob from ``path`` (server-side)."""
        self._send({"cmd": "load", "side": side.lower(), "path": path})

    def _recv_loop(self):
        poller = zmq.Poller()
        poller.register(self._sub_socket, zmq.POLLIN)

        while self._running:
            events = dict(poller.poll(timeout=100))
            if self._sub_socket not in events:
                continue
            try:
                topic_bytes = self._sub_socket.recv()  # noqa: F841 (topic ignored)
                payload_bytes = self._sub_socket.recv()
                data = json.loads(payload_bytes.decode("utf-8"))
                status = _parse_calibration_status(data)

                with self._lock:
                    self._status = status

                cb = self._callback
                if cb is not None:
                    try:
                        cb(status)
                    except Exception as e:
                        print(f"Callback error: {e}")
            except zmq.ZMQError:
                if not self._running:
                    break
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Parse error: {e}")


class ManusGlove:
    """ZMQ client for receiving Manus glove data.

    Connects to the manus_zmq_server and receives glove skeleton,
    ergonomics, and raw sensor data at ~120Hz.

    Args:
        host: Hostname or IP of the machine running manus_zmq_server.
        pub_port: ZMQ PUB port on the server (default 5555).
        haptic_port: ZMQ port for sending haptic commands (default 5556).
        sides: Which glove sides to subscribe to. Default is both.

    Example::

        from manus_glove import ManusGlove

        with ManusGlove() as glove:
            while True:
                data = glove.get_data("right")
                if data:
                    tips = data.get_fingertip_positions()
                    print(tips)
    """

    def __init__(
        self,
        host: str = "localhost",
        pub_port: int = 5555,
        haptic_port: int = 5556,
        sides: Optional[List[str]] = None,
    ):
        self._host = host
        self._pub_port = pub_port
        self._haptic_port = haptic_port
        self._sides = sides or ["left", "right"]

        self._context: Optional[zmq.Context] = None
        self._sub_socket: Optional[zmq.Socket] = None
        self._haptic_socket: Optional[zmq.Socket] = None

        self._lock = threading.Lock()
        self._latest: Dict[str, GloveData] = {}
        self._callback: Optional[Callable[[GloveData], None]] = None

        self._recv_thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self):
        """Connect to the ZMQ server and start receiving data."""
        self._context = zmq.Context()

        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.connect(f"tcp://{self._host}:{self._pub_port}")
        for side in self._sides:
            topic = f"manus_glove_{side}"
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)

        self._haptic_socket = self._context.socket(zmq.PUSH)
        self._haptic_socket.connect(f"tcp://{self._host}:{self._haptic_port}")

        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def disconnect(self):
        """Stop receiving and close all sockets."""
        self._running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._sub_socket is not None:
            self._sub_socket.close()
            self._sub_socket = None
        if self._haptic_socket is not None:
            self._haptic_socket.close()
            self._haptic_socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def get_data(self, side: str = "right") -> Optional[GloveData]:
        """Get the latest data for the given hand side.

        Args:
            side: "left" or "right" (case-insensitive).

        Returns:
            GloveData if data has been received, else None.
        """
        key = side.lower()
        with self._lock:
            return self._latest.get(key)

    def get_both(self) -> Dict[str, Optional[GloveData]]:
        """Get latest data for both hands.

        Returns:
            Dict with keys "left" and "right", values are GloveData or None.
        """
        with self._lock:
            return {
                "left": self._latest.get("left"),
                "right": self._latest.get("right"),
            }

    def set_callback(self, callback: Optional[Callable[[GloveData], None]]):
        """Set a callback invoked on every new glove message.

        The callback receives a GloveData and is called from the receive thread.
        Set to None to remove.
        """
        self._callback = callback

    def send_haptic(
        self,
        left_fingers: Optional[List[float]] = None,
        right_fingers: Optional[List[float]] = None,
    ):
        """Send haptic vibration commands to the gloves.

        Args:
            left_fingers: 5 float values [thumb, index, middle, ring, pinky] in 0-1.
            right_fingers: 5 float values [thumb, index, middle, ring, pinky] in 0-1.
        """
        if self._haptic_socket is None:
            return

        msg = {}
        if left_fingers is not None:
            msg["left_fingers"] = [float(v) for v in left_fingers[:5]]
        if right_fingers is not None:
            msg["right_fingers"] = [float(v) for v in right_fingers[:5]]

        if msg:
            self._haptic_socket.send_string(json.dumps(msg))

    def _recv_loop(self):
        """Background thread: receive ZMQ messages and update latest data."""
        poller = zmq.Poller()
        poller.register(self._sub_socket, zmq.POLLIN)

        while self._running:
            events = dict(poller.poll(timeout=100))  # 100ms timeout
            if self._sub_socket not in events:
                continue

            try:
                topic_bytes = self._sub_socket.recv()
                payload_bytes = self._sub_socket.recv()
                topic = topic_bytes.decode("utf-8")
                data = json.loads(payload_bytes.decode("utf-8"))
                glove_data = _parse_glove_data(data)

                side_key = "left" if "left" in topic else "right"
                with self._lock:
                    self._latest[side_key] = glove_data

                cb = self._callback
                if cb is not None:
                    try:
                        cb(glove_data)
                    except Exception as e:
                        print(f"Callback error: {e}")

            except zmq.ZMQError:
                if not self._running:
                    break
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Parse error: {e}")
