"""Drive the ABC sim robot over Adamo with the abc-sim fine-tuned GR00T N1.7 policy (in-process).

Adapted from openpi/examples/abc_sim/adamo_client.py (same protocol), with the openpi websocket call
replaced by Gr00tPolicy.get_action. Needs adamo[video]>=1.0.9 (announce_control) in the GR00T venv.

    ADAMO_API_KEY=... python runpod/abc_sim/adamo_bridge.py --model-path <checkpoint-2000> --robot abc-inference
    python runpod/abc_sim/adamo_bridge.py --model-path <checkpoint-2000> --warmup-only   # no robot, no control

It claims the robot the way operate.adamohq.com does (acquired, then a heartbeat a second on
{robot}/control/json/operator_control) and, while in control, holds the {robot}/operator/active/policy/{sessionId}
token (announce_control). Cameras: head (left view of the stereo pair), wrist_left, wrist_right, letterboxed to
224x224 as in training. State: {robot}/action/joints, the joint target the sim applies (the training state).
Each inference returns a 40-step chunk of absolute joint targets (the processor adds the state back to the
relative arm actions); its first execute_steps are published at 30 Hz as JointState JSON on
{robot}/control/joint_state, and the next chunk is requested inference_lead_steps before that.
SIGINT/SIGTERM release the robot. If another operator acquires the robot, the bridge stops and exits.
"""

import dataclasses
import json
import logging
import os
import signal
import struct
import threading
import time
import uuid

import numpy as np
from PIL import Image
import tyro

logger = logging.getLogger("adamo_bridge")

JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(6)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(6)),
    "right_gripper",
]
# GR00T video key -> Adamo video track.
CAMERAS = {"head": "head", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
STEREO_TRACKS = ("head",)
# GR00T state/action groups -> slices of the 14-d joint vector (meta/modality.json).
GROUPS = {
    "left_arm": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_arm": slice(7, 13),
    "right_gripper": slice(13, 14),
}
HEARTBEAT_S = 1.0


@dataclasses.dataclass
class Args:
    model_path: str
    robot: str = "abc-inference"
    prompt: str = "sim put the plastic bottles in the bin"
    embodiment_tag: str = "new_embodiment"
    control_hz: float = 30.0
    execute_steps: int = 20
    """Actions executed from each 40-step chunk before the next one takes over."""
    inference_lead_steps: int = 8
    """Request the next chunk this many steps before execute_steps run out."""
    state_topic: str = "{robot}/action/joints"
    control_topic: str = "{robot}/control/joint_state"
    max_seconds: float | None = 900.0
    """Release the robot and exit after this long."""
    warmup_only: bool = False
    """Load the policy, time a few dummy inferences, and exit without touching the robot."""


def resize_with_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox like openpi_client.image_tools.resize_with_pad (PIL bilinear, zero pad, centered)."""
    h, w = img.shape[:2]
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = np.asarray(Image.fromarray(img).resize((rw, rh), resample=Image.BILINEAR))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - rh) // 2, (width - rw) // 2
    out[top : top + rh, left : left + rw] = resized
    return out


class Policy:
    """Gr00tPolicy wrapped to take a 14-d state and 224x224 images and return a (40, 14) chunk."""

    def __init__(self, args: Args) -> None:
        import torch

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device="cuda", strict=True)
        self.language_key = self.policy.language_key
        self.prompt = args.prompt
        self._torch = torch

    def __call__(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        obs = {
            "video": {k: img[None, None] for k, img in images.items()},
            "state": {g: state[None, None, s].astype(np.float32) for g, s in GROUPS.items()},
            "language": {self.language_key: [[self.prompt]]},
        }
        action, _ = self.policy.get_action(obs)
        return np.concatenate([action[g][0] for g in GROUPS], axis=-1).astype(np.float32)

    def warmup(self, n: int = 5) -> list[float]:
        rng = np.random.default_rng(0)
        images = {k: rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for k in CAMERAS}
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = 1.0
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            chunk = self(images, state)
            self._torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        logger.warning("warmup chunk %s, latency ms %s", chunk.shape, [round(t, 1) for t in times])
        return times


class Latest:
    """Thread-safe latest value with a count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.value = None
        self.count = 0

    def set(self, value) -> None:
        with self._lock:
            self.value, self.count = value, self.count + 1

    def get(self):
        with self._lock:
            return self.value


def decode_joints(payload: bytes) -> np.ndarray:
    """[ts_us: u64 BE][f32 BE] * N."""
    n = (len(payload) - 8) // 4
    return np.asarray(struct.unpack_from(f">{n}f", payload, 8), dtype=np.float32)


class Bridge:
    def __init__(self, args: Args, policy: Policy) -> None:
        import adamo
        from adamo import Priority

        self.adamo = adamo
        self.args = args
        self.robot = args.robot
        self.stop = threading.Event()
        self.policy = policy

        self.session = adamo.connect(api_key=os.environ["ADAMO_API_KEY"])
        self.frames = {key: Latest() for key in CAMERAS}
        self.state = Latest()
        self._receivers = {
            key: self.session.video_receiver(self.robot, track, max_queued_frames=1) for key, track in CAMERAS.items()
        }
        self._state_sub = self.session.subscribe(
            args.state_topic.format(robot=self.robot), callback=lambda s: self.state.set(decode_joints(bytes(s.payload)))
        )
        self._control_pub = self.session.publisher(
            args.control_topic.format(robot=self.robot), priority=Priority.REAL_TIME, express=True
        )

        self.operator = {"userId": "gr00t-policy", "sessionId": uuid.uuid4().hex, "displayName": "GR00T N1.7 policy"}
        self.in_control = False
        self._announcement = None  # adamo.ControlAnnouncement while in control
        ownership_topic = f"{self.robot}/control/json/operator_control"
        self._ownership_pub = self.session.publisher(ownership_topic, priority=Priority.DATA_HIGH, reliable=True)
        self._ownership_sub = self.session.subscribe(ownership_topic, callback=self._on_ownership)

        self._lock = threading.Lock()  # guards chunk, chunk_start, step and in_control
        self._chunk: np.ndarray | None = None
        self._chunk_start = 0  # control step at which chunk[0] is due
        self._step = 0
        self.infer_ms: list[float] = []
        self.published = 0

    # -- inputs -------------------------------------------------------------------

    def _camera_loop(self, key: str) -> None:
        while not self.stop.is_set():
            try:
                frame = self._receivers[key].recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                logger.exception("camera %s receive failed; reopening", key)
                if self.stop.wait(0.5):
                    return
                self._receivers[key].close()
                self._receivers[key] = self.session.video_receiver(self.robot, CAMERAS[key], max_queued_frames=1)
                continue
            rgb = np.asarray(frame.rgba)[..., :3]
            if CAMERAS[key] in STEREO_TRACKS:
                rgb = rgb[:, : rgb.shape[1] // 2]
            self.frames[key].set(np.ascontiguousarray(rgb))

    def observation(self) -> tuple[dict, np.ndarray] | None:
        state = self.state.get()
        images = {key: slot.get() for key, slot in self.frames.items()}
        if state is None or any(img is None for img in images.values()):
            return None
        # Letterboxed to 224x224 as in training (the dataset's images were stored that way).
        return {key: resize_with_pad(img, 224, 224) for key, img in images.items()}, state

    # -- operator control ---------------------------------------------------------

    def _publish_ownership(self, event: str, active: bool) -> None:
        msg = {
            "version": 1,
            "active": active,
            "event": event,
            "operator": self.operator,
            "timestampMs": self.session.fabric_now_us() / 1000,
        }
        self._ownership_pub.put(json.dumps(msg).encode())

    def _on_ownership(self, sample) -> None:
        try:
            msg = json.loads(sample.payload)
        except ValueError:
            return
        operator = msg.get("operator") or {}
        if operator.get("sessionId") == self.operator["sessionId"]:
            return
        if msg.get("active") and msg.get("event") == "acquired" and self.in_control:
            logger.warning("control taken by %s; stopping", operator.get("displayName"))
            with self._lock:
                self.in_control = False
            self._end_announcement()
            self.stop.set()

    def _end_announcement(self) -> None:
        announcement, self._announcement = self._announcement, None
        if announcement is not None:
            announcement.close()

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(HEARTBEAT_S):
            if self.in_control:
                self._publish_ownership("heartbeat", active=True)

    # -- inference and control ----------------------------------------------------

    def _inference_loop(self) -> None:
        a = self.args
        while not self.stop.is_set():
            with self._lock:
                chunk, start, step = self._chunk, self._chunk_start, self._step
            if chunk is not None and step - start < a.execute_steps - a.inference_lead_steps:
                time.sleep(0.002)
                continue
            obs = self.observation()
            if obs is None:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            try:
                actions = self.policy(*obs)
            except Exception:
                logger.exception("inference failed; stopping")
                self.stop.set()
                return
            self.infer_ms = [*self.infer_ms[-99:], (time.perf_counter() - t0) * 1e3]
            with self._lock:
                # chunk[0] was predicted for the step the observation was taken at; the control loop skips
                # the actions whose time passed while inference ran.
                self._chunk, self._chunk_start = actions, step

    def _control_loop(self) -> None:
        period = 1.0 / self.args.control_hz
        next_tick = time.monotonic()
        while not self.stop.is_set():
            next_tick += period
            with self._lock:
                if self._chunk is not None and self.in_control:
                    idx = min(self._step - self._chunk_start, len(self._chunk) - 1)
                    msg = self.adamo.JointState(names=JOINT_NAMES, positions=self._chunk[idx].tolist())
                    self._control_pub.put(msg.to_json())
                    self._step += 1
                    self.published += 1
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def run(self) -> None:
        # SIGTERM (and SIGINT) end the main loop so close() releases the robot.
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, lambda *_: self.stop.set())
        self._publish_ownership("acquired", active=True)
        self._announcement = self.session.announce_control(self.robot, "policy", self.operator["sessionId"])
        with self._lock:
            self.in_control = True
        logger.warning("acquired %s as %s", self.robot, self.operator)
        threads = [threading.Thread(target=self._camera_loop, args=(key,), daemon=True) for key in CAMERAS]
        threads += [threading.Thread(target=f, daemon=True) for f in (self._inference_loop, self._control_loop)]
        threads.append(threading.Thread(target=self._heartbeat_loop, daemon=True))
        for t in threads:
            t.start()
        t_start = time.monotonic()
        try:
            while not self.stop.wait(5.0):
                ms = self.infer_ms
                logger.warning(
                    "frames %s state msgs %d | chunks %d, infer mean %s ms | actions published %d",
                    {CAMERAS[k]: s.count for k, s in self.frames.items()},
                    self.state.count,
                    len(ms),
                    round(float(np.mean(ms))) if ms else None,
                    self.published,
                )
                if self.args.max_seconds is not None and time.monotonic() - t_start > self.args.max_seconds:
                    logger.warning("max_seconds reached")
                    break
        finally:
            self.close(threads)

    def close(self, threads: list[threading.Thread]) -> None:
        was_in_control = self.in_control
        with self._lock:
            self.in_control = False
        self._end_announcement()
        if was_in_control:
            self._publish_ownership("released", active=False)
        self.stop.set()
        for t in threads:
            t.join(timeout=2)
        for closable in (*self._receivers.values(), self._state_sub, self._ownership_sub, self._control_pub,
                         self._ownership_pub, self.session):  # fmt: skip
            try:
                closable.close()
            except Exception:
                logger.exception("close failed")
        logger.warning("released %s after %d actions", self.robot, self.published)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", force=True)
    policy = Policy(args)
    policy.warmup()
    if args.warmup_only:
        return
    Bridge(args, policy).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
