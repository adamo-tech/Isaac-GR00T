"""Serve the abc-sim GR00T N1.7 policy (FastGr00t, training-server LoRA merged) to robots over Adamo.

The HTTP counterpart of adamo_bridge.py, with the routes of abc's deploy/adamo_server.py. The
policy loads once at startup on its own thread (TRT backbone + compiled action head in one CUDA
graph per prompt, ~20 ms per chunk on an RTX 4090). Then:

    POST /start   {"robot_name": "abc-sim", "adamo_api_key": "ak_...", "prompt": ..., "lora": ...}
                  connect, (optionally) swap in another LoRA, compile the prompt if it is new,
                  claim the robot and drive it;
    POST /lora    {"lora": "abc-sim-r64-b256-500"}  write another LoRA into the live weights (~0.1 s,
                  no recompile), also during a run;
    POST /take_control   reclaim the robot after another operator took it;
    POST /stop    release the robot and disconnect;
    GET  /status  policy phase, LoRA, cameras, state, inference and control;
    GET  /loras   LoRA directories under the LoRA root.

Cameras, state and control are as in adamo_bridge.py: head (left half of the stereo pair),
wrist_left and wrist_right letterboxed to 224x224; state from {robot}/action/joints; each 40-step
chunk of absolute joint targets is published at 30 Hz on {robot}/control/joint_state, execute_steps
of it before the next chunk takes over.

    python -m runpod.abc_sim.adamo_server --model-path <base weights + abc-sim processor> \\
        --lora-root <dir of LoRA saves> --lora abc-sim-r64-b256-500 --engine-dir trt_fast/engines

modal/gr00t_fast_adamo.py serves this on Modal.
"""

from collections.abc import Callable
from concurrent.futures import Future
import dataclasses
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from runpod.abc_sim.adamo_bridge import (
    CAMERAS,
    GROUPS,
    HEARTBEAT_S,
    JOINT_NAMES,
    STEREO_TRACKS,
    decode_joints,
    resize_with_pad,
)
import tyro


logger = logging.getLogger("adamo_server")

RES = 224
CONTROL_LEASE_S = 3.0  # another operator's claim lapses without a heartbeat for this long


@dataclasses.dataclass
class Args:
    model_path: str
    """Base weights the LoRA was trained from plus the abc-sim processor files."""
    engine_dir: str
    """FastGr00t engines for 224x224 (scripts/deployment/fast/build_engines.py)."""
    lora_root: str = "."
    """Relative LoRA names (/start, /lora) resolve here."""
    lora: str | None = None
    """LoRA loaded at startup (name under lora_root or a path); None serves the base weights."""
    prompt: str = "sim put the plastic bottles in the bin"
    """Default prompt, compiled at startup."""
    embodiment_tag: str = "new_embodiment"
    host: str = "0.0.0.0"
    port: int = 8000


class StartRequest(BaseModel):
    robot_name: str
    adamo_api_key: str | None = None
    """Defaults to the ADAMO_API_KEY environment variable."""
    prompt: str | None = None
    """Overrides the server's default prompt; a new prompt compiles first (~1 min)."""
    lora: str | None = None
    """Swap this LoRA in before the run (name under the LoRA root or a path)."""
    state_topic: str = "{robot}/action/joints"
    control_topic: str = "{robot}/control/joint_state"
    control_hz: float = 30.0
    execute_steps: int = 20
    """Actions executed from each 40-step chunk before the next one takes over."""
    inference_lead_steps: int = 8
    """Request the next chunk this many steps before execute_steps run out."""
    max_seconds: float | None = 900.0
    """Release the robot after this long; None runs until /stop."""
    frame_timeout_s: float | None = 30.0
    """Stop when a camera or the state has been silent this long (counted from /start)."""


class LoraRequest(BaseModel):
    lora: str


class FastPolicy:
    """FastGr00t with a merged LoRA: 224x224 frames + a 14-d state in, a (40, 14) chunk out."""

    def __init__(self, args: Args, lora_dir: Path | None) -> None:
        from scripts.deployment.fast.fast_gr00t import FastGr00t
        from scripts.deployment.fast.lora import load_lora_policy

        policy, self.swapper = load_lora_policy(
            args.model_path, None if lora_dir is None else str(lora_dir), args.embodiment_tag
        )
        self.language_key = policy.language_key
        self.fast = FastGr00t(policy, args.engine_dir, H=RES, W=RES)
        self.prompts: set[str] = set()

    def __call__(self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> np.ndarray:
        obs = {
            "video": {k: img[None, None] for k, img in images.items()},
            "state": {g: state[None, None, s].astype(np.float32) for g, s in GROUPS.items()},
            "language": {self.language_key: [[prompt]]},
        }
        action = self.fast.get_action(obs)
        return np.concatenate([action[g][0] for g in GROUPS], axis=-1).astype(np.float32)

    def warmup(self, prompt: str, n: int = 5) -> list[float]:
        """Compile and capture the graph for ``prompt`` (the first call) and time a few calls."""
        import torch

        rng = np.random.default_rng(0)
        images = {k: rng.integers(0, 255, (RES, RES, 3), dtype=np.uint8) for k in CAMERAS}
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = 1.0
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self(images, state, prompt)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        self.prompts.add(prompt)
        logger.warning("prompt %r ready, latency ms %s", prompt, [round(t, 1) for t in times])
        return times

    def swap(self, lora_dir: Path) -> None:
        if self.swapper is None:
            raise ValueError("serving the base weights; start the server with a LoRA to swap LoRAs")
        self.swapper.swap(lora_dir)
        self.fast.refresh_weights()


class Latest:
    """Thread-safe latest value with arrival time and count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.value = None
        self.stamp = 0.0
        self.count = 0

    def set(self, value) -> None:
        with self._lock:
            self.value, self.stamp, self.count = value, time.monotonic(), self.count + 1

    def get(self):
        with self._lock:
            return self.value


class AdamoRun:
    """One robot connection: camera decoders, state, operator control and the control loop.
    Inference runs on the worker thread (``inference_step``)."""

    def __init__(self, req: StartRequest, prompt: str) -> None:
        import adamo
        from adamo import Priority

        self.adamo = adamo
        self.req = req
        self.prompt = prompt
        self.robot = req.robot_name
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.started = time.monotonic()

        api_key = req.adamo_api_key or os.environ.get("ADAMO_API_KEY")
        if not api_key:
            raise ValueError("pass adamo_api_key or set ADAMO_API_KEY")
        self.session = adamo.connect(api_key=api_key)
        self.frames = {key: Latest() for key in CAMERAS}
        self.state = Latest()
        self._receivers = {
            key: self.session.video_receiver(self.robot, track, max_queued_frames=1)
            for key, track in CAMERAS.items()
        }
        self._state_sub = self.session.subscribe(
            req.state_topic.format(robot=self.robot), callback=self._on_state
        )
        self._control_pub = self.session.publisher(
            req.control_topic.format(robot=self.robot), priority=Priority.REAL_TIME, express=True
        )

        self.operator = {
            "userId": "policy",
            "sessionId": uuid.uuid4().hex,
            "displayName": "GR00T N1.7 policy",
        }
        self.in_control = False
        self.controller: dict | None = None  # another operator holding the robot
        self._controller_seen = 0.0
        self._announcement = None  # adamo.ControlAnnouncement while in control
        ownership_topic = f"{self.robot}/control/json/operator_control"
        self._ownership_pub = self.session.publisher(
            ownership_topic, priority=Priority.DATA_HIGH, reliable=True
        )
        self._ownership_sub = self.session.subscribe(ownership_topic, callback=self._on_ownership)

        self._lock = threading.Lock()  # guards chunk, chunk_start, step, generation and in_control
        self._generation = 0  # bumped on every ownership change
        self._chunk: np.ndarray | None = None
        self._chunk_start = 0  # control step at which chunk[0] is due
        self._step = 0
        self.infer_ms: list[float] = []
        self.published = 0

        self._threads = [
            threading.Thread(target=self._camera_loop, args=(key,), daemon=True) for key in CAMERAS
        ]
        self._threads += [
            threading.Thread(target=f, daemon=True)
            for f in (self._control_loop, self._heartbeat_loop)
        ]
        for t in self._threads:
            t.start()

    # -- inputs -------------------------------------------------------------------

    def _camera_loop(self, key: str) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self._receivers[key].recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception as e:
                logger.exception("camera %s receive failed; reopening", key)
                self.error = f"camera {key}: {e!r}"
                if self.stop_event.wait(0.5):
                    return
                try:
                    self._receivers[key].close()
                    self._receivers[key] = self.session.video_receiver(
                        self.robot, CAMERAS[key], max_queued_frames=1
                    )
                except Exception as e:
                    self.error = f"camera {key}: reopen failed: {e!r}"
                continue
            rgb = np.asarray(frame.rgba)[..., :3]
            if CAMERAS[key] in STEREO_TRACKS:
                rgb = rgb[:, : rgb.shape[1] // 2]
            self.frames[key].set(np.ascontiguousarray(rgb))

    def _on_state(self, sample) -> None:
        try:
            self.state.set(decode_joints(bytes(sample.payload)))
        except Exception as e:
            self.error = f"joint state on {sample.key}: {e!r}"

    def observation(self) -> tuple[dict, np.ndarray] | None:
        state = self.state.get()
        images = {key: slot.get() for key, slot in self.frames.items()}
        if state is None or any(img is None for img in images.values()):
            return None
        if len(state) != len(JOINT_NAMES):
            raise ValueError(f"state has {len(state)} joints, the policy takes {len(JOINT_NAMES)}")
        # Letterboxed to 224x224 as in training (the dataset's images were stored that way).
        return {key: resize_with_pad(img, RES, RES) for key, img in images.items()}, state

    def timed_out(self) -> str | None:
        """Why the run should stop: an input went silent or max_seconds passed; else None."""
        now = time.monotonic()
        if self.req.max_seconds is not None and now - self.started > self.req.max_seconds:
            return f"max_seconds ({self.req.max_seconds:.0f} s) reached"
        timeout = self.req.frame_timeout_s
        if timeout is None:
            return None
        for name, slot in [*self.frames.items(), ("state", self.state)]:
            age = now - (slot.stamp if slot.count else self.started)
            if age > timeout:
                return f"nothing on {name} for {age:.0f} s"
        return None

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
        if msg.get("active"):
            if self.in_control and msg.get("event") == "acquired":
                logger.warning("control taken by %s", operator.get("displayName"))
                self._set_in_control(False)
            self.controller, self._controller_seen = operator, time.monotonic()
        elif self.controller and operator.get("sessionId") == self.controller.get("sessionId"):
            self.controller = None

    def take_control(self) -> None:
        self._publish_ownership("acquired", active=True)
        self._set_in_control(True)
        self.controller = None

    def release_control(self) -> None:
        if self.in_control:
            self._set_in_control(False)
            self._publish_ownership("released", active=False)

    def _set_in_control(self, in_control: bool) -> None:
        """Change ownership and drop the chunk, so nothing inferred before the change is sent."""
        with self._lock:
            self.in_control = in_control
            self._chunk = None
            self._generation += 1
        if in_control and self._announcement is None:
            try:
                self._announcement = self.session.announce_control(
                    self.robot, self.operator["userId"], self.operator["sessionId"]
                )
            except Exception as e:
                logger.exception("could not announce control")
                self.error = (
                    f"announce_control failed (the operate page won't show the policy): {e!r}"
                )
        elif not in_control and self._announcement is not None:
            announcement, self._announcement = self._announcement, None
            try:
                announcement.close()
            except Exception:
                logger.exception("could not withdraw the control announcement")

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(HEARTBEAT_S):
            if self.in_control:
                self._publish_ownership("heartbeat", active=True)
            if self.controller and time.monotonic() - self._controller_seen > CONTROL_LEASE_S:
                self.controller = None

    # -- inference and control ----------------------------------------------------

    def inference_step(self, policy: FastPolicy) -> bool:
        """Infer the next chunk if one is due. Returns False when idle."""
        req = self.req
        with self._lock:
            if not self.in_control:
                return False
            chunk, start, step, generation = (
                self._chunk,
                self._chunk_start,
                self._step,
                self._generation,
            )
        if chunk is not None and step - start < req.execute_steps - req.inference_lead_steps:
            return False
        obs = self.observation()
        if obs is None:
            return False
        t0 = time.perf_counter()
        actions = policy(*obs, self.prompt)
        self.infer_ms = [*self.infer_ms[-99:], (time.perf_counter() - t0) * 1e3]
        with self._lock:
            if self._generation == generation:
                # chunk[0] is for the step the observation was taken at; the control loop skips
                # the actions whose time passed while inference ran.
                self._chunk, self._chunk_start = actions, step
        return True

    def _control_loop(self) -> None:
        period = 1.0 / self.req.control_hz
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            next_tick += period
            with self._lock:
                if self._chunk is not None and self.in_control:
                    idx = min(self._step - self._chunk_start, len(self._chunk) - 1)
                    msg = self.adamo.JointState(
                        names=JOINT_NAMES, positions=self._chunk[idx].tolist()
                    )
                    self._control_pub.put(msg.to_json())
                    self._step += 1
                    self.published += 1
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def status(self) -> dict:
        now = time.monotonic()
        ms = self.infer_ms
        return {
            "robot": self.robot,
            "prompt": self.prompt,
            "error": self.error,
            "elapsed_s": round(now - self.started),
            "control": {
                "in_control": self.in_control,
                "operator": self.operator,
                "held_by": self.controller,
            },
            "cameras": {
                CAMERAS[k]: {
                    "frames": s.count,
                    "age_ms": round((now - s.stamp) * 1e3) if s.count else None,
                    "shape": None if s.value is None else list(s.value.shape),
                }
                for k, s in self.frames.items()
            },
            "state": {
                "msgs": self.state.count,
                "age_ms": round((now - self.state.stamp) * 1e3) if self.state.count else None,
            },
            "inference": {
                "chunks": len(ms),
                "last_ms": round(ms[-1], 1) if ms else None,
                "mean_ms": round(float(np.mean(ms)), 1) if ms else None,
            },
            "actions_published": self.published,
        }

    def close(self) -> None:
        try:
            self.release_control()
        except Exception:
            logger.exception("release failed")
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=2)
        for closable in (*self._receivers.values(), self._state_sub, self._ownership_sub,
                         self._control_pub, self._ownership_pub, self.session):  # fmt: skip
            try:
                closable.close()
            except Exception:
                logger.exception("close failed")
        logger.warning("released %s after %d actions", self.robot, self.published)


class PolicyWorker:
    """Owns the GPU on one thread: load, compile prompts, swap LoRAs and run inference for the
    active run. Requests from HTTP threads are queued as (fn, Future) and run between inferences."""

    def __init__(
        self,
        args: Args,
        resolve_lora: Callable[[str], Path],
        on_ready: Callable[[], None] | None = None,
        prepare: Callable[[], None] | None = None,
    ):
        self.args = args
        self.resolve_lora = resolve_lora
        self.on_ready = on_ready
        self.prepare = prepare
        self.policy: FastPolicy | None = None
        self.phase = "preparing" if prepare else "loading"
        self.error: str | None = None
        self.lora: str | None = None
        self.run: AdamoRun | None = None
        self.last_stop: str | None = None
        self._lock = threading.RLock()
        self._tasks: queue.Queue[tuple[Callable, Future]] = queue.Queue()
        threading.Thread(target=self._main, daemon=True).start()

    def _main(self) -> None:
        try:
            t0 = time.perf_counter()
            if self.prepare is not None:
                self.prepare()
                self.phase = "loading"
            lora_dir = None if self.args.lora is None else self.resolve_lora(self.args.lora)
            self.policy = FastPolicy(self.args, lora_dir)
            self.lora = None if lora_dir is None else str(lora_dir)
            self.phase = "compiling"
            self.policy.warmup(self.args.prompt)
            logger.warning("policy ready in %.0f s", time.perf_counter() - t0)
            self.phase = "ready"
            if self.on_ready is not None:
                self.on_ready()
        except Exception as e:
            logger.exception("policy load failed")
            self.phase, self.error = "failed", repr(e)
            return
        while True:
            try:
                fn, fut = self._tasks.get_nowait()
            except queue.Empty:
                pass
            else:
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(fn())
                    except BaseException as e:
                        fut.set_exception(e)
                continue
            run = self.run
            if run is None or run.stop_event.is_set():
                time.sleep(0.005)
                continue
            reason = run.timed_out()
            if reason is not None:
                logger.warning("stopping %s: %s", run.robot, reason)
                with self._lock:
                    if self.run is run:
                        self.last_stop = reason
                        self.stop()
                continue
            try:
                if not run.inference_step(self.policy):
                    time.sleep(0.002)
            except Exception as e:
                logger.exception("inference failed")
                run.error = f"inference: {e!r}"
                time.sleep(0.5)

    def submit(self, fn: Callable, timeout: float | None = None):
        """Run ``fn`` on the worker thread and return its result."""
        if self.policy is None or self.phase == "failed":
            raise HTTPException(503, f"policy not ready ({self.phase}: {self.error})")
        fut: Future = Future()
        self._tasks.put((fn, fut))
        return fut.result(timeout)

    def swap(self, lora: str) -> str:
        lora_dir = self.resolve_lora(lora)

        def do_swap():
            t0 = time.perf_counter()
            self.policy.swap(lora_dir)
            self.lora = str(lora_dir)
            logger.warning("swapped in %s in %.0f ms", lora_dir, (time.perf_counter() - t0) * 1e3)
            return self.lora

        return self.submit(do_swap)

    def start(self, req: StartRequest) -> AdamoRun:
        prompt = req.prompt or self.args.prompt
        with self._lock:
            self.stop()
            self.last_stop = None
            if req.lora is not None:
                self.swap(req.lora)
            if prompt not in self.policy.prompts:
                previous, self.phase = self.phase, "compiling"
                try:
                    self.submit(lambda: self.policy.warmup(prompt))
                finally:
                    self.phase = previous
            run = AdamoRun(req, prompt)
            run.take_control()
            self.run = run
            return run

    def stop(self) -> None:
        with self._lock:
            run, self.run = self.run, None
            if run is not None:
                run.close()


def resolve_under(root: str | Path) -> Callable[[str], Path]:
    def resolve(lora: str) -> Path:
        path = Path(lora) if Path(lora).is_absolute() else Path(root) / lora
        if (
            not (path / "trainable.safetensors").is_file()
            or not (path / "lora_config.json").is_file()
        ):
            raise ValueError(
                f"{path} is not a LoRA save (trainable.safetensors + lora_config.json)"
            )
        return path

    return resolve


def list_loras(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(p.parent.name for p in root.glob("*/lora_config.json"))


def create_app(
    args: Args,
    resolve_lora: Callable[[str], Path] | None = None,
    lora_root: Callable[[], Path] | None = None,
    on_ready: Callable[[], None] | None = None,
    prepare: Callable[[], None] | None = None,
) -> FastAPI:
    """``resolve_lora`` maps a LoRA name to its directory (default: under ``args.lora_root``);
    ``lora_root`` returns the directory /loras lists. ``prepare`` runs on the worker thread before
    the policy loads (e.g. to build engines; it may still set ``args``), ``on_ready`` after."""
    resolve_lora = resolve_lora or resolve_under(args.lora_root)
    lora_root = lora_root or (lambda: Path(args.lora_root))
    worker = PolicyWorker(args, resolve_lora, on_ready, prepare)
    app = FastAPI(title="GR00T N1.7 Adamo server")
    app.state.worker = worker

    def bad_request(fn):
        try:
            return fn()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, repr(e)) from e

    @app.post("/start")
    def start(req: StartRequest) -> dict:
        run = bad_request(lambda: worker.start(req))
        return {"started": True, "robot": run.robot, "prompt": run.prompt, "lora": worker.lora}

    @app.post("/lora")
    def lora(req: LoraRequest) -> dict:
        return {"lora": bad_request(lambda: worker.swap(req.lora))}

    @app.post("/take_control")
    def take_control() -> dict:
        run = worker.run
        if run is None:
            raise HTTPException(409, "no active run; POST /start first")
        run.take_control()
        return {"in_control": True, "robot": run.robot}

    @app.post("/stop")
    def stop() -> dict:
        worker.stop()
        return {"stopped": True}

    @app.get("/status")
    def status() -> dict:
        run = worker.run
        return {
            "policy": worker.phase,
            "policy_error": worker.error,
            "lora": worker.lora,
            "prompts": sorted(worker.policy.prompts) if worker.policy else [],
            "last_stop": worker.last_stop,
            "run": None if run is None else run.status(),
        }

    @app.get("/loras")
    def loras() -> dict:
        root = lora_root()
        return {"root": str(root), "loras": list_loras(root)}

    return app


def main(args: Args) -> None:
    import uvicorn

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", force=True)
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
