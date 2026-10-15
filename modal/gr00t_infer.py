"""GR00T N1.7 LIBERO inference on Modal, driving a robot over adamo.

Mirrors openpi's modal/openpi_infer.py: the container joins adamo, decodes the
robot's h264 camera track itself, reads proprioception off
`{robot}/proprioception/joints`, and publishes actions to
`{robot}/control/{channel}` as adamo JointState. Modal's `.remote()` is used
only to warm the container and to start the loop.

  modal run    modal/gr00t_infer.py::fill     # once, after `hf auth login`
  modal deploy modal/gr00t_infer.py
  python modal/run_adamo.py                                         # libero-sim/cam

The backbone, nvidia/Cosmos-Reason2-2B, is a GATED Hugging Face repo that every
GR00T checkpoint loads at construction (gr00t/model/modules/qwen3_backbone.py).
`download` caches it on the Volume with the caller's token; the serve container
then runs with HF_HUB_OFFLINE=1 and needs no token at all.
"""

import modal


app = modal.App("gr00t-infer")
vol = modal.Volume.from_name("gr00t-assets", create_if_missing=True)

REPO = "/opt/gr00t"
ASSETS = "/gr00t_assets"
HF_HOME = f"{ASSETS}/hf"
CKPT_REPO = "nvidia/GR00T-N1.7-LIBERO"
BACKBONE = "nvidia/Cosmos-Reason2-2B"
LOCAL = "/home/h/workspace/inference/Isaac-GR00T"

# LIBERO_PANDA (gr00t/configs/data/embodiment_configs.py:193): two 256x256
# views, an 8-d state split over 7 keys, a 7-d action over 16 steps.
EMBODIMENT = "libero_sim"
RES = 256
GUARD = 16  # libero-sim's black seam between the stitched views
VIDEO_KEYS = ("image", "wrist_image")
STATE_KEYS = (("x", 1), ("y", 1), ("z", 1), ("roll", 1), ("pitch", 1), ("yaw", 1), ("gripper", 2))
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LANG_KEY = "annotation.human.action.task_description"

image = (
    modal.Image.debian_slim(python_version="3.12")  # gr00t is requires-python ==3.12
    .apt_install(
        "git",
        # adamo's native extension links GStreamer and decodes h264 through it.
        "libgstreamer1.0-0",
        "libgstreamer-plugins-base1.0-0",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-libav",
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",
    )
    .pip_install("uv")
    # Only the lock goes into the build layer, so editing gr00t/ does not
    # repeat the dependency install.
    .add_local_file(f"{LOCAL}/pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(f"{LOCAL}/uv.lock", f"{REPO}/uv.lock", copy=True)
    # Install through uv.lock (torch 2.9.0+cu128, flash-attn 2.8.3, transformers
    # 4.57.3), minus the TensorRT and deepspeed stacks inference never imports.
    # The aarch64 torchcodec entry is a repo-relative wheel path that is not in
    # this layer, so it is dropped before install.
    .run_commands(
        f"cd {REPO} && uv export --frozen --no-dev --no-emit-project --no-hashes"
        " --no-emit-package deepspeed"
        " --no-emit-package tensorrt-cu12 --no-emit-package tensorrt-cu12-libs"
        " --no-emit-package tensorrt-cu12-bindings"
        " --no-emit-package tensorrt-cu13 --no-emit-package tensorrt-cu13-libs"
        " --no-emit-package tensorrt-cu13-bindings"
        " | grep -v 'aarch64.whl' > /tmp/req.txt",
        "uv pip install --system -r /tmp/req.txt"
        " --extra-index-url https://download.pytorch.org/whl/cu128"
        " --index-strategy unsafe-best-match",
    )
    # numpy is pinned alongside so adamo cannot float it off gr00t's 1.26.4.
    .run_commands("uv pip install --system 'adamo[video]>=0.4.58' numpy==1.26.4")
    .env({"HF_HOME": HF_HOME, "PYTHONPATH": REPO, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(f"{LOCAL}/gr00t", f"{REPO}/gr00t", ignore=["**/__pycache__", "eval/sim/**"])
)


def ckpt_dir(suite: str) -> str:
    return f"{ASSETS}/GR00T-N1.7-LIBERO/{suite}"


@app.function(image=image, volumes={ASSETS: vol}, timeout=3600, cpu=8.0, memory=16384)
def download(hf_token: str = "", suite: str = "libero_spatial") -> None:
    """Fill the Volume: one LIBERO suite's weights plus the gated backbone.

    Only inference files are pulled. The HF repo also carries 16 ZeRO optimizer
    shards and a 9 GiB model_states.pt per suite, none of which Gr00tPolicy reads.
    """
    import os

    from huggingface_hub import snapshot_download

    if not hf_token:
        raise ValueError(f"download needs hf_token with access to {BACKBONE} (gated)")
    snapshot_download(
        CKPT_REPO,
        allow_patterns=[
            f"{suite}/{f}"
            for f in (
                "config.json",
                "embodiment_id.json",
                "model-*.safetensors",
                "model.safetensors.index.json",
                "processor_config.json",
                "statistics.json",
            )
        ],
        local_dir=f"{ASSETS}/GR00T-N1.7-LIBERO",
        token=hf_token,
    )
    snapshot_download(BACKBONE, token=hf_token)
    vol.commit()

    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ASSETS) for f in fs)
    print(f"volume holds {total / 2**30:.2f} GiB; {suite}: {sorted(os.listdir(ckpt_dir(suite)))}")


@app.cls(
    image=image,
    # A 3B model at bf16 is ~7 GiB, so every entry fits. Ampere or newer only:
    # use_flash_attention is on in the checkpoint config and flash-attn 2 needs it.
    gpu=["L40S", "A100-40GB", "A10G", "L4"],
    volumes={ASSETS: vol},
    # Exactly one loop may drive a robot's control channel.
    max_containers=1,
    scaledown_window=1200,
    timeout=3600,
    startup_timeout=1800,
)
# Lets progress() answer while serve_stream holds the container.
@modal.concurrent(max_inputs=4)
class Gr00tLibero:
    suite: str = modal.parameter(default="libero_spatial")

    @modal.enter()
    def load(self):
        import os
        import time

        # The backbone came off the Volume in download(); never reach for the hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
        # transformers probes the hub for "is this a Mistral tokenizer" on every
        # tokenizer load; gr00t/policy/__init__.py skips that for non-Mistral ids.
        os.environ["GROOT_PATCH_MISTRAL"] = "1"
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        import numpy as np
        import torch

        self._serving = False
        self._progress: dict = {}

        t0 = time.time()
        self.policy = Gr00tPolicy(
            embodiment_tag=EMBODIMENT, model_path=ckpt_dir(self.suite), device="cuda"
        )
        self.load_s = time.time() - t0

        t1 = time.time()
        zeros = np.zeros((RES, RES, 3), np.uint8)
        out = self.infer(zeros, zeros, np.zeros(8), "pick up the black bowl")
        torch.cuda.synchronize()
        self.warmup_ms = (time.time() - t1) * 1000
        print(f"loaded {self.suite} in {self.load_s:.1f}s, warm-up {self.warmup_ms:.0f} ms")
        print(f"device: {torch.cuda.get_device_name()}  chunk: {out.shape}")

    def infer(self, base, wrist, state8, prompt: str):
        """One observation in, a (16, 7) action chunk out, in LIBERO's command space.

        gr00t/eval/sim/LIBERO/libero_env.py:188-201 post-processes every action
        before env.step: the gripper is trained in [0, 1] (0 = close), and LIBERO
        wants -1 open / +1 close. That conversion is applied here, so the chunk
        can go straight onto a robot that speaks raw LIBERO, as libero-sim does.
        """
        import numpy as np

        obs = {
            "video": {k: img[None, None] for k, img in zip(VIDEO_KEYS, (base, wrist))},
            "state": {},
            "language": {LANG_KEY: [[prompt]]},
        }
        i = 0
        for key, n in STATE_KEYS:
            obs["state"][key] = np.asarray(state8[i : i + n], np.float32)[None, None]
            i += n
        action, _ = self.policy.get_action(obs)
        chunk = np.concatenate([action[k][0] for k in ACTION_KEYS], axis=-1)
        chunk[:, -1] = -np.sign(2.0 * chunk[:, -1] - 1.0)
        return chunk

    @modal.method()
    def stats(self) -> dict:
        import torch

        return {
            "device": torch.cuda.get_device_name(),
            "suite": self.suite,
            "load_s": self.load_s,
            "warmup_ms": self.warmup_ms,
        }

    @modal.method()
    def progress(self) -> dict:
        """Snapshot of the running loop; the launcher's only view inside it."""
        return dict(self._progress)

    @modal.method()
    def serve_stream(
        self,
        robot: str = "libero-sim",
        track: str = "cam",
        prompt: str = "pick up the black bowl between the plate and the ramekin and place it on the plate",
        control_channel: str = "gr00t",
        mode: str = "async",
        action_steps: int = 16,
        hz: float = 30.0,
        queue_depth: int = 2,
        max_queue: int = 8,
        seconds: float = 3600.0,
        dry_run: bool = False,
        adamo_api_key: str = "",
    ) -> dict:
        """Closed loop against a robot that queues its commands.

        libero-sim (robot/libero_stream.py) queues each command (at most
        QUEUE_MAX=8, dropping the oldest past that), runs exactly one env.step
        per command, and publishes proprioception after every step and before
        the frame. There is no way to clear its queue, so whatever is sent will
        be executed. Two ways to drive it:

        mode="async" (default): inference runs back to back on a thread, always
        on the newest state and the first frame captured after it. The main
        loop trickles one action per tick at `hz`, keeping at most
        `queue_depth` commands queued on the robot, so a new chunk takes effect
        within a couple of steps and the arm never waits on the GPU. A chunk is
        indexed by robot step: chunk[i] is for the i-th step after its
        observation, so the steps executed while it was being inferred are
        skipped (latency compensation). `action_steps` caps how far into a
        chunk the loop will go before holding for the next one.

        mode="lockstep": what rollout_policy.py does with the env in-process.
        Send `action_steps` actions, wait until the robot has executed them,
        observe, infer. Every observation sees exactly the result of the last
        chunk, but the arm stands still for the whole observe+infer interval.
        """
        import struct
        import threading
        import time

        import adamo
        from adamo.operate.control import JointState
        import numpy as np

        if not adamo_api_key:
            raise ValueError("serve_stream needs adamo_api_key; run_adamo.py passes it")
        if mode not in ("async", "lockstep"):
            raise ValueError(f"mode must be 'async' or 'lockstep', got {mode!r}")
        if not 1 <= action_steps <= 16:
            raise ValueError(f"action_steps must be 1..16 (the chunk length), got {action_steps}")
        if self._serving:
            raise RuntimeError("a serve loop is already running in this container")
        self._serving = True
        prog = {
            "robot": robot, "track": track, "prompt": prompt, "mode": mode,
            "state": "connecting", "inferences": 0, "actions_sent": 0,
            "state_timeouts": 0, "frame_timeouts": 0, "starved": 0, "resyncs": 0,
            "infer_p50": 0.0, "infer_p95": 0.0, "cycle_p50": 0.0, "lag_p50": 0,
            "last_action": None, "last_state": None, "elapsed_s": 0.0, "dry_run": dry_run,
        }  # fmt: skip
        self._progress = prog

        sess = adamo.connect(api_key=adamo_api_key)

        # Joint updates: [ts_us u64 BE][f32 BE x N] (adamo/joints.py). One per
        # executed step, so the count is how the loop tracks the robot's queue.
        joints = {"n": 0, "ts_us": None, "vec": None, "at": 0.0}
        cond = threading.Condition()

        def _on_joints(sample):
            b = bytes(sample.payload)
            n = (len(b) - 8) // 4
            if n <= 0:
                return
            ts, *vals = struct.unpack(f">Q{n}f", b[: 8 + 4 * n])
            with cond:
                joints.update(
                    n=joints["n"] + 1, ts_us=ts, vec=np.asarray(vals), at=time.monotonic()
                )
                cond.notify_all()

        joints_sub = sess.subscribe(f"{robot}/proprioception/joints", callback=_on_joints)
        rx = sess.video_receiver(robot, track)
        # Reliable: a lost command would desync the step count from the robot.
        act_pub = sess.publisher(
            f"{robot}/control/{control_channel}", priority=250, express=True, reliable=True
        )

        def send(action):
            if not dry_run:
                act_pub.put(
                    JointState(
                        names=list(ACTION_KEYS),
                        positions=[float(x) for x in action],
                        frame_id=f"gr00t/{self.suite}",
                    ).to_json()
                )
            prog["actions_sent"] += 1
            prog["last_action"] = [round(float(x), 3) for x in action]

        def wait_joints(since: int, need: int, timeout: float) -> bool:
            """Until `need` joint updates past `since`, or the stream goes quiet."""
            deadline = time.monotonic() + timeout
            with cond:
                while joints["n"] - since < need:
                    left = deadline - time.monotonic()
                    quiet = joints["n"] > since and time.monotonic() - joints["at"] > 0.3
                    if left <= 0 or quiet:
                        return False
                    cond.wait(min(left, 0.05))
            return True

        def frame_after(ts_us, arrived: float, timeout: float):
            """First decoded frame captured at/after `ts_us` (sender clock)."""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    f = rx.recv(timeout=0.2)
                except TimeoutError:
                    rx.request_keyframe()
                    continue
                while (nxt := rx.try_recv()) is not None:
                    f = nxt
                cap = f.capture_unix_us
                fresh = cap >= ts_us if (cap is not None and ts_us is not None) else (
                    time.monotonic() > arrived + 0.1
                )  # fmt: skip
                if fresh:
                    return np.asarray(f.rgba)[:, :, :3]
            return None

        def observe_and_infer(snapshot, timeout: float):
            """(ts_us, state, arrived) -> action chunk, or None if no fresh frame."""
            ts_us, state, arrived = snapshot
            rgb = frame_after(ts_us, arrived, timeout)
            if rgb is None:
                prog["frame_timeouts"] += 1
                return None
            if rgb.shape[1] != 2 * RES + GUARD or rgb.shape[0] != RES:
                raise ValueError(
                    f"{robot}/{track} is {rgb.shape[1]}x{rgb.shape[0]}; expected the "
                    f"stitched {2 * RES + GUARD}x{RES} pair libero-sim publishes"
                )
            if state.size != 8:
                raise ValueError(f"{robot} publishes {state.size}-d state; LIBERO needs 8")
            base = np.ascontiguousarray(rgb[:, :RES])
            wrist = np.ascontiguousarray(rgb[:, RES + GUARD :])
            t0 = time.perf_counter()
            chunk = self.infer(base, wrist, state, prompt)
            record(infer_ms, (time.perf_counter() - t0) * 1000, "infer")
            prog["inferences"] += 1
            prog["last_state"] = [round(float(x), 3) for x in state]
            return chunk

        infer_ms: list[float] = []
        cycle_ms: list[float] = []
        lags: list[float] = []

        def record(xs: list, x: float, name: str):
            xs.append(x)
            s = sorted(xs[-500:])
            prog[f"{name}_p50"] = s[len(s) // 2]
            if name == "infer":
                prog["infer_p95"] = s[min(int(len(s) * 0.95), len(s) - 1)]

        t_start = time.perf_counter()
        hold = [0.0] * 6 + [-1.0]  # zero delta, gripper open
        stop = threading.Event()
        planner = None
        try:
            # libero-sim publishes state only after a step, so a fresh sim has
            # none yet. One zero-delta step makes it publish without moving.
            prog["state"] = "waiting for state"
            while joints["vec"] is None and time.perf_counter() - t_start < seconds:
                n0 = joints["n"]
                act_pub.put(JointState(names=list(ACTION_KEYS), positions=hold).to_json())
                wait_joints(n0, 1, 2.0)

            if mode == "lockstep":
                while time.perf_counter() - t_start < seconds:
                    t_cycle = time.perf_counter()
                    prog["elapsed_s"] = t_cycle - t_start
                    with cond:
                        snapshot = (joints["ts_us"], joints["vec"], joints["at"])
                    prog["state"] = "observing"
                    chunk = observe_and_infer(snapshot, 3.0)
                    if chunk is None:
                        continue
                    prog["state"] = "running"
                    # Past `max_queue`, one more per joint update so libero-sim's
                    # QUEUE_MAX never drops an action; a lost update costs 0.2 s.
                    n0 = joints["n"]
                    for i, action in enumerate(chunk[:action_steps]):
                        if i >= max_queue and not dry_run:
                            wait_joints(n0, i - max_queue + 1, 0.2)
                        send(action)
                    if dry_run:
                        time.sleep(action_steps / hz)  # nothing will step
                    elif not wait_joints(n0, action_steps, 1.0 + action_steps / 10):
                        prog["state_timeouts"] += 1
                    record(cycle_ms, (time.perf_counter() - t_cycle) * 1000, "cycle")
            else:
                # Robot step bookkeeping. `sent` counts actions published;
                # executed() estimates how many the robot has run, from the joint
                # count. Joint updates are lossy, so `offset` absorbs drift: a
                # robot with commands queued publishes every step, so silence
                # means its queue has drained and everything sent has executed.
                steps = {"sent": 0, "offset": -joints["n"], "last_send": 0.0}
                plan = {"chunk": None, "g_obs": 0, "at": 0.0}
                plan_lock = threading.Lock()
                failure: list[BaseException] = []
                quiet_s = 0.15

                def executed() -> int:
                    return min(joints["n"] + steps["offset"], steps["sent"])

                def plan_loop():
                    try:
                        while not stop.is_set():
                            with cond:
                                snapshot = (joints["ts_us"], joints["vec"], joints["at"])
                                g_obs = executed()
                            chunk = observe_and_infer(snapshot, 1.0)
                            if chunk is None:
                                continue
                            now = time.perf_counter()
                            with plan_lock:
                                if plan["at"]:
                                    record(cycle_ms, (now - plan["at"]) * 1000, "cycle")
                                plan.update(chunk=chunk, g_obs=g_obs, at=now)
                                # Steps the robot ran while this chunk was inferred.
                                record(lags, steps["sent"] - g_obs, "lag")
                    except BaseException as exc:
                        failure.append(exc)

                planner = threading.Thread(target=plan_loop, name="gr00t-planner", daemon=True)
                planner.start()
                prog["state"] = "running"
                period = 1.0 / hz
                next_tick = time.perf_counter()
                while time.perf_counter() - t_start < seconds:
                    if failure:
                        raise failure[0]
                    prog["elapsed_s"] = time.perf_counter() - t_start
                    now = time.monotonic()
                    with cond:
                        n, last_update = joints["n"], joints["at"]
                    if n + steps["offset"] > steps["sent"]:
                        steps["offset"] = steps["sent"] - n  # someone else stepped it
                    outstanding = steps["sent"] - (n + steps["offset"])
                    if (
                        outstanding > 0
                        and now - last_update > quiet_s
                        and now - steps["last_send"] > quiet_s
                    ):
                        steps["offset"] = steps["sent"] - n
                        outstanding = 0
                        prog["resyncs"] += 1

                    if outstanding < queue_depth:
                        with plan_lock:
                            chunk, g_obs = plan["chunk"], plan["g_obs"]
                        i = steps["sent"] - g_obs
                        if chunk is not None and i < action_steps:
                            send(chunk[i])
                            steps["sent"] += 1
                            steps["last_send"] = now
                        elif chunk is not None:
                            prog["starved"] += 1  # ran off the end of the newest chunk

                    next_tick += period
                    slack = next_tick - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.perf_counter()
        finally:
            stop.set()
            if planner is not None:
                planner.join(timeout=5.0)
            prog["state"] = "stopped"
            for h in (rx, joints_sub, act_pub):
                try:
                    h.close()
                except Exception:
                    pass
            sess.close()
            self._serving = False

        print(f"serve_stream: {prog}")
        return prog


@app.local_entrypoint()
def fill(suite: str = "libero_spatial"):
    """`modal run modal/gr00t_infer.py::fill` -- download() with the local HF login."""
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("no Hugging Face token: run `hf auth login` first")
    download.remote(hf_token=token, suite=suite)


@app.local_entrypoint()
def main(suite: str = "libero_spatial"):
    for k, v in Gr00tLibero(suite=suite).stats.remote().items():
        print(f"  {k:10s}: {v:.2f}" if isinstance(v, float) else f"  {k:10s}: {v}")
