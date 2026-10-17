"""GR00T N1.7 abc-sim inference over Adamo on Modal: FastGr00t with a training-server LoRA merged in.

Serves runpod/abc_sim/adamo_server.py (routes /start, /stop, /status, /take_control, /lora, /loras)
at a public *.modal.run URL, like abc's deploy/modal_app.py serves its adamo_server. The model is
the base nvidia/GR00T-N1.7-3B from the gr00t-assets Volume, the abc-sim processor built from
runpod/abc_sim/abc_sim_config.py + modal/abc_sim_statistics.json (as the training server builds
it), and, per run, a LoRA from the gr00t-train-lora-checkpoints Volume merged into the action
head, so it costs nothing at inference. /start or /lora with a LoRA name writes it into the live
weights in ~0.1 s without recompiling (the Volume is reloaded first, so LoRAs saved after the
container started are found); /start without "lora" runs the base weights, restored the same way.

    modal deploy modal/gr00t_fast_adamo.py
    GR00T_LORA=abc-sim-r64-b256-300 modal deploy modal/gr00t_fast_adamo.py   # another template LoRA
    uv run --no-project --with modal python modal/run_fast_adamo.py abc-sim --lora abc-sim-r64-b256-500
    modal app stop gr00t-fast-adamo                             # take the endpoint down

The first start on a GPU type builds the ViT and LLM TensorRT engines (~3-5 min) and fills the
Inductor cache (the first graph capture takes ~6 min, later ones ~1 min); both are kept on the
gr00t-assets Volume per GPU and TensorRT version. `modal run modal/gr00t_fast_adamo.py::build`
does the engine build without serving. Every LoRA of the base model shares the engines.

The container scales to zero SCALEDOWN_S after the last request; while the policy loads or a run
is active, _keepalive requests its own public /status, because Modal sees a robot run (Adamo
traffic, no HTTP) as idle.

Environment, read at deploy time: GR00T_GPU (default L40S), GR00T_REGION (default us; empty for
any), GR00T_LORA (the LoRA loaded at startup, default abc-sim-r64-b256-500, a directory name on
gr00t-train-lora-checkpoints; runs can swap in any LoRA with its target layers and projector slot
settings, or the base weights; empty loads only the base weights, with no LoRA swaps).
"""

import os
from pathlib import Path
import time

import modal


APP_NAME = "gr00t-fast-adamo"
GPU = os.environ.get("GR00T_GPU", "L40S")
REGION = os.environ.get("GR00T_REGION", "us") or None
LORA = os.environ.get("GR00T_LORA", "abc-sim-r64-b256-500") or None
SCALEDOWN_S = 180
KEEPALIVE_S = 30

app = modal.App(APP_NAME)
assets = modal.Volume.from_name("gr00t-assets", create_if_missing=True)
loras = modal.Volume.from_name("gr00t-train-lora-checkpoints")

LOCAL = Path(__file__).resolve().parents[1]
REPO = "/opt/gr00t"
ASSETS = "/gr00t_assets"
HF_HOME = f"{ASSETS}/hf"
LORAS = "/loras"
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
EMBODIMENT_TAG = "new_embodiment"
MODALITY_CONFIG = f"{REPO}/runpod/abc_sim/abc_sim_config.py"
STATS = f"{REPO}/modal/abc_sim_statistics.json"
MODEL_DIR = "/root/abc_model"  # base weights (symlinked from the Volume) + abc-sim processor
PROMPT = "sim put the plastic bottles in the bin"
RES = 224

image = (
    modal.Image.debian_slim(python_version="3.12")  # gr00t requires python 3.12
    .apt_install(
        "git",
        # Triton (torch.compile) builds its launchers with a C compiler.
        "build-essential",
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
    # Only the lock goes into the build layer, so editing the code does not repeat the install.
    .add_local_file(LOCAL / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(LOCAL / "uv.lock", f"{REPO}/uv.lock", copy=True)
    # uv.lock plus the train-server extra (fastapi, uvicorn), with TensorRT (cu12) for the
    # FastGr00t engines, minus deepspeed and the aarch64 TensorRT. The aarch64 torchcodec entry is
    # a repo-relative wheel path that is not in this layer, so it is dropped.
    .run_commands(
        f"cd {REPO} && uv export --frozen --no-dev --extra train-server --no-emit-project"
        " --no-hashes --no-emit-package deepspeed"
        " --no-emit-package tensorrt-cu13 --no-emit-package tensorrt-cu13-libs"
        " --no-emit-package tensorrt-cu13-bindings"
        " | grep -v 'aarch64.whl' > /tmp/req.txt",
        "uv pip install --system -r /tmp/req.txt"
        " --extra-index-url https://download.pytorch.org/whl/cu128"
        " --extra-index-url https://pypi.nvidia.com"
        " --index-strategy unsafe-best-match",
    )
    # announce_control needs adamo 1.0.9; numpy is pinned so adamo cannot float it off 1.26.4.
    .run_commands("uv pip install --system 'adamo[video]>=1.0.9' numpy==1.26.4")
    .env(
        {
            "HF_HOME": HF_HOME,
            "PYTHONPATH": REPO,
            "TOKENIZERS_PARALLELISM": "false",
            # Weights come off the Volume; never reach for the hub. GROOT_PATCH_MISTRAL skips
            # transformers' Mistral-tokenizer probe, which also goes to the hub.
            "HF_HUB_OFFLINE": "1",
            "GROOT_PATCH_MISTRAL": "1",
            # Compile and load Inductor kernels in-process. With the default worker pool, a
            # container that hits the compile cache on the Volume waits forever on the workers'
            # futures (idle, 0% CPU) while loading the cached graph.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "GR00T_GPU": GPU,
            "GR00T_LORA": LORA or "",
        }
    )
    .add_local_dir(LOCAL / "gr00t", f"{REPO}/gr00t", ignore=["**/__pycache__", "eval/sim/**"])
    .add_local_file(
        LOCAL / "scripts/deployment/__init__.py", f"{REPO}/scripts/deployment/__init__.py"
    )
    .add_local_dir(
        LOCAL / "scripts/deployment/fast",
        f"{REPO}/scripts/deployment/fast",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(
        LOCAL / "scripts/deployment/trt",
        f"{REPO}/scripts/deployment/trt",
        ignore=["**/__pycache__"],
    )
    .add_local_dir(
        LOCAL / "runpod/abc_sim", f"{REPO}/runpod/abc_sim", ignore=["**/__pycache__", "*.html"]
    )
    .add_local_file(LOCAL / "modal/abc_sim_statistics.json", STATS)
)


def gpu_cache_dir(kind: str) -> Path:
    """Per-GPU (and, for engines, per-TensorRT) cache directory on the assets Volume."""
    import tensorrt
    import torch

    gpu = torch.cuda.get_device_name().replace(" ", "_")
    suffix = f"trt{tensorrt.__version__}" if kind == "engines" else f"torch{torch.__version__}"
    return Path(ASSETS) / "fast_adamo" / kind / f"{gpu}-{suffix}"


def assemble_model() -> None:
    """MODEL_DIR: the base model's weights next to the abc-sim processor the LoRA was trained with
    (built as gr00t.train_server.launch.build builds it)."""
    import tempfile

    from gr00t.configs.finetune_config import FinetuneConfig
    from gr00t.experiment.launch_finetune import build_config
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from gr00t.train_server.launch import load_statistics
    from huggingface_hub import snapshot_download
    from scripts.deployment.fast.lora import assemble_model_dir

    ft = FinetuneConfig(
        base_model_path=BASE_MODEL,
        dataset_path=STATS,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config_path=MODALITY_CONFIG,
    )
    config = build_config(ft)
    config.validate()
    tag = ft.embodiment_tag
    with tempfile.TemporaryDirectory() as tmp:
        processor = Gr00tN1d7Pipeline(config, Path(tmp))._create_processor()
        processor.set_statistics({tag.value: load_statistics(STATS, tag, config)}, override=True)
        processor.save_pretrained(tmp)
        assemble_model_dir(snapshot_download(BASE_MODEL), tmp, MODEL_DIR)


def ensure_engines() -> Path:
    """The engine directory for this GPU, building the engines into it if they are missing."""
    import subprocess
    import sys

    engines = gpu_cache_dir("engines")
    names = [f"vit_bf16_{RES}x{RES}.engine", f"llm_static_bf16_{RES}x{RES}.engine"]
    if all((engines / n).is_file() for n in names):
        return engines
    print(f"building FastGr00t engines into {engines} (~3-5 min)", flush=True)
    t0 = time.time()
    tmp = engines.with_name(f".{engines.name}.tmp")
    subprocess.run(
        [sys.executable, "-m", "scripts.deployment.fast.build_engines",
         "--model-path", MODEL_DIR, "--embodiment-tag", EMBODIMENT_TAG,
         "--height", str(RES), "--width", str(RES),
         "--engine-dir", str(tmp), "--onnx-dir", "/tmp/trt_fast_onnx"],
        cwd=REPO,
        check=True,
    )  # fmt: skip
    tmp.rename(engines)
    assets.commit()
    print(f"engines built in {time.time() - t0:.0f} s", flush=True)
    return engines


def use_compile_cache() -> None:
    cache = gpu_cache_dir("inductor")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))


@app.function(image=image, gpu=GPU, volumes={ASSETS: assets}, cpu=8.0, memory=32768, timeout=3600)
def build() -> str:
    """Build the engines for this GPU type ahead of the first serve (optional)."""
    assemble_model()
    return str(ensure_engines())


@app.function(
    image=image,
    gpu=GPU,
    region=REGION,
    volumes={ASSETS: assets, LORAS: loras},
    cpu=8.0,
    memory=32768,
    # One stateful server: the policy, its compiled graphs and the robot connection live here.
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
    timeout=600,  # per request; runs live on the server's threads
    startup_timeout=1800,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label=APP_NAME)
def serve():
    import logging
    import threading

    from runpod.abc_sim.adamo_server import Args, create_app, resolve_under
    import torch

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", force=True)
    # Half the CPUs; the rest decode video. (The image defaults torch to one thread.)
    torch.set_num_threads(4)
    use_compile_cache()

    def resolve_lora(name: str) -> Path:
        try:
            loras.reload()  # LoRAs saved since the container started
        except Exception as e:
            print(f"LoRA volume reload failed ({e!r}); using the mounted state")
        return resolve_under(LORAS)(name)

    args = Args(
        model_path=MODEL_DIR,
        engine_dir="",  # set once the engines are ready (prepare below)
        lora_root=LORAS,
        lora=os.environ.get("GR00T_LORA") or None,
        prompt=PROMPT,
        embodiment_tag=EMBODIMENT_TAG,
    )

    def prepare() -> None:
        assemble_model()
        args.engine_dir = str(ensure_engines())

    def on_ready() -> None:
        try:
            assets.commit()  # the Inductor/Triton cache from the first graph capture
        except Exception as e:
            print(f"assets commit failed: {e!r}")

    web = create_app(
        args,
        resolve_lora=resolve_lora,
        lora_root=lambda: Path(LORAS),
        on_ready=on_ready,
        prepare=prepare,
    )
    threading.Thread(target=_keepalive, args=(web.state.worker,), daemon=True).start()
    return web


def _keepalive(worker) -> None:
    """Request our own public /status while the policy loads or a run is active.

    Modal counts only HTTP requests through its proxy as activity, so without this the container
    scales down SCALEDOWN_S after /start, mid-run, or during a first engine build."""
    import urllib.request

    url = None
    while True:
        time.sleep(KEEPALIVE_S)
        if worker.phase in ("ready", "failed") and worker.run is None:
            continue
        try:
            url = url or serve.get_web_url()
            with urllib.request.urlopen(f"{url}/status", timeout=30):
                pass
        except Exception as e:
            print(f"keepalive request failed ({e!r}); the container may scale down mid-run")
