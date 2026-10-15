"""GR00T N1.7 Tinker-protocol training server on Modal: serverless, one H100.

The FastAPI app from gr00t/train_server is served as a Modal ASGI endpoint. Nothing runs
until a client calls it: the first request (normally create_model) starts an H100
container, and the container shuts down once no request has arrived for IDLE_SECONDS
(clients poll retrieve_future while training, which keeps it up). The next client gets
a fresh container and a fresh model. Within one container, a new create_model also
starts from scratch: it replaces the previous model, whose later calls fail.

max_containers=1 keeps every request on the container that holds the model in memory.

  modal secret create gr00t-train-api-key GR00T_TRAIN_API_KEY=ak_...
  modal run    modal/gr00t_train.py::download --hf-token "$(cat ~/.cache/huggingface/token)"
  modal deploy modal/gr00t_train.py
  # base URL: https://<workspace>--gr00t-train.modal.run (printed by deploy)

Weights come from the gr00t-assets Volume (shared with gr00t_infer.py); checkpoints go to
the gr00t-train-checkpoints Volume, committed after every save, so they outlive the
container. Normalization stats are abc_sim_statistics.json next to this file, exported from
the abc-sim GR00T dataset; swap EMBODIMENT_TAG, MODALITY_CONFIG and STATS for another robot.
"""

from pathlib import Path

import modal


app = modal.App("gr00t-train")
assets = modal.Volume.from_name("gr00t-assets", create_if_missing=True)
checkpoints = modal.Volume.from_name("gr00t-train-checkpoints", create_if_missing=True)

LOCAL = Path(__file__).resolve().parents[1]
REPO = "/opt/gr00t"
ASSETS = "/gr00t_assets"
HF_HOME = f"{ASSETS}/hf"
CHECKPOINTS = "/checkpoints"
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
BACKBONE = "nvidia/Cosmos-Reason2-2B"

EMBODIMENT_TAG = "new_embodiment"
MODALITY_CONFIG = f"{REPO}/runpod/abc_sim/abc_sim_config.py"
STATS = f"{REPO}/modal/abc_sim_statistics.json"

IDLE_SECONDS = 180  # shut the H100 down after 3 minutes without a request

image = (
    modal.Image.debian_slim(python_version="3.12")  # gr00t requires python 3.12
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("uv")
    # Only the lock goes into the build layer, so editing gr00t/ does not repeat the install.
    .add_local_file(LOCAL / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(LOCAL / "uv.lock", f"{REPO}/uv.lock", copy=True)
    # uv.lock plus the train-server extra (fastapi, uvicorn), minus the TensorRT and
    # deepspeed stacks training never imports. The aarch64 torchcodec entry is a
    # repo-relative wheel path that is not in this layer, so it is dropped.
    .run_commands(
        f"cd {REPO} && uv export --frozen --no-dev --extra train-server --no-emit-project"
        " --no-hashes --no-emit-package deepspeed"
        " --no-emit-package tensorrt-cu12 --no-emit-package tensorrt-cu12-libs"
        " --no-emit-package tensorrt-cu12-bindings"
        " --no-emit-package tensorrt-cu13 --no-emit-package tensorrt-cu13-libs"
        " --no-emit-package tensorrt-cu13-bindings"
        " | grep -v 'aarch64.whl' > /tmp/req.txt",
        "uv pip install --system -r /tmp/req.txt"
        " --extra-index-url https://download.pytorch.org/whl/cu128"
        " --index-strategy unsafe-best-match",
    )
    .env({"HF_HOME": HF_HOME, "PYTHONPATH": REPO, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(LOCAL / "gr00t", f"{REPO}/gr00t", ignore=["**/__pycache__", "eval/sim/**"])
    .add_local_file(LOCAL / "runpod/abc_sim/abc_sim_config.py", MODALITY_CONFIG)
    .add_local_file(LOCAL / "modal/abc_sim_statistics.json", STATS)
)


@app.function(image=image, volumes={ASSETS: assets}, timeout=3600, cpu=8.0, memory=16384)
def download(hf_token: str = "") -> None:
    """Cache the base model and the gated backbone on the Volume (run once)."""
    from huggingface_hub import snapshot_download

    if not hf_token:
        raise ValueError(f"download needs hf_token with access to {BACKBONE} (gated)")
    for repo in (BASE_MODEL, BACKBONE):
        print(repo, "->", snapshot_download(repo, token=hf_token))
    assets.commit()


@app.function(
    image=image,
    gpu="H100",
    volumes={ASSETS: assets, CHECKPOINTS: checkpoints},
    secrets=[modal.Secret.from_name("gr00t-train-api-key")],
    max_containers=1,
    scaledown_window=IDLE_SECONDS,
    timeout=600,  # per request; the work itself runs on the server's worker thread
    startup_timeout=900,
)
@modal.concurrent(max_inputs=256)
@modal.asgi_app(label="gr00t-train")
def serve():
    import os

    # Weights come off the Volume; never reach for the hub. GROOT_PATCH_MISTRAL skips
    # transformers' Mistral-tokenizer probe, which also goes to the hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GROOT_PATCH_MISTRAL"] = "1"

    from gr00t.train_server.app import create_app
    from gr00t.train_server.launch import ServerConfig, build

    cfg = ServerConfig(
        base_model_path=BASE_MODEL,
        embodiment_tag=EMBODIMENT_TAG,
        stats=STATS,
        modality_config_path=MODALITY_CONFIG,
        checkpoint_dir=CHECKPOINTS,
        replace_model=True,
    )
    spec, create_session = build(cfg)
    return create_app(
        api_key=os.environ["GR00T_TRAIN_API_KEY"],
        spec=spec,
        create_session=create_session,
        checkpoint_dir=CHECKPOINTS,
        first_seq_id=cfg.first_seq_id,
        replace_model=True,
        on_save=lambda path: checkpoints.commit(),
    )
