# GR00T N1.7 training server (Tinker HTTP protocol)

A FastAPI server that fine-tunes GR00T N1.7 through Tinker-style calls: one process, one GPU,
one model. It reuses GR00T's own model loading (as `launch_finetune.py`), processor (image
transforms, tokenization, normalization, relative actions) and flow-matching loss. The VLM is
always frozen. By default it is a full fine-tune of the projector and diffusion action head, and
`lora_config.rank` is accepted and ignored. With `--lora` the action head trains through LoRA
adapters instead (see [LoRA](#lora)). `lora_config.seed` seeds the run either way.

## Requirements

- One NVIDIA GPU with 80 GB of memory (tested on an H100 80GB). The 3B model is loaded in fp32,
  and the trainable action head carries AdamW state. abc-sim (3 cameras, 40-step chunks) at
  batch 16 used about 40 GB and took about 1 s per forward_backward + optim_step.
- CUDA 12.8+ driver, Python 3.12, [uv](https://docs.astral.sh/uv/).
- A Hugging Face token with access to the gated `nvidia/Cosmos-Reason2-2B` backbone (accept
  its license on the model page first).
- About 40 GB of disk for the venv and the base models, plus room for checkpoints: a
  `save_weights_for_sampler` directory is about 12 GB and a `save_weights` directory (weights +
  AdamW state) about 24 GB on N1.7-3B. A 150 GB container disk leaves room for a few of each.
- Normalization stats for your embodiment: either the GR00T-format LeRobot dataset you train on,
  or a `statistics.json` (see [Stats](#stats)).

## Launch on a new RunPod pod

### 1. Create the pod

In the RunPod console (or with `runpodctl`), create a Secure Cloud pod with:

- GPU: 1x H100 SXM (`NVIDIA H100 80GB HBM3`), with a CUDA 12.8 minimum.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- Container disk: 150 GB.
- Exposed ports: `22/tcp` for SSH. Also expose `8000/http` if clients should reach the server
  through the RunPod proxy instead of an SSH tunnel (see step 6).
- Your SSH public key in the `PUBLIC_KEY` env var, or registered on the account with "SSH" enabled.

Note the pod's direct SSH command from the console, e.g. `ssh root@<ip> -p <port>`. It changes
every time the pod is stopped and started. The steps below use these shell variables:

```bash
POD=root@<ip>
PORT=<ssh port>
SSH="ssh -i ~/.ssh/id_ed25519 -p $PORT"
```

### 2. Copy the repo, your data and your HF token

From your machine, in this repository's root:

```bash
$SSH $POD 'mkdir -p /root/gr00t /root/data /root/.cache/huggingface'
rsync -az -e "$SSH" --exclude .venv --exclude checkpoints --exclude __pycache__ --exclude .git \
    ./ $POD:/root/gr00t/
# Your GR00T-format LeRobot dataset (used for normalization stats):
rsync -az -e "$SSH" /path/to/your_dataset/ $POD:/root/data/your_dataset/
# Hugging Face token for the gated backbone (or run `hf auth login` on the pod):
$SSH $POD 'umask 077; cat > /root/.cache/huggingface/token' < ~/.cache/huggingface/token
```

Keep everything on the container disk (`/root`). If you attach a network volume at
`/workspace`, use it only for checkpoints: building a venv there is slow, and it rejects `chown`.

### 3. Install the packages with uv

On the pod (`$SSH $POD`):

```bash
export LC_ALL=C.UTF-8
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

cd /root/gr00t
export UV_PROJECT_ENVIRONMENT=/root/venv   # venv on container disk
uv sync --extra train-server               # GR00T + fastapi/uvicorn, pinned by uv.lock
```

`uv sync` takes a few minutes, mostly for torch, flash-attn and TensorRT wheels. Add
`--extra dev` if you also want to run the tests on the pod.

### 4. Download the models

```bash
/root/venv/bin/hf download nvidia/GR00T-N1.7-3B
/root/venv/bin/hf download nvidia/Cosmos-Reason2-2B   # gated; needs the token from step 2
```

These go to the Hugging Face cache (`/root/.cache/huggingface/hub`), where the server looks for
them. To run offline later, set `HF_HUB_OFFLINE=1`.

### 5. Start the server

Pick an API key. Clients send it as `X-API-Key`; the adamo client expects the `ak_...` format.

```bash
cd /root/gr00t
export GR00T_TRAIN_API_KEY=ak_$(openssl rand -hex 16)
echo "$GR00T_TRAIN_API_KEY" > /root/train_server_api_key   # so you can read it back later

nohup /root/venv/bin/python -m gr00t.train_server.launch \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag new_embodiment \
    --modality-config-path runpod/abc_sim/abc_sim_config.py \
    --stats /root/data/your_dataset \
    --checkpoint-dir /root/train_server_checkpoints \
    --host 0.0.0.0 --port 8000 \
    > /root/train_server.log 2>&1 &

tail -f /root/train_server.log   # wait for "Uvicorn running on http://0.0.0.0:8000"
```

The server is ready in about 20 s (it computes stats but doesn't load the model yet). The
model loads on the GPU when a client calls `create_model`, and `retrieve_future` returns
`try_again` until it's done. If you start the server from a one-shot `ssh $POD '...'` command
instead of an interactive shell, use `setsid nohup ... < /dev/null &` so the SSH command can
return.

Use your own `--embodiment-tag` and `--modality-config-path` for other robots (see the
`examples/*/` configs). The first `--stats` run on a dataset writes `meta/stats.json` and
`meta/relative_stats.json` into it; later runs reuse them.

### 6. Connect a client

Choose one:

- **SSH tunnel** (private, works with only `22/tcp` exposed). On the client machine:

  ```bash
  ssh -N -L 8000:localhost:8000 -i ~/.ssh/id_ed25519 -p $PORT $POD
  # base URL: http://127.0.0.1:8000
  ```

- **RunPod proxy** (needs `8000/http` exposed at pod creation): the base URL is
  `https://<pod-id>-8000.proxy.runpod.net`. The proxy URL is public, so the API key is the
  only protection.

Check that it works:

```bash
curl -s http://127.0.0.1:8000/api/v1/healthz                      # {"status":"ok"}
curl -s -H "X-API-Key: $KEY" -X POST http://127.0.0.1:8000/api/v1/get_info \
     -H 'Content-Type: application/json' -d '{"model_id": "x"}'   # 404 unknown model_id = auth OK
```

To exercise the whole API with real frames from your dataset (run it on the pod, or anywhere
with the repo and the dataset):

```bash
/root/venv/bin/python scripts/train_server_smoke.py --url http://127.0.0.1:8000 \
    --api-key "$(cat /root/train_server_api_key)" \
    --dataset /root/data/your_dataset \
    --modality-config-path runpod/abc_sim/abc_sim_config.py \
    --steps 20 --batch-size 16 --policy-check
```

The server holds one model. To start over, restart the server process, or launch it with
`--replace-model` so a new `create_model` replaces the old model.

### 7. Stop

Kill the server (`pkill -f '^/root/venv/bin/python -m gr00t.train_server.launch'`; the
anchored pattern keeps `pkill` from matching your own shell). Copy any checkpoints you want to keep
off the pod (`rsync -az -e "$SSH" $POD:/root/train_server_checkpoints/<name> ./`), then stop or
terminate the pod. Container disk is wiped when a pod stops.

## Launch serverless on Modal

`modal/gr00t_train.py` serves the same app as a Modal ASGI endpoint on one H100, billed only
while a container runs:

- The first request (normally `create_model`) starts the container. Booting takes a minute or
  two before `create_model` can begin loading the model.
- The container shuts down after 3 minutes with no request (`IDLE_SECONDS`). Clients poll
  `retrieve_future` while training, which keeps it alive.
- Every client starts from scratch. A cold container has no model, and inside a live
  container a new `create_model` replaces the previous model (`--replace-model`). The old
  model's queued and later calls then fail with a "replaced" error.
- `max_containers=1` keeps all requests on the container that holds the model.
- Checkpoints go to the `gr00t-train-checkpoints` Volume at `/checkpoints/<path>`, committed
  after every save, so they survive the container. `load_weights` with a relative path
  reads from there.
- Saves are slow on Modal, because the 12 GB (sampler) or 24 GB (with optimizer) checkpoint is
  written to the network Volume. While one runs, a `retrieve_future` poll can occasionally
  come back `408 Request Timeout` from Modal's proxy. Polling is safe to repeat, so clients
  should treat 408, 5xx and network errors on `retrieve_future` as "still pending"
  (`scripts/train_server_smoke.py` does).

One-time setup, from this repository's root:

```bash
modal secret create gr00t-train-api-key GR00T_TRAIN_API_KEY=ak_$(openssl rand -hex 16)
modal run modal/gr00t_train.py::download --hf-token "$(cat ~/.cache/huggingface/token)"
modal deploy modal/gr00t_train.py
```

`download` caches `nvidia/GR00T-N1.7-3B` and the gated `nvidia/Cosmos-Reason2-2B` on the
`gr00t-assets` Volume. Containers then run with `HF_HUB_OFFLINE=1` and need no token.
`deploy` prints the base URL, `https://<workspace>--gr00t-train.modal.run`. The app is set up
for abc-sim. For another robot, change `EMBODIMENT_TAG`, `MODALITY_CONFIG` and `STATS` at the
top of the file; for stats, export a statistics JSON as described in [Stats](#stats).

Check it (the first call wakes the container):

```bash
URL=https://<workspace>--gr00t-train.modal.run
curl -s $URL/api/v1/healthz
python scripts/train_server_smoke.py --url $URL --api-key ak_... \
    --dataset ~/.cache/huggingface/lerobot/local/abc_sim_bottles_gr00t \
    --modality-config-path runpod/abc_sim/abc_sim_config.py --steps 3
```

## Launch locally

```bash
uv sync --extra train-server
GR00T_TRAIN_API_KEY=ak_... uv run python -m gr00t.train_server.launch \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag new_embodiment \
    --modality-config-path runpod/abc_sim/abc_sim_config.py \
    --stats ~/.cache/huggingface/lerobot/local/abc_sim_bottles_gr00t \
    --checkpoint-dir ./train_server_checkpoints \
    --port 8000
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--base-model-path` | required | Checkpoint behind the API name `nvidia/GR00T-N1.7` (HF repo id or local dir). |
| `--embodiment-tag` | required | Embodiment tag, e.g. `new_embodiment`. |
| `--modality-config-path` | none | Python file that registers the modality config for that tag. |
| `--stats` | required | Dataset dir or statistics JSON (see below). |
| `--checkpoint-dir` | `./train_server_checkpoints` | Where saves go; relative `load_weights` paths resolve here. |
| `--api-key` | `$GR00T_TRAIN_API_KEY` | Expected `X-API-Key`. |
| `--host` / `--port` | `0.0.0.0` / `8000` | Bind address. |
| `--first-seq-id` | `1` | seq_id of each model's first ordered call (Tinker: `create_model` is 0). |
| `--replace-model` | off | A new `create_model` replaces the held model instead of being refused. |
| `--augment` / `--no-augment` | on | GR00T's training-time augmentation (image crop/jitter, processor state dropout). |
| `--state-dropout-prob` | `0.2` | Same default as `launch_finetune.py`. |
| `--device` | `cuda` | Torch device. |
| `--lora` / `--no-lora` | off | Train the action head (DiT, VL self-attention, embodiment projector slot) through LoRA, rank from `create_model` (see [LoRA](#lora)). |
| `--lora-alpha` | `2 * rank` | LoRA alpha. |
| `--lora-dropout` | `0.0` | LoRA dropout. |
| `--projector-init-tag` | none | With `--lora`: start the projector slot from this embodiment's slot, e.g. `xdof`. |
| `--projector-init-layout` | none | With `--projector-init-tag`: `key=offset,...`, where each of your keys starts in that embodiment's padded layout. |

## LoRA

With `--lora`, the action head trains through LoRA adapters of rank `lora_config.rank` (which
must be > 0). The VLM stays frozen, as without LoRA.

- **Projectors.** GR00T's state encoder, action encoder and action decoder hold one weight slot
  per embodiment (`CategorySpecificLinear`, 32 slots). Only your embodiment's slot is kept on the
  GPU, as a plain `nn.Linear`; the other 31 slots are removed from the network (parked on the
  CPU, to rebuild the full layout at save time). That slot gets LoRA adapters like any other
  linear layer. For `new_embodiment` it is slot 10, which in `nvidia/GR00T-N1.7-3B` is still at
  its random initialization, so the adapters learn the embodiment mapping on top of a fixed
  random projection.
- **Starting from a pretrained slot.** `--projector-init-tag xdof` starts from another
  embodiment's pretrained slot instead (xdof is slot 27). Its padded state/action vectors are
  laid out differently, so `--projector-init-layout` says where each of your keys starts in that
  layout, and the slot's input rows (state and action encoders) and output columns (action
  decoder) are permuted to match yours. The trained slot is still saved into your embodiment's
  own slot (10), so inference uses the `new_embodiment` tag as usual and slot 27 is unchanged.
  xdof's layout (state and action): left EEF 0-8, right EEF 9-17, left gripper 18, right
  gripper 19, left joints 20-25, right joints 26-31. For abc-sim (6 joints + 1 gripper per arm,
  like xdof):

  ```bash
  --lora --projector-init-tag xdof \
      --projector-init-layout left_arm=20,left_gripper=18,right_arm=26,right_gripper=19
  ```

  The xdof EEF dims get no input (they map to your padding). Joint order, sign and gripper
  convention should match xdof's for the pretrained weights to help.
- **DiT and VL self-attention.** Every `nn.Linear` gets an adapter; base weights are frozen.
- **Still trained in full:** the position embedding and `vlln` (1.6 M parameters).

On N1.7-3B at rank 32 that is 40.8 M trainable parameters (39.2 M LoRA, of which 33.9 M in the
DiT and 4.7 M in the VL self-attention), and about 300 M fewer parameters on the GPU.

Checkpoints change as follows:

- `save_weights` writes only the trainable tensors to `trainable.safetensors` (156 MB at
  rank 32), plus `lora_config.json` and `optimizer.pt` (311 MB). It holds no base weights
  and no processor.
- `save_weights_for_sampler` merges the adapters into the base weights, writes the projector
  slot back into the 32-slot tensors (other slots unchanged), and saves a normal full checkpoint
  that `Gr00tPolicy` loads directly.
- `load_weights` accepts only `save_weights` checkpoints from a LoRA model with the same rank,
  alpha, dropout, embodiment slot and projector init, on top of the same `--base-model-path`.

On Modal, `GR00T_TRAIN_LORA=1 modal deploy modal/gr00t_train.py` deploys the LoRA server as a
separate app, `gr00t-train-lora` (base URL `https://<workspace>--gr00t-train-lora.modal.run`),
with its own `gr00t-train-lora-checkpoints` Volume. It starts from the xdof slot; set
`PROJECTOR_INIT_TAG = None` in the file to start from `new_embodiment`'s own slot.

## Stats

`--stats` is either:

- a GR00T LeRobot dataset directory. Stats and relative-action stats are computed and cached
  in its `meta/`, exactly as fine-tuning does.
- a JSON file in GR00T's statistics format (`{"state", "action", "relative_action"}`),
  optionally nested under the embodiment tag, like a checkpoint's `statistics.json`.

The stats are also written into every checkpoint (`statistics.json`), so `Gr00tPolicy` loads a
`save_weights_for_sampler` directory directly.

## Routes (prefix `/api/v1`, header `X-API-Key`)

| Route | Body | Result (via `retrieve_future` unless noted) |
|---|---|---|
| `GET/POST healthz` | – | `{"status": "ok"}` (no key needed, synchronous) |
| `create_model` | `base_model: "nvidia/GR00T-N1.7"`, `lora_config {rank, seed}`, `session_id?`, `model_seq_id?`, `user_metadata?` | `{"type": "create_model", "model_id"}` |
| `forward_backward` | `model_id`, `seq_id`, `forward_backward_input {data, loss_fn: "flow_matching"}` | `{"type": "forward_backward", "loss_fn_output_type": "ArrayRecord", "loss_fn_outputs": [{"loss": TensorData}], "metrics": {"loss:mean"}}` |
| `optim_step` | `model_id`, `seq_id`, `adam_params {learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm}` | `{"type": "optim_step", "metrics": {"grad_norm", "learning_rate", "step"}}` |
| `save_weights` | `model_id`, `seq_id`, `path` (name), `ttl_seconds` (ignored) | `{"type": "save_weights", "path"}`: weights + processor/stats + `optimizer.pt` (with `--lora`: trainable tensors + `optimizer.pt` only) |
| `save_weights_for_sampler` | same | `{"type": "save_weights_for_sampler", "path"}`: weights + processor/stats (loads in `Gr00tPolicy`) |
| `load_weights` | `model_id`, `seq_id`, `path` (name or absolute dir), `optimizer` | `{"type": "load_weights", "path"}` |
| `get_info` | `model_id` | synchronous: `{"model_id", "model_data": {"base_model", "action_horizon", "image_names", "state_names", "action_names", "state_dims", "action_dims"}}` |
| `retrieve_future` | `request_id` | synchronous: the result or `{"error", "category"}` once the call is done, or `{"type": "try_again", ...}` if it is still running after 20 s |

Every other call answers `{"request_id", "model_id"}` immediately.

## Semantics

- **Ordering.** Calls with a seq_id run on one GPU worker strictly in seq_id order per model,
  waiting for gaps. A repeated seq_id returns the original request_id without running again.
- **Validation.** Submissions are checked before queuing (base model, datum names and shapes,
  image decoding, save/load paths, seq_id already past). A rejected submission returns 4xx with
  `{"error", "category": "user"}` and does not use up its seq_id.
- **Failures.** A queued call that fails returns `{"error", "category"}` (`user` for bad input,
  `server` otherwise). Every later call for that model then fails with an error naming the first
  failed call. Failed saves do not poison the model.
- **Gradients.** Each `forward_backward` backpropagates GR00T's masked-mean flow-matching loss
  over its own batch; several calls before an `optim_step` sum their gradients. `optim_step`
  runs AdamW with the request's hyperparameters, clips only when `grad_clip_norm > 0`, then
  zeroes gradients. bf16 autocast and TF32 match fine-tuning defaults.

## Datum

```json
{
  "model_input": {"chunks": [
    {"type": "image", "name": "head", "format": "png", "data": "<base64>"},
    {"type": "state", "name": "left_arm", "data": {"data": [...], "dtype": "float32", "shape": [6]}},
    {"type": "text", "text": "put the bottles in the bin"}
  ]},
  "loss_fn_inputs": {"left_arm": {"data": [...], "dtype": "float32", "shape": [40, 6]}}
}
```

Images, states and action keys must match the modality config exactly (see `get_info`); one
image per camera per video delta index, one text chunk. States are `[dim]` (or
`[steps, dim]`); actions are raw absolute chunks `[action_horizon, dim]` (`shape` may be
omitted for a flat list). Keys configured as RELATIVE are converted using the datum's state,
as in GR00T fine-tuning.
