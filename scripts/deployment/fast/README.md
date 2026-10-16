# FastGr00t: low-latency GR00T N1.7 inference (with merged LoRA)

The whole policy, from uint8 camera frames to normalized actions, runs as **one CUDA graph** per
instruction:

GPU image preprocessing → bf16 ViT (TRT) → image-token scatter → bf16 LLM (TRT) → VLLN →
VL self-attention → state encoder → 4 flow-matching steps (torch.compile).

| RTX 4090, bf16, batch 1, 3 cameras × 224×224, 40-step chunk | `get_action` |
|---|---:|
| `Gr00tPolicy` eager | 167 ms |
| `Gr00tPolicy` eager, LoRA unmerged (peft) | 280 ms |
| **FastGr00t, LoRA merged** | **20.6 ms** |
| FastGr00t, LoRA unmerged | 24.2 ms |

Normalized actions match eager PyTorch with the same noise at cosine 0.99999. On the GPU the time
splits into ViT 4.5 ms, LLM 3.3 ms, VL self-attention 0.8 ms, cross-attention K/V 1.4 ms and four
DiT steps ~9.7 ms.

## How

- **GPU preprocessing** (`gpu_preproc.py`): for a fixed input size the resize → crop → resize →
  Qwen resize chain is linear per axis, so it is extracted once and applied as matmuls (same uint8
  rounding points; max difference 2 LSB).
- **bf16 ViT engine** instead of the stock fp32 one (2× faster).
- **Static LLM engine**: deepstack features are dense inputs added in place of `masked_scatter`,
  which removes the data-dependent `NonZero`, so the engine can be captured in a CUDA graph.
- **Action head rewrite** (`FastActionHead`, same math as `AlternateVLDiT`): cross-attention K/V
  once per call instead of once per step, text/image blocks gather their tokens instead of masking,
  timestep embeddings and AdaLN/output modulations precomputed, `torch.compile` (max-autotune).
- Instruction tokens, text embeddings and M-RoPE position ids are cached per instruction;
  relative→absolute action decoding is vectorized.

## LoRA

LoRA adapters from the training server (`gr00t/train_server`, `--lora`; a `save_weights` directory
with `trainable.safetensors` + `lora_config.json`) train only the action head, so:

- they are **merged** into the action-head weights at load (`lora.load_lora_policy`), and inference
  costs exactly what the base model costs;
- the ViT and LLM engines are shared by every LoRA of the same base model, and the VL
  self-attention runs in PyTorch inside the graph by default, so a LoRA needs no engine of its own;
- `LoraSwapper.swap(dir)` + `FastGr00t.refresh_weights()` write another LoRA into the live weights
  in place (~0.1 s including reading the file; ~40 ms from tensors already on the GPU) — no
  recompile, no graph recapture.

`model_path` must hold the base weights the LoRA was trained from **and** the embodiment's
processor files. A LoRA save has neither; `lora.assemble_model_dir(base, processor_dir, out)`
symlinks the base weights next to the processor files of e.g. a sampler save of the same run.

## Usage

```bash
# once per GPU type / TensorRT version / camera size (~3 min)
python -m scripts.deployment.fast.build_engines --model-path <ckpt> --embodiment-tag new_embodiment \
    --height 224 --width 224 --engine-dir trt_fast/engines

# check against eager PyTorch and time it
python -m scripts.deployment.fast.benchmark --model-path <ckpt> --embodiment-tag new_embodiment \
    --lora-dir <lora dir> --engine-dir trt_fast/engines --swap-lora-dir <lora dir>
```

```python
from scripts.deployment.fast.fast_gr00t import FastGr00t
from scripts.deployment.fast.lora import load_lora_policy

policy, swapper = load_lora_policy(model_path, lora_dir, "new_embodiment")
fast = FastGr00t(policy, "trt_fast/engines", H=224, W=224)
action = fast.get_action(observation)   # same observation/return format as Gr00tPolicy.get_action

swapper.swap(other_lora_dir)            # later: switch LoRA in place
fast.refresh_weights()
```

The first call for a new instruction compiles and captures the graph (~1 min with a warm Inductor
cache, ~6 min on a fresh machine); later calls replay it. Frames must have the size the engines
were built for, and batch size is 1.

`python -m runpod.abc_sim.adamo_bridge --engine-dir ... --lora-dir ...` drives the abc-sim robot over
Adamo with this path.
