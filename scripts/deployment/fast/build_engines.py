# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the backbone TRT engines FastGr00t needs for one camera size.

    python -m scripts.deployment.fast.build_engines --model-path <ckpt> --embodiment-tag new_embodiment \
        --height 224 --width 224 --engine-dir trt_fast/engines

vit  bf16 Qwen3-VL ViT for a fixed pixel_values shape (the stock export is fp32: 2x slower).
llm  bf16 LLM whose deepstack inputs are dense (B, S, H) tensors added in place of the stock
     masked_scatter, so the engine has no data-dependent shapes and can be captured in a CUDA graph.

Both depend only on the frozen backbone, so every action-head fine-tune or LoRA of the same base
model reuses them. Engines are specific to the GPU model and TensorRT version.
"""

from dataclasses import dataclass
import inspect
import os
import re
import textwrap

from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import onnx
from scripts.deployment.trt import export_onnx_n1d7 as ex
from scripts.deployment.trt.build_tensorrt_engine import build_engine
import torch
import tyro


@dataclass
class Config:
    model_path: str
    """Checkpoint with the embodiment's processor (base weights suffice: the backbone is frozen)."""
    embodiment_tag: str
    height: int = 224
    width: int = 224
    engine_dir: str = "trt_fast/engines"
    onnx_dir: str = "trt_fast/onnx"
    parts: tuple[str, ...] = ("vit", "llm")
    seq_margin: tuple[int, int] = (20, 100)
    """The LLM engine accepts the captured sequence length minus/plus these (other prompts)."""
    workspace_mb: int = 8192


def dummy_observation(policy: Gr00tPolicy, height: int, width: int) -> dict:
    """Random frames of the given size and a mid-range state; only shapes matter for the export."""
    mc = policy.get_modality_config()
    rng = np.random.default_rng(0)
    T = len(mc["video"].delta_indices)
    video = {
        k: rng.integers(0, 255, (1, T, height, width, 3), dtype=np.uint8)
        for k in mc["video"].modality_keys
    }
    stats = policy.processor.state_action_processor.statistics[policy.embodiment_tag.value]["state"]
    Ts = len(mc["state"].delta_indices)
    state = {}
    for k in mc["state"].modality_keys:
        lo, hi = np.asarray(stats[k]["min"]), np.asarray(stats[k]["max"])
        state[k] = ((lo + hi) / 2).astype(np.float32)[None, None].repeat(Ts, 1)
    lang = {mc["language"].modality_keys[0]: [["pick up the object"]]}
    return {"video": video, "state": state, "language": lang}


def static_llm_exporter():
    """The stock export_llm_to_onnx with dense deepstack inputs (same math, no NonZero)."""
    src = textwrap.dedent(inspect.getsource(ex.export_llm_to_onnx))
    src, n = re.subn(
        r"( +)B, S, H = hidden_states\.shape\n.*?\n\1return hidden_states\n",
        lambda m: m.group(1)
        + "return hidden_states + visual_embeds  # dense: zeros off the image tokens\n",
        src,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError("export_llm_to_onnx changed: _deepstack_add not found")
    replacements = {
        'ds_dummy = torch.randn_like(ds, dtype=dtype, device="cuda")':
            'ds_dummy = torch.randn(batch_size, seq_len, ds.shape[-1], dtype=dtype, device="cuda")',
        '            name = f"deepstack_{i}"\n            input_names.append(name)':
            '            name = f"deepstack_{i}"\n            input_names.append(name)\n'
            '            llm_dynamic_axes[name] = {1: "seq_len"}',
    }  # fmt: skip
    for old, new in replacements.items():
        if old not in src:
            raise RuntimeError(f"export_llm_to_onnx changed: {old!r} not found")
        src = src.replace(old, new)
    ns = dict(ex.__dict__)
    exec(compile(src, "export_llm_static", "exec"), ns)
    return ns["export_llm_to_onnx"]


def main(cfg: Config) -> None:
    os.makedirs(cfg.engine_dir, exist_ok=True)
    size = f"{cfg.height}x{cfg.width}"
    policy = Gr00tPolicy(cfg.embodiment_tag, cfg.model_path, device="cuda:0", strict=False)
    vit_cap, llm_cap = ex.ViTInputCapture(), ex.LLMInputCapture()
    inner = policy.model.backbone.model.model
    hooks = [
        inner.visual.register_forward_hook(vit_cap.hook_fn, with_kwargs=True),
        inner.language_model.register_forward_pre_hook(llm_cap.hook_fn, with_kwargs=True),
    ]
    with torch.inference_mode():
        policy.get_action(dummy_observation(policy, cfg.height, cfg.width))
    for h in hooks:
        h.remove()

    if "vit" in cfg.parts:
        path = ex.export_vit_to_onnx(policy, f"{cfg.onnx_dir}/vit_{size}", vit_cap, use_bf16=True)
        shape = {"pixel_values": tuple(vit_cap.pixel_values_shape)}
        engine = f"{cfg.engine_dir}/vit_bf16_{size}.engine"
        build_engine(onnx_path=path, engine_path=engine, precision="bf16", workspace_mb=cfg.workspace_mb,
                     min_shapes=shape, opt_shapes=shape, max_shapes=shape)  # fmt: skip
        print("built", engine)

    if "llm" in cfg.parts:
        path = static_llm_exporter()(
            policy, llm_cap, f"{cfg.onnx_dir}/llm_static_{size}", use_bf16=True
        )
        hidden, seq = llm_cap.inputs_embeds.shape[-1], llm_cap.inputs_embeds.shape[1]
        names = {i.name for i in onnx.load(path, load_external_data=False).graph.input}

        def shapes(s):
            all_shapes = {
                "inputs_embeds": (1, s, hidden),
                "attention_mask": (1, s),
                "position_ids": (3, 1, s),
                "visual_pos_masks": (1, s),
                **{f"deepstack_{i}": (1, s, hidden) for i in range(3)},
            }
            return {k: v for k, v in all_shapes.items() if k in names}

        engine = f"{cfg.engine_dir}/llm_static_bf16_{size}.engine"
        lo, hi = cfg.seq_margin
        build_engine(onnx_path=path, engine_path=engine, precision="bf16", workspace_mb=cfg.workspace_mb,
                     min_shapes=shapes(max(seq - lo, 1)), opt_shapes=shapes(seq), max_shapes=shapes(seq + hi))  # fmt: skip
        print("built", engine, "sequence length", seq)


if __name__ == "__main__":
    main(tyro.cli(Config))
