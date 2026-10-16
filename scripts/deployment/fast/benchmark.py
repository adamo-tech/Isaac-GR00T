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

"""Check FastGr00t (optionally with a merged LoRA) against eager PyTorch and time it.

    python -m scripts.deployment.fast.benchmark --model-path <ckpt> --embodiment-tag new_embodiment \
        --lora-dir <train-server LoRA dir> --height 224 --width 224 --engine-dir trt_fast/engines

The reference is the eager model with the same noise; with a LoRA it keeps the peft adapters
unmerged, i.e. the trained model exactly as the training server ran it.
"""

from dataclasses import dataclass
import time

import numpy as np
import torch
import tyro

from .build_engines import dummy_observation
from .fast_gr00t import FastGr00t
from .lora import attach_lora, load_lora_policy, merge_lora, read_lora


@dataclass
class Config:
    model_path: str
    embodiment_tag: str
    lora_dir: str | None = None
    height: int = 224
    width: int = 224
    engine_dir: str = "trt_fast/engines"
    iters: int = 200
    swap_lora_dir: str | None = None
    """Also time LoraSwapper.swap to this LoRA (written into the running graph's weights)."""


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    return f"mean {np.mean(ts):.2f} ms  p50 {np.median(ts):.2f}  p99 {np.percentile(ts, 99):.2f}"


def main(cfg: Config) -> None:
    policy, swapper = load_lora_policy(cfg.model_path, None, cfg.embodiment_tag, strict=False)
    obs = dummy_observation(policy, cfg.height, cfg.width)
    instruction = obs["language"][policy.language_key][0][0]
    ah = policy.model.action_head
    noise = torch.randn(
        1, ah.config.action_horizon, ah.action_dim, generator=torch.Generator().manual_seed(0)
    )
    noise = noise.cuda().bfloat16()

    # eager reference: unmerged adapters, fixed noise
    if cfg.lora_dir:
        attach_lora(policy.model, *read_lora(cfg.lora_dir))
    captured = {}
    get_action, randn = policy.model.get_action, torch.randn

    def hooked(inputs, options=None):
        out = get_action(inputs, options)
        captured["pred"] = out["action_pred"].float()
        return out

    policy.model.get_action = hooked
    torch.randn = (
        lambda *a, **k: noise.clone()
        if tuple(k.get("size", a[0] if a else ())) == tuple(noise.shape)
        else randn(*a, **k)
    )  # noqa: E731
    try:
        policy.get_action(obs)
    finally:
        policy.model.get_action, torch.randn = get_action, randn
    if cfg.lora_dir:
        from .lora import LoraSwapper

        swapper = LoraSwapper(policy.model, read_lora(cfg.lora_dir)[0], merge_lora(policy.model))

    fast = FastGr00t(policy, cfg.engine_dir, H=cfg.height, W=cfg.width)
    t0 = time.time()
    fast.get_action(obs)
    print(f"first call (compile + capture): {time.time() - t0:.1f} s")

    def compare(label):
        pred = fast.infer_normalized(fast.state.clone(), instruction, noise=noise).float()
        ref = captured["pred"][..., : pred.shape[-1]]
        cos = torch.nn.functional.cosine_similarity(pred.flatten(), ref.flatten(), dim=0).item()
        print(
            f"{label}: cosine {cos:.5f}, max |d| {(pred - ref).abs().max().item():.4f} (normalized actions)"
        )

    compare("fast vs eager")
    c = fast.graphs[instruction]
    print("graph replay:     ", bench(lambda: fast.replay(c), cfg.iters))
    print("get_action (numpy):", bench(lambda: fast.get_action(obs), cfg.iters))

    if cfg.swap_lora_dir and swapper is not None:

        def swap_and_refresh():
            swapper.swap(cfg.swap_lora_dir)
            fast.refresh_weights()

        print("LoRA swap + refresh:", bench(swap_and_refresh, 10, warmup=1))
        if cfg.swap_lora_dir == cfg.lora_dir:
            compare("after swapping the same LoRA back in")


if __name__ == "__main__":
    main(tyro.cli(Config))
