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
        --lora-dir <adamo_model save_weights dir> --height 224 --width 224 --engine-dir trt_fast/engines

The reference is the eager model with the same noise; with a LoRA it keeps the peft adapters
unmerged, i.e. the trained model exactly as the training server ran it.
"""

from dataclasses import dataclass
from pathlib import Path
import time

from adamo_model.policy import attach_lora, merge_lora, read_lora
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import tyro

from .build_engines import dummy_observation
from .fast_gr00t import FastGr00t


@dataclass
class Config:
    model_path: str
    embodiment_tag: str
    lora_dir: str | None = None
    height: int = 224
    width: int = 224
    engine_dir: str = "trt_fast/engines"
    iters: int = 200


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
    policy = Gr00tPolicy(cfg.embodiment_tag, cfg.model_path, device="cuda:0", strict=False)
    obs = dummy_observation(policy, cfg.height, cfg.width)
    instruction = obs["language"][policy.language_key][0][0]
    ah = policy.model.action_head
    noise = torch.randn(
        1, ah.config.action_horizon, ah.action_dim, generator=torch.Generator().manual_seed(0)
    )
    noise = noise.cuda().bfloat16()

    # eager reference: unmerged adapters, fixed noise
    if cfg.lora_dir:
        attach_lora(policy.model, *read_lora(Path(cfg.lora_dir)))
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
        merge_lora(policy.model)

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
    fah = c["fah"]
    with torch.inference_mode():
        state_features = ah.state_encoder(fast.state, fast.emb_id)
        kv = fah.cross_kv(fast.vlsa.out["output"])
        actions = fast.noise.clone()
        step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(step_graph):
            c["step"](actions, state_features, kv, fah.step_indices[:1], fast.emb_id, fast.velocity_strength)
    print("one flow step:    ", bench(step_graph.replay, cfg.iters))
    print("get_action (numpy):", bench(lambda: fast.get_action(obs), cfg.iters))


if __name__ == "__main__":
    main(tyro.cli(Config))
