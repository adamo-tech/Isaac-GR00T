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

"""Drive a running GR00T training server with real LeRobot frames and check the result.

Builds Tinker datums from a GR00T-format dataset (the same steps fine-tuning would sample),
queues forward_backward + optim_step rounds without waiting, prints loss and step time,
saves for the sampler, and optionally loads that checkpoint with Gr00tPolicy.

  python scripts/train_server_smoke.py --url http://127.0.0.1:8000 --api-key ak_... \\
      --dataset /path/to/lerobot_dataset --embodiment-tag new_embodiment \\
      --modality-config-path runpod/abc_sim/abc_sim_config.py --steps 20 --batch-size 16
"""

import argparse
import base64
import importlib
import io
from pathlib import Path
import sys
import time

import httpx
import numpy as np
from PIL import Image


def png_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tensor(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float32)
    return {"data": array.ravel().tolist(), "dtype": "float32", "shape": list(array.shape)}


def to_datum(step) -> dict:
    """VLAStepData -> Tinker datum (images as PNG, absolute actions as loss_fn_inputs)."""
    chunks = [
        {"type": "image", "name": name, "format": "png", "data": png_b64(frame)}
        for name, frames in step.images.items()
        for frame in frames
    ]
    chunks += [
        {"type": "state", "name": name, "data": tensor(value)}
        for name, value in step.states.items()
    ]
    chunks.append({"type": "text", "text": step.text})
    return {
        "model_input": {"chunks": chunks},
        "loss_fn_inputs": {name: tensor(value) for name, value in step.actions.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--modality-config-path")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--policy-check", action="store_true")
    args = p.parse_args()

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.modality_config_path:
        path = Path(args.modality_config_path)
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
    modality_configs = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(args.dataset, modality_configs)
    rng = np.random.default_rng(0)
    # A few decoded episodes are enough for a smoke test (each holds every video frame).
    episodes = [loader[int(i)] for i in rng.choice(len(loader), min(3, len(loader)), replace=False)]
    horizon = len(modality_configs["action"].delta_indices)
    steps = []
    for _ in range(args.batch_size * args.steps):
        episode = episodes[int(rng.integers(len(episodes)))]
        index = int(rng.integers(len(episode) - horizon))
        steps.append(extract_step_data(episode, index, modality_configs, tag))
    batches = [
        [to_datum(s) for s in steps[i : i + args.batch_size]]
        for i in range(0, len(steps), args.batch_size)
    ]
    print(f"built {len(batches)} batches of {args.batch_size} datums")

    client = httpx.Client(base_url=args.url, headers={"X-API-Key": args.api_key}, timeout=600)

    def post(route: str, body: dict) -> dict:
        response = client.post(f"/api/v1/{route}", json=body)
        response.raise_for_status()
        return response.json()

    def wait(request_id: str) -> dict:
        delay = 0.1
        while True:
            try:
                result = post("retrieve_future", {"request_id": request_id})
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                # Polling is safe to repeat. Modal can answer 408 while a save keeps the
                # container busy; treat that, 5xx and network errors as "still pending".
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and status != 408 and status < 500:
                    raise
                result = {"type": "try_again"}
            if result.get("type") != "try_again":
                if "error" in result and "category" in result:
                    raise RuntimeError(result)
                return result
            time.sleep(delay)
            delay = min(1.0, delay * 1.5)

    t0 = time.monotonic()
    created = post(
        "create_model", {"base_model": "nvidia/GR00T-N1.7", "lora_config": {"rank": 32, "seed": 0}}
    )
    model_id = created["model_id"]
    wait(created["request_id"])
    print(
        f"create_model: {time.monotonic() - t0:.1f}s, info {post('get_info', {'model_id': model_id})}"
    )

    adam = {
        "learning_rate": args.lr,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 1e-5,
    }
    seq_id, fb_ids, step_ids = 1, [], []
    t0 = time.monotonic()
    for batch in batches:
        fb = {"forward_backward_input": {"data": batch, "loss_fn": "flow_matching"}}
        fb_ids.append(
            post("forward_backward", {"model_id": model_id, "seq_id": seq_id, **fb})["request_id"]
        )
        step = {"model_id": model_id, "seq_id": seq_id + 1, "adam_params": adam}
        step_ids.append(post("optim_step", step)["request_id"])
        seq_id += 2
    print(f"queued {len(batches)} rounds in {time.monotonic() - t0:.1f}s")
    for i, (fb_id, step_id) in enumerate(zip(fb_ids, step_ids)):
        fb, step = wait(fb_id), wait(step_id)
        print(
            f"step {i + 1:3d}  loss {fb['metrics']['loss:mean']:.4f}  "
            f"grad_norm {step['metrics']['grad_norm']:.3f}  t {time.monotonic() - t0:6.1f}s"
        )

    t0 = time.monotonic()
    sampler = wait(
        post("save_weights_for_sampler", {"model_id": model_id, "seq_id": seq_id, "path": "smoke"})[
            "request_id"
        ]
    )
    state = wait(
        post("save_weights", {"model_id": model_id, "seq_id": seq_id + 1, "path": "smoke-state"})[
            "request_id"
        ]
    )
    print(f"saved {sampler['path']} and {state['path']} in {time.monotonic() - t0:.1f}s")
    loaded = wait(
        post(
            "load_weights",
            {"model_id": model_id, "seq_id": seq_id + 2, "path": "smoke-state", "optimizer": True},
        )["request_id"]
    )
    print(f"load_weights: {loaded}")

    if args.policy_check:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(tag, sampler["path"], device="cuda:0")
        step = steps[0]
        observation = {
            "video": {k: np.stack(v)[None] for k, v in step.images.items()},
            "state": {k: np.asarray(v, np.float32)[None] for k, v in step.states.items()},
            "language": {
                modality_configs["language"].modality_keys[0]: [[step.text]],
            },
        }
        action, _ = policy.get_action(observation)
        print("Gr00tPolicy action shapes:", {k: v.shape for k, v in action.items()})


if __name__ == "__main__":
    main()
