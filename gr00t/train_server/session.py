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

"""One fine-tuning model on one GPU: forward/backward, AdamW steps, save and load."""

import contextlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from safetensors.torch import load_file
import torch

from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

from .datum import UserError


OPTIMIZER_FILE = "optimizer.pt"


class TrainingSession:
    """Holds the model, its GR00T processor and an AdamW optimizer over the trainable params.

    Gradients accumulate across ``forward_backward`` calls (each backpropagates GR00T's
    masked-mean flow-matching loss for its own batch) until ``optim_step``.
    """

    def __init__(
        self,
        model: Gr00tN1d7,
        processor: Gr00tN1d7Processor,
        *,
        device: str | torch.device,
        augment: bool = True,
        bf16: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.train()
        self.processor = processor
        if augment:
            self.processor.train()
        else:
            self.processor.eval()
        self.collator = processor.collator
        self.bf16 = bf16 and self.device.type == "cuda"
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.params:
            raise RuntimeError("Model has no trainable parameters")
        # Hyperparameters are overwritten from each optim_step request.
        self.optimizer = torch.optim.AdamW(self.params, lr=0.0)
        self.step = 0

    def _autocast(self):
        if self.bf16:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def forward_backward(self, steps: list[VLAStepData]) -> dict[str, Any]:
        try:
            processed = [
                self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
                for step in steps
            ]
        except (KeyError, ValueError) as e:
            raise UserError(f"GR00T processor rejected the batch: {e}") from e
        batch = self.collator(processed)["inputs"]

        with self._autocast():
            out = self.model(batch)
        loss = out["loss"]
        loss.backward()

        action_loss = out["action_loss"].detach().float()
        mask = out["action_mask"].detach().float()
        per_datum = (action_loss.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)).cpu().tolist()
        return {
            "type": "forward_backward",
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": [
                {"loss": {"data": [value], "dtype": "float32", "shape": [1]}} for value in per_datum
            ],
            "metrics": {"loss:mean": loss.item()},
        }

    def optim_step(self, adam_params: dict[str, float]) -> dict[str, Any]:
        for group in self.optimizer.param_groups:
            group["lr"] = adam_params["learning_rate"]
            group["betas"] = (adam_params["beta1"], adam_params["beta2"])
            group["eps"] = adam_params["eps"]
            group["weight_decay"] = adam_params["weight_decay"]
        clip = adam_params["grad_clip_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, clip if clip > 0 else math.inf)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return {
            "type": "optim_step",
            "metrics": {
                "grad_norm": float(grad_norm),
                "learning_rate": adam_params["learning_rate"],
                "step": self.step,
            },
        }

    def save(self, path: Path, *, optimizer: bool) -> None:
        """Write a checkpoint ``Gr00tPolicy`` can load (+ optimizer state), replacing ``path``."""
        tmp = path.with_name(f".{path.name}.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp)
        self.processor.save_pretrained(tmp)
        if optimizer:
            torch.save(
                {"optimizer": self.optimizer.state_dict(), "step": self.step}, tmp / OPTIMIZER_FILE
            )
        shutil.rmtree(path, ignore_errors=True)
        tmp.rename(path)

    def load(self, path: Path, *, optimizer: bool) -> None:
        """Load weights saved by ``save`` (or any GR00T checkpoint of this architecture)."""
        index = path / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        state: dict[str, torch.Tensor] = {}
        for name in files:
            if not (path / name).exists():
                raise UserError(f"{path} has no {name}")
            state.update(load_file(path / name))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # save_pretrained stores tied weights (e.g. lm_head tied to the embeddings) once, so a
        # missing key is fine when it shares storage with a key that was loaded.
        aliases: dict[int, list[str]] = {}
        for name, tensor in self.model.state_dict().items():
            aliases.setdefault(tensor.data_ptr(), []).append(name)
        tied = {
            name for names in aliases.values() if any(n in state for n in names) for name in names
        }
        missing = [name for name in missing if name not in tied]
        if missing or unexpected:
            raise UserError(
                f"{path} does not match the model: {len(missing)} missing keys "
                f"{missing[:5]}, {len(unexpected)} unexpected keys {unexpected[:5]}"
            )
        if optimizer:
            if not (path / OPTIMIZER_FILE).exists():
                raise UserError(f"{path} has no optimizer state (saved for sampler?)")
            saved = torch.load(path / OPTIMIZER_FILE, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step = saved["step"]
        self.optimizer.zero_grad(set_to_none=True)
