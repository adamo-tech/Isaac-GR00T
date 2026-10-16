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

from safetensors.torch import load_file, save_file
import torch

from gr00t.data.types import MessageType, VLAStepData
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

from .datum import UserError
from .lora import LoraSettings, apply_lora, merged_state_dict


OPTIMIZER_FILE = "optimizer.pt"
# LoRA training checkpoints (save_weights) hold only the trainable tensors.
ADAPTER_FILE = "trainable.safetensors"
ADAPTER_CONFIG_FILE = "lora_config.json"


class TrainingSession:
    """Holds the model, its GR00T processor and an AdamW optimizer over the trainable params.

    Gradients accumulate across ``forward_backward`` calls (each backpropagates GR00T's
    masked-mean flow-matching loss for its own batch) until ``optim_step``.

    With ``lora``, the action head (DiT, VL self-attention and the trained embodiment's projector
    slot) trains through LoRA adapters (see ``lora.py``). ``save(optimizer=True)`` then writes
    only the trainable tensors (adapters, position embedding, ``vlln``) and the optimizer state;
    ``save(optimizer=False)`` merges the adapters into a full checkpoint that ``Gr00tPolicy``
    loads.
    """

    def __init__(
        self,
        model: Gr00tN1d7,
        processor: Gr00tN1d7Processor,
        *,
        device: str | torch.device,
        augment: bool = True,
        bf16: bool = True,
        lora: LoraSettings | None = None,
        chunk_size: int = 256,
    ):
        self.lora = lora
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = chunk_size  # most datums per forward/backward pass
        if lora is not None:
            if lora.rank <= 0:
                raise UserError("this server trains LoRA; create_model needs lora_config.rank > 0")
            self._full_projectors = apply_lora(model, lora)
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

    def prepare(self, steps: list[VLAStepData]) -> list[dict[str, Any]]:
        """The model inputs for ``steps``, collated in chunks of at most ``chunk_size`` datums
        (processor + collator; CPU only, so the scheduler runs it ahead of the GPU work, on one
        thread and in call order)."""
        try:
            processed = [
                self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
                for step in steps
            ]
        except (KeyError, ValueError) as e:
            raise UserError(f"GR00T processor rejected the batch: {e}") from e
        return [
            self.collator(processed[i : i + self.chunk_size])["inputs"]
            for i in range(0, len(processed), self.chunk_size)
        ]

    def forward_backward(
        self, steps: list[VLAStepData], batches: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Accumulate the gradient of the batch's masked-mean loss; ``batches`` is
        ``prepare(steps)`` if already built.

        A batch larger than ``chunk_size`` runs as several forward/backward passes, each chunk's
        loss weighted by its share of the batch's valid action elements, so the accumulated
        gradient and the reported loss equal those of one pass over the whole batch."""
        if batches is None:
            batches = self.prepare(steps)
        total = sum(float(b["action_mask"].sum()) for b in batches)
        loss_sum, per_datum = 0.0, []
        for batch in batches:
            with self._autocast():
                out = self.model(batch)
            action_loss = out["action_loss"].detach().float()
            mask = out["action_mask"].detach().float()
            weight = float(mask.sum()) / total if len(batches) > 1 else 1.0
            (out["loss"] * weight).backward()
            loss_sum += out["loss"].item() * weight
            per_datum += (action_loss.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-6)).tolist()
            del out, action_loss, mask
        return {
            "type": "forward_backward",
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": [
                {"loss": {"data": [value], "dtype": "float32", "shape": [1]}} for value in per_datum
            ],
            "metrics": {"loss:mean": loss_sum},
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
        if self.lora is not None and optimizer:
            tmp.mkdir(parents=True)
            trainable = {
                name: p.detach().contiguous()
                for name, p in self.model.named_parameters()
                if p.requires_grad
            }
            save_file(trainable, tmp / ADAPTER_FILE)
            (tmp / ADAPTER_CONFIG_FILE).write_text(json.dumps(self.lora.to_json()))
        elif self.lora is not None:
            state = merged_state_dict(self.model, self._full_projectors, self.lora.embodiment_id)
            self.model.save_pretrained(tmp, state_dict=state)
            self.processor.save_pretrained(tmp)
        else:
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
        if self.lora is not None:
            self._load_trainable(path)
        else:
            self._load_full(path)
        if optimizer:
            if not (path / OPTIMIZER_FILE).exists():
                raise UserError(f"{path} has no optimizer state (saved for sampler?)")
            saved = torch.load(path / OPTIMIZER_FILE, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step = saved["step"]
        self.optimizer.zero_grad(set_to_none=True)

    def _load_trainable(self, path: Path) -> None:
        if not (path / ADAPTER_FILE).exists():
            raise UserError(
                f"{path} is not a LoRA training checkpoint (no {ADAPTER_FILE}); "
                "a LoRA model loads only checkpoints from save_weights"
            )
        saved = json.loads((path / ADAPTER_CONFIG_FILE).read_text())
        if saved != self.lora.to_json():
            raise UserError(f"{path} was saved with LoRA {saved}, this model uses {self.lora}")
        state = load_file(path / ADAPTER_FILE)
        params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        missing = sorted(set(params) - set(state))
        unexpected = sorted(set(state) - set(params))
        if missing or unexpected:
            raise UserError(
                f"{path} does not match the model: {len(missing)} missing keys "
                f"{missing[:5]}, {len(unexpected)} unexpected keys {unexpected[:5]}"
            )
        with torch.no_grad():
            for name, p in params.items():
                if state[name].shape != p.shape:
                    raise UserError(f"{path}: {name} has shape {list(state[name].shape)}")
                p.copy_(state[name])

    def _load_full(self, path: Path) -> None:
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
