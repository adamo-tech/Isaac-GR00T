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

"""Inference with LoRA adapters saved by the training server (gr00t/train_server, ``save_weights``).

A LoRA checkpoint directory holds ``trainable.safetensors`` (adapters plus the fully trained
position embedding and ``vlln``) and ``lora_config.json`` (``LoraSettings``). It applies to the base
model it was trained from, with the processor (modality config + statistics) of the embodiment.

``load_lora_policy`` builds a ``Gr00tPolicy`` with the adapters merged into the action-head weights,
so inference costs exactly what the base model costs. ``LoraSwapper`` writes another LoRA of the
same base model into those weights in place (about 0.1 s on an RTX 4090, file read included), or
restores the base model's own weights (``use_base``), which a running ``FastGr00t`` picks up after
``refresh_weights()`` without recompiling or recapturing.
"""

import json
from pathlib import Path

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.train_server.lora import ADAPTER, LoraSettings, apply_lora
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
import torch


ADAPTER_FILE = "trainable.safetensors"
ADAPTER_CONFIG_FILE = "lora_config.json"
PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


def assemble_model_dir(base_model: str, processor_dir: str, out_dir: str) -> Path:
    """A checkpoint directory with the base model's weights (symlinked) and another checkpoint's
    processor files, e.g. those of a sampler save of the LoRA run (which has the embodiment's
    modality config and statistics, unlike the base model)."""
    from huggingface_hub import snapshot_download

    base = Path(base_model) if Path(base_model).is_dir() else Path(snapshot_download(base_model))
    processor = Path(processor_dir)
    if not (processor / "processor_config.json").exists() and (processor / "processor").is_dir():
        processor = processor / "processor"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name in PROCESSOR_FILES or f.name == "processor":
            continue
        dst = out / f.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f.resolve())
    for name in PROCESSOR_FILES:
        if (processor / name).exists():
            (out / name).write_bytes((processor / name).read_bytes())
    return out


def read_lora(lora_dir: str | Path) -> tuple[LoraSettings, dict[str, torch.Tensor]]:
    lora_dir = Path(lora_dir)
    settings = LoraSettings(**json.loads((lora_dir / ADAPTER_CONFIG_FILE).read_text()))
    return settings, load_file(str(lora_dir / ADAPTER_FILE))


@torch.no_grad()
def base_weights(
    model: torch.nn.Module, settings: LoraSettings, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The base model's values, before ``attach_lora``, of what a LoRA changes besides the adapted
    layers' weights: each projector's ``settings.embodiment_id`` slot (as the ``SlotLinear``
    weight and bias that replace it; a LoRA may start from another slot) and the fully trained
    tensors (position embedding, ``vlln``). Keyed by parameter name in the merged model."""
    out = {}
    for name, module in model.action_head.named_modules():
        if isinstance(module, CategorySpecificLinear):
            slot = settings.embodiment_id
            out[f"action_head.{name}.weight"] = module.W.detach()[slot].T.contiguous().clone()
            out[f"action_head.{name}.bias"] = module.b.detach()[slot].clone()
    params = dict(model.named_parameters())
    trained = [k for k in state if "lora_" not in k]
    missing = [k for k in trained if k not in params]
    if missing:
        raise ValueError(f"LoRA trains tensors the base model lacks: {missing[:5]}")
    out.update({k: params[k].detach().clone() for k in trained})
    return out


def lora_layers(model: torch.nn.Module) -> dict[str, LoraLayer]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoraLayer)}


@torch.no_grad()
def attach_lora(
    model: torch.nn.Module, settings: LoraSettings, state: dict[str, torch.Tensor]
) -> None:
    """Cut the projectors to the trained slot, inject adapters and load the trained tensors (bf16)."""
    apply_lora(model, settings)
    params = dict(model.named_parameters())
    unknown = sorted(k for k in state if k not in params)
    missing = sorted(k for k in params if "lora_" in k and k not in state)
    if unknown or missing:
        raise ValueError(
            f"LoRA does not match the model: unknown {unknown[:5]}, missing {missing[:5]}"
        )
    for name, value in state.items():
        p = params[name]
        p.data = value.to(device=p.device, dtype=torch.bfloat16 if "lora_" in name else p.dtype)
    model.requires_grad_(False)
    model.eval()


@torch.no_grad()
def merge_lora(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Fold the adapters into the base weights and remove the LoRA wrappers. Returns each adapted
    layer's pre-merge (base) weight, which ``LoraSwapper`` writes later LoRAs onto."""
    base = {}
    for name, m in lora_layers(model).items():
        base[name] = m.get_base_layer().weight.detach().clone()
        m.merge(adapter_names=[ADAPTER])
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, m.get_base_layer())
    return base


def load_lora_policy(
    model_path: str,
    lora_dir: str | None,
    embodiment_tag: str,
    *,
    device: str = "cuda:0",
    strict: bool = True,
) -> tuple[Gr00tPolicy, "LoraSwapper | None"]:
    """``Gr00tPolicy`` with the LoRA in ``lora_dir`` merged in (or the plain checkpoint if None).

    ``model_path`` must hold the base weights the LoRA was trained from and the embodiment's
    processor files (see ``assemble_model_dir``)."""
    policy = Gr00tPolicy(embodiment_tag, model_path, device=device, strict=strict)
    if lora_dir is None:
        return policy, None
    settings, state = read_lora(lora_dir)
    original = base_weights(policy.model, settings, state)
    attach_lora(policy.model, settings, state)
    base = merge_lora(policy.model)
    return policy, LoraSwapper(policy.model, settings, base, original)


class LoraSwapper:
    """Writes LoRAs (of the same base model, rank-independent but with the same target layers and
    projector slot settings) into the merged model's weights in place, or restores the base
    model's weights (``use_base``, given ``original`` from ``base_weights``)."""

    def __init__(
        self,
        model: torch.nn.Module,
        settings: LoraSettings,
        base: dict[str, torch.Tensor],
        original: dict[str, torch.Tensor] | None = None,
    ):
        self.model = model
        self.settings = settings
        self.base = base
        self.original = original
        self.params = dict(model.named_parameters())
        # Adapted layers keep their base bias under a LoRA; use_base may overwrite the projectors'.
        self.bias = {
            name: model.get_submodule(name).bias.detach().clone()
            for name in base
            if model.get_submodule(name).bias is not None
        }

    @torch.no_grad()
    def use_base(self) -> None:
        """Restore the base model's weights (no LoRA)."""
        if self.original is None:
            raise ValueError("no base weights were kept; build the swapper with load_lora_policy")
        for name, W0 in self.base.items():
            self.model.get_submodule(name).weight.copy_(W0)
        for name, value in self.original.items():
            self.params[name].copy_(value)

    @torch.no_grad()
    def swap(self, lora_dir: str | Path) -> None:
        settings, state = read_lora(lora_dir)
        keep = ("embodiment_id", "init_slot", "state_map", "action_map")
        if any(getattr(settings, k) != getattr(self.settings, k) for k in keep):
            raise ValueError(f"{lora_dir}: projector slot settings differ from the loaded LoRA")
        scale = settings.alpha / settings.rank
        device = next(iter(self.base.values())).device
        state = {k: v.to(device, non_blocking=True) for k, v in state.items()}
        for name, W0 in self.base.items():
            A = state[f"{name}.lora_A.{ADAPTER}.weight"].float()
            B = state[f"{name}.lora_B.{ADAPTER}.weight"].float()
            self.model.get_submodule(name).weight.copy_((W0.float() + scale * (B @ A)).to(W0.dtype))
        for name, bias in self.bias.items():
            self.model.get_submodule(name).bias.copy_(bias)
        for name, value in state.items():
            if "lora_" not in name:
                self.params[name].copy_(value)
        self.settings = settings
