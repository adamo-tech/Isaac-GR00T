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

"""LoRA on the action head: DiT, VL self-attention and the trained embodiment's projectors.

The VLM stays frozen as in full fine-tuning. The projectors (``CategorySpecificLinear``) hold
one weight slot per embodiment; only the trained embodiment's slot is kept on the device, as a
plain ``nn.Linear`` (``SlotLinear``), and the full multi-slot tensors are parked on the CPU to
rebuild a standard checkpoint at save time. Every ``nn.Linear`` in the DiT, the VL
self-attention and the projectors then gets a LoRA adapter, with its base weight frozen. The
position embedding and ``vlln`` keep training in full.

The slot can start from another (pretrained) embodiment's slot, ``init_slot``. Its padded
state/action layout generally differs, so ``state_map`` / ``action_map`` give, for each of this
embodiment's state/action dims in order, the index of the matching dim in the source layout;
the projector's input rows (state and action encoders) and output columns (action decoder) are
permuted accordingly. The trained slot is always written back to ``embodiment_id``, in this
embodiment's layout, so the source embodiment's slot is left untouched.
"""

from dataclasses import asdict, dataclass

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
from peft import LoraConfig, inject_adapter_in_model
import torch
from torch import nn


ADAPTER = "default"
ADAPTER_FILE = "trainable.safetensors"
ADAPTER_CONFIG_FILE = "lora_config.json"
# Submodules of the action head whose linear layers get adapters instead of full training.
LORA_MODULES = ("model", "vl_self_attention", "state_encoder", "action_encoder", "action_decoder")
# Projector layers whose input rows / output columns are indexed by state or action dims.
STATE_INPUTS = ("state_encoder.layer1",)
ACTION_INPUTS = ("action_encoder.W1",)
ACTION_OUTPUTS = ("action_decoder.layer2",)


@dataclass
class LoraSettings:
    rank: int
    alpha: float
    embodiment_id: int
    """Projector slot of the trained embodiment (the processor's embodiment_id_mapping)."""
    dropout: float = 0.0
    init_slot: int | None = None
    """Projector slot to start from; None starts from ``embodiment_id``'s own slot."""
    state_map: list[int] | None = None
    """With ``init_slot``: source-layout index of each of this embodiment's state dims."""
    action_map: list[int] | None = None
    """With ``init_slot``: source-layout index of each of this embodiment's action dims."""

    def to_json(self) -> dict:
        return asdict(self)


class SlotLinear(nn.Linear):
    """One embodiment's slot of a ``CategorySpecificLinear``; ignores the category ids."""

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


def layout_permutation(index_map: list[int], size: int) -> torch.Tensor:
    """Permutation ``p`` of ``range(size)`` with ``p[i] = index_map[i]`` for the mapped dims; the
    unmapped source dims fill the padding positions, in order."""
    if len(set(index_map)) != len(index_map) or not all(0 <= j < size for j in index_map):
        raise ValueError(f"layout map must be distinct indices in [0, {size}), got {index_map}")
    used = set(index_map)
    return torch.tensor(index_map + [j for j in range(size) if j not in used])


def extract_slot(
    model: nn.Module, settings: LoraSettings
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Replace every ``CategorySpecificLinear`` in the action head by one slot as a
    ``SlotLinear`` (``settings.init_slot``, remapped to this embodiment's layout, or else
    ``settings.embodiment_id``). Returns the original ``(W, b)`` tensors, on the CPU, by module
    name."""
    remap = settings.init_slot is not None
    slot = settings.init_slot if remap else settings.embodiment_id
    if remap and (settings.state_map is None or settings.action_map is None):
        raise ValueError("init_slot needs state_map and action_map")
    full = {}
    for name, module in list(model.action_head.named_modules()):
        if not isinstance(module, CategorySpecificLinear):
            continue
        for s in (slot, settings.embodiment_id):
            if not 0 <= s < module.num_categories:
                raise ValueError(f"projector slot {s} out of range for {module.num_categories}")
        W, b = module.W.detach()[slot], module.b.detach()[slot]  # [in, out], [out]
        if remap and name in STATE_INPUTS + ACTION_INPUTS:
            index_map = settings.state_map if name in STATE_INPUTS else settings.action_map
            W = W[layout_permutation(index_map, W.shape[0]).to(W.device)]
        if remap and name in ACTION_OUTPUTS:
            perm = layout_permutation(settings.action_map, W.shape[1]).to(W.device)
            W, b = W[:, perm], b[perm]
        linear = SlotLinear(W.shape[0], W.shape[1], device=W.device, dtype=W.dtype)
        with torch.no_grad():
            linear.weight.copy_(W.T)
            linear.bias.copy_(b)
        W, b = module.W.detach(), module.b.detach()
        full[f"action_head.{name}"] = (W.cpu().clone(), b.cpu().clone())
        parent_name, _, child = name.rpartition(".")
        parent = model.action_head.get_submodule(parent_name) if parent_name else model.action_head
        setattr(parent, child, linear)
    return full


def apply_lora(model: nn.Module, settings: LoraSettings) -> dict:
    """Cut the projectors to one slot and add adapters, in place.

    Returns the projectors' original multi-slot tensors (see ``extract_slot``)."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    full = extract_slot(model, settings)
    targets = []
    for sub in LORA_MODULES:
        module = getattr(model.action_head, sub)
        module.requires_grad_(False)
        targets += [
            f"action_head.{sub}.{name}"
            for name, child in module.named_modules()
            if isinstance(child, nn.Linear)
        ]
    if not targets:
        raise RuntimeError("No linear layers found for LoRA in the action head")
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=targets,
    )
    inject_adapter_in_model(config, model, adapter_name=ADAPTER)
    # peft leaves only adapters trainable; restore full training of the parameters outside the
    # LoRA modules that were trainable before.
    lora_prefixes = tuple(f"action_head.{sub}." for sub in LORA_MODULES)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif name in trainable and not name.startswith(lora_prefixes):
            p.requires_grad = True
    return full
