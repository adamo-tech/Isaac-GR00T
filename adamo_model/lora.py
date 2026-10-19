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

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear
from gr00t.policy.gr00t_policy import Gr00tPolicy
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
import torch
from torch import nn


ADAPTER = "default"
ADAPTER_FILE = "trainable.safetensors"
ADAPTER_CONFIG_FILE = "lora_config.json"
PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")
LORA_MODULES = ("model", "vl_self_attention", "state_encoder", "action_encoder", "action_decoder")
STATE_INPUTS = ("state_encoder.layer1",)
ACTION_INPUTS = ("action_encoder.W1",)
ACTION_OUTPUTS = ("action_decoder.layer2",)


@dataclass
class LoraSettings:
    rank: int
    alpha: float
    embodiment_id: int
    dropout: float = 0.0
    init_slot: int | None = None
    state_map: list[int] | None = None
    action_map: list[int] | None = None

    def to_json(self) -> dict:
        return asdict(self)


class SlotLinear(nn.Linear):
    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


def layout_permutation(index_map: list[int], size: int) -> torch.Tensor:
    used = set(index_map)
    return torch.tensor(index_map + [j for j in range(size) if j not in used])


def extract_slot(model: nn.Module, settings: LoraSettings) -> None:
    is_remapped = settings.init_slot is not None
    slot = settings.init_slot if is_remapped else settings.embodiment_id
    for name, module in list(model.action_head.named_modules()):
        if not isinstance(module, CategorySpecificLinear):
            continue
        weight, bias = module.W.detach()[slot], module.b.detach()[slot]
        if is_remapped and name in STATE_INPUTS + ACTION_INPUTS:
            index_map = settings.state_map if name in STATE_INPUTS else settings.action_map
            weight = weight[layout_permutation(index_map, weight.shape[0]).to(weight.device)]
        if is_remapped and name in ACTION_OUTPUTS:
            permutation = layout_permutation(settings.action_map, weight.shape[1]).to(weight.device)
            weight, bias = weight[:, permutation], bias[permutation]
        linear = SlotLinear(
            weight.shape[0], weight.shape[1], device=weight.device, dtype=weight.dtype
        )
        with torch.no_grad():
            linear.weight.copy_(weight.T)
            linear.bias.copy_(bias)
        parent_name, _, child = name.rpartition(".")
        parent = model.action_head.get_submodule(parent_name) if parent_name else model.action_head
        setattr(parent, child, linear)


def apply_lora(model: nn.Module, settings: LoraSettings) -> None:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    extract_slot(model, settings)
    targets = []
    for module_name in LORA_MODULES:
        module = getattr(model.action_head, module_name)
        module.requires_grad_(False)
        targets += [
            f"action_head.{module_name}.{name}"
            for name, child in module.named_modules()
            if isinstance(child, nn.Linear)
        ]
    config = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=targets,
    )
    inject_adapter_in_model(config, model, adapter_name=ADAPTER)
    lora_prefixes = tuple(f"action_head.{module_name}." for module_name in LORA_MODULES)
    for name, parameter in model.named_parameters():
        if "lora_" in name or (name in trainable and not name.startswith(lora_prefixes)):
            parameter.requires_grad = True


def assemble_model_dir(base_directory: Path, processor_directory: Path, directory: Path) -> Path:
    for file in base_directory.iterdir():
        if file.name not in PROCESSOR_FILES:
            (directory / file.name).symlink_to(file.resolve())
    for name in PROCESSOR_FILES:
        (directory / name).write_bytes((processor_directory / name).read_bytes())
    return directory


def read_lora(lora_directory: Path) -> tuple[LoraSettings, dict[str, torch.Tensor]]:
    settings = LoraSettings(**json.loads((lora_directory / ADAPTER_CONFIG_FILE).read_text()))
    return settings, load_file(str(lora_directory / ADAPTER_FILE))


@torch.no_grad()
def attach_lora(model: nn.Module, settings: LoraSettings, state: dict[str, torch.Tensor]) -> None:
    apply_lora(model, settings)
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameter = parameters[name]
        parameter.data = value.to(
            device=parameter.device,
            dtype=torch.bfloat16 if "lora_" in name else parameter.dtype,
        )
    model.requires_grad_(False)
    model.eval()


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLayer):
            module.merge(adapter_names=[ADAPTER])
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, module.get_base_layer())


def load_lora_policy(
    model_directory: Path, lora_directory: Path, embodiment_tag: str
) -> Gr00tPolicy:
    policy = Gr00tPolicy(embodiment_tag, str(model_directory), device="cuda:0")
    attach_lora(policy.model, *read_lora(lora_directory))
    merge_lora(policy.model)
    return policy
