import hashlib
import json
from pathlib import Path
import tempfile
import uuid

import gr00t.model
from gr00t.policy.gr00t_policy import Gr00tPolicy
from peft.tuners.lora import LoraLayer
from runtime import model_storage
from safetensors.torch import load_file
from scripts.deployment import fast
from scripts.deployment.fast.build_engines import Config, main as build_engines
from scripts.deployment.fast.fast_gr00t import FastGr00t
import tensorrt
import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor.runtime.triton_heuristics import (
    CachingAutotuner,
    check_autotune_cache,
    triton_config_to_hashable,
)

from adamo_model import embodiment
from adamo_model.lora import ADAPTER, ADAPTER_CONFIG_FILE, ADAPTER_FILE, LoraSettings, apply_lora


ENGINES = model_storage.CACHE_PATH / "gr00t" / "engines"
COMPILE_CACHE = model_storage.CACHE_PATH / "gr00t" / "compile_cache"
GRAPH_SOURCES = (Path(gr00t.model.__file__).parent, Path(fast.__file__).parent)


PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


def recheck_autotune_cache(self, reload_kernel_from_src):
    configs = [result.config for result in self.compile_results]
    cached_configs, _, self.autotune_cache_info = check_autotune_cache(
        configs, self.filename, self.inductor_meta
    )
    if len(cached_configs) == 1:
        best_config = cached_configs[0]
        found_by_coordesc = getattr(best_config, "found_by_coordesc", False)
        best_config_hash = triton_config_to_hashable(best_config)
        for compile_result in self.compile_results:
            if triton_config_to_hashable(compile_result.config) == best_config_hash:
                compile_result.config.found_by_coordesc = found_by_coordesc
                self.compile_results = [compile_result]
                return
        if self.fn.fn is None:
            self.fn = reload_kernel_from_src().fn
        self.compile_results = [self._precompile_config(best_config)]


CachingAutotuner.recheck_autotune_cache = recheck_autotune_cache


def compile_cache_directory(prompt):
    paths = sorted(path for directory in GRAPH_SOURCES for path in directory.rglob("*.py"))
    sources = b"".join(path.read_bytes() for path in paths)
    digest = hashlib.sha256(sources + prompt.encode()).hexdigest()[:16]
    return COMPILE_CACHE / torch.cuda.get_device_name() / torch.__version__ / digest


def load_compile_cache(prompt):
    for path in compile_cache_directory(prompt).glob("*.bin"):
        torch.compiler.load_cache_artifacts(path.read_bytes())


def save_compile_cache(prompt):
    directory = compile_cache_directory(prompt)
    directory.mkdir(parents=True, exist_ok=True)
    partial = directory / f"{uuid.uuid4().hex}.partial"
    partial.write_bytes(torch.compiler.save_cache_artifacts()[0])
    partial.rename(partial.with_suffix(".bin"))


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


class Gr00tN17Policy:
    def __init__(self, fast, mapping):
        self.fast = fast
        self.mapping = {name: mapping[name] for name in ("images", "state", "action", "frame_rate")}
        self.action_sizes = mapping["action_sizes"]
        self.parts = mapping["embodiment"]
        self.refill_step_index = mapping["action_horizon"] // 2
        self.saved_miss_count = 0

    def infer(self, observation, prompt, prefix=None, inference_delay=0, real_time_chunking=True):
        if prompt not in self.fast.graphs:
            load_compile_cache(prompt)
        action_prefix = None if prefix is None or not real_time_chunking else {name: prefix[:, start:end] for name, (start, end) in embodiment.ranges(self.parts).items()}
        actions = self.fast.get_action(
            embodiment.gr00t_observation(observation, prompt, self.mapping["images"], self.parts), action_prefix, inference_delay
        )
        miss_count = counters["inductor"]["fxgraph_cache_miss"]
        if miss_count > self.saved_miss_count:
            save_compile_cache(prompt)
            self.saved_miss_count = miss_count
        return embodiment.robot_actions(actions, self.parts)


def engine_directory(model_directory, camera_count):
    directory = (
        ENGINES
        / torch.cuda.get_device_name()
        / tensorrt.__version__
        / f"{embodiment.IMAGE_SIZE}x{embodiment.IMAGE_SIZE}"
        / f"{camera_count}-cameras"
    )

    if not directory.exists():
        partial = directory.with_name(f"{directory.name}.{uuid.uuid4().hex}.partial")
        build_engines(
            Config(
                model_path=str(model_directory),
                embodiment_tag=embodiment.EMBODIMENT_TAG.value,
                height=embodiment.IMAGE_SIZE,
                width=embodiment.IMAGE_SIZE,
                engine_dir=str(partial),
                onnx_dir=tempfile.mkdtemp(),
            )
        )
        partial.rename(directory)

    return directory


def load(policy):
    weights = model_storage.fetch(policy["model_weights"])
    mapping = json.loads((weights / "mapping.json").read_text())
    model_directory = assemble_model_dir(
        embodiment.base_checkpoint(), weights, Path(tempfile.mkdtemp())
    )
    gr00t_policy = load_lora_policy(model_directory, weights, embodiment.EMBODIMENT_TAG.value)
    fast = FastGr00t(
        gr00t_policy,
        str(engine_directory(model_directory, len(mapping["images"]))),
        H=embodiment.IMAGE_SIZE,
        W=embodiment.IMAGE_SIZE,
    )
    return Gr00tN17Policy(fast, mapping)
