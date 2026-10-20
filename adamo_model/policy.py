import json
from pathlib import Path
import tempfile

from gr00t.policy.gr00t_policy import Gr00tPolicy
from peft.tuners.lora import LoraLayer
from runtime import model_storage
from safetensors.torch import load_file
from scripts.deployment.fast.build_engines import Config, main as build_engines
from scripts.deployment.fast.fast_gr00t import FastGr00t
import tensorrt
import torch
from torch import nn

from adamo_model import embodiment
from adamo_model.lora import ADAPTER, ADAPTER_CONFIG_FILE, ADAPTER_FILE, LoraSettings, apply_lora


ENGINES = model_storage.CACHE_PATH / "gr00t" / "engines"


PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


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

    def infer(self, observation, prompt, prefix=None, inference_delay=0, real_time_chunking=False):
        actions = self.fast.get_action(
            embodiment.gr00t_observation(observation, prompt, self.mapping["images"], self.parts)
        )
        return embodiment.robot_actions(actions, self.parts)


def engine_directory(model_directory):
    directory = (
        ENGINES
        / torch.cuda.get_device_name()
        / tensorrt.__version__
        / f"{embodiment.IMAGE_SIZE}x{embodiment.IMAGE_SIZE}"
    )

    if not directory.exists():
        partial = directory.with_name(f"{directory.name}.partial")
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
        str(engine_directory(model_directory)),
        H=embodiment.IMAGE_SIZE,
        W=embodiment.IMAGE_SIZE,
    )
    return Gr00tN17Policy(fast, mapping)
