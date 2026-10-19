import json
from pathlib import Path
import tempfile

from runtime import model_storage
from scripts.deployment.fast.build_engines import Config, main as build_engines
from scripts.deployment.fast.fast_gr00t import FastGr00t
import tensorrt
import torch

from adamo_model import embodiment
from adamo_model.lora import assemble_model_dir, load_lora_policy


ENGINES = model_storage.CACHE_PATH / "gr00t" / "engines"


class Gr00tN17Policy:
    def __init__(self, fast, mapping, state_parts, action_parts):
        self.fast = fast
        self.mapping = {name: mapping[name] for name in ("images", "state", "action", "frame_rate")}
        self.action_sizes = mapping["action_sizes"]
        self.state_parts = state_parts
        self.action_parts = action_parts
        self.refill_step_index = mapping["action_horizon"] // 2

    def infer(self, observation, prompt, prefix=None, inference_delay=0, real_time_chunking=False):
        actions = self.fast.get_action(
            embodiment.gr00t_observation(
                observation, prompt, self.mapping["images"], self.state_parts
            )
        )
        return embodiment.robot_actions(actions, self.action_parts)


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
    norm_stats = json.loads((weights / "norm_stats.json").read_text())
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
    return Gr00tN17Policy(
        fast,
        mapping,
        embodiment.parts(norm_stats, mapping, "state"),
        embodiment.parts(norm_stats, mapping, "action"),
    )
