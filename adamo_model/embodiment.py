import pathlib
import tempfile

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.experiment.launch_finetune import build_config
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import numpy
from PIL import Image
from runtime import model_storage

from adamo_model.lora import LoraSettings


BASE_CHECKPOINT = "hf://nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BASE_CHECKPOINT_CHECKSUM = "339454305dedcf2729dde2e39cca0856f2d4abcb52776dedc39496050b8b4469"
EMBODIMENT_TAG = EmbodimentTag.NEW_EMBODIMENT
LANGUAGE_KEY = "annotation.human.task_description"
IMAGE_SIZE = 224
YAM_PARTS = {
    "left_arm": (0, 6),
    "left_gripper": (6, 7),
    "right_arm": (7, 13),
    "right_gripper": (13, 14),
}
RELATIVE_PARTS = ("left_arm", "right_arm")
RELATIVE_ACTION_KEY = "relative_action"
XDOF_TAG = "xdof_relative_eef_relative_joint"
XDOF_LAYOUT = [20, 21, 22, 23, 24, 25, 18, 26, 27, 28, 29, 30, 31, 19]


def is_bimanual_yam(norm_stats, mapping):
    return all(
        sum(len(norm_stats[key]["q01"]) for key in mapping[modality]) == len(XDOF_LAYOUT)
        for modality in ("state", "action")
    )


def is_relative(norm_stats, mapping):
    return is_bimanual_yam(norm_stats, mapping) and RELATIVE_ACTION_KEY in norm_stats


def parts(norm_stats, mapping, modality):
    if is_bimanual_yam(norm_stats, mapping):
        return YAM_PARTS

    ends = numpy.cumsum([len(norm_stats[key]["q01"]) for key in mapping[modality]])
    return {
        key: (int(end) - len(norm_stats[key]["q01"]), int(end))
        for key, end in zip(mapping[modality], ends)
    }


def modality_config(norm_stats, mapping, action_horizon):
    is_relative_run = is_relative(norm_stats, mapping)
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.RELATIVE
            if is_relative_run and name in RELATIVE_PARTS
            else ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        for name in parts(norm_stats, mapping, "action")
    ]
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=mapping["images"]),
        "state": ModalityConfig(
            delta_indices=[0], modality_keys=list(parts(norm_stats, mapping, "state"))
        ),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=list(parts(norm_stats, mapping, "action")),
            action_configs=action_configs,
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }


def part_statistics(q01, q99):
    return {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "min": q01.tolist(),
        "max": q99.tolist(),
        "mean": numpy.zeros_like(q01).tolist(),
        "std": numpy.ones_like(q01).tolist(),
    }


def statistics(norm_stats, mapping, action_horizon):
    result = {}

    for modality in ("state", "action"):
        q01, q99 = (
            numpy.concatenate(
                [numpy.asarray(norm_stats[key][bound], dtype=float) for key in mapping[modality]]
            )
            for bound in ("q01", "q99")
        )
        result[modality] = {
            name: part_statistics(q01[start:end], q99[start:end])
            for name, (start, end) in parts(norm_stats, mapping, modality).items()
        }

    if is_relative(norm_stats, mapping):
        q01, q99 = (
            numpy.asarray(norm_stats[RELATIVE_ACTION_KEY][bound], dtype=float).reshape(
                action_horizon, -1
            )
            for bound in ("q01", "q99")
        )
        result[RELATIVE_ACTION_KEY] = {
            name: part_statistics(q01[:, start:end], q99[:, start:end])
            for name, (start, end) in YAM_PARTS.items()
            if name in RELATIVE_PARTS
        }

    return result


def lora_settings(norm_stats, mapping, rank):
    is_yam = is_bimanual_yam(norm_stats, mapping)
    return LoraSettings(
        rank=rank,
        alpha=2 * rank,
        embodiment_id=EMBODIMENT_TAG_TO_PROJECTOR_INDEX[EMBODIMENT_TAG.value],
        init_slot=EMBODIMENT_TAG_TO_PROJECTOR_INDEX[XDOF_TAG] if is_yam else None,
        state_map=XDOF_LAYOUT if is_yam else None,
        action_map=XDOF_LAYOUT if is_yam else None,
    )


def letterbox(image):
    height, width = image.shape[:2]
    ratio = max(width / IMAGE_SIZE, height / IMAGE_SIZE)
    resized_height, resized_width = int(height / ratio), int(width / ratio)
    resized = numpy.asarray(
        Image.fromarray(numpy.ascontiguousarray(image)).resize(
            (resized_width, resized_height), resample=Image.BILINEAR
        )
    )
    placed = numpy.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=numpy.uint8)
    top, left = (IMAGE_SIZE - resized_height) // 2, (IMAGE_SIZE - resized_width) // 2
    placed[top : top + resized_height, left : left + resized_width] = resized
    return placed


def gr00t_observation(observation, prompt, image_keys, state_parts):
    state = numpy.asarray(observation["state"], dtype=numpy.float32)
    return {
        "video": {key: letterbox(observation["images"][key])[None, None] for key in image_keys},
        "state": {name: state[start:end][None, None] for name, (start, end) in state_parts.items()},
        "language": {LANGUAGE_KEY: [[prompt]]},
    }


def robot_actions(actions, action_parts):
    return numpy.concatenate([actions[name][0] for name in action_parts], axis=1).astype(
        numpy.float32
    )


def pipeline(base_checkpoint):
    config = build_config(
        FinetuneConfig(
            base_model_path=str(base_checkpoint),
            dataset_path="",
            embodiment_tag=EMBODIMENT_TAG.value,
        )
    )
    return Gr00tN1d7Pipeline(config, pathlib.Path(tempfile.mkdtemp()))


def processor(model_pipeline, norm_stats, mapping, action_horizon):
    model_pipeline.config.data.modality_configs = {
        EMBODIMENT_TAG.value: modality_config(norm_stats, mapping, action_horizon)
    }
    built = model_pipeline._create_processor()
    built.set_statistics(
        {EMBODIMENT_TAG.value: statistics(norm_stats, mapping, action_horizon)}, override=True
    )
    return built


def base_checkpoint():
    return model_storage.fetch(BASE_CHECKPOINT, BASE_CHECKPOINT_CHECKSUM)
