import json
import pathlib
import tempfile

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.action_chunking import JointActionChunk
from gr00t.data.state_action.pose import JointPose
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.experiment.launch_finetune import build_config
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
import numpy
from PIL import Image
from runtime import model_storage

from adamo_model.lora import LoraSettings


BASE_CHECKPOINT = "hf://nvidia/GR00T-N1.7-3B@2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BASE_CHECKPOINT_CHECKSUM = "339454305dedcf2729dde2e39cca0856f2d4abcb52776dedc39496050b8b4469"
EMBODIMENT_TAG = EmbodimentTag.NEW_EMBODIMENT
LANGUAGE_KEY = "annotation.human.task_description"
ACTION_HORIZON = 40
IMAGE_SIZE = 224


def ranges(parts):
    ends = numpy.cumsum([part["size"] for part in parts])
    return {part["name"]: (int(end) - part["size"], int(end)) for part, end in zip(parts, ends)}


def modality_config(image_keys, parts):
    names = [part["name"] for part in parts]
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=image_keys),
        "state": ModalityConfig(delta_indices=[0], modality_keys=names),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=names,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE
                    if part["kind"] == "joints"
                    else ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for part in parts
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
    }


def columns(episode, keys):
    return numpy.concatenate(
        [
            numpy.asarray(episode[key], dtype=numpy.float32).reshape(len(episode[key]), -1)
            for key in keys
        ],
        axis=1,
    )


def summary(values):
    return {
        "mean": numpy.mean(values, axis=0).tolist(),
        "std": numpy.std(values, axis=0).tolist(),
        "min": numpy.min(values, axis=0).tolist(),
        "max": numpy.max(values, axis=0).tolist(),
        "q01": numpy.quantile(values, 0.01, axis=0).tolist(),
        "q99": numpy.quantile(values, 0.99, axis=0).tolist(),
    }


def relative_chunks(states, actions, start, end):
    chunks = []

    for state, action in zip(states, actions):
        for index in range(len(action) - (ACTION_HORIZON - 1)):
            chunk = JointActionChunk(
                [JointPose(joints) for joints in action[index : index + ACTION_HORIZON, start:end]]
            ).relative_chunking(reference_frame=JointPose(state[index, start:end]))
            chunks.append(numpy.stack([pose.joints for pose in chunk.poses], dtype=numpy.float32))

    return chunks


def statistics(episodes, mapping, parts):
    states = [columns(episode, mapping["state"]) for episode in episodes]
    actions = [columns(episode, mapping["action"]) for episode in episodes]
    all_states, all_actions = numpy.vstack(states), numpy.vstack(actions)
    part_ranges = ranges(parts)
    return {
        "state": {
            name: summary(all_states[:, start:end]) for name, (start, end) in part_ranges.items()
        },
        "action": {
            name: summary(all_actions[:, start:end]) for name, (start, end) in part_ranges.items()
        },
        "relative_action": {
            part["name"]: summary(relative_chunks(states, actions, *part_ranges[part["name"]]))
            for part in parts
            if part["kind"] == "joints"
        },
    }


def width(values):
    return numpy.shape(values)[-1]


def pretrained_layouts(base_checkpoint):
    configs = json.loads((base_checkpoint / "processor_config.json").read_text())[
        "processor_kwargs"
    ]["modality_configs"]
    statistics = json.loads((base_checkpoint / "statistics.json").read_text())
    layouts = {}

    for tag, config in configs.items():
        if config["state"]["modality_keys"] != config["action"]["modality_keys"]:
            continue

        offset = 0
        layouts[tag] = []

        for key, action_config in zip(
            config["action"]["modality_keys"], config["action"]["action_configs"]
        ):
            relative = statistics[tag].get("relative_action", {})
            size = (
                width(relative[key]["min"])
                if key in relative
                else width(statistics[tag]["action"][key]["min"])
            )
            kind = (
                "joints"
                if action_config["type"] == "NON_EEF" and action_config["rep"] == "RELATIVE"
                else "gripper"
                if "gripper" in key
                else None
            )
            side = (
                "left" if key.startswith("left_") else "right" if key.startswith("right_") else None
            )
            layouts[tag].append({"kind": kind, "side": side, "size": size, "offset": offset})
            offset += size

    return layouts


def pretrained_start(base_checkpoint, parts):
    for tag, layout in pretrained_layouts(base_checkpoint).items():
        free = list(layout)
        offsets = []

        for part in parts:
            match = next(
                (
                    candidate
                    for candidate in free
                    if (candidate["kind"], candidate["side"], candidate["size"])
                    == (part["kind"], part["side"], part["size"])
                ),
                None,
            )

            if match is None:
                break

            free.remove(match)
            offsets.append(match["offset"])
        else:
            return tag, [
                offset + index
                for part, offset in zip(parts, offsets)
                for index in range(part["size"])
            ]

    return None, None


def lora_settings(rank, processor, base_checkpoint, parts):
    tag, layout = pretrained_start(base_checkpoint, parts)
    return LoraSettings(
        rank=rank,
        alpha=2 * rank,
        embodiment_id=processor.embodiment_id_mapping[EMBODIMENT_TAG.value],
        init_slot=None if tag is None else processor.embodiment_id_mapping[tag],
        state_map=layout,
        action_map=layout,
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


def gr00t_observation(observation, prompt, image_keys, parts):
    state = numpy.asarray(observation["state"], dtype=numpy.float32)
    return {
        "video": {key: letterbox(observation["images"][key])[None, None] for key in image_keys},
        "state": {
            name: state[start:end][None, None] for name, (start, end) in ranges(parts).items()
        },
        "language": {LANGUAGE_KEY: [[prompt]]},
    }


def robot_actions(actions, parts):
    return numpy.concatenate([actions[part["name"]][0] for part in parts], axis=1).astype(
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


def processor(model_pipeline, image_keys, parts, statistics):
    model_pipeline.config.data.modality_configs = {
        EMBODIMENT_TAG.value: modality_config(image_keys, parts)
    }
    built = model_pipeline._create_processor()
    built.set_statistics({EMBODIMENT_TAG.value: statistics}, override=True)
    return built


def base_checkpoint():
    return model_storage.fetch(BASE_CHECKPOINT, BASE_CHECKPOINT_CHECKSUM)
