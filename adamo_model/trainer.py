import concurrent.futures
import io
import itertools
import json
import math
import random

from gr00t.data.types import MessageType, VLAStepData
import numpy
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from transformers import set_seed

from adamo_model import embodiment
from adamo_model.lora import ADAPTER_CONFIG_FILE, ADAPTER_FILE, apply_lora


PREPARE_THREAD_COUNT = 8
CHUNK_SIZE = 256
OPTIMIZER_FILE = "optimizer.pt"
MAPPING_KEYS = ("images", "state", "action", "frame_rate", "state_sizes", "action_sizes")


def step_data(datum, frames, part_ranges):
    frame_images = {}
    images = {}

    for key, image in datum["images"].items():
        if isinstance(image, dict):
            pointer = (image["dataset"], image["episode"], image["frame"])
            frame_images[pointer] = frame_images.get(pointer) or frames.frame(*pointer)
            image = frame_images[pointer][key]

        if isinstance(image, bytes):
            image = numpy.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

        images[key] = [image]

    state = numpy.concatenate(
        [
            numpy.ravel(numpy.asarray(value, dtype=numpy.float32))
            for value in datum["state"].values()
        ]
    )
    actions = numpy.concatenate(
        [numpy.asarray(value, dtype=numpy.float32) for value in datum["loss_fn_inputs"].values()],
        axis=1,
    )
    return VLAStepData(
        images=images,
        states={name: state[start:end][None] for name, (start, end) in part_ranges.items()},
        actions={name: actions[:, start:end] for name, (start, end) in part_ranges.items()},
        text=datum["prompt"],
        embodiment=embodiment.EMBODIMENT_TAG,
    )


class Trainer:
    base_model = "nvidia/GR00T-N1.7"
    action_horizon = embodiment.ACTION_HORIZON

    def __init__(self, rank, frames, state_directory=None, seed=None, parts=None):
        self.rank = rank
        self.frames = frames
        self.parts = parts
        self.seed = random.randrange(2**31) if seed is None else seed
        self.mapping = None
        self.episodes = []
        self.processor = None
        self.step_count = 0
        self.preparers = concurrent.futures.ThreadPoolExecutor(max_workers=PREPARE_THREAD_COUNT)

        if state_directory:
            saved = json.loads((state_directory / "mapping.json").read_text())
            self.mapping = {name: saved[name] for name in MAPPING_KEYS}
            self.seed = saved["seed"]
            self.step_count = saved["step_count"]
            self.parts = saved["embodiment"]

        self.part_ranges = embodiment.ranges(self.parts)
        set_seed(self.seed)
        self.base_checkpoint = embodiment.base_checkpoint()
        self.pipeline = embodiment.pipeline(self.base_checkpoint)

        if self.pipeline.config.training.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = self.pipeline._create_model()

        if state_directory:
            statistics = json.loads((state_directory / "statistics.json").read_text())
            self.start(statistics[embodiment.EMBODIMENT_TAG.value])
            trainable = load_file(state_directory / ADAPTER_FILE, device="cuda")

            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(trainable[name])

            self.optimizer.load_state_dict(
                torch.load(state_directory / OPTIMIZER_FILE, map_location="cuda", weights_only=True)
            )

    def set_norm_stats(self, norm_stats):
        pass

    def add_episodes(self, episodes):
        self.episodes += episodes

    def start(self, statistics):
        self.processor = embodiment.processor(
            self.pipeline, self.mapping["images"], self.parts, statistics
        )
        self.processor.train()
        self.lora = embodiment.lora_settings(
            self.rank, self.processor, self.base_checkpoint, self.parts
        )
        apply_lora(self.model, self.lora)
        self.model.to("cuda").train()
        self.params = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(self.params, lr=0.0)

    def statistics_episodes(self, data):
        if self.episodes:
            return self.episodes

        image = next(iter(data[0]["images"].values()))

        if not isinstance(image, dict):
            raise ValueError(
                "This run has no normalization stats. Call `training.compute_norm_stats(episodes)` before `forward_backward`."
            )

        return self.frames.episodes(image["dataset"])

    def check_mapping(self, datum):
        mapping = {
            "images": list(datum["images"]),
            "state": list(datum["state"]),
            "action": list(datum["loss_fn_inputs"]),
            "frame_rate": datum["frame_rate"],
            "state_sizes": [numpy.size(value) for value in datum["state"].values()],
            "action_sizes": [len(value[0]) for value in datum["loss_fn_inputs"].values()],
        }
        self.mapping = self.mapping or mapping

        if mapping != self.mapping:
            raise ValueError(
                f"This run reads {self.mapping}, and a datum has {mapping}. A run keeps one set of image, state and action keys and one frame rate."
            )

        part_size = sum(part["size"] for part in self.parts)

        if sum(mapping["state_sizes"]) != part_size or sum(mapping["action_sizes"]) != part_size:
            raise ValueError(
                f"The embodiment has {part_size} values ({', '.join(f'{part["name"]}: {part["size"]}' for part in self.parts)}), and a datum has {sum(mapping['state_sizes'])} state and {sum(mapping['action_sizes'])} action values. State and action follow the embodiment's parts, in order."
            )

        horizon = len(next(iter(datum["loss_fn_inputs"].values())))

        if horizon != self.action_horizon:
            raise ValueError(
                f"Action chunks have {horizon} steps, and this base model takes {self.action_horizon}."
            )

    def prepare(self, data):
        for datum in data:
            self.check_mapping(datum)

        if self.processor is None:
            self.start(
                embodiment.statistics(self.statistics_episodes(data), self.mapping, self.parts)
            )

        examples = [
            self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            for step in self.preparers.map(
                step_data, data, itertools.repeat(self.frames), itertools.repeat(self.part_ranges)
            )
        ]
        return [
            self.processor.collator(examples[index : index + CHUNK_SIZE])["inputs"]
            for index in range(0, len(examples), CHUNK_SIZE)
        ]

    def forward_backward(self, batches, example_count):
        total = sum(float(batch["action_mask"].sum()) for batch in batches)
        loss = 0.0

        for batch in batches:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.pipeline.config.training.bf16
            ):
                output = self.model(batch)

            weight = float(output["action_mask"].detach().float().sum()) / total
            (output["loss"] * weight).backward()
            loss += output["loss"].item() * weight

        return {"loss": loss, "metrics": {}}

    def optim_step(self, adam_params):
        for group in self.optimizer.param_groups:
            group.update(
                lr=adam_params["learning_rate"],
                betas=(adam_params["beta1"], adam_params["beta2"]),
                eps=adam_params["eps"],
                weight_decay=adam_params["weight_decay"],
            )

        clip = adam_params["grad_clip_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, clip if clip > 0 else math.inf)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        return {"grad_norm": float(grad_norm), "step": self.step_count}

    def save_weights(self, directory):
        save_file(
            {
                name: parameter.detach().contiguous()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            },
            directory / ADAPTER_FILE,
        )
        (directory / ADAPTER_CONFIG_FILE).write_text(json.dumps(self.lora.to_json()))
        self.processor.save_pretrained(directory)
        mapping = {
            **self.mapping,
            "base_model": self.base_model,
            "rank": self.rank,
            "action_horizon": self.action_horizon,
            "embodiment": self.parts,
            "step_count": self.step_count,
            "seed": self.seed,
        }
        (directory / "mapping.json").write_text(json.dumps(mapping))
        return mapping

    def save_state(self, directory):
        torch.save(self.optimizer.state_dict(), directory / OPTIMIZER_FILE)
        return self.save_weights(directory)
