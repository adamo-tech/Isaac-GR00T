import concurrent.futures
import io
import itertools
import json
import math
import multiprocessing
import os
import random

from gr00t.data.types import MessageType, VLAStepData
import numpy
from PIL import Image
from runtime.embodiment import action_parts, state_parts
from safetensors.torch import load_file, save_file
import torch
from training.flops import counted_flops
from transformers import set_seed

from adamo_model import embodiment
from adamo_model.lora import ADAPTER_CONFIG_FILE, ADAPTER_FILE, apply_lora


PREPARE_THREAD_COUNT = 8
PREPARE_PROCESS_COUNT = 8
CHUNK_SIZE = 256
OPTIMIZER_FILE = "optimizer.pt"
MAPPING_KEYS = ("images", "state", "action", "frame_rate", "state_sizes", "action_sizes")


def step_data(datum, frames, state_ranges, action_ranges):
    frame_images = {}
    images = {}
    image_keys = list(datum["images"])

    for key, image in datum["images"].items():
        if isinstance(image, dict):
            pointer = (image["dataset"], image["episode"], image["frame"])
            frame_images[pointer] = frame_images.get(pointer) or frames.frame(*pointer, image_keys)
            image = frame_images[pointer][key]

        if isinstance(image, bytes):
            image = numpy.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

        images[key] = [embodiment.letterbox(image)]

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
        states={name: state[start:end][None] for name, (start, end) in state_ranges.items()},
        actions={name: actions[:, start:end] for name, (start, end) in action_ranges.items()},
        text=datum["prompt"],
        embodiment=embodiment.EMBODIMENT_TAG,
    )


def start_processor_worker(base_checkpoint, image_keys, parts, statistics, seed):
    global worker_processor
    worker_processor = embodiment.processor(
        embodiment.pipeline(base_checkpoint), image_keys, parts, statistics
    )
    worker_processor.train()
    worker_seed = (seed + os.getpid()) % 2**32
    random.seed(worker_seed)
    numpy.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def processed(step):
    return worker_processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])


class Trainer:
    base_model = "nvidia/GR00T-N1.7"
    action_horizon = embodiment.ACTION_HORIZON
    is_episode_end_padded = False

    def __init__(self, rank, frames, state_directory=None, seed=None, parts=None):
        self.rank = rank
        self.frames = frames
        self.parts = parts
        self.seed = random.randrange(2**31) if seed is None else seed
        self.mapping = None
        self.episodes = []
        self.processor = None
        self.step_count = 0
        self.loss_weight = 0.0
        self.example_flops = None
        self.preparers = concurrent.futures.ThreadPoolExecutor(max_workers=PREPARE_THREAD_COUNT)

        if state_directory:
            saved = json.loads((state_directory / "mapping.json").read_text())
            self.mapping = {name: saved[name] for name in MAPPING_KEYS}
            self.seed = saved["seed"]
            self.step_count = saved["step_count"]
            self.parts = saved["embodiment"]

        self.state_ranges = embodiment.ranges(state_parts(self.parts))
        self.action_ranges = embodiment.ranges(action_parts(self.parts))
        set_seed(self.seed)
        self.base_checkpoint = embodiment.base_checkpoint()
        self.pipeline = embodiment.pipeline(self.base_checkpoint)

        if self.pipeline.config.training.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = self.pipeline._create_model()
        self.vision_language = self.model.backbone.model

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

    def add_episodes(self, episodes):
        self.episodes += episodes

    def start(self, statistics):
        self.processor = embodiment.processor(
            self.pipeline, self.mapping["images"], self.parts, statistics
        )
        self.processor.train()
        self.processors = concurrent.futures.ProcessPoolExecutor(
            PREPARE_PROCESS_COUNT,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=start_processor_worker,
            initargs=(
                self.base_checkpoint,
                self.mapping["images"],
                self.parts,
                statistics,
                self.seed,
            ),
        )
        self.lora = embodiment.lora_settings(
            self.rank, self.processor, self.base_checkpoint, self.parts
        )
        apply_lora(self.model, self.lora)
        self.model.to("cuda").train()

        for module in [
            *(block.mlp for block in self.vision_language.visual.blocks),
            *(
                module
                for layer in self.vision_language.language_model.layers
                for module in (layer.input_layernorm, layer.post_attention_layernorm, layer.mlp)
            ),
        ]:
            module.compile()

        self.params = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(self.params, lr=0.0, fused=True)

    def statistics_episodes(self, data):
        if self.episodes:
            return self.episodes

        images = [image for datum in data for image in datum["images"].values()]

        if not isinstance(images[0], dict):
            raise ValueError(
                "This run has no normalization stats. Call `training.compute_norm_stats(episodes)` before `forward_backward`."
            )

        return [
            episode
            for dataset_id in dict.fromkeys(image["dataset"] for image in images)
            for episode in self.frames.episodes(dataset_id)
        ]

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

        state_size = sum(part["size"] for part in state_parts(self.parts))
        action_size = sum(part["size"] for part in action_parts(self.parts))

        if sum(mapping["state_sizes"]) != state_size or sum(mapping["action_sizes"]) != action_size:
            raise ValueError(
                f"The embodiment has {state_size} state and {action_size} action values ({', '.join(f'{part["name"]}: {part["size"]} {part.get("vector", "both")}' for part in self.parts)}), and a datum has {sum(mapping['state_sizes'])} state and {sum(mapping['action_sizes'])} action values. State and action follow the embodiment's parts, in order."
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

        steps = list(
            self.preparers.map(
                step_data,
                data,
                itertools.repeat(self.frames),
                itertools.repeat(self.state_ranges),
                itertools.repeat(self.action_ranges),
            )
        )
        examples = list(self.processors.map(processed, steps))
        return self.batches(examples), self.batches(examples[:1])

    def batches(self, examples):
        return [
            self.batch(examples[index : index + CHUNK_SIZE])
            for index in range(0, len(examples), CHUNK_SIZE)
        ]

    def batch(self, examples):
        inputs = self.processor.collator(examples)["inputs"]
        inputs["position_ids"] = self.vision_language.model.get_rope_index(
            inputs["input_ids"], inputs["image_grid_thw"], None, inputs["attention_mask"]
        )[0]
        return {
            key: value.pin_memory() if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def local_forward_backward(self, batches):
        loss_sum = 0.0
        weight_sum = 0.0

        for batch in batches:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.pipeline.config.training.bf16
            ):
                output = self.model(batch)

            weight = float(output["action_mask"].detach().float().sum())
            (output["loss"] * weight).backward()
            loss_sum += output["loss"].detach() * weight
            weight_sum += weight

        return float(loss_sum), weight_sum

    def forward_backward(self, prepared, example_count):
        batches, example_batches = prepared
        self.example_flops = self.example_flops or counted_flops(
            lambda: self.local_forward_backward(example_batches), self.params
        )
        loss_sum, weight_sum = self.local_forward_backward(batches)
        self.loss_weight += weight_sum
        return {
            "loss": loss_sum / weight_sum,
            "flops": self.example_flops * example_count,
            "metrics": {},
        }

    def optim_step(self, adam_params):
        for group in self.optimizer.param_groups:
            group.update(
                lr=adam_params["learning_rate"],
                betas=(adam_params["beta1"], adam_params["beta2"]),
                eps=adam_params["eps"],
                weight_decay=adam_params["weight_decay"],
            )

        for parameter in self.params:
            if parameter.grad is not None:
                parameter.grad /= self.loss_weight

        self.loss_weight = 0.0
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
