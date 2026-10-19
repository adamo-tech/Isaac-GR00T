import concurrent.futures
import io
import itertools
import json
import math
import random
import threading

from gr00t.data.types import MessageType, VLAStepData
import numpy
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from transformers import set_seed

from adamo_model import embodiment
from adamo_model.lora import ADAPTER_CONFIG_FILE, ADAPTER_FILE, apply_lora


MICRO_BATCH_SIZE = 32
PREPARE_THREAD_COUNT = 8
OPTIMIZER_FILE = "optimizer.pt"


def step_data(datum, frames, state_parts, action_parts):
    frame_images = {}
    images = {}

    for key, image in datum["images"].items():
        if isinstance(image, dict):
            pointer = (image["dataset"], image["episode"], image["frame"])
            frame_images[pointer] = frame_images.get(pointer) or frames.frame(*pointer)
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
    states = {name: state[start:end][None] for name, (start, end) in state_parts.items()}
    return VLAStepData(
        images=images,
        states=states,
        actions={name: actions[:, start:end] for name, (start, end) in action_parts.items()},
        text=datum["prompt"],
        embodiment=embodiment.EMBODIMENT_TAG,
    )


class Trainer:
    base_model = "nvidia/GR00T-N1.7"
    action_horizon = 40

    def __init__(self, rank, frames, state_directory=None, seed=None):
        self.rank = rank
        self.seed = random.randrange(2**31) if seed is None else seed
        self.frames = frames
        self.mapping = None
        self.norm_stats = None
        self.processor = None
        self.step_count = 0
        self.example_count = 0
        self.processor_lock = threading.Lock()
        self.preparers = concurrent.futures.ThreadPoolExecutor(max_workers=PREPARE_THREAD_COUNT)

        if state_directory:
            saved = json.loads((state_directory / "mapping.json").read_text())
            self.mapping = {
                name: saved[name] for name in ("images", "state", "action", "frame_rate")
            }
            self.norm_stats = json.loads((state_directory / "norm_stats.json").read_text())
            self.step_count = saved["step_count"]
            self.seed = saved["seed"]

        set_seed(self.seed)
        self.pipeline = embodiment.pipeline(embodiment.base_checkpoint())
        torch.backends.cuda.matmul.allow_tf32 = self.pipeline.config.training.tf32
        torch.backends.cudnn.allow_tf32 = self.pipeline.config.training.tf32
        self.model = self.pipeline._create_model()

        if state_directory:
            self.built_processor()
            trainable = load_file(state_directory / ADAPTER_FILE, device="cuda")

            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(trainable[name])

            self.optimizer.load_state_dict(
                torch.load(state_directory / OPTIMIZER_FILE, map_location="cuda", weights_only=True)
            )

    def set_norm_stats(self, norm_stats):
        if self.norm_stats not in (None, norm_stats):
            raise ValueError(
                "This run already has normalization stats, and these are different. A run keeps one set of stats."
            )

        self.norm_stats = norm_stats

    def check_mapping(self, datum):
        mapping = {
            "images": list(datum["images"]),
            "state": list(datum["state"]),
            "action": list(datum["loss_fn_inputs"]),
            "frame_rate": datum["frame_rate"],
        }
        self.mapping = self.mapping or mapping

        if mapping != self.mapping:
            raise ValueError(
                f"This run reads {self.mapping}, and a datum has {mapping}. A run keeps one set of image, state and action keys and one frame rate."
            )

        horizon = len(next(iter(datum["loss_fn_inputs"].values())))

        if horizon != self.action_horizon:
            raise ValueError(
                f"Action chunks have {horizon} steps, and this base model takes {self.action_horizon}."
            )

    def built_processor(self):
        missing = [
            key
            for key in self.mapping["state"] + self.mapping["action"]
            if key not in (self.norm_stats or {})
        ]

        if missing:
            raise ValueError(
                f"There are no normalization stats for {', '.join(f'`{key}`' for key in missing)}. Pass `norm_stats` to `adamo.LoraTraining`, or use `dataset.batches`."
            )

        with self.processor_lock:
            if self.processor is None:
                self.state_parts, self.action_parts = (
                    embodiment.parts(self.norm_stats, self.mapping, "state"),
                    embodiment.parts(self.norm_stats, self.mapping, "action"),
                )
                self.lora_settings = embodiment.lora_settings(
                    self.norm_stats, self.mapping, self.rank
                )
                apply_lora(self.model, self.lora_settings)
                self.model.to("cuda").train()
                self.params = [
                    parameter for parameter in self.model.parameters() if parameter.requires_grad
                ]
                self.optimizer = torch.optim.AdamW(self.params, lr=0.0)
                processor = embodiment.processor(
                    self.pipeline, self.norm_stats, self.mapping, self.action_horizon
                )
                processor.train()
                self.processor = processor

        return self.processor

    def prepare(self, data):
        for datum in data:
            self.check_mapping(datum)

        processor = self.built_processor()
        examples = [
            processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
            for step in self.preparers.map(
                step_data,
                data,
                itertools.repeat(self.frames),
                itertools.repeat(self.state_parts),
                itertools.repeat(self.action_parts),
            )
        ]
        return [
            (
                processor.collator(examples[start : start + MICRO_BATCH_SIZE])["inputs"],
                len(examples[start : start + MICRO_BATCH_SIZE]),
            )
            for start in range(0, len(examples), MICRO_BATCH_SIZE)
        ]

    def forward_backward(self, prepared, example_count):
        self.step_count += 1
        loss_sum = 0.0

        for inputs, micro_batch_size in prepared:
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.pipeline.config.training.bf16
            ):
                micro_batch_loss = self.model(inputs)["loss"] * micro_batch_size

            micro_batch_loss.backward()
            loss_sum += micro_batch_loss.item()

        self.example_count += example_count

        return {"loss": loss_sum / example_count, "metrics": {"example_count": self.example_count}}

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
                parameter.grad /= self.example_count

        torch.nn.utils.clip_grad_norm_(
            self.params,
            adam_params["grad_clip_norm"] if adam_params["grad_clip_norm"] > 0 else math.inf,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.example_count = 0

    def saved_mapping(self):
        action_sizes = [len(self.norm_stats[key]["q01"]) for key in self.mapping["action"]]
        return {
            **self.mapping,
            "base_model": self.base_model,
            "rank": self.rank,
            "action_horizon": self.action_horizon,
            "action_sizes": action_sizes,
            "step_count": self.step_count,
            "seed": self.seed,
        }

    def save_weights(self, directory):
        save_file(
            {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            },
            directory / ADAPTER_FILE,
        )
        (directory / ADAPTER_CONFIG_FILE).write_text(json.dumps(self.lora_settings.to_json()))
        self.processor.save_pretrained(directory)
        (directory / "norm_stats.json").write_text(json.dumps(self.norm_stats))
        (directory / "mapping.json").write_text(json.dumps(self.saved_mapping()))
        return self.saved_mapping()

    def save_state(self, directory):
        torch.save(self.optimizer.state_dict(), directory / OPTIMIZER_FILE)
        return self.save_weights(directory)
