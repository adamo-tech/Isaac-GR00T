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

"""A tiny GR00T N1.7 (stub VLM backbone + tokenizer, real processor and action head) on CPU."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
import numpy as np
from PIL import Image
import torch
from transformers.feature_extraction_utils import BatchFeature


TAG = EmbodimentTag.NEW_EMBODIMENT
HORIZON = 4
STATE_DIMS = {"arm": 5, "gripper": 1}
CAMERAS = ["ego_view", "wrist"]
API_KEY = "ak_test_key"
EMBED_DIM = 64

MODALITY_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=CAMERAS),
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_DIMS)),
    "action": ModalityConfig(
        delta_indices=list(range(HORIZON)),
        modality_keys=list(STATE_DIMS),
        action_configs=[
            ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, ActionFormat.DEFAULT),
            ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}


def _stats(shape) -> dict[str, list]:
    ones = np.ones(shape)
    return {
        "min": (-ones).tolist(),
        "max": ones.tolist(),
        "mean": (0 * ones).tolist(),
        "std": ones.tolist(),
        "q01": (-ones).tolist(),
        "q99": ones.tolist(),
    }


STATISTICS = {
    "state": {k: _stats(d) for k, d in STATE_DIMS.items()},
    "action": {k: _stats(d) for k, d in STATE_DIMS.items()},
    "relative_action": {"arm": _stats((HORIZON, STATE_DIMS["arm"]))},
}


class FakeVLMProcessor:
    """Stands in for the Qwen3-VL processor: fixed-length token ids, no pixel tensors."""

    tokenizer = SimpleNamespace(padding_side="left")

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        return "prompt"

    def __call__(self, text, images, return_tensors="pt", padding=True):
        n = len(text)
        return {
            "input_ids": torch.ones(n, 8, dtype=torch.long),
            "attention_mask": torch.ones(n, 8, dtype=torch.long),
        }


class FakeBackbone:
    """Constant VL features, so the action head can fit a fixed batch."""

    def __init__(self, seq_len: int = 8):
        self.features = torch.randn(
            1, seq_len, EMBED_DIM, generator=torch.Generator().manual_seed(0)
        )

    def prepare_input(self, inputs):
        return BatchFeature(data=inputs)

    def __call__(self, inputs):
        batch = inputs["state"].shape[0]
        features = self.features.to(inputs["state"].device).expand(batch, -1, -1)
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": torch.ones(batch, features.shape[1], dtype=torch.long),
                "image_mask": torch.ones(batch, features.shape[1], dtype=torch.bool),
            }
        )


def make_session(seed: int | None = None):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from gr00t.train_server.session import TrainingSession

    if seed is not None:
        torch.manual_seed(seed)
    config = Gr00tN1d7Config(
        model_name="nvidia/Cosmos-Reason2-2B",
        backbone_model_type="qwen",
        backbone_embedding_dim=EMBED_DIM,
        hidden_size=EMBED_DIM,
        input_embedding_dim=EMBED_DIM,
        max_state_dim=8,
        max_action_dim=8,
        action_horizon=HORIZON,
        state_history_length=1,
        max_num_embodiments=4,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        select_layer=1,
        use_flash_attention=False,
        load_bf16=False,
        state_dropout_prob=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": EMBED_DIM,
            "interleave_self_attention": True,
        },
    )
    with (
        patch("gr00t.model.gr00t_n1d7.gr00t_n1d7.get_backbone_cls") as get_backbone_cls,
        patch(
            "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
            return_value=FakeVLMProcessor(),
        ),
    ):
        get_backbone_cls.return_value = lambda **kwargs: FakeBackbone()
        model = Gr00tN1d7(config)
        processor = Gr00tN1d7Processor(
            modality_configs={TAG.value: MODALITY_CONFIG},
            max_state_dim=8,
            max_action_dim=8,
            max_action_horizon=HORIZON,
            use_albumentations=True,
            shortest_image_edge=32,
            crop_fraction=0.95,
            use_relative_action=True,
            embodiment_id_mapping={TAG.value: 0},
        )
    processor.set_statistics({TAG.value: STATISTICS}, override=True)
    return TrainingSession(model, processor, device="cpu", augment=False)


def png_b64(rng: np.random.Generator, size: int = 32) -> str:
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def tensor(array: np.ndarray) -> dict:
    return {"data": array.ravel().tolist(), "dtype": "float32", "shape": list(array.shape)}


# Every datum asks for the same action chunk, so a model that sees near-identical inputs has
# something to fit and the flow-matching loss can fall well below its starting value.
ACTION_TARGET = {
    k: np.random.default_rng(1).choice([-0.9, 0.9], (HORIZON, d)).astype(np.float32)
    for k, d in STATE_DIMS.items()
}


def make_datum(rng: np.random.Generator, cameras=CAMERAS) -> dict:
    chunks = [{"type": "image", "name": c, "format": "png", "data": png_b64(rng)} for c in cameras]
    chunks += [
        {"type": "state", "name": k, "data": tensor(rng.uniform(-0.05, 0.05, d).astype(np.float32))}
        for k, d in STATE_DIMS.items()
    ]
    chunks.append({"type": "text", "text": "stack the cubes"})
    return {
        "model_input": {"chunks": chunks},
        "loss_fn_inputs": {k: tensor(target) for k, target in ACTION_TARGET.items()},
    }


def make_spec():
    from gr00t.train_server.datum import EmbodimentSpec

    return EmbodimentSpec.from_modality_config(TAG, MODALITY_CONFIG, STATISTICS)
