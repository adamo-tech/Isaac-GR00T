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

"""Launch the Tinker-protocol training server for GR00T N1.7 (one process, one GPU).

  GR00T_TRAIN_API_KEY=ak_... python -m gr00t.train_server.launch \\
      --base-model-path nvidia/GR00T-N1.7-3B \\
      --embodiment-tag new_embodiment \\
      --modality-config-path examples/rebot-arm-dm/rebot_config.py \\
      --stats /path/to/lerobot_dataset \\
      --checkpoint-dir ./train_server_checkpoints

The model is built exactly as ``launch_finetune.py`` builds it (frozen VLM, trained projector
and diffusion action head, same augmentations). Normalization statistics come from ``--stats``:
either a GR00T LeRobot dataset directory (stats and relative-action stats are computed and
cached in its ``meta/`` the same way fine-tuning does) or a JSON file in GR00T's statistics
format, ``{"state": ..., "action": ..., "relative_action": ...}``, optionally nested under the
embodiment tag (e.g. a checkpoint's ``statistics.json``).
"""

from dataclasses import dataclass
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed
import tyro
import uvicorn

from gr00t.configs.base_config import Config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.launch_finetune import build_config
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

from .app import create_app
from .datum import EmbodimentSpec
from .session import TrainingSession


@dataclass
class ServerConfig:
    base_model_path: str
    """Checkpoint behind the API's "nvidia/GR00T-N1.7" (HF repo id or local dir)."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive)."""

    stats: str
    """LeRobot dataset directory, or a statistics JSON file (see module docstring)."""

    modality_config_path: str | None = None
    """Python file registering the modality config; None uses a pre-registered tag."""

    checkpoint_dir: str = "./train_server_checkpoints"
    """Where save_weights / save_weights_for_sampler write, and where relative load paths resolve."""

    api_key: str | None = None
    """Expected X-API-Key; defaults to the GR00T_TRAIN_API_KEY environment variable."""

    host: str = "0.0.0.0"
    port: int = 8000

    first_seq_id: int = 1
    """seq_id of each model's first ordered call (Tinker: create_model is 0, so 1)."""

    replace_model: bool = False
    """Let a new create_model replace the held model instead of being refused."""

    augment: bool = True
    """Apply GR00T's training-time augmentations (image transforms, state dropout)."""

    state_dropout_prob: float = 0.2
    """Same default as launch_finetune.py."""

    device: str = "cuda"


def load_statistics(stats: str, tag: EmbodimentTag, config: Config) -> dict[str, Any]:
    path = Path(stats)
    if path.is_dir():
        generate_stats(path)
        generate_rel_stats(path, tag)
        loaded = LeRobotEpisodeLoader(
            path, config.data.modality_configs[tag.value]
        ).get_dataset_statistics()
        return {
            modality: merge_statistics(
                per_dataset_stats=[loaded[modality]],
                dataset_sampling_weights=[1.0],
                is_relative_stats=modality == "relative_action",
            )
            for modality in ("state", "action", "relative_action")
            if modality in loaded
        }
    loaded = json.loads(path.read_text())
    return loaded.get(tag.value, loaded)


def build(cfg: ServerConfig):
    """Return (spec, create_session) for create_app."""
    ft = FinetuneConfig(
        base_model_path=cfg.base_model_path,
        dataset_path=cfg.stats,
        embodiment_tag=cfg.embodiment_tag,
        modality_config_path=cfg.modality_config_path,
        state_dropout_prob=cfg.state_dropout_prob,
    )
    config = build_config(ft)
    config.validate()
    tag = ft.embodiment_tag
    modality_config = config.data.modality_configs[tag.value]
    statistics = load_statistics(cfg.stats, tag, config)
    spec = EmbodimentSpec.from_modality_config(tag, modality_config, statistics)

    def create_session(seed: int | None) -> TrainingSession:
        # A replaced model's session has just been dropped; give its GPU memory back first.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if seed is not None:
            set_seed(seed)
        if config.training.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        save_cfg_dir = Path(cfg.checkpoint_dir) / ".server"
        save_cfg_dir.mkdir(parents=True, exist_ok=True)
        pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir)
        model = pipeline._create_model()
        processor = pipeline._create_processor()
        processor.set_statistics({tag.value: statistics}, override=True)
        return TrainingSession(
            model,
            processor,
            device=cfg.device,
            augment=cfg.augment,
            bf16=config.training.bf16,
        )

    return spec, create_session


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = tyro.cli(ServerConfig, description=__doc__)
    api_key = cfg.api_key or os.environ.get("GR00T_TRAIN_API_KEY")
    if not api_key:
        raise SystemExit("Set --api-key or GR00T_TRAIN_API_KEY")
    spec, create_session = build(cfg)
    app = create_app(
        api_key=api_key,
        spec=spec,
        create_session=create_session,
        checkpoint_dir=cfg.checkpoint_dir,
        first_seq_id=cfg.first_seq_id,
        replace_model=cfg.replace_model,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
