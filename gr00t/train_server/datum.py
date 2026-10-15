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

"""Decode Tinker-style datums into GR00T ``VLAStepData``.

A datum is ``{"model_input": {"chunks": [...]}, "loss_fn_inputs": {action_key: TensorData}}``
where chunks are ``image`` (base64 PNG/JPEG), ``state`` (TensorData) or ``text``. Names map
onto the embodiment's modality config keys; everything here only reshapes and decodes, the
GR00T processor does the actual preprocessing (normalization, relative actions, image
transforms, tokenization).
"""

import base64
import binascii
from dataclasses import dataclass
import io
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VLAStepData


class UserError(ValueError):
    """Bad client input; reported with category "user"."""


IMAGE_FORMATS = {"png": "PNG", "jpeg": "JPEG"}
TENSOR_DTYPES = {"float32": np.float32, "int64": np.int64}


@dataclass(frozen=True)
class EmbodimentSpec:
    """What a datum must contain for one embodiment, derived from its modality config + stats."""

    embodiment_tag: EmbodimentTag
    image_names: list[str]
    image_frames: int
    state_dims: dict[str, int]
    state_steps: int
    action_dims: dict[str, int]
    action_horizon: int

    @classmethod
    def from_modality_config(
        cls,
        embodiment_tag: EmbodimentTag,
        modality_config: dict[str, ModalityConfig],
        statistics: dict[str, Any],
    ) -> "EmbodimentSpec":
        def dims(modality: str) -> dict[str, int]:
            keys = modality_config[modality].modality_keys
            missing = [k for k in keys if k not in statistics.get(modality, {})]
            if missing:
                raise ValueError(f"Statistics have no {modality} entries for {missing}")
            return {k: len(statistics[modality][k]["min"]) for k in keys}

        return cls(
            embodiment_tag=embodiment_tag,
            image_names=list(modality_config["video"].modality_keys),
            image_frames=len(modality_config["video"].delta_indices),
            state_dims=dims("state"),
            state_steps=len(modality_config["state"].delta_indices),
            action_dims=dims("action"),
            action_horizon=len(modality_config["action"].delta_indices),
        )

    def info(self) -> dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "image_names": self.image_names,
            "state_names": list(self.state_dims),
            "action_names": list(self.action_dims),
            "state_dims": self.state_dims,
            "action_dims": self.action_dims,
        }


def decode_tensor(value: Any, what: str) -> np.ndarray:
    """TensorData ``{"data": [...], "dtype": ..., "shape"?: [...]}`` -> ndarray (1-D if no shape)."""
    if not isinstance(value, dict) or "data" not in value:
        raise UserError(f"{what}: expected TensorData {{data, dtype, shape}}, got {value!r:.100}")
    dtype = value.get("dtype", "float32")
    if dtype not in TENSOR_DTYPES:
        raise UserError(f"{what}: dtype {dtype!r} not in {sorted(TENSOR_DTYPES)}")
    try:
        array = np.asarray(value["data"], dtype=TENSOR_DTYPES[dtype])
    except (TypeError, ValueError) as e:
        raise UserError(f"{what}: data is not a flat list of numbers ({e})") from None
    if array.ndim != 1:
        raise UserError(f"{what}: data must be a flat list, got {array.ndim}-D")
    shape = value.get("shape")
    if shape is None:
        return array
    if int(np.prod(shape)) != array.size:
        raise UserError(f"{what}: shape {shape} does not match {array.size} values")
    return array.reshape(shape)


def decode_image(chunk: dict, what: str) -> np.ndarray:
    fmt = chunk.get("format")
    if fmt not in IMAGE_FORMATS:
        raise UserError(f"{what}: format {fmt!r} not in {sorted(IMAGE_FORMATS)}")
    try:
        raw = base64.b64decode(chunk.get("data", ""), validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as e:
        raise UserError(f"{what}: could not decode {fmt} image ({e})") from None
    if image.format != IMAGE_FORMATS[fmt]:
        raise UserError(f"{what}: declared {fmt} but bytes are {image.format}")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _check_names(got: list[str], expected: list[str], what: str) -> None:
    unknown = [n for n in got if n not in expected]
    missing = [n for n in expected if n not in got]
    if unknown or missing:
        raise UserError(
            f"{what}: unknown {unknown}, missing {missing}; expected exactly {expected}"
        )


def decode_datum(datum: Any, spec: EmbodimentSpec, index: int = 0) -> VLAStepData:
    where = f"datum {index}"
    if not isinstance(datum, dict) or not isinstance(datum.get("model_input"), dict):
        raise UserError(f"{where}: expected {{model_input: {{chunks}}, loss_fn_inputs}}")
    chunks = datum["model_input"].get("chunks")
    if not isinstance(chunks, list):
        raise UserError(f"{where}: model_input.chunks must be a list")

    images: dict[str, list[np.ndarray]] = {}
    states: dict[str, np.ndarray] = {}
    texts: list[str] = []
    for i, chunk in enumerate(chunks):
        what = f"{where} chunk {i}"
        kind = chunk.get("type") if isinstance(chunk, dict) else None
        if kind == "image":
            name = chunk.get("name")
            if name not in spec.image_names:
                raise UserError(f"{what}: unknown camera {name!r}; expected {spec.image_names}")
            images.setdefault(name, []).append(decode_image(chunk, f"{what} ({name})"))
        elif kind == "state":
            name = chunk.get("name")
            if name not in spec.state_dims:
                raise UserError(f"{what}: unknown state {name!r}; expected {list(spec.state_dims)}")
            if name in states:
                raise UserError(f"{what}: state {name!r} given twice")
            states[name] = _state(decode_tensor(chunk.get("data"), what), spec, name, what)
        elif kind == "text":
            if not isinstance(chunk.get("text"), str):
                raise UserError(f"{what}: text chunk needs a string 'text'")
            texts.append(chunk["text"])
        else:
            raise UserError(f"{what}: type {kind!r} not in ['image', 'state', 'text']")

    _check_names(list(images), spec.image_names, f"{where} cameras")
    for name, frames in images.items():
        if len(frames) != spec.image_frames:
            raise UserError(
                f"{where}: camera {name!r} has {len(frames)} frames, expected {spec.image_frames}"
            )
    _check_names(list(states), list(spec.state_dims), f"{where} states")
    if len(texts) != 1:
        raise UserError(f"{where}: expected exactly one text chunk, got {len(texts)}")

    loss_fn_inputs = datum.get("loss_fn_inputs")
    if not isinstance(loss_fn_inputs, dict):
        raise UserError(f"{where}: loss_fn_inputs must map action keys to TensorData")
    _check_names(list(loss_fn_inputs), list(spec.action_dims), f"{where} loss_fn_inputs")
    actions = {
        name: _action(decode_tensor(value, f"{where} action {name!r}"), spec, name, where)
        for name, value in loss_fn_inputs.items()
    }

    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text=texts[0],
        embodiment=spec.embodiment_tag,
    )


def _state(array: np.ndarray, spec: EmbodimentSpec, name: str, what: str) -> np.ndarray:
    dim, steps = spec.state_dims[name], spec.state_steps
    if array.ndim == 1 and steps == 1:
        array = array[None]
    if array.shape != (steps, dim):
        raise UserError(
            f"{what}: state {name!r} has shape {list(array.shape)}, expected [{dim}]"
            if steps == 1
            else f"{what}: state {name!r} has shape {list(array.shape)}, expected [{steps}, {dim}]"
        )
    return array.astype(np.float32)


def _action(array: np.ndarray, spec: EmbodimentSpec, name: str, where: str) -> np.ndarray:
    expected = (spec.action_horizon, spec.action_dims[name])
    if array.ndim == 1 and array.size == expected[0] * expected[1]:
        array = array.reshape(expected)
    if array.shape != expected:
        raise UserError(
            f"{where}: action {name!r} has shape {list(array.shape)}, expected {list(expected)}"
        )
    return array.astype(np.float32)
