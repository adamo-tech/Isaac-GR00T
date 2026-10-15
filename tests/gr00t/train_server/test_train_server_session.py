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

"""Save/load round trips of TrainingSession checkpoints."""

from gr00t.train_server.datum import UserError
import pytest
from test_support.train_server import make_session
import torch


def test_load_accepts_tied_weights_saved_once(tmp_path):
    # Like the backbone's lm_head, which shares the embedding matrix: save_pretrained writes the
    # shared tensor once, so loading must not report the alias as a missing key.
    session = make_session(seed=0)
    source = session.model.action_head.action_decoder
    session.model.tied_head = torch.nn.Linear(1, 1)
    first = next(p for p in source.parameters() if p.dim() == 2)
    session.model.tied_head.weight = first
    session.model.tied_head._dynamic_tied_weights_keys = ["weight"]

    session.save(tmp_path / "ckpt", optimizer=True)
    with torch.no_grad():
        first.add_(1.0)
    session.load(tmp_path / "ckpt", optimizer=True)
    assert session.model.tied_head.weight is first


def test_load_rejects_mismatched_checkpoint(tmp_path):
    session = make_session(seed=0)
    session.save(tmp_path / "ckpt", optimizer=False)
    session.model.extra = torch.nn.Linear(2, 2)
    with pytest.raises(UserError, match="missing keys"):
        session.load(tmp_path / "ckpt", optimizer=False)
    with pytest.raises(UserError, match="no optimizer state"):
        del session.model.extra
        session.load(tmp_path / "ckpt", optimizer=True)
