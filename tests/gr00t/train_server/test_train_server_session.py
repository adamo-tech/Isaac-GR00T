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

from gr00t.data.types import MessageType
from gr00t.train_server.datum import UserError, decode_datum
import numpy as np
import pytest
from test_support.train_server import SLOT, make_datum, make_session, make_spec
import torch


ADAM = {
    "learning_rate": 1e-2,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


def make_steps(n=4):
    rng = np.random.default_rng(0)
    return [decode_datum(make_datum(rng), make_spec(), i) for i in range(n)]


def train(session, steps, n=3):
    for _ in range(n):
        session.forward_backward(steps)
        session.optim_step(ADAM)


def eval_loss(session, steps) -> float:
    """Flow-matching loss with fixed noise and no augmentation."""
    session.model.eval()
    processed = [
        session.processor([{"type": MessageType.EPISODE_STEP.value, "content": s}]) for s in steps
    ]
    batch = session.collator(processed)["inputs"]
    torch.manual_seed(123)
    with torch.no_grad():
        loss = session.model(batch)["loss"].item()
    session.model.train()
    return loss


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


def test_lora_keeps_one_projector_slot_and_trains_adapters_only():
    session = make_session(seed=0, lora_rank=4, lora=True)
    names = {n for n, _ in session.model.named_parameters()}
    assert not any(n.endswith((".W", ".b")) for n in names), "multi-slot projectors remain"
    assert "action_head.action_decoder.layer2.lora_A.default.weight" in names
    trainable = {n for n, p in session.model.named_parameters() if p.requires_grad}
    assert any(".lora_A." in n for n in trainable)
    lora_modules = ("model", "vl_self_attention", "state_encoder", "action_encoder")
    lora_modules += ("action_decoder",)
    for n in trainable:
        if n.startswith(tuple(f"action_head.{m}." for m in lora_modules)):
            assert "lora_" in n, f"{n} should be frozen"
    base = {
        n: p.detach().clone() for n, p in session.model.named_parameters() if ".base_layer." in n
    }
    train(session, make_steps())
    for n, p in session.model.named_parameters():
        if n in base:
            assert torch.equal(p, base[n]), n


def test_lora_save_weights_holds_only_trainable_tensors(tmp_path):
    session = make_session(seed=0, lora_rank=4, lora=True)
    train(session, make_steps())
    session.save(tmp_path / "ckpt", optimizer=True)
    files = sorted(f.name for f in (tmp_path / "ckpt").iterdir())
    assert files == ["lora_config.json", "optimizer.pt", "trainable.safetensors"]

    expected = {n: p.detach().clone() for n, p in session.model.named_parameters()}
    step = session.step
    with torch.no_grad():
        for p in session.model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    session.load(tmp_path / "ckpt", optimizer=True)
    for n, p in session.model.named_parameters():
        assert torch.equal(p, expected[n]), n
    assert session.step == step

    other_rank = make_session(seed=0, lora_rank=8, lora=True)
    with pytest.raises(UserError, match="saved with LoRA"):
        other_rank.load(tmp_path / "ckpt", optimizer=True)


def test_lora_sampler_checkpoint_is_merged_full_model(tmp_path):
    steps = make_steps()
    session = make_session(seed=0, lora_rank=4, lora=True)
    train(session, steps)
    session.save(tmp_path / "sampler", optimizer=False)
    assert (tmp_path / "sampler" / "model.safetensors").exists()
    assert (tmp_path / "sampler" / "statistics.json").exists()

    # A plain (non-LoRA) model loads it with every key matched and computes the same loss.
    plain = make_session(seed=1)
    plain.load(tmp_path / "sampler", optimizer=False)
    assert eval_loss(plain, steps) == pytest.approx(eval_loss(session, steps), rel=1e-4)

    # Only the trained slot of the projectors changed; the other embodiments' slots did not.
    original = make_session(seed=0).model.action_head.action_decoder.layer2
    merged = plain.model.action_head.action_decoder.layer2
    others = [i for i in range(original.W.shape[0]) if i != SLOT]
    assert torch.equal(merged.W[others], original.W[others])
    assert torch.equal(merged.b[others], original.b[others])
    assert not torch.equal(merged.W[SLOT], original.W[SLOT])

    # A LoRA model resumes only from save_weights checkpoints.
    with pytest.raises(UserError, match="not a LoRA training checkpoint"):
        session.load(tmp_path / "sampler", optimizer=False)


def test_lora_projector_slot_starts_from_another_embodiment_remapped(tmp_path):
    # Our state/action layout is [arm(5), gripper(1)]; in the source embodiment's slot the arm
    # sits at dims 3-7 and the gripper at dim 0.
    source, index_map = 1, [3, 4, 5, 6, 7, 0]
    perm = [3, 4, 5, 6, 7, 0, 1, 2]
    original = make_session(seed=0).model.action_head
    session = make_session(
        seed=0,
        lora_rank=4,
        lora=True,
        init_slot=source,
        state_map=index_map,
        action_map=index_map,
    )
    head = session.model.action_head
    state_in = head.state_encoder.layer1.base_layer
    assert torch.equal(state_in.weight.T, original.state_encoder.layer1.W[source][perm])
    action_in = head.action_encoder.W1.base_layer
    assert torch.equal(action_in.weight.T, original.action_encoder.W1.W[source][perm])
    action_out = head.action_decoder.layer2.base_layer
    assert torch.equal(action_out.weight.T, original.action_decoder.layer2.W[source][:, perm])
    assert torch.equal(action_out.bias, original.action_decoder.layer2.b[source][perm])
    hidden = head.action_decoder.layer1.base_layer
    assert torch.equal(hidden.weight.T, original.action_decoder.layer1.W[source])

    steps = make_steps()
    train(session, steps)
    session.save(tmp_path / "sampler", optimizer=False)
    plain = make_session(seed=1)
    plain.load(tmp_path / "sampler", optimizer=False)
    assert eval_loss(plain, steps) == pytest.approx(eval_loss(session, steps), rel=1e-4)
    # The trained slot lands in our own slot; the source embodiment's slot is untouched.
    merged = plain.model.action_head.action_decoder.layer2
    assert torch.equal(merged.W[source], original.action_decoder.layer2.W[source])
    assert not torch.equal(merged.W[SLOT], original.action_decoder.layer2.W[SLOT])


def test_layout_maps_from_offsets():
    from gr00t.train_server.launch import layout_maps

    state_map, action_map = layout_maps(make_spec(), "arm=3, gripper=0")
    assert state_map == action_map == [3, 4, 5, 6, 7, 0]
    with pytest.raises(SystemExit, match="gripper"):
        layout_maps(make_spec(), "arm=3")


class _MaskedLossModel(torch.nn.Module):
    """Deterministic stand-in returning GR00T's loss outputs (masked mean over action elements)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 3))

    def forward(self, batch):
        mask = batch["action_mask"]
        action_loss = (batch["x"] * self.w) ** 2 * mask
        return {
            "loss": action_loss.sum() / (mask.sum() + 1e-6),
            "action_loss": action_loss,
            "action_mask": mask,
        }


def test_chunked_forward_backward_matches_one_pass():
    """Split batches accumulate the same gradient and report the same loss as one pass, also when
    chunks hold different numbers of valid action elements."""
    session = make_session()
    session.model = _MaskedLossModel()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(5, 4, 3, generator=gen)
    mask = torch.ones(5, 4, 3)
    mask[3:, 2:] = 0  # the last chunk has fewer valid elements

    def run(chunks):
        session.model.w.grad = None
        batches = [{"x": x[a:b], "action_mask": mask[a:b]} for a, b in chunks]
        out = session.forward_backward(None, batches)
        return out, session.model.w.grad.clone()

    whole, grad_whole = run([(0, 5)])
    split, grad_split = run([(0, 2), (2, 4), (4, 5)])
    torch.testing.assert_close(grad_split, grad_whole)
    assert split["metrics"]["loss:mean"] == pytest.approx(whole["metrics"]["loss:mean"], rel=1e-6)
    per_datum = lambda o: [d["loss"]["data"][0] for d in o["loss_fn_outputs"]]  # noqa: E731
    assert per_datum(split) == pytest.approx(per_datum(whole), rel=1e-6)


def test_prepare_collates_in_chunks():
    session = make_session()
    session.chunk_size = 3
    batches = session.prepare(make_steps(7))
    assert [len(b["action_mask"]) for b in batches] == [3, 3, 1]
