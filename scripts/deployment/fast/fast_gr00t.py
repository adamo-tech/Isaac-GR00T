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

"""Low-latency GR00T N1.7 inference for fixed-size cameras.

One CUDA graph per instruction covers the whole model:
  uint8 frames -> GPU preprocessing (gpu_preproc) -> ViT (TRT) -> image-token scatter -> LLM (TRT)
  -> VLLN -> VL self-attention (TRT or PyTorch) -> state encoder -> 4 flow-matching steps -> normalized actions.

Exact rewrites of the action head (same math as AlternateVLDiT):
  - cross-attention K/V of the VL tokens are computed once per call, not once per denoising step;
  - text-only / image-only cross-attention blocks gather their token subset instead of masking;
  - timestep embeddings, AdaLN scale/shift and output modulation are precomputed for the fixed schedule;
  - instruction-dependent inputs (token ids, text embeddings, M-RoPE position ids) are cached.

The ViT and LLM engines depend only on the (frozen) backbone, so every action-head fine-tune or
LoRA of the same base model shares them. The action head runs in PyTorch (torch.compile) inside the
graph.
"""

import numpy as np
from scripts.deployment.trt.trt_torch import torch_type
import tensorrt as trt
import torch
import torch.nn.functional as F

from .gpu_preproc import GpuImagePreprocessor


def _rot6d_to_matrix(r6):
    """(..., 6) first two rows of a rotation matrix -> (..., 3, 3), Gram-Schmidt as in pose.py."""
    r1, r2 = r6[..., :3], r6[..., 3:6]
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = r2 - np.sum(r1 * r2, -1, keepdims=True) * r1
    r2 = r2 / np.linalg.norm(r2, axis=-1, keepdims=True)
    return np.stack([r1, r2, np.cross(r1, r2)], -2)


def install_fast_absolute_conversion(state_action_processor):
    """Vectorized replacement for StateActionProcessor._convert_to_absolute_action.

    The stock version builds a Pose object (with scipy rotations) per timestep, ~20 ms per call for XDOF.
    EEF/XYZ_ROT6D: T_abs = T_ref @ T_rel; NON_EEF: abs = ref + rel. Other formats use the original."""
    from gr00t.data.types import ActionFormat, ActionType

    orig = state_action_processor._convert_to_absolute_action

    def convert(action, reference_state, action_type, action_format):
        if action_type == ActionType.NON_EEF:
            return (action + reference_state[None]).astype(np.float64)
        if action_type == ActionType.EEF and action_format == ActionFormat.XYZ_ROT6D:
            a = action.astype(np.float64)
            ref = reference_state.astype(np.float64)
            R_ref = _rot6d_to_matrix(ref[3:9])
            R_rel = _rot6d_to_matrix(a[:, 3:9])
            p = ref[:3][None] + a[:, :3] @ R_ref.T
            R = R_ref[None] @ R_rel
            return np.concatenate([p, R[:, :2, :].reshape(-1, 6)], -1)
        return orig(
            action=action,
            reference_state=reference_state,
            action_type=action_type,
            action_format=action_format,
        )

    state_action_processor._convert_to_absolute_action = convert
    return orig


class TrtRunner:
    """TRT engine with fixed shapes and persistent I/O buffers; enqueue only, no host sync."""

    _logger = trt.Logger(trt.Logger.ERROR)

    def __init__(self, path):
        trt.init_libnvinfer_plugins(self._logger, "")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self._logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        self.out = {}
        self._keep = {}

    def dtype(self, name):
        return torch_type(self.engine.get_tensor_dtype(name))

    def bind(self, **inputs):
        """Bind input tensors and allocate fresh outputs.

        Returns (outputs, keepalive). A CUDA graph captured after this call bakes in these addresses, so
        the caller must keep both alive for the graph's lifetime (rebinding for another graph must not
        free them)."""
        self.out, self._keep = {}, {}
        for n in self.inputs:
            x = inputs[n]
            assert x.is_contiguous() and x.dtype == self.dtype(n), (n, x.dtype, self.dtype(n))
            self.ctx.set_input_shape(n, tuple(x.shape))
            self.ctx.set_tensor_address(n, x.data_ptr())
            self._keep[n] = x
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            self.out[n] = torch.empty(shape, dtype=self.dtype(n), device="cuda")
            self.ctx.set_tensor_address(n, self.out[n].data_ptr())
        return self.out, self._keep

    def use(self, io):
        """Point the context back at buffers from an earlier bind() (for enqueues outside a graph)."""
        out, keep = io
        for n, x in keep.items():
            self.ctx.set_input_shape(n, tuple(x.shape))
            self.ctx.set_tensor_address(n, x.data_ptr())
        for n, x in out.items():
            self.ctx.set_tensor_address(n, x.data_ptr())
        self.out = out

    def __call__(self):
        assert self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


class TorchModuleRunner:
    """Runs a torch module with TrtRunner's interface (persistent input/output buffers), so it is
    captured in the CUDA graph like an engine enqueue and keeps reading the module's live weights."""

    def __init__(self, module, compile_module=True):
        self.module = (
            torch.compile(module, mode="max-autotune-no-cudagraphs", dynamic=False)
            if compile_module
            else module
        )
        self.inputs = ["hidden_states"]
        self.out = {}

    def bind(self, hidden_states):
        self.x = hidden_states
        self.out = {"output": torch.empty_like(hidden_states)}
        return self.out, {"hidden_states": hidden_states}

    def use(self, io):
        self.out = io[0]
        self.x = io[1]["hidden_states"]

    def __call__(self):
        self.out["output"].copy_(self.module(self.x))
        return self.out


class FastActionHead(torch.nn.Module):
    """AlternateVLDiT flow-matching sampler with per-call K/V and fixed-schedule precomputation."""

    def __init__(self, ah, text_idx, image_idx):
        super().__init__()
        self.ah = ah
        dit = ah.model
        self.blocks = dit.transformer_blocks
        self.heads = self.blocks[0].attn1.heads
        self.n_steps = ah.num_inference_timesteps
        self.horizon = ah.action_horizon
        self.dt = 1.0 / self.n_steps
        dev = ah.model.proj_out_2.weight.device
        self.register_buffer("text_idx", text_idx.to(dev))
        self.register_buffer("image_idx", image_idx.to(dev))
        every = 2 * dit.attend_text_every_n_blocks
        # which encoder subset each even (cross-attn) block attends to
        self.cross_src = {
            i: ("text" if i % every == 0 else "image") for i in range(0, len(self.blocks), 2)
        }
        self.precompute()

    @torch.no_grad()
    def precompute(self):
        """Compute the weight-dependent constants."""
        ah, dit = self.ah, self.ah.model
        dev, dt = dit.proj_out_2.weight.device, dit.proj_out_2.weight.dtype
        bufs = {}
        t_disc = [
            int(t / float(self.n_steps) * ah.num_timestep_buckets) for t in range(self.n_steps)
        ]
        ts = torch.tensor(t_disc, device=dev)
        temb = dit.timestep_encoder(ts)  # (S, D)
        mods = []
        for blk in self.blocks:
            scale, shift = blk.norm1.linear(blk.norm1.silu(temb)).chunk(2, dim=1)
            mods.append(torch.stack([scale, shift], 1))  # (S, 2, D)
        bufs["mods"] = torch.stack(mods, 1).contiguous()  # (S, L, 2, D)
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        bufs["out_mod"] = torch.stack([scale, shift], 1).contiguous()  # (S, 2, D)
        bufs["timesteps"] = ts.view(self.n_steps, 1)
        bufs["step_indices"] = torch.arange(self.n_steps, device=dev)
        bufs["pos_embs"] = (
            ah.position_embedding(torch.arange(self.horizon, device=dev)).unsqueeze(0).to(dt)
        )
        for name, value in bufs.items():
            self.register_buffer(name, value)

    def _heads(self, x):
        B, S, D = x.shape
        return x.view(B, S, self.heads, D // self.heads).transpose(1, 2)

    def cross_kv(self, vl):
        subsets = {
            "text": vl.index_select(1, self.text_idx),
            "image": vl.index_select(1, self.image_idx),
        }
        kv = []
        for i, src in self.cross_src.items():
            a = self.blocks[i].attn1
            enc = subsets[src]
            kv.append((self._heads(a.to_k(enc)), self._heads(a.to_v(enc))))
        return kv

    def step(self, actions, state_features, kv, t: torch.Tensor, emb_id):
        ah = self.ah
        timestep = self.timesteps.index_select(0, t)[0]
        mods = self.mods.index_select(0, t)[0]
        out_mod = self.out_mod.index_select(0, t)[0]
        a_feat = ah.action_encoder(actions, timestep, emb_id) + self.pos_embs
        x = torch.cat((state_features, a_feat), dim=1)
        D = x.shape[-1]
        for i, blk in enumerate(self.blocks):
            scale, shift = mods[i, 0], mods[i, 1]
            h = F.layer_norm(x, (D,), eps=blk.norm1.norm.eps) * (1 + scale) + shift
            a = blk.attn1
            q = self._heads(a.to_q(h))
            if i % 2 == 1:
                k, v = self._heads(a.to_k(h)), self._heads(a.to_v(h))
            else:
                k, v = kv[i // 2]
            o = F.scaled_dot_product_attention(q, k, v, scale=a.scale)
            o = o.transpose(1, 2).reshape(x.shape)
            x = x + a.to_out[0](o)
            h = F.layer_norm(x, (D,), eps=blk.norm3.eps)
            x = x + blk.ff.net[2](blk.ff.net[0](h))
        dit = ah.model
        scale, shift = out_mod[0], out_mod[1]
        x = F.layer_norm(x, (D,), eps=dit.norm_out.eps) * (1 + scale) + shift
        out = dit.proj_out_2(x)
        pred = ah.action_decoder(out, emb_id)
        return actions + self.dt * pred[:, -self.horizon :]


class FastGr00t:
    """Drop-in fast path for Gr00tPolicy.get_action with fixed camera size and batch size 1."""

    def __init__(
        self,
        policy,
        engine_dir,
        H=720,
        W=1280,
        compile_step=True,
        vit_engine=None,
        llm_engine=None,
        vlsa_engine=None,
    ):
        """Engines default to build_engines.py's names for this H x W. With ``vlsa_engine=None`` the
        VL self-attention runs in PyTorch inside the graph (same speed on an RTX 4090), so
        fine-tunes of the action head, LoRA included, need no engine of their own."""
        vit_engine = vit_engine or f"vit_bf16_{H}x{W}.engine"
        llm_engine = llm_engine or f"llm_static_bf16_{H}x{W}.engine"
        self.policy = policy
        self.tag = policy.embodiment_tag
        self.mc = policy.get_modality_config()
        proc = policy.processor
        self.proc = proc
        install_fast_absolute_conversion(proc.state_action_processor)
        model = policy.model
        self.model = model
        self.backbone = model.backbone
        self.ah = model.action_head
        self.dtype = torch.bfloat16
        self.cams = self.mc["video"].modality_keys
        self.T = len(self.mc["video"].delta_indices)
        self.n_img = self.T * len(self.cams)
        self.preproc = GpuImagePreprocessor(proc, H, W)
        self.vit = TrtRunner(f"{engine_dir}/{vit_engine}")
        self.llm = TrtRunner(f"{engine_dir}/{llm_engine}")
        # static LLM export: dense deepstack inputs, no data-dependent ops -> fully capturable
        self.llm_static = "visual_pos_masks" not in self.llm.inputs
        self.vlsa = (
            TrtRunner(f"{engine_dir}/{vlsa_engine}")
            if vlsa_engine
            else TorchModuleRunner(self.ah.vl_self_attention, compile_module=compile_step)
        )
        self.emb_id = torch.tensor(
            [proc.embodiment_id_mapping[self.tag.value]], device="cuda", dtype=torch.long
        )
        self.frames = torch.zeros((self.n_img, H, W, 3), dtype=torch.uint8, device="cuda")
        self.frames_host = torch.zeros_like(self.frames, device="cpu").pin_memory()
        self.state = torch.zeros(
            (1, 1, proc.max_state_dim * self.ah.config.state_history_length),
            dtype=self.dtype,
            device="cuda",
        )
        self.noise = torch.zeros(
            (1, self.ah.config.action_horizon, self.ah.action_dim), dtype=self.dtype, device="cuda"
        )
        self.gen = torch.Generator(device="cuda").manual_seed(0)
        self.compile_step = compile_step
        self.graphs = {}

    # ---------- instruction-dependent constants ----------
    def _prepare_instruction(self, instruction):
        proc = self.proc
        lang = instruction
        if proc.formalize_language:
            import re

            lang = re.sub(r"[^\w\s]", "", lang.lower())
        dummy = [torch.zeros(3, 32, 32, dtype=torch.uint8)] * self.n_img
        conv = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f} for f in dummy],
                    {"type": "text", "text": lang},
                ],
            }
        ]
        text = proc.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        pv, grid = self.preproc(self.frames)
        # expand image pads using the true grid (tokens per image = t*h*w / merge^2)
        merge = proc.processor.image_processor.merge_size
        n_tok = int(grid[0].prod()) // (merge * merge)
        text = text.replace("<|image_pad|>", "<|placeholder|>" * n_tok).replace(
            "<|placeholder|>", "<|image_pad|>"
        )
        ids = proc.processor.tokenizer([text], return_tensors="pt")["input_ids"].cuda()
        inner = self.backbone.model.model
        image_token_id = self.backbone.model.config.image_token_id
        attn = torch.ones_like(ids)
        pos_ids, _ = inner.get_rope_index(
            ids, grid.cuda(), video_grid_thw=None, attention_mask=attn
        )
        img_mask = ids[0] == image_token_id
        with torch.no_grad():
            text_embeds = inner.get_input_embeddings()(ids).to(self.dtype)
        c = {
            "ids": ids,
            "attn": attn.to(torch.int64).contiguous(),
            "pos_ids": pos_ids.to(torch.int64).contiguous(),
            "vis_mask": img_mask[None].contiguous(),
            "text_embeds": text_embeds.contiguous(),
            "image_pos": img_mask.nonzero().squeeze(1),
            "text_idx": (~img_mask).nonzero().squeeze(1),
            "grid": grid,
            "inputs_embeds": text_embeds.clone().contiguous(),
        }
        return c

    # ---------- the captured computation ----------
    def _forward(self, c, fah):
        self._part_a(c)
        self.llm()
        self._part_b(c, fah)

    def _part_a(self, c):
        pv, _ = self.preproc(self.frames)
        c["pv"].copy_(pv)
        vit = self.vit()
        img = vit["image_embeds"].to(self.dtype)
        ds = vit["deepstack_features"].to(self.dtype)
        if self.llm_static:
            for i in range(ds.shape[0]):
                c["ds_dense"][i, 0].index_copy_(0, c["image_pos"], ds[i])
        else:
            c["ds"].copy_(ds)
        emb = c["inputs_embeds"]
        emb.copy_(c["text_embeds"])
        emb[0].index_copy_(0, c["image_pos"], img)

    def _part_b(self, c, fah):
        feats = self.llm.out["embeddings"]
        c["vlsa_in"].copy_(self.ah.vlln(feats.to(self.dtype)))
        vl = self.vlsa()["output"]
        state_features = self.ah.state_encoder(self.state, self.emb_id)
        kv = fah.cross_kv(vl)
        actions = self.noise.clone()
        for t in range(fah.n_steps):
            actions = c["step"](
                actions, state_features, kv, fah.step_indices[t : t + 1], self.emb_id
            )
        c["out"].copy_(actions)

    def _build(self, instruction):
        c = self._prepare_instruction(instruction)
        fah = FastActionHead(self.ah, c["text_idx"], c["image_pos"])
        c["fah"] = fah
        c["step"] = (
            torch.compile(fah.step, mode="max-autotune-no-cudagraphs", dynamic=False)
            if self.compile_step
            else fah.step
        )
        pv_shape = (
            c["grid"].prod(1).sum().item(),
            self.vit.engine.get_tensor_shape("pixel_values")[-1],
        )
        c["pv"] = torch.zeros(pv_shape, dtype=self.vit.dtype("pixel_values"), device="cuda")
        c["vit_io"] = self.vit.bind(pixel_values=c["pv"])
        n_ds = self.vit.out["deepstack_features"].shape[0]
        c["ds"] = torch.zeros(
            self.vit.out["deepstack_features"].shape, dtype=self.dtype, device="cuda"
        )
        llm_in = dict(
            inputs_embeds=c["inputs_embeds"],
            attention_mask=c["attn"],
            position_ids=c["pos_ids"],
            visual_pos_masks=c["vis_mask"],
        )
        S = c["ids"].shape[1]
        if self.llm_static:
            # zeros at non-visual positions are never written, so the dense add is exact
            c["ds_dense"] = torch.zeros(
                (n_ds, 1, S, c["inputs_embeds"].shape[-1]), dtype=self.dtype, device="cuda"
            )
            for i in range(n_ds):
                llm_in[f"deepstack_{i}"] = c["ds_dense"][i]
        else:
            for i in range(n_ds):
                llm_in[f"deepstack_{i}"] = c["ds"][i]
        c["llm_io"] = self.llm.bind(**{k: v for k, v in llm_in.items() if k in self.llm.inputs})
        S = c["ids"].shape[1]
        c["vlsa_in"] = torch.zeros(
            (1, S, self.ah.vlln.normalized_shape[0]), dtype=self.dtype, device="cuda"
        )
        c["vlsa_io"] = self.vlsa.bind(hidden_states=c["vlsa_in"])
        c["out"] = torch.zeros_like(self.noise)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):
                self._forward(c, fah)
        torch.cuda.current_stream().wait_stream(side)
        if self.llm_static:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"), torch.inference_mode():
                self._forward(c, fah)
            c["graph"] = g
            return c
        # The stock LLM engine is not stream-capturable (data-dependent shapes in the deepstack
        # injection), so it is enqueued between two graphs; nothing waits on the host in between.
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga, capture_error_mode="thread_local"), torch.inference_mode():
            self._part_a(c)
        with torch.inference_mode():
            self.llm()
        with torch.cuda.graph(gb, capture_error_mode="thread_local"), torch.inference_mode():
            self._part_b(c, fah)
        c["graph_a"], c["graph_b"] = ga, gb
        return c

    def replay(self, c):
        if "graph" in c:
            c["graph"].replay()
            return
        c["graph_a"].replay()
        self.llm.use(c["llm_io"])
        self.llm()
        c["graph_b"].replay()

    # ---------- public API ----------
    def set_frames(self, frames_by_cam):
        """frames_by_cam: {cam_key: (T, H, W, 3) uint8}. Written into the pinned staging buffer."""
        i = 0
        for t in range(self.T):
            for k in self.cams:
                self.frames_host[i].numpy()[...] = frames_by_cam[k][t]
                i += 1

    def infer_normalized(self, state_normalized, instruction, noise=None):
        if instruction not in self.graphs:
            self.graphs[instruction] = self._build(instruction)
        c = self.graphs[instruction]
        self.frames.copy_(self.frames_host, non_blocking=True)
        self.state.copy_(state_normalized, non_blocking=True)
        if noise is None:
            self.noise.normal_(generator=self.gen)
        else:
            self.noise.copy_(noise)
        self.replay(c)
        return c["out"]

    def get_action(self, observation):
        """Same observation / return format as Gr00tPolicy.get_action (batch size 1)."""
        video = observation["video"]
        # frame order expected by the processor: time-major over (t, view)
        i = 0
        for t in range(self.T):
            for k in self.cams:
                self.frames_host[i].numpy()[...] = video[k][0, t]
                i += 1
        state_keys = self.mc["state"].modality_keys
        raw_state = {k: observation["state"][k][0] for k in state_keys}
        norm = self.proc.state_action_processor.apply_state(
            state=raw_state, embodiment_tag=self.tag.value
        )
        s = np.concatenate([norm[k] for k in state_keys], -1).reshape(-1)
        st = np.zeros(self.state.shape[-1], np.float32)
        st[: s.shape[0]] = s  # (T*max_state_dim) flattening matches view(B, 1, -1) only for T=1
        lang = observation["language"][self.mc["language"].modality_keys[0]][0][0]
        out = self.infer_normalized(torch.from_numpy(st).view(1, 1, -1).to(self.dtype), lang)
        norm_action = out.float().cpu().numpy()
        batched_states = {k: observation["state"][k] for k in state_keys}
        return self.proc.decode_action(norm_action, self.tag, batched_states)
