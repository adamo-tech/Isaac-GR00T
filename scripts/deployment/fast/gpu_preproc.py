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

"""GPU image preprocessing for GR00T N1.7 with fixed-size camera frames.

The reference host pipeline per frame is
  albumentations: SmallestMaxSize(256, INTER_AREA) -> FractionalCenterCrop(0.95) -> SmallestMaxSize(256, INTER_AREA)
  -> uint8 -> Qwen fast image processor: bicubic antialiased resize to a multiple of 32 -> uint8
  -> rescale/normalize -> temporal duplication -> 16x16 patchify.
Every resize/crop is separable and linear, so for a fixed input size each stage is `Ay @ img @ Ax.T`.
The matrices are extracted once by pushing float basis images through the reference ops, then applied
on the GPU with the same uint8 rounding points as the reference.
"""

import hashlib
import os

import numpy as np
import torch


def _probe_axis_matrices(fn, H, W):
    """fn: float32 (H, W) image -> float32 (h, w). Returns Ay (h, H), Ax (w, W) for a separable linear fn."""
    cols = []
    for k in range(H):
        img = np.zeros((H, W), np.float32)
        img[k, :] = 1.0
        cols.append(fn(img)[:, 0])
    Ay = np.stack(cols, 1)
    cols = []
    for k in range(W):
        img = np.zeros((H, W), np.float32)
        img[:, k] = 1.0
        cols.append(fn(img)[0, :])
    Ax = np.stack(cols, 1)
    return Ay, Ax


DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/gr00t/fast_preproc")


class GpuImagePreprocessor:
    def __init__(self, processor, H=720, W=1280, device="cuda", cache_dir=DEFAULT_CACHE_DIR):
        self.device = device
        self.H, self.W = H, W
        ip = processor.processor.image_processor
        self.patch, self.tps, self.merge = ip.patch_size, ip.temporal_patch_size, ip.merge_size
        self.mean = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
        # the matrices depend on the processor's transform settings, not just the input size
        key = hashlib.sha1(
            repr((processor.eval_image_transform, ip.size, ip.resample)).encode()
        ).hexdigest()[:12]
        path = os.path.join(cache_dir, f"preproc_{H}x{W}_{key}.npz")
        if os.path.exists(path):
            z = np.load(path)
            mats = [z[k] for k in ("a1y", "a1x", "a2y", "a2x")]
        else:
            mats = self._build(processor, ip)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(path, a1y=mats[0], a1x=mats[1], a2y=mats[2], a2x=mats[3])
        self.a1y, self.a1x, self.a2y, self.a2x = (
            torch.from_numpy(m).float().to(device) for m in mats
        )
        self.out_h, self.out_w = self.a2y.shape[0], self.a2x.shape[0]

    def _build(self, processor, ip):
        from transformers.image_utils import SizeDict
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        tf = processor.eval_image_transform

        def stage1(img2d):
            x = np.repeat(img2d[:, :, None], 3, 2)
            return tf(image=x)["image"][:, :, 0].astype(np.float32)

        a1y, a1x = _probe_axis_matrices(stage1, self.H, self.W)
        h1, w1 = a1y.shape[0], a1x.shape[0]
        h2, w2 = smart_resize(
            h1,
            w1,
            factor=self.patch * self.merge,
            min_pixels=ip.size["shortest_edge"],
            max_pixels=ip.size["longest_edge"],
        )

        from transformers.image_processing_utils_fast import pil_torch_interpolation_mapping
        from transformers.image_utils import PILImageResampling

        interp = pil_torch_interpolation_mapping[PILImageResampling(ip.resample)]

        def stage2(img2d):
            x = torch.from_numpy(img2d)[None, None].repeat(1, 3, 1, 1)
            y = ip.resize(image=x, size=SizeDict(height=h2, width=w2), interpolation=interp)
            return y[0, 0].numpy().astype(np.float32)

        a2y, a2x = _probe_axis_matrices(stage2, h1, w1)
        return a1y, a1x, a2y, a2x

    @torch.no_grad()
    def __call__(self, frames_u8):
        """frames_u8: (N, H, W, 3) uint8 tensor (any device). Returns (pixel_values fp32 (N*tokens, C*t*p*p), grid_thw)."""
        x = frames_u8.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()  # N,3,H,W
        x = (self.a1y @ x @ self.a1x.T).round_().clamp_(0, 255)
        x = (self.a2y @ x @ self.a2x.T).round_().clamp_(0, 255)
        x = (x / 255.0 - self.mean) / self.std
        N, C, h, w = x.shape
        p, m, t = self.patch, self.merge, self.tps
        x = x.unsqueeze(1).expand(N, t, C, h, w)
        gh, gw = h // p, w // p
        x = x.reshape(N, 1, t, C, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(N * gh * gw, C * t * p * p)
        grid = torch.tensor([[1, gh, gw]] * N, dtype=torch.int64)
        return x, grid
