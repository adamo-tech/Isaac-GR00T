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

"""End-to-end test of the Tinker-protocol training server on a tiny CPU model."""

import time

from fastapi.testclient import TestClient
import numpy as np
import pytest
from test_support.train_server import API_KEY, HORIZON, make_datum, make_session, make_spec


HEADERS = {"X-API-Key": API_KEY}


def make_client(tmp_path, **kwargs):
    from gr00t.train_server.app import create_app

    app = create_app(
        api_key=API_KEY,
        spec=make_spec(),
        create_session=make_session,
        checkpoint_dir=tmp_path / "checkpoints",
        **kwargs,
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as client:
        yield client


def post(client, route, body, status=200):
    response = client.post(f"/api/v1/{route}", json=body, headers=HEADERS)
    assert response.status_code == status, response.text
    return response.json()


def wait(client, request_id, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = post(client, "retrieve_future", {"request_id": request_id})
        if result.get("type") != "try_again":
            return result
        time.sleep(0.05)
    raise TimeoutError(request_id)


def create_model(client):
    body = {"base_model": "nvidia/GR00T-N1.7", "lora_config": {"rank": 32, "seed": 0}}
    created = post(client, "create_model", body)
    assert wait(client, created["request_id"]) == {
        "type": "create_model",
        "model_id": created["model_id"],
    }
    return created["model_id"]


def test_healthz_and_auth(client):
    assert client.get("/api/v1/healthz").json() == {"status": "ok"}
    response = client.post("/api/v1/get_info", json={"model_id": "x"})
    assert response.status_code == 401
    assert response.json()["category"] == "user"


def test_rejects_other_base_models(client):
    error = post(client, "create_model", {"base_model": "Qwen/Qwen3-8B"}, status=400)
    assert error["category"] == "user" and "nvidia/GR00T-N1.7" in error["error"]


def test_train_queue_save_and_load(client, tmp_path):
    model_id = create_model(client)
    info = post(client, "get_info", {"model_id": model_id})
    assert info["model_data"]["action_horizon"] == HORIZON
    assert info["model_data"]["image_names"] == ["ego_view", "wrist"]
    assert info["model_data"]["action_names"] == ["arm", "gripper"]

    rng = np.random.default_rng(0)
    batch = [make_datum(rng) for _ in range(16)]
    fb_body = {"data": batch, "loss_fn": "flow_matching", "loss_fn_config": None}
    adam = {"learning_rate": 3e-3, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8}

    # Queue every call before waiting on any of them.
    fb_ids, step_ids = [], []
    seq_id = 1
    for _ in range(30):
        fb = {"model_id": model_id, "seq_id": seq_id, "forward_backward_input": fb_body}
        fb_ids.append(post(client, "forward_backward", fb)["request_id"])
        step = {"model_id": model_id, "seq_id": seq_id + 1, "adam_params": adam}
        step_ids.append(post(client, "optim_step", step)["request_id"])
        seq_id += 2

    # A retried seq_id returns the original request instead of running again.
    retry = {"model_id": model_id, "seq_id": 1, "forward_backward_input": fb_body}
    assert post(client, "forward_backward", retry)["request_id"] == fb_ids[0]

    results = [wait(client, rid) for rid in fb_ids]
    for result in results:
        assert result["type"] == "forward_backward"
        assert result["loss_fn_output_type"] == "ArrayRecord"
        assert len(result["loss_fn_outputs"]) == len(batch)
    losses = [r["metrics"]["loss:mean"] for r in results]
    assert np.mean(losses[-5:]) < 0.85 * np.mean(losses[:5]), losses
    step = wait(client, step_ids[-1])
    assert step["type"] == "optim_step" and step["metrics"]["grad_norm"] > 0

    # A bad camera name is refused synchronously and does not use up the seq_id.
    bad = {
        "model_id": model_id,
        "seq_id": seq_id,
        "forward_backward_input": {**fb_body, "data": [make_datum(rng, ["ego_view", "nope"])]},
    }
    error = post(client, "forward_backward", bad, status=400)
    assert error["category"] == "user"
    assert "'nope'" in error["error"] and "ego_view" in error["error"]

    saved = wait(
        client,
        post(client, "save_weights", {"model_id": model_id, "seq_id": seq_id, "path": "s1"})[
            "request_id"
        ],
    )
    sampler = wait(
        client,
        post(
            client,
            "save_weights_for_sampler",
            {"model_id": model_id, "seq_id": seq_id + 1, "path": "s1-sampler"},
        )["request_id"],
    )
    checkpoints = tmp_path / "checkpoints"
    assert saved == {"type": "save_weights", "path": str(checkpoints / "s1")}
    assert sampler == {"type": "save_weights_for_sampler", "path": str(checkpoints / "s1-sampler")}
    for name in ("model.safetensors", "config.json", "processor_config.json", "statistics.json"):
        assert (checkpoints / "s1" / name).exists(), name
        assert (checkpoints / "s1-sampler" / name).exists(), name
    assert (checkpoints / "s1" / "optimizer.pt").exists()
    assert not (checkpoints / "s1-sampler" / "optimizer.pt").exists()

    loaded = wait(
        client,
        post(
            client,
            "load_weights",
            {"model_id": model_id, "seq_id": seq_id + 2, "path": "s1", "optimizer": True},
        )["request_id"],
    )
    assert loaded == {"type": "load_weights", "path": str(checkpoints / "s1")}

    error = post(
        client,
        "load_weights",
        {"model_id": model_id, "seq_id": seq_id + 3, "path": "../elsewhere"},
        status=400,
    )
    assert error["category"] == "user"


def test_one_model_per_server(client):
    create_model(client)
    error = post(client, "create_model", {"base_model": "nvidia/GR00T-N1.7"}, status=400)
    assert error["category"] == "user"


def test_new_create_model_replaces_the_old_one(tmp_path):
    saved = []
    with make_client(tmp_path, replace_model=True, on_save=saved.append) as client:
        old = create_model(client)
        new = create_model(client)
        assert new != old

        datum = make_datum(np.random.default_rng(0))
        fb = {"data": [datum], "loss_fn": "flow_matching"}
        stale = post(
            client,
            "forward_backward",
            {"model_id": old, "seq_id": 1, "forward_backward_input": fb},
        )
        error = wait(client, stale["request_id"])
        assert error["category"] == "user" and "replaced" in error["error"]

        fresh = post(
            client,
            "forward_backward",
            {"model_id": new, "seq_id": 1, "forward_backward_input": fb},
        )
        assert wait(client, fresh["request_id"])["type"] == "forward_backward"
        save = post(client, "save_weights_for_sampler", {"model_id": new, "seq_id": 2, "path": "s"})
        assert wait(client, save["request_id"])["type"] == "save_weights_for_sampler"
        assert saved == [tmp_path / "checkpoints" / "s"]
