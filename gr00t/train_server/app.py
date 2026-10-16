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

"""FastAPI app speaking Tinker's training HTTP protocol for GR00T N1.7.

Submissions are validated synchronously: a 4xx means nothing was queued and the seq_id is
still free. Accepted calls return ``{"request_id", "model_id"}`` at once and run on the
scheduler's worker; ``/retrieve_future`` returns ``try_again`` until the result (or
``{"error", "category"}``) is ready.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import threading
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .datum import EmbodimentSpec, UserError, decode_datum
from .scheduler import Scheduler
from .session import TrainingSession


BASE_MODEL = "nvidia/GR00T-N1.7"


class ApiError(Exception):
    def __init__(self, status: int, error: str, category: str = "user"):
        super().__init__(error)
        self.status, self.error, self.category = status, error, category


class LoraConfig(BaseModel):
    rank: int = 0
    seed: int | None = None


class CreateModelRequest(BaseModel):
    session_id: str | None = None
    model_seq_id: int | None = None
    base_model: str
    lora_config: LoraConfig = Field(default_factory=LoraConfig)
    user_metadata: dict[str, Any] | None = None


class ForwardBackwardInput(BaseModel):
    data: list[Any]
    loss_fn: str
    loss_fn_config: dict[str, Any] | None = None


class ForwardBackwardRequest(BaseModel):
    model_id: str
    seq_id: int
    forward_backward_input: ForwardBackwardInput


class AdamParams(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepRequest(BaseModel):
    model_id: str
    seq_id: int
    adam_params: AdamParams = Field(default_factory=AdamParams)


class SaveWeightsRequest(BaseModel):
    model_id: str
    seq_id: int
    path: str
    ttl_seconds: int | None = None


class LoadWeightsRequest(BaseModel):
    model_id: str
    seq_id: int
    path: str
    optimizer: bool = False


class ModelIdRequest(BaseModel):
    model_id: str


class RetrieveFutureRequest(BaseModel):
    request_id: str


def create_app(
    *,
    api_key: str,
    spec: EmbodimentSpec,
    create_session: Callable[[int | None, int], TrainingSession],
    checkpoint_dir: str | Path,
    first_seq_id: int = 1,
    replace_model: bool = False,
    on_save: Callable[[Path], None] | None = None,
) -> FastAPI:
    """Build the app. ``create_session(seed, lora_rank)`` loads the model on the GPU worker.

    The server holds one model. With ``replace_model``, a new ``create_model`` replaces it (the
    old model's calls then fail); otherwise a second ``create_model`` is refused. ``on_save``
    runs on the worker after every successful save, e.g. to commit a network volume.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scheduler = Scheduler(first_seq_id=first_seq_id)
    created: dict[tuple[str | None, int | None], tuple[str, str]] = {}
    current: list[str] = []  # model_id of the model this server holds
    create_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        scheduler.stop()

    app = FastAPI(title="GR00T N1.7 training server", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.exception_handler(ApiError)
    async def api_error(_: Request, e: ApiError):
        return JSONResponse({"error": e.error, "category": e.category}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, e: RequestValidationError):
        return JSONResponse(
            {"error": f"invalid request: {e.errors()}", "category": "user"}, status_code=422
        )

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key is None or not secrets.compare_digest(x_api_key, api_key):
            raise ApiError(401, "missing or invalid X-API-Key")

    def model_or_404(model_id: str) -> None:
        if not scheduler.has_model(model_id):
            raise ApiError(404, f"unknown model_id {model_id!r}")

    def submit(model_id, seq_id, description, validate, run, poisons=True) -> dict[str, str]:
        """Return the original request for a retried seq_id; else validate, then queue."""
        model_or_404(model_id)
        request_id = scheduler.existing_request(model_id, seq_id)
        if request_id is None:
            try:
                scheduler.check_seq_id(model_id, seq_id)
                payload = validate()
            except UserError as e:
                raise ApiError(400, str(e)) from None
            request_id = scheduler.submit(
                model_id,
                seq_id,
                description,
                lambda session: run(session, payload),
                poisons=poisons,
            )
        return {"request_id": request_id, "model_id": model_id}

    def checkpoint_path(name: str) -> Path:
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts or name.startswith("."):
            raise UserError(f"path must be a plain relative name, got {name!r}")
        return checkpoint_dir / path

    def load_path(name: str) -> Path:
        path = Path(name) if Path(name).is_absolute() else checkpoint_path(name)
        if not path.is_dir():
            raise UserError(f"no checkpoint at {path}")
        return path

    @app.api_route("/api/v1/healthz", methods=["GET", "POST"])
    def healthz():
        return {"status": "ok"}

    v1 = "/api/v1"
    auth = [Depends(check_key)]

    @app.post(f"{v1}/create_model", dependencies=auth)
    def create_model(body: CreateModelRequest):
        if body.base_model != BASE_MODEL:
            raise ApiError(400, f"base_model must be {BASE_MODEL!r}, got {body.base_model!r}")
        # A retried create (same session_id + model_seq_id) gets the original request back.
        key = (body.session_id, body.model_seq_id)
        with create_lock:
            if None not in key and key in created and created[key][1] in current:
                request_id, model_id = created[key]
            elif current and not replace_model:
                raise ApiError(400, "this server holds one model and it already exists")
            else:
                if current:
                    old = current.pop()
                    scheduler.retire(old, f"model {old} was replaced by a newer create_model")
                seed, rank = body.lora_config.seed, body.lora_config.rank
                future = scheduler.create_model(lambda: create_session(seed, rank))
                request_id, model_id = created[key] = (future.request_id, future.model_id)
                current.append(model_id)
        return {"request_id": request_id, "model_id": model_id}

    @app.post(f"{v1}/forward_backward", dependencies=auth)
    def forward_backward(body: ForwardBackwardRequest):
        fb = body.forward_backward_input

        def validate():
            if fb.loss_fn != "flow_matching":
                raise UserError(f"loss_fn {fb.loss_fn!r} not supported; use 'flow_matching'")
            if not fb.data:
                raise UserError("forward_backward_input.data is empty")
            return [decode_datum(d, spec, i) for i, d in enumerate(fb.data)]

        return submit(
            body.model_id,
            body.seq_id,
            "forward_backward",
            validate,
            lambda session, steps: session.forward_backward(steps),
        )

    @app.post(f"{v1}/optim_step", dependencies=auth)
    def optim_step(body: OptimStepRequest):
        params = body.adam_params.model_dump()
        return submit(
            body.model_id,
            body.seq_id,
            "optim_step",
            lambda: params,
            lambda session, p: session.optim_step(p),
        )

    def save_route(kind: str, optimizer: bool):
        def save(body: SaveWeightsRequest):
            def run(session: TrainingSession, path: Path):
                session.save(path, optimizer=optimizer)
                if on_save is not None:
                    on_save(path)
                return {"type": kind, "path": str(path)}

            return submit(
                body.model_id,
                body.seq_id,
                kind,
                lambda: checkpoint_path(body.path),
                run,
                poisons=False,
            )

        app.post(f"{v1}/{kind}", dependencies=auth, name=kind)(save)

    save_route("save_weights", optimizer=True)
    save_route("save_weights_for_sampler", optimizer=False)

    @app.post(f"{v1}/load_weights", dependencies=auth)
    def load_weights(body: LoadWeightsRequest):
        def run(session: TrainingSession, path: Path):
            session.load(path, optimizer=body.optimizer)
            return {"type": "load_weights", "path": str(path)}

        return submit(body.model_id, body.seq_id, "load_weights", lambda: load_path(body.path), run)

    @app.post(f"{v1}/get_info", dependencies=auth)
    def get_info(body: ModelIdRequest):
        model_or_404(body.model_id)
        return {
            "model_id": body.model_id,
            "model_data": {"base_model": BASE_MODEL, **spec.info()},
        }

    @app.post(f"{v1}/retrieve_future", dependencies=auth)
    def retrieve_future(body: RetrieveFutureRequest):
        future = scheduler.get(body.request_id)
        if future is None:
            raise ApiError(404, f"unknown request_id {body.request_id!r}")
        if not future.done:
            return {"type": "try_again", "request_id": future.request_id, "queue_state": "active"}
        if future.error is not None:
            return {"error": future.error, "category": future.category}
        return future.result

    return app
