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

"""Futures plus a single worker thread that runs each model's calls strictly in seq_id order.

Clients queue many calls without waiting; every call gets a request_id at once and its result
is read later through ``get``. Per model, a call runs only after every lower seq_id has run, so
calls arriving out of order wait for the gap. A repeated seq_id returns the original
request_id. Once a call fails, later calls for that model fail too (unless the failed call was
marked ``poisons=False``, as saves are). A retired (replaced) model fails everything queued.
"""

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
from typing import Any, Callable
import uuid

from .datum import UserError


logger = logging.getLogger(__name__)


@dataclass
class Future:
    request_id: str
    model_id: str
    description: str
    done: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    category: str | None = None


@dataclass
class _CreateRun:
    create: Callable[[], Any]
    model_id: str


@dataclass
class _Job:
    future: Future
    run: Callable[[Any], dict[str, Any]] | _CreateRun
    poisons: bool


@dataclass
class _Model:
    next_seq_id: int
    session: Any = None
    pending: dict[int, _Job] = field(default_factory=dict)
    request_ids: dict[int, str] = field(default_factory=dict)
    failure: Future | None = None
    retired: str | None = None


class Scheduler:
    def __init__(self, first_seq_id: int = 1):
        self.first_seq_id = first_seq_id
        self._cond = threading.Condition()
        self._futures: dict[str, Future] = {}
        self._models: dict[str, _Model] = {}
        self._creates: deque[_Job] = deque()
        self._stopped = False
        self._thread = threading.Thread(target=self._work, name="gr00t-train-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join()

    def has_model(self, model_id: str) -> bool:
        with self._cond:
            return model_id in self._models

    def get(self, request_id: str) -> Future | None:
        with self._cond:
            return self._futures.get(request_id)

    def existing_request(self, model_id: str, seq_id: int) -> str | None:
        """request_id already registered for this seq_id (a client retry), if any."""
        with self._cond:
            return self._models[model_id].request_ids.get(seq_id)

    def check_seq_id(self, model_id: str, seq_id: int) -> None:
        with self._cond:
            model = self._models[model_id]
            if seq_id < self.first_seq_id or (
                seq_id < model.next_seq_id and seq_id not in model.request_ids
            ):
                raise UserError(
                    f"seq_id {seq_id} is already past for model {model_id} "
                    f"(next expected {model.next_seq_id}; sequence starts at {self.first_seq_id})"
                )

    def create_model(self, create: Callable[[], Any]) -> Future:
        """Queue ``create()`` (returns the session object) under a fresh model_id."""
        model_id = uuid.uuid4().hex
        with self._cond:
            future = self._new_future(model_id, "create_model")
            self._models[model_id] = _Model(next_seq_id=self.first_seq_id)
            self._creates.append(_Job(future, _CreateRun(create, model_id), poisons=True))
            self._cond.notify_all()
        return future

    def retire(self, model_id: str, reason: str) -> None:
        """Drop a model: its session is released and its queued and later calls fail."""
        with self._cond:
            model = self._models[model_id]
            model.retired = reason
            model.session = None
            for job in model.pending.values():
                self._set(job.future, error=reason, category="user")
            model.pending.clear()

    def submit(
        self,
        model_id: str,
        seq_id: int,
        description: str,
        run: Callable[[Any], dict[str, Any]],
        *,
        poisons: bool = True,
    ) -> str:
        """Queue ``run(session)`` as seq_id of model_id; returns its request_id."""
        with self._cond:
            model = self._models[model_id]
            if seq_id in model.request_ids:
                return model.request_ids[seq_id]
            future = self._new_future(model_id, f"{description} (seq_id {seq_id})")
            model.request_ids[seq_id] = future.request_id
            if model.retired is not None:
                self._set(future, error=model.retired, category="user")
                return future.request_id
            model.pending[seq_id] = _Job(future, run, poisons)
            self._cond.notify_all()
            return future.request_id

    def _new_future(self, model_id: str, description: str) -> Future:
        future = Future(request_id=uuid.uuid4().hex, model_id=model_id, description=description)
        self._futures[future.request_id] = future
        return future

    def _next_job(self) -> tuple[_Model, _Job] | None:
        if self._creates:
            job = self._creates.popleft()
            return self._models[job.future.model_id], job
        for model in self._models.values():
            if model.retired is not None or (model.session is None and model.failure is None):
                continue  # replaced, or still being created
            job = model.pending.pop(model.next_seq_id, None)
            if job is not None:
                model.next_seq_id += 1
                return model, job
        return None

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._stopped and (picked := self._next_job()) is None:
                    self._cond.wait()
                if self._stopped:
                    return
            model, job = picked
            self._run(model, job)

    def _run(self, model: _Model, job: _Job) -> None:
        future = job.future
        if model.retired is not None:
            self._finish(future, error=model.retired, category="user")
            return
        if model.failure is not None:
            self._finish(
                future,
                error=f"{future.description} not run: earlier call {model.failure.description} "
                f"(request_id {model.failure.request_id}) failed: {model.failure.error}",
                category=model.failure.category,
            )
            return
        try:
            if isinstance(job.run, _CreateRun):
                session = job.run.create()
                with self._cond:
                    if model.retired is None:
                        model.session = session
                del session
                result = {"type": "create_model", "model_id": job.run.model_id}
            else:
                result = job.run(model.session)
        except UserError as e:
            self._fail(model, job, str(e), "user")
        except Exception as e:
            logger.exception("%s failed", future.description)
            self._fail(model, job, f"{type(e).__name__}: {e}", "server")
        else:
            self._finish(future, result=result)

    def _fail(self, model: _Model, job: _Job, error: str, category: str) -> None:
        self._finish(job.future, error=error, category=category)
        if job.poisons and model.failure is None:
            model.failure = job.future

    def _finish(self, future: Future, **values: Any) -> None:
        with self._cond:
            self._set(future, **values)

    @staticmethod
    def _set(future: Future, **values: Any) -> None:
        for key, value in values.items():
            setattr(future, key, value)
        future.done = True
