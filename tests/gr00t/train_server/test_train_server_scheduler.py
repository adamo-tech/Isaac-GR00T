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

"""Ordering, retry and failure semantics of the training server's scheduler."""

import time

from gr00t.train_server.datum import UserError
from gr00t.train_server.scheduler import Scheduler
import pytest


@pytest.fixture
def scheduler():
    scheduler = Scheduler(first_seq_id=1)
    yield scheduler
    scheduler.stop()


def wait_done(scheduler, request_ids, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        futures = [scheduler.get(r) for r in request_ids]
        if all(f.done for f in futures):
            return futures
        time.sleep(0.01)
    raise TimeoutError(request_ids)


def make_model(scheduler):
    created = scheduler.create_model(lambda: "session")
    wait_done(scheduler, [created.request_id])
    return created.model_id


def recorder(ran, seq_id):
    def run(session):
        assert session == "session"
        ran.append(seq_id)
        return {"seq_id": seq_id}

    return run


def test_out_of_order_arrival_runs_in_seq_order(scheduler):
    model_id = make_model(scheduler)
    ran = []
    late = [scheduler.submit(model_id, s, "call", recorder(ran, s)) for s in (4, 2, 3)]
    time.sleep(0.2)
    assert ran == [], "calls after a gap must wait for seq_id 1"

    first = scheduler.submit(model_id, 1, "call", recorder(ran, 1))
    futures = wait_done(scheduler, [first, *late])
    assert ran == [1, 2, 3, 4]
    assert [f.result for f in futures] == [{"seq_id": s} for s in (1, 4, 2, 3)]


def test_repeated_seq_id_returns_original_request(scheduler):
    model_id = make_model(scheduler)
    ran = []
    first = scheduler.submit(model_id, 1, "call", recorder(ran, 1))
    again = scheduler.submit(model_id, 1, "call", recorder(ran, 1))
    assert again == first
    wait_done(scheduler, [first])
    assert scheduler.submit(model_id, 1, "call", recorder(ran, 1)) == first
    assert scheduler.existing_request(model_id, 1) == first
    assert ran == [1]


def test_seq_id_before_start_is_rejected(scheduler):
    model_id = make_model(scheduler)
    with pytest.raises(UserError, match="already past"):
        scheduler.check_seq_id(model_id, 0)


def test_failure_poisons_later_calls(scheduler):
    model_id = make_model(scheduler)
    ran = []

    def bad(session):
        raise UserError("unknown camera 'nope'")

    ok = scheduler.submit(model_id, 1, "forward_backward", recorder(ran, 1))
    failed = scheduler.submit(model_id, 2, "forward_backward", bad)
    later = scheduler.submit(model_id, 3, "optim_step", recorder(ran, 3))
    ok_f, failed_f, later_f = wait_done(scheduler, [ok, failed, later])

    assert ok_f.error is None
    assert (failed_f.error, failed_f.category) == ("unknown camera 'nope'", "user")
    assert later_f.category == "user"
    assert "forward_backward (seq_id 2)" in later_f.error and failed in later_f.error
    assert ran == [1]


def test_failed_save_does_not_poison(scheduler):
    model_id = make_model(scheduler)
    ran = []

    def disk_full(session):
        raise OSError("No space left on device")

    save = scheduler.submit(model_id, 1, "save_weights", disk_full, poisons=False)
    after = scheduler.submit(model_id, 2, "forward_backward", recorder(ran, 2))
    save_f, after_f = wait_done(scheduler, [save, after])
    assert save_f.category == "server" and "No space left" in save_f.error
    assert after_f.error is None and ran == [2]


def test_calls_wait_for_create_model(scheduler):
    ran = []

    def slow_create():
        time.sleep(0.2)
        return "session"

    created = scheduler.create_model(slow_create)
    call = scheduler.submit(created.model_id, 1, "call", recorder(ran, 1))
    create_f, call_f = wait_done(scheduler, [created.request_id, call])
    assert create_f.result == {"type": "create_model", "model_id": created.model_id}
    assert call_f.result == {"seq_id": 1}


def test_failed_create_fails_later_calls(scheduler):
    def broken():
        raise RuntimeError("CUDA out of memory")

    created = scheduler.create_model(broken)
    call = scheduler.submit(created.model_id, 1, "call", lambda session: {})
    create_f, call_f = wait_done(scheduler, [created.request_id, call])
    assert create_f.category == "server" and "CUDA out of memory" in create_f.error
    assert call_f.error is not None and "create_model" in call_f.error


def test_retired_model_fails_queued_and_later_calls(scheduler):
    model_id = make_model(scheduler)
    ran = []
    waiting = scheduler.submit(model_id, 2, "call", recorder(ran, 2))  # waits for seq_id 1
    scheduler.retire(model_id, "model replaced")
    later = scheduler.submit(model_id, 1, "call", recorder(ran, 1))
    waiting_f, later_f = wait_done(scheduler, [waiting, later])
    assert (waiting_f.error, waiting_f.category) == ("model replaced", "user")
    assert later_f.error == "model replaced"
    assert ran == []


def test_model_retired_while_creating_never_gets_a_session(scheduler):
    def slow_create():
        time.sleep(0.2)
        return "session"

    created = scheduler.create_model(slow_create)
    scheduler.retire(created.model_id, "model replaced")
    (create_f,) = wait_done(scheduler, [created.request_id])
    assert create_f.done
    assert scheduler._models[created.model_id].session is None
