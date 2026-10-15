#!/usr/bin/env python
"""Run GR00T N1.7 (LIBERO) on a Modal GPU against a live adamo robot. Ctrl-C spins it down.

    export ADAMO_API_KEY=ak_...
    uv run --no-project --with modal python modal/run_adamo.py            # libero-sim/cam
    uv run --no-project --with modal python modal/run_adamo.py --dry-run  # infer, publish nothing

    modal run    modal/gr00t_infer.py::fill   # once: weights + gated backbone onto the Volume
    modal deploy modal/gr00t_infer.py         # after any change to that file

Ctrl-C cancels the call with terminate_containers=True, so the GPU is released
immediately rather than idling out its 20 min scaledown window.
"""

import argparse
import os
import signal
import sys
import time

import modal


APP, CLS = "gr00t-infer", "Gr00tLibero"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("robot", nargs="?", default="libero-sim")
    p.add_argument("track", nargs="?", default="cam")
    p.add_argument(
        "--prompt",
        default="pick up the black bowl between the plate and the ramekin and place it on the plate",
        help="language instruction; the default is libero-sim's libero_spatial task 0",
    )
    p.add_argument("--suite", default="libero_spatial", help="checkpoint subfolder to load")
    p.add_argument("--control-channel", default="gr00t")
    p.add_argument(
        "--mode",
        default="async",
        choices=["async", "lockstep"],
        help="async: infer back to back, trickle actions from the newest chunk (smooth). "
        "lockstep: execute a chunk, then observe and infer (how the benchmark runs)",
    )
    p.add_argument(
        "--action-steps", type=int, default=16, help="max actions used from one chunk of 16"
    )
    p.add_argument("--hz", type=float, default=30.0, help="async: action rate (libero-sim: 30)")
    p.add_argument(
        "--queue-depth", type=int, default=2, help="async: max commands queued on the robot"
    )
    p.add_argument("--seconds", type=float, default=3600.0, help="hard cap in the container")
    p.add_argument("--dry-run", action="store_true", help="infer but publish no actions")
    p.add_argument("--poll", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("ADAMO_API_KEY")
    if not api_key:
        print("ADAMO_API_KEY is not set: export the key for the robot's org", file=sys.stderr)
        return 2

    print(f"starting {APP}/{CLS}[{args.suite}] (a cold start loads ~9 GiB)...")
    gr00t = modal.Cls.from_name(APP, CLS)(suite=args.suite)
    t0 = time.perf_counter()
    try:
        st = gr00t.stats.remote()
    except Exception as exc:
        print(
            f"cannot reach {APP}/{CLS}: {exc}\ndeploy it first:  modal deploy modal/gr00t_infer.py",
            file=sys.stderr,
        )
        return 2
    print(
        f"ready on {st['device']} in {time.perf_counter() - t0:.1f}s "
        f"(load {st['load_s']:.1f}s, warm-up {st['warmup_ms']:.0f} ms)"
    )

    call = gr00t.serve_stream.spawn(
        robot=args.robot,
        track=args.track,
        prompt=args.prompt,
        control_channel=args.control_channel,
        mode=args.mode,
        action_steps=args.action_steps,
        hz=args.hz,
        queue_depth=args.queue_depth,
        seconds=args.seconds,
        dry_run=args.dry_run,
        adamo_api_key=api_key,
    )
    print(
        f"serving {args.robot}/{args.track} -> {args.robot}/control/{args.control_channel}"
        + f"  [{args.mode}]"
        + ("  [DRY RUN]" if args.dry_run else "")
        + f"\nprompt: {args.prompt!r}\nCtrl-C to stop and release the GPU.\n"
    )

    # Ctrl-C sets a flag rather than raising inside a progress.remote() round
    # trip: a poll still queued when the cancel lands gets rescheduled onto a
    # fresh container (see openpi modal/run_adamo.py for the full story).
    stopping = []

    def _on_sigint(_sig, _frm):
        stopping.append(True)
        print("\nstopping...", flush=True)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _on_sigint)

    last = {}
    try:
        while not stopping:
            try:
                print(f"\nserve loop finished on its own: {call.get(timeout=0)}")
                return 0
            except TimeoutError:
                pass
            except Exception as exc:
                print(f"\nserve loop failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                break
            try:
                p = gr00t.progress.remote()
            except Exception as exc:
                print(f"progress unavailable: {exc}")
                p = None
            if p and not stopping:
                last = p
                line = (
                    f"[{p['elapsed_s']:6.0f}s] {p['state']:<18} "
                    f"infer {p['inferences']:4d} (p50 {p['infer_p50']:4.0f} ms, "
                    f"cycle {p['cycle_p50']:4.0f} ms)  sent {p['actions_sent']:5d}  "
                    + (
                        f"lag {p['lag_p50']:2.0f} steps  starved {p['starved']}  "
                        f"resync {p['resyncs']}  frame-to {p['frame_timeouts']}  "
                        if p.get("mode") == "async"
                        else f"timeouts s{p['state_timeouts']}/f{p['frame_timeouts']}  "
                    )
                    + f"act {p['last_action'] or '-'}"
                )
                print(f"\r{line:<170}"[:171], end="", flush=True)
            for _ in range(max(1, int(args.poll * 10))):
                if stopping:
                    break
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            call.cancel(terminate_containers=True)
            print("\ncontainer terminated.")
        except Exception as exc:
            print(f"cancel failed: {exc}\nrun:  modal app stop {APP} -y", file=sys.stderr)

    if last:
        print("last snapshot:")
        for k in ("inferences", "actions_sent", "infer_p50", "infer_p95", "cycle_p50",
                  "lag_p50", "starved", "resyncs", "state_timeouts", "frame_timeouts",
                  "last_state", "elapsed_s"):  # fmt: skip
            print(f"  {k:15s}: {last.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
