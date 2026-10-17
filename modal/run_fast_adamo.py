#!/usr/bin/env python
"""Drive an Adamo robot with the gr00t-fast-adamo Modal app (FastGr00t + LoRA). Ctrl-C releases it.

    export ADAMO_API_KEY=ak_...
    uv run --no-project --with modal python modal/run_fast_adamo.py abc-sim
    uv run --no-project --with modal python modal/run_fast_adamo.py abc-sim --lora abc-sim-r64-b256-300
    uv run --no-project --with modal python modal/run_fast_adamo.py --list-loras

    modal deploy modal/gr00t_fast_adamo.py   # after any change to the app or the server

The first request starts the container; this waits until /status reports the policy ready (a
first start on a new GPU type builds engines and compiles, ~10 min; later ~2 min), then POSTs
/start and prints the run's counters. Ctrl-C POSTs /stop, which releases the robot; the container
scales down 3 minutes later.
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

import modal


APP = "gr00t-fast-adamo"


def call(url: str, path: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}", data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')}") from None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("robot", nargs="?", default="abc-sim")
    p.add_argument("--prompt", default=None, help="default: the server's (abc-sim bottles task)")
    p.add_argument(
        "--lora",
        default=None,
        help="LoRA on gr00t-train-lora-checkpoints; default: the base weights, no LoRA",
    )
    p.add_argument("--execute-steps", type=int, default=20)
    p.add_argument("--inference-lead-steps", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=900.0)
    p.add_argument("--url", default=None, help="server URL; default: the deployed app's")
    p.add_argument("--list-loras", action="store_true", help="list the LoRAs and exit")
    p.add_argument("--poll", type=float, default=2.0)
    args = p.parse_args(argv)

    url = args.url or modal.Function.from_name(APP, "serve").get_web_url()
    print(f"{APP}: {url}")
    if args.list_loras:
        print("\n".join(call(url, "/loras")["loras"]))
        return 0
    api_key = os.environ.get("ADAMO_API_KEY")
    if not api_key:
        print("ADAMO_API_KEY is not set: export the key for the robot's org", file=sys.stderr)
        return 2

    t0, phase = time.monotonic(), None
    while True:
        st = call(url, "/status")
        if st["policy"] != phase:
            phase = st["policy"]
            print(f"[{time.monotonic() - t0:5.0f}s] policy {phase}  lora {st['lora']}")
        if phase == "ready":
            break
        if phase == "failed":
            print(f"policy failed: {st['policy_error']}", file=sys.stderr)
            return 1
        time.sleep(5)

    body = {
        "robot_name": args.robot,
        "adamo_api_key": api_key,
        "prompt": args.prompt,
        "lora": args.lora,
        "execute_steps": args.execute_steps,
        "inference_lead_steps": args.inference_lead_steps,
        "max_seconds": args.max_seconds,
    }
    print(f"starting on {args.robot} (a new prompt compiles first, ~1 min)...")
    started = call(url, "/start", body)
    print(f"driving {started['robot']}: prompt {started['prompt']!r}, lora {started['lora']}")
    print("Ctrl-C to stop and release the robot.\n")

    stopping = []
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    try:
        while not stopping:
            st = call(url, "/status", timeout=30)
            run = st["run"]
            if run is None:
                print(f"\nrun ended: {st['last_stop']}")
                return 0
            cams = " ".join(f"{k}:{v['frames']}" for k, v in run["cameras"].items())
            inf = run["inference"]
            line = (
                f"[{run['elapsed_s']:5d}s] control {'yes' if run['control']['in_control'] else 'NO '}"
                f"  frames {cams}  state {run['state']['msgs']}  chunks {inf['chunks']}"
                f" (mean {inf['mean_ms']} ms)  actions {run['actions_published']}"
                + (f"  error {run['error']}" if run["error"] else "")
            )
            print(f"\r{line:<160}"[:161], end="", flush=True)
            for _ in range(int(args.poll * 10)):
                if stopping:
                    break
                time.sleep(0.1)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("\nstopping...")
        print(call(url, "/stop", {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
