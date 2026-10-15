"""Build an HTML page comparing GR00T N1.7, pi0.5 and DM0.5 training losses on abc-sim.

  python runpod/abc_sim/plot_losses.py <gr00t losses.jsonl> <out.html>
"""

import json
import sys
from pathlib import Path

RUNS = Path("/home/h/workspace/inference/openpi/data/abc_sim/runs")


def read(path, key):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [[r["step"], r[key]] for r in rows if r.get("step", 0) > 0 and key in r]


series = [
    {"name": "GR00T N1.7", "key": "gr00t", "data": read(sys.argv[1], "loss")},
    # pi0.5 averages over 32 padded dims; loss*32/14 is its 14-dim figure (an upper bound).
    {"name": "π0.5 (LoRA)", "key": "pi05", "data": read(RUNS / "pi05_lora_bs56/losses.jsonl", "loss_14d_upper")},
    {"name": "DM0.5 (LoRA)", "key": "dm05", "data": read(RUNS / "dm05_lora_bs56/losses.jsonl", "loss")},
]
html = Path(__file__).with_name("loss_page.html").read_text().replace("__DATA__", json.dumps(series))
Path(sys.argv[2]).write_text(html)
print(sys.argv[2], {s["key"]: (len(s["data"]), s["data"][-1]) for s in series})
