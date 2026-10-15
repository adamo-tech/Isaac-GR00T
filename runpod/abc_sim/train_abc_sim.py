"""Fine-tune GR00T N1.7 on abc-sim bottles-in-bin (one GPU) and log losses as JSONL.

Matches the pi0.5 / DM0.5 runs on the same 9 episodes (batch 56, 2000 steps).
Each logging step appends {"step", "loss", "grad_norm", "learning_rate"} to
<run_dir>/losses.jsonl, the format openpi/data/abc_sim/runs/*/losses.jsonl uses.

  python runpod/abc_sim/train_abc_sim.py --base <GR00T-N1.7-3B dir> --dataset <dir> --run-dir <dir>
"""

import argparse
import json
import runpy
import sys
from pathlib import Path

import transformers

REPO = Path(__file__).resolve().parents[2]

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--run-dir", required=True)
p.add_argument("--batch-size", type=int, default=56)
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--lr", type=float, default=1e-4)
args = p.parse_args()

run_dir = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
losses = (run_dir / "losses.jsonl").open("w")
orig_log = transformers.Trainer.log


def log(self, logs, *a, **k):
    if "loss" in logs:
        row = {"step": self.state.global_step}
        row.update({key: logs[key] for key in ("loss", "grad_norm", "learning_rate") if key in logs})
        losses.write(json.dumps(row) + "\n")
        losses.flush()
    return orig_log(self, logs, *a, **k)


transformers.Trainer.log = log

sys.argv = [
    "launch_finetune.py",
    "--base_model_path", args.base,
    "--dataset_path", args.dataset,
    "--embodiment_tag", "new_embodiment",
    "--modality_config_path", str(REPO / "runpod/abc_sim/abc_sim_config.py"),
    "--num_gpus", "1",
    "--output_dir", str(run_dir / "ckpt"),
    "--save_steps", "1000",
    "--save_total_limit", "2",
    "--save_only_model",
    "--max_steps", str(args.max_steps),
    "--warmup_ratio", "0.05",
    "--weight_decay", "1e-5",
    "--learning_rate", str(args.lr),
    "--global_batch_size", str(args.batch_size),
    "--dataloader_num_workers", "8",
    "--shard_size", "1024",
    "--num_shards_per_epoch", "100000",
    "--episode_sampling_rate", "0.1",
]
try:
    runpy.run_path(str(REPO / "gr00t/experiment/launch_finetune.py"), run_name="__main__")
finally:
    losses.close()
