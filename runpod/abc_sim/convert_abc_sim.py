"""Convert openpi's abc-sim LeRobot v2.1 dataset into GR00T-flavored LeRobot v2.

The source (~/.cache/huggingface/lerobot/local/abc_sim_bottles, owned by the
openpi session, read-only here) stores images inline as PNG bytes; GR00T's
loader decodes mp4 per camera, so each camera is re-encoded to h264 and the
parquet columns are renamed to GR00T's `observation.state` / `action` layout.

  .venv/bin/python runpod/abc_sim/convert_abc_sim.py
"""

import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

SRC = Path.home() / ".cache/huggingface/lerobot/local/abc_sim_bottles"
DST = Path.home() / ".cache/huggingface/lerobot/local/abc_sim_bottles_gr00t"
CAMS = {"head_image": "head", "wrist_left_image": "wrist_left", "wrist_right_image": "wrist_right"}
RES = 224
FPS = 30
LANG_COL = "annotation.human.task_description"

# Joint order from the openpi session: left_joint_0..5, left_gripper, right_joint_0..5, right_gripper.
GROUPS = {"left_arm": (0, 6), "left_gripper": (6, 7), "right_arm": (7, 13), "right_gripper": (13, 14)}


def encode(frames: list[bytes], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{RES}x{RES}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "2", str(out)],
        stdin=subprocess.PIPE,
    )
    for b in frames:
        proc.stdin.write(np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, out


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "meta").mkdir(parents=True)
    for f in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy(SRC / "meta" / f, DST / "meta" / f)

    info = json.loads((SRC / "meta/info.json").read_text())
    for ep_file in sorted((SRC / "data/chunk-000").glob("episode_*.parquet")):
        t = pq.read_table(ep_file).to_pandas()
        ep = int(t.episode_index.iloc[0])
        for col, name in CAMS.items():
            encode(
                [r["bytes"] for r in t[col]],
                DST / f"videos/chunk-000/observation.images.{name}/episode_{ep:06d}.mp4",
            )
        n = len(t)
        df = pd.DataFrame({
            "observation.state": [np.asarray(s, np.float32) for s in t.state],
            "action": [np.asarray(a, np.float32) for a in t.actions],
            "timestamp": t.timestamp.astype(np.float32),
            "frame_index": t.frame_index,
            "episode_index": t.episode_index,
            "index": t["index"],
            "task_index": t.task_index,
            LANG_COL: t.task_index,
            "next.reward": np.zeros(n, np.float32),
            "next.done": np.arange(n) == n - 1,
        })
        out = DST / f"data/chunk-000/episode_{ep:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print(f"episode {ep}: {n} frames")

    feats = {k: v for k, v in info["features"].items() if k not in CAMS and k not in ("state", "actions")}
    names = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"] + \
            [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
    feats["observation.state"] = {"dtype": "float32", "shape": [14], "names": names}
    feats["action"] = {"dtype": "float32", "shape": [14], "names": names}
    for name in CAMS.values():
        feats[f"observation.images.{name}"] = {
            "dtype": "video", "shape": [RES, RES, 3], "names": ["height", "width", "channel"],
            "info": {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.height": RES, "video.width": RES, "video.channels": 3},
        }
    feats[LANG_COL] = {"dtype": "int64", "shape": [1], "names": None}
    feats["next.reward"] = {"dtype": "float32", "shape": [1], "names": None}
    feats["next.done"] = {"dtype": "bool", "shape": [1], "names": None}
    info.update(features=feats, total_videos=len(CAMS) * info["total_episodes"])
    (DST / "meta/info.json").write_text(json.dumps(info, indent=4))

    span = {k: {"start": a, "end": b} for k, (a, b) in GROUPS.items()}
    modality = {
        "state": span,
        "action": span,
        "video": {n: {"original_key": f"observation.images.{n}"} for n in CAMS.values()},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (DST / "meta/modality.json").write_text(json.dumps(modality, indent=4))
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
