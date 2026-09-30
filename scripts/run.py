"""Run one mlx_lm.lora config in a fresh process and log its metrics.

    python scripts/run.py configs/r1_lora_fp.yaml [--notes "..."]

Writes results/logs/<run_id>.log (raw stdout), results/logs/<run_id>_metrics.csv
(per-report losses), and upserts one row into results/runs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
RUNS_CSV = ROOT / "results" / "runs.csv"

TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([0-9.]+), Learning Rate [0-9.eE+-]+, It/sec ([0-9.]+), "
    r"Tokens/sec ([0-9.]+), Trained Tokens \d+, Peak mem ([0-9.]+) GB"
)
VAL_RE = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")
TEST_RE = re.compile(r"Test loss ([0-9.]+)")
OOM_RE = re.compile(r"Insufficient Memory|OutOfMemory|metal::malloc|out of memory", re.I)

COLUMNS = [
    "run_id", "base_model", "bits", "group_size", "lora_rank", "seq_len", "grad_ckpt",
    "iters", "final_train_loss", "final_val_loss", "test_loss",
    "peak_mem_gb", "tokens_per_sec", "wall_clock_min", "status", "notes",
]


def quant_info(model: str) -> tuple[str, str]:
    cfg_path = ROOT / model / "config.json"
    if cfg_path.exists():
        q = json.loads(cfg_path.read_text()).get("quantization")
        if q:
            return str(q["bits"]), str(q["group_size"])
    return "32", ""  # unquantized fp32 base (see docs/baseline.md)


def upsert(row: dict) -> None:
    rows = []
    if RUNS_CSV.exists():
        with RUNS_CSV.open() as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] != row["run_id"]]
    rows.append(row)
    with RUNS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    run_id = config_path.stem.split("_")[0]
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{run_id}.log"

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    print("Running:", " ".join(cmd))

    train, val, test_loss = [], [], ""
    oom_seen = False
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            if m := TRAIN_RE.search(line):
                train.append([int(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5])])
            elif m := VAL_RE.search(line):
                val.append([int(m[1]), float(m[2])])
            elif m := TEST_RE.search(line):
                test_loss = float(m[1])
            if OOM_RE.search(line):
                oom_seen = True
        ret = proc.wait()
    wall_min = (time.time() - start) / 60

    with (LOGS / f"{run_id}_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "train_loss", "val_loss", "its_per_sec", "tokens_per_sec", "peak_mem_gb"])
        writer.writerows([i, t, "", s, tok, mem] for i, t, s, tok, mem in train)
        writer.writerows([i, "", v, "", "", ""] for i, v in val)

    if ret == 0:
        status = "ok"
    elif oom_seen or ret == -9:  # -9: killed by macOS under memory pressure
        status = "oom"
    else:
        status = "error"

    bits, group = quant_info(cfg["model"])
    upsert({
        "run_id": run_id,
        "base_model": cfg["model"],
        "bits": bits,
        "group_size": group,
        "lora_rank": cfg["lora_parameters"]["rank"],
        "seq_len": cfg["max_seq_length"],
        "grad_ckpt": cfg["grad_checkpoint"],
        "iters": cfg["iters"],
        "final_train_loss": train[-1][1] if train else "",
        "final_val_loss": val[-1][1] if val else "",
        "test_loss": test_loss,
        "peak_mem_gb": max(r[4] for r in train) if train else "",
        "tokens_per_sec": round(sum(r[3] for r in train) / len(train), 1) if train else "",
        "wall_clock_min": round(wall_min, 1),
        "status": status,
        "notes": args.notes,
    })
    print(f"{run_id}: {status} in {wall_min:.1f} min (exit {ret}) -> {RUNS_CSV}")


if __name__ == "__main__":
    main()
