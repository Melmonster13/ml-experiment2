"""Generate fixed-prompt samples from each successful run's adapter.

    python scripts/eval_samples.py            # every run with status=ok
    python scripts/eval_samples.py r1 r2      # specific runs

Prompts use the same Alpaca template as training. Greedy decoding (temp 0),
fixed seed and max tokens. Output: results/samples/<run_id>.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "results" / "samples"
MAX_TOKENS = 256
SEED = 0

TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def ok_runs() -> list[str]:
    with (ROOT / "results" / "runs.csv").open() as f:
        return [r["run_id"] for r in csv.DictReader(f) if r["status"] == "ok"]


def main() -> None:
    run_ids = sys.argv[1:] or ok_runs()
    prompts = (ROOT / "prompts" / "eval_prompts.txt").read_text().strip().splitlines()
    configs = {p.stem.split("_")[0]: p for p in (ROOT / "configs").glob("*.yaml")}
    SAMPLES.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids:
        cfg = yaml.safe_load(configs[run_id].read_text())
        print(f"== {run_id}: {cfg['model']} + {cfg['adapter_path']}")
        model, tokenizer = load(
            str(ROOT / cfg["model"]) if (ROOT / cfg["model"]).exists() else cfg["model"],
            adapter_path=str(ROOT / cfg["adapter_path"]),
        )
        sampler = make_sampler(temp=0.0)

        lines = [
            f"# {run_id} samples\n",
            f"Base: `{cfg['model']}` · adapter: `{cfg['adapter_path']}` (unfused) · "
            f"greedy, max_tokens={MAX_TOKENS}, seed={SEED}\n",
        ]
        for i, instruction in enumerate(prompts, 1):
            mx.random.seed(SEED)
            text = generate(
                model, tokenizer, prompt=TEMPLATE.format(instruction=instruction),
                max_tokens=MAX_TOKENS, sampler=sampler,
            )
            lines += [f"## {i}. {instruction}\n", f"```\n{text.strip()}\n```\n"]

        (SAMPLES / f"{run_id}.md").write_text("\n".join(lines))
        del model, tokenizer
        mx.clear_cache()


if __name__ == "__main__":
    main()
