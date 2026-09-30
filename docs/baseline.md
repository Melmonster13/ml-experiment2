# Baseline (extracted from `ml-experiment`)

Source repo: `/Users/melcadd/Developer/ml-experiment` (HEAD `30dae09`).
Every original config snapshot lives in `experiments/<timestamp>/adapter_config.json`.

## Environment
| | |
|---|---|
| Hardware | M4 MacBook Pro, 16 GB unified memory |
| Python | 3.11.15 (original `.venv`) |
| mlx | 0.31.2 (installed 2026-04-30) |
| mlx-lm | 0.31.3 (installed 2026-04-30) |

Both packages were installed before the first run (2026-04-30 23:03), so every original run used these versions.

## Shared hyperparameters (identical across all original runs)
| Param | Value |
|---|---|
| fine_tune_type | lora |
| optimizer | adam (default config) |
| learning_rate | 1e-4, constant (`lr_schedule: null`) |
| batch_size | 1 |
| grad_accumulation_steps | 1 |
| grad_checkpoint | **true** |
| seed | 0 |
| lora rank / scale / dropout | 8 / 10.0 / 0.05 |
| lora alpha | 16 (**ignored by mlx-lm**, which only reads `scale`) |
| lora keys | `self_attn.{q,k,v}_proj`, `self_attn.dense` |
| mask_prompt | false |
| steps_per_report / steps_per_eval / save_every | 10 / 100 / 100 |
| test | false (no test loss was ever recorded) |

## Best runs
| | Phi-2 | StarCoder2-3B |
|---|---|---|
| Run | `2026-05-04-0014` | `2026-05-04-2233` |
| Model | `microsoft/phi-2` | `bigcode/starcoder2-3b` |
| Dataset | `tatsu-lab/alpaca` | `iamtarun/python_code_instructions_18k_alpaca` |
| num_layers | 16 | 4 |
| max_seq_length | 1024 | 256 |
| iters | 200 | 200 |
| val_batches | 25 | 5 |
| Val loss @100 / @200 | 0.892 / **0.817** | 0.935 / **0.569** |
| Final train loss | 0.692 | 0.566 |
| It/sec (last report) | 1.80 | 1.62 |

- The README's "0.81" is 0.817 truncated. Its "0.82" row (16 layers / rank 8 / 300 iters, `2026-05-03-2339`) hit the same 0.817 at iter 200, because that run is the same seed and config up to that point. So the two numbers are the same measurement.
- Runs are deterministic. Phi-2 runs with the same config give identical val loss at shared checkpoints. That makes the R1 reproducibility gate meaningful.
- The original logs didn't record peak memory or tokens/sec. They only parsed `It/sec`.

## OOM configs (StarCoder2-3B, for R8)
| Run | num_layers | max_seq_length | val_batches | Outcome |
|---|---|---|---|---|
| `2026-05-04-2214` | 16 | 1024 | 25 | OOM at startup (no metrics written) |
| `2026-05-04-2228` | 8 | 512 | 10 | OOM at iter 1 (after initial val 1.454) |

`2026-05-04-2240` (4 layers / 256 / 400 iters) was **stopped manually** at iter 190. It was not an OOM.

## Data
- Format: `{"text": ...}` JSONL (completions style), Alpaca prompt template with an optional `### Input` block.
- Split: shuffled with `random.Random(0)`. Validation and test each get `max(500, n // 20)` examples, and the rest go to train.
- The original `data/` now holds the **Python-instructions** dataset (train/valid/test = 16752/930/930):
  - train `bf3f42cf…c2f5`
  - valid `135cc2ae…d91c`
  - test `345ff8c1…600c6b`
- The Phi-2 Alpaca data was overwritten. It would need regenerating with the `load_dataset` line switched back.

## CLI notes (mlx-lm 0.31.3)
- `python -m mlx_lm.<cmd>` is deprecated. Use `python -m mlx_lm <cmd>`.
- convert: `--hf-path`, `--mlx-path`, `-q`, `--q-bits`, `--q-group-size`, `--q-mode affine` (the default)
- lora: `-c/--config`, `--grad-checkpoint`, `--test`, `--test-batches`. LoRA rank, scale and dropout can only be set via YAML `lora_parameters`.
- fuse: `--model`, `--adapter-path`, `--save-path`, `--dequantize`
- Log lines to parse:
  - `Val loss X, Val took Y`
  - `Train loss …, Learning Rate …, It/sec …, Tokens/sec …, Trained Tokens …, Peak mem X GB`
  - `Test loss X, Test ppl Y.`

## Base model precision
The cached `bigcode/starcoder2-3b` snapshot (`733247c5`) stores all 483 tensors as **F32** (12.1 GB), and its `config.json` has no `torch_dtype`. mlx-lm loads the weights as stored, so the original runs, and R1, train on an **fp32** base. Because `config.json` has no `torch_dtype`, `mlx_lm.convert` also keeps the non-quantized parameters and the quantization scales and biases in fp32. The quantized runs therefore differ from R1 only in bit width and group size. No `--dtype` flag is passed.

## GPU memory cap (required for R1)
By default, Metal caps GPU memory at **11.84 GB** (`max_recommended_working_set_size`) on this machine. fp32 StarCoder2 training peaks at **12.64 GB**, so R1 sat over the cap:
- Attempt 1 OOM'd at about iter 55.
- Attempt 2 OOM'd before iter 10, even with 30 GB of free disk.

The original May run completing was luck. **Every run in this project uses `sudo sysctl iogpu.wired_limit_mb=14336`** (a 14 GB cap). The setting resets on reboot, so re-apply it before each session and check it with `sysctl iogpu.wired_limit_mb`.

## R1 reproducibility check: passed
R1 matches `2026-05-04-2233` exactly:
- Val loss 1.407 / 0.935 / **0.569** at iters 1 / 100 / 200
- Final train loss 0.566
- New: test loss **0.595** (all 930 test examples), peak memory 12.64 GB, ~269 tokens/sec, 10.0 min wall-clock

## Decisions (confirmed by Mel, 2026-09-29)
1. **Baseline model: StarCoder2-3B**, replaying `2026-05-04-2233`. The data is copied from the original `data/` and verified against the checksums above.
2. **R6 = best quantized base with `grad_checkpoint: false`.** Every original run had checkpointing on, so R6 shows whether quantization alone makes it unnecessary.
3. **R8 is split into two runs:**
   - R8a: 16 layers / seq length 1024
   - R8b: 8 layers / seq length 512
   - Both use the best quantized base and are otherwise baseline.
4. **Test loss added to every run** with `test: true` and `test_batches: -1` (all 930 examples). Validation stays at `val_batches: 5` for a faithful replay.
