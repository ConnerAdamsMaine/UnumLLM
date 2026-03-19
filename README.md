# OneBitLLM

The repo-root Python code is now a thin frontend only. It does three jobs:

1. host the GUI
2. pull training corpora
3. dispatch commands to the Rust engine under [Engine/](./Engine)

It does not implement its own model, tokenizer, quantization, training loop, or inference stack anymore.

## Current Python Surface

- `gui.py`: Tkinter frontend for corpus download plus Rust `train`/`generate`
- `corpus.py`: corpus downloader with preset support
- `config.py`: Rust model-config module plus config CLI utilities
- `engine_client.py`: Rust engine module plus engine CLI utilities
- `train.py`: single wrapper for `onebitllm train`
- `generate.py`: single wrapper for `onebitllm generate`

## Build The Rust Engine

```bash
cd Engine
cargo build --release -p onebitllm-cli
```

Optional Python bindings:

```bash
cd Engine
pip install -e .
```

If the CLI is not at the default location, set `ONEBITLLM_ENGINE_BIN=/abs/path/to/onebitllm`.

## Download A Corpus

```bash
python corpus.py --preset tinyshakespeare
```

## Inspect Engine Status

```bash
python engine_client.py status
```

The status output reports the selected binary, whether it is stale relative to local Rust source edits, and whether `train`, `generate`, `quantize`, and `benchmark` are real commands or validate-only stubs.

## Write A Rust Model Config

```bash
python config.py write-model-config output/model_config.json --hidden-size 768 --num-layers 12
```

## Launch The GUI

```bash
python gui.py
```

The GUI writes Rust model config JSON and streams the Rust CLI output directly.

Optional GUI file logging:

```bash
python gui.py --log
python gui.py --log logs/gui-session.log
```

`--log` writes to `onebitllm-gui.log` in the repo root. `--log PATH` writes to the specified file.

## Run Rust Train Through Python

```bash
python train.py \
  --data corpora/tinyshakespeare.txt \
  --output output \
  --hidden-size 768 \
  --num-layers 12 \
  --num-attention-heads 12
```

Teacher-guided adaptation for the bigram path:

```bash
python train.py \
  --data corpora/tinyshakespeare.txt \
  --output output/student \
  --teacher-model output/teacher/model.obm \
  --distill-alpha 0.5 \
  --distill-temperature 1.5 \
  --eval-data corpora/tinyshakespeare.txt \
  --architecture bigram \
  --hidden-size 256 \
  --num-layers 1 \
  --num-attention-heads 1 \
  --num-kv-heads 1 \
  --intermediate-size 256 \
  --vocab-size 256
```

## Run Rust Generate Through Python

```bash
python generate.py \
  --model /path/to/existing-model.obm \
  --prompt "The future of 1-bit models is"
```

## Notes

- The Rust CLI and bindings are still incomplete in parts. The Python frontend does not mask that; it just forwards the real Rust output.
- The Rust CLI now has a real end-to-end path for `architecture = "bigram"` with explicit `fp32`, strict `binary`, and `ternary` train/save modes, teacher distillation, deployed-model eval, and workload benchmarking. Larger architectures still have validation-only train/generate surfaces.
- `train` can now write runnable `.obm` models for the byte-level bigram path. Other architectures still stop after validation.
- `quantize` can convert bigram `.obm` models between `fp32`, strict `binary`, and `ternary` weight formats and report conversion/eval drift.
- `benchmark` measures cold load, p50/p95/p99 latency, request throughput, and token throughput on real prompts for bigram `.obm` models.
- The old repo-root Python model/tokenizer/quantization stack was removed on purpose to keep architecture boundaries clear.
