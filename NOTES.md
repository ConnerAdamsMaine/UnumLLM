# Notes

## Scope of the cleanup

The repo-root Python code was reduced to a thin frontend layer.

Python now only:

1. hosts the GUI
2. downloads corpora
3. writes Rust model config JSON
4. dispatches train/generate commands to the Rust engine
5. reports Rust engine/binding status

Python no longer owns a separate model, tokenizer, quantization, training, inference, or chat runtime at repo root.

## Files added or rewritten

- `config.py`
  - Rust model-config dataclass
  - train/generate command dataclasses
  - CLI helpers for showing, writing, and inspecting model config JSON
- `engine_client.py`
  - Rust CLI discovery
  - engine status inspection
  - command streaming helper
  - CLI helpers for `status`, `resolve-binary`, and `run`
- `corpus.py`
  - corpus preset registry
  - corpus downloader
  - CLI for preset listing and downloading
- `train.py`
  - single Python wrapper for `onebitllm train`
  - importable helpers for building/running train jobs
- `generate.py`
  - single Python wrapper for `onebitllm generate`
  - importable helpers for building/running generate jobs
- `gui.py`
  - Tkinter frontend for corpus download, Rust config writing, Rust train, Rust generate, and engine diagnostics
  - uses the shared module APIs instead of duplicating train/generate command logic
- `README.md`
  - updated to describe the current Python boundary and current entry points
- `requirements.txt`
  - simplified to reflect that the repo-root frontend has no third-party Python runtime dependency requirement

## Files removed

- `train_v2.py`
- `generate_v2.py`
- `chat.py`
- old repo-root Python model/tokenizer/quantization/training files from the previous parallel Python runtime

## Current repo-root Python surface

- `config.py`
- `engine_client.py`
- `corpus.py`
- `train.py`
- `generate.py`
- `gui.py`

Each remaining file has both:

- a module face for import/use from other Python code
- a CLI face for direct shell usage

## Behavioral notes

- The GUI is only a frontend; it does not implement its own training or generation logic.
- The Python wrappers forward real Rust CLI output directly.
- The installed Rust Python bindings are detectable, but the Rust train/generate path still contains placeholder behavior in places.
- That placeholder behavior is now surfaced honestly instead of being masked by a second Python implementation.

## Verification performed

- `rg -n "\\bv1\\b|\\bv2\\b|train_v2|generate_v2|chat.py" . --glob '!Engine/**' --glob '!dataset/**' --glob '!__pycache__/**'`
- `.venv/bin/python -m compileall config.py engine_client.py corpus.py gui.py train.py generate.py`
- `.venv/bin/python config.py show-model-defaults --compact`
- `.venv/bin/python engine_client.py status`
- `.venv/bin/python corpus.py --list-presets`
- `.venv/bin/python train.py --help`
- `.venv/bin/python generate.py --help`
- import smoke check for the remaining repo-root Python modules

## Result

There is now one current version of the repo-root Python frontend, with a strict boundary:

- GUI
- corpus
- Rust command/config orchestration

No repo-root Python v1/v2 split remains.
