#!/usr/bin/env python3
"""Single Python wrapper for `onebitllm train`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from config import (
    RustModelConfig,
    TrainCommandConfig,
    add_model_config_arguments,
    build_model_config_from_args,
)
from dataset_installer import REDPAJAMA_SENTINEL, materialize_redpajama_corpus
from engine_client import stream_engine_command

MODEL_CONFIG_FLAGS = (
    "--architecture",
    "--hidden-size",
    "--num-layers",
    "--num-attention-heads",
    "--num-kv-heads",
    "--intermediate-size",
    "--vocab-size",
    "--max-seq-len",
    "--rms-norm-eps",
    "--activation",
    "--positional-encoding",
    "--rope-theta",
    "--use-bias",
    "--no-use-bias",
    "--quant-group-size",
    "--weight-format",
    "--training-weight-format",
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rust training through the Python frontend")
    parser.add_argument("--engine", default=None, help="Path to the Rust CLI binary")
    parser.add_argument("--data", required=True, help="Training corpus path")
    parser.add_argument("--config", default=None, help="Existing Rust model config JSON")
    parser.add_argument("--config-out", default=None, help="Where to write a generated Rust model config JSON")
    parser.add_argument("--output", default="output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument(
        "--train-weight-format",
        default="same-as-config",
        choices=("same-as-config", "fp32", "binary", "ternary"),
    )
    parser.add_argument(
        "--save-weight-format",
        default="same-as-train",
        choices=("same-as-train", "fp32", "binary", "ternary"),
    )
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    add_model_config_arguments(parser)
    return parser


def find_inline_model_config_flags(argv: list[str]) -> list[str]:
    """Return explicit inline model-config flags present in the raw argv."""

    seen: list[str] = []
    for token in argv:
        for flag in MODEL_CONFIG_FLAGS:
            if token == flag or token.startswith(f"{flag}="):
                seen.append(flag)
                break
    return sorted(set(seen))


def validate_train_inputs(job: TrainCommandConfig, config_path: str | Path) -> Path:
    """Validate Python-side paths before spawning Rust."""

    data_path = Path(job.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {data_path}")

    config_file = Path(config_path).expanduser()
    if not config_file.is_file():
        raise FileNotFoundError(f"Model config JSON not found: {config_file}")
    RustModelConfig.load_json(config_file)

    if job.resume:
        resume_path = Path(job.resume).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if job.teacher_model:
        teacher_path = Path(job.teacher_model).expanduser()
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    if job.eval_data:
        eval_path = Path(job.eval_data).expanduser()
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation corpus not found: {eval_path}")

    return config_file


def _is_redpajama_sentinel(value: str | None) -> bool:
    return bool(value) and value.strip().lower() == REDPAJAMA_SENTINEL


def resolve_special_dataset_inputs(job: TrainCommandConfig) -> TrainCommandConfig:
    if _is_redpajama_sentinel(job.data):
        job.data = str(materialize_redpajama_corpus(Path(job.output) / "datasets" / "redpajama", "train"))
    if _is_redpajama_sentinel(job.eval_data):
        job.eval_data = str(materialize_redpajama_corpus(Path(job.output) / "datasets" / "redpajama", "eval"))
    return job


def resolve_model_config_path(args: argparse.Namespace) -> Path:
    """Resolve an existing config path or write one from CLI flags."""

    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config JSON not found: {config_path}")
        RustModelConfig.load_json(config_path)
        return config_path

    model_config = build_model_config_from_args(args)
    config_path = Path(args.config_out or (Path(args.output) / "model_config.json")).expanduser()
    model_config.save_json(config_path)
    print(f"Wrote Rust model config to {config_path}")
    return config_path


def build_train_job(args: argparse.Namespace) -> TrainCommandConfig:
    """Build a train job from parsed CLI args."""

    return TrainCommandConfig(
        data=args.data,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        resume=args.resume,
        teacher_model=args.teacher_model,
        eval_data=args.eval_data,
        train_weight_format=args.train_weight_format,
        save_weight_format=args.save_weight_format,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
    )


def run_train_command(
    job: TrainCommandConfig,
    *,
    config_path: str | Path,
    engine: str | Path | None = None,
    on_output=None,
) -> int:
    """Run a Rust training job from Python."""

    job = resolve_special_dataset_inputs(job)
    config_path = validate_train_inputs(job, config_path)
    return stream_engine_command(
        job.build_args(config_path),
        explicit_binary=engine,
        on_output=on_output,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the train wrapper."""

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    inline_flags = find_inline_model_config_flags(raw_argv)
    if args.config and inline_flags:
        parser.error(
            "--config cannot be combined with inline model-config flags: "
            + ", ".join(inline_flags)
        )
    try:
        config_path = resolve_model_config_path(args)
        job = build_train_job(args)
        return run_train_command(job, config_path=config_path, engine=args.engine)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
