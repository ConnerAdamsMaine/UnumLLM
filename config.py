#!/usr/bin/env python3
"""Thin Python-side config objects for driving the Rust engine."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

SUPPORTED_ACTIVATIONS = ("gelu", "mish", "relu", "silu", "swiglu")
SUPPORTED_POSITIONAL_ENCODINGS = ("alibi", "learned", "rope")
SUPPORTED_WEIGHT_FORMATS = ("fp32", "binary", "ternary")


@dataclass(slots=True)
class RustModelConfig:
    """Model config format expected by the Rust CLI."""

    architecture: str = "bitnet-b1.58"
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 12
    intermediate_size: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    activation: str = "silu"
    positional_encoding: str = "rope"
    rope_theta: float = 10000.0
    use_bias: bool = False
    quant_group_size: int = 0
    weight_format: str = "binary"
    training_weight_format: str = "binary"
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        positive_fields = {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than 0 (got {value})")

        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be greater than 0 (got {self.rms_norm_eps})")

        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be greater than 0 (got {self.rope_theta})")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})"
            )

        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError(
                "num_kv_heads must be less than or equal to num_attention_heads "
                f"(got num_kv_heads={self.num_kv_heads}, num_attention_heads={self.num_attention_heads})"
            )

        if self.activation not in SUPPORTED_ACTIVATIONS:
            supported = ", ".join(SUPPORTED_ACTIVATIONS)
            raise ValueError(f"activation must be one of {supported} (got {self.activation!r})")

        if self.positional_encoding not in SUPPORTED_POSITIONAL_ENCODINGS:
            supported = ", ".join(SUPPORTED_POSITIONAL_ENCODINGS)
            raise ValueError(
                "positional_encoding must be one of "
                f"{supported} (got {self.positional_encoding!r})"
            )

        if self.quant_group_size < 0:
            raise ValueError(f"quant_group_size must be 0 or a positive power of two (got {self.quant_group_size})")

        if self.quant_group_size and self.quant_group_size & (self.quant_group_size - 1):
            raise ValueError(
                "quant_group_size must be 0 (per-tensor) or a positive power of two "
                f"(got {self.quant_group_size})"
            )

        if self.weight_format not in SUPPORTED_WEIGHT_FORMATS:
            supported = ", ".join(SUPPORTED_WEIGHT_FORMATS)
            raise ValueError(f"weight_format must be one of {supported} (got {self.weight_format!r})")

        if self.training_weight_format not in SUPPORTED_WEIGHT_FORMATS:
            supported = ", ".join(SUPPORTED_WEIGHT_FORMATS)
            raise ValueError(
                "training_weight_format must be one of "
                f"{supported} (got {self.training_weight_format!r})"
            )

    def save_json(self, path: str | Path) -> Path:
        self.validate()
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load_json(cls, path: str | Path) -> "RustModelConfig":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        config = cls(**data)
        config.validate()
        return config


@dataclass(slots=True)
class TrainCommandConfig:
    """Python wrapper args for `onebitllm train`."""

    data: str
    output: str = "output"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 0
    save_every: int = 500
    log_every: int = 10
    seed: int | None = None
    resume: str | None = None
    teacher_model: str | None = None
    eval_data: str | None = None
    train_weight_format: str = "same-as-config"
    save_weight_format: str = "same-as-train"
    distill_alpha: float = 0.0
    distill_temperature: float = 1.0

    def build_args(self, config_path: str | Path) -> list[str]:
        args = [
            "train",
            "--data",
            self.data,
            "--config",
            str(config_path),
            "--output",
            self.output,
            "--epochs",
            str(self.epochs),
            "--batch-size",
            str(self.batch_size),
            "--lr",
            str(self.lr),
            "--weight-decay",
            str(self.weight_decay),
            "--max-grad-norm",
            str(self.max_grad_norm),
            "--warmup-steps",
            str(self.warmup_steps),
            "--max-steps",
            str(self.max_steps),
            "--save-every",
            str(self.save_every),
            "--log-every",
            str(self.log_every),
        ]
        if self.seed is not None:
            args.extend(["--seed", str(self.seed)])
        if self.resume:
            args.extend(["--resume", self.resume])
        if self.teacher_model:
            args.extend(["--teacher-model", self.teacher_model])
        if self.eval_data:
            args.extend(["--eval-data", self.eval_data])
        args.extend(["--train-weight-format", self.train_weight_format])
        args.extend(["--save-weight-format", self.save_weight_format])
        args.extend(["--distill-alpha", str(self.distill_alpha)])
        args.extend(["--distill-temperature", str(self.distill_temperature)])
        return args


@dataclass(slots=True)
class GenerateCommandConfig:
    """Python wrapper args for `onebitllm generate`."""

    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    device: str = "cpu"
    stream: bool = True

    def build_args(self) -> list[str]:
        args = [
            "generate",
            "--model",
            self.model,
            "--prompt",
            self.prompt,
            "--max-tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
            "--top-k",
            str(self.top_k),
            "--top-p",
            str(self.top_p),
            "--repetition-penalty",
            str(self.repetition_penalty),
            "--device",
            self.device,
        ]
        if self.seed is not None:
            args.extend(["--seed", str(self.seed)])
        args.extend(["--stream", "true" if self.stream else "false"])
        return args


def build_model_config_from_args(args: argparse.Namespace) -> RustModelConfig:
    """Construct a Rust model config from parsed CLI args."""

    config = RustModelConfig(
        architecture=args.architecture,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        rms_norm_eps=args.rms_norm_eps,
        activation=args.activation,
        positional_encoding=args.positional_encoding,
        rope_theta=args.rope_theta,
        use_bias=args.use_bias,
        quant_group_size=args.quant_group_size,
        weight_format=args.weight_format,
        training_weight_format=args.training_weight_format,
    )
    config.validate()
    return config


def add_model_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach Rust model-config flags to a parser."""

    defaults = RustModelConfig()
    parser.add_argument("--architecture", default=defaults.architecture)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--num-attention-heads", type=int, default=defaults.num_attention_heads)
    parser.add_argument("--num-kv-heads", type=int, default=defaults.num_kv_heads)
    parser.add_argument("--intermediate-size", type=int, default=defaults.intermediate_size)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--rms-norm-eps", type=float, default=defaults.rms_norm_eps)
    parser.add_argument("--activation", default=defaults.activation, choices=SUPPORTED_ACTIVATIONS)
    parser.add_argument(
        "--positional-encoding",
        default=defaults.positional_encoding,
        choices=SUPPORTED_POSITIONAL_ENCODINGS,
    )
    parser.add_argument("--rope-theta", type=float, default=defaults.rope_theta)
    parser.add_argument("--use-bias", action=argparse.BooleanOptionalAction, default=defaults.use_bias)
    parser.add_argument("--quant-group-size", type=int, default=defaults.quant_group_size)
    parser.add_argument("--weight-format", default=defaults.weight_format, choices=SUPPORTED_WEIGHT_FORMATS)
    parser.add_argument(
        "--training-weight-format",
        default=defaults.training_weight_format,
        choices=SUPPORTED_WEIGHT_FORMATS,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for config utilities."""

    parser = argparse.ArgumentParser(description="Config utilities for the OneBitLLM Python frontend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_defaults = subparsers.add_parser("show-model-defaults", help="Print default Rust model config JSON")
    show_defaults.add_argument("--compact", action="store_true", help="Print compact JSON")

    write_config = subparsers.add_parser("write-model-config", help="Write a Rust model config JSON file")
    write_config.add_argument("output", help="Destination JSON path")
    add_model_config_arguments(write_config)

    inspect_config = subparsers.add_parser("inspect-model-config", help="Read a Rust model config JSON file")
    inspect_config.add_argument("path", help="Config JSON path")
    inspect_config.add_argument("--compact", action="store_true", help="Print compact JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for config helpers."""

    args = build_parser().parse_args(argv)

    try:
        if args.command == "show-model-defaults":
            config = RustModelConfig()
            print(json.dumps(config.to_dict(), indent=None if args.compact else 2))
            return 0

        if args.command == "write-model-config":
            config = build_model_config_from_args(args)
            output = config.save_json(args.output)
            print(output)
            return 0

        if args.command == "inspect-model-config":
            config = RustModelConfig.load_json(args.path)
            print(json.dumps(config.to_dict(), indent=None if args.compact else 2))
            return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc))

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
