#!/usr/bin/env python3
"""GUI-only Python frontend for driving the Rust engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from config import GenerateCommandConfig, RustModelConfig, TrainCommandConfig
from corpus import CORPUS_PRESETS, default_destination_for_preset, download_corpus, resolve_download_url
from engine_client import REPO_ROOT, detect_engine_status, format_engine_status, translate_engine_output
from generate import run_generate_command
from train import run_train_command

DEFAULT_GUI_LOG_PATH = REPO_ROOT / "onebitllm-gui.log"
LOGGER = logging.getLogger("onebitllm.gui")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the GUI frontend."""

    parser = argparse.ArgumentParser(description="Launch the OneBitLLM GUI frontend")
    parser.add_argument(
        "--log",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help=(
            "Write GUI logs to a file. Use `--log` for the default path "
            f"({DEFAULT_GUI_LOG_PATH}) or `--log path/to/file.log` for a custom path."
        ),
    )
    return parser


def resolve_gui_log_path(log_arg: str | None) -> Path | None:
    """Resolve the CLI log argument into a filesystem path."""

    if log_arg is None:
        return None
    if log_arg == "__default__":
        return DEFAULT_GUI_LOG_PATH
    return Path(log_arg).expanduser()


def configure_gui_logging(log_arg: str | None) -> Path | None:
    """Configure optional file logging for the GUI."""

    log_path = resolve_gui_log_path(log_arg)
    if log_path is None:
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
    LOGGER.info("GUI logging enabled")
    return log_path


class OneBitLLMGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OneBitLLM Python Frontend")
        self.root.geometry("1120x900")
        LOGGER.info("Initializing GUI")

        status = detect_engine_status()
        self.engine_status = status
        self.engine_path_var = tk.StringVar(
            value=str(status.binary_path) if status.binary_path else ""
        )

        self._build_train_defaults()
        self._build_generate_defaults()
        self._build_corpus_defaults()

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.corpus_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.generate_tab = ttk.Frame(notebook)
        self.engine_tab = ttk.Frame(notebook)

        notebook.add(self.corpus_tab, text="Corpus")
        notebook.add(self.train_tab, text="Train")
        notebook.add(self.generate_tab, text="Generate")
        notebook.add(self.engine_tab, text="Engine")

        self._build_corpus_tab()
        self._build_train_tab()
        self._build_generate_tab()
        self._build_engine_tab(status)
        self._apply_engine_status(status, clear_log=False)

    def _build_train_defaults(self) -> None:
        model = RustModelConfig()
        self.architecture_var = tk.StringVar(value=model.architecture)
        self.hidden_size_var = tk.IntVar(value=model.hidden_size)
        self.num_layers_var = tk.IntVar(value=model.num_layers)
        self.num_heads_var = tk.IntVar(value=model.num_attention_heads)
        self.num_kv_heads_var = tk.IntVar(value=model.num_kv_heads)
        self.intermediate_size_var = tk.IntVar(value=model.intermediate_size)
        self.vocab_size_var = tk.IntVar(value=model.vocab_size)
        self.max_seq_len_var = tk.IntVar(value=model.max_seq_len)
        self.activation_var = tk.StringVar(value=model.activation)

        self.train_data_var = tk.StringVar(value=str(REPO_ROOT / "corpora" / "tinyshakespeare.txt"))
        self.train_output_var = tk.StringVar(value=str(REPO_ROOT / "output"))
        self.train_config_path_var = tk.StringVar(value=str(REPO_ROOT / "output" / "model_config.json"))
        self.train_epochs_var = tk.IntVar(value=3)
        self.train_batch_size_var = tk.IntVar(value=8)
        self.train_lr_var = tk.DoubleVar(value=1e-4)
        self.train_weight_decay_var = tk.DoubleVar(value=0.01)
        self.train_max_grad_norm_var = tk.DoubleVar(value=1.0)
        self.train_warmup_steps_var = tk.IntVar(value=100)
        self.train_max_steps_var = tk.IntVar(value=0)
        self.train_save_every_var = tk.IntVar(value=500)
        self.train_log_every_var = tk.IntVar(value=10)
        self.train_seed_var = tk.StringVar(value="")
        self.train_resume_var = tk.StringVar(value="")

    def _build_generate_defaults(self) -> None:
        self.generate_model_var = tk.StringVar(value="")
        self.generate_max_tokens_var = tk.IntVar(value=256)
        self.generate_temperature_var = tk.DoubleVar(value=0.7)
        self.generate_top_k_var = tk.IntVar(value=0)
        self.generate_top_p_var = tk.DoubleVar(value=1.0)
        self.generate_repetition_penalty_var = tk.DoubleVar(value=1.0)
        self.generate_seed_var = tk.StringVar(value="")
        self.generate_stream_var = tk.BooleanVar(value=True)

    def _build_corpus_defaults(self) -> None:
        default_preset = "tinyshakespeare"
        self.corpus_preset_var = tk.StringVar(value=default_preset)
        self.corpus_url_var = tk.StringVar(value=CORPUS_PRESETS[default_preset].url)
        self.corpus_output_var = tk.StringVar(
            value=str(REPO_ROOT / default_destination_for_preset(default_preset))
        )

    def _build_train_tab(self) -> None:
        model_frame = ttk.LabelFrame(self.train_tab, text="Rust Model Config")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        self._add_entry(model_frame, "Architecture", self.architecture_var)
        self._add_entry(model_frame, "Hidden Size", self.hidden_size_var)
        self._add_entry(model_frame, "Layers", self.num_layers_var)
        self._add_entry(model_frame, "Attention Heads", self.num_heads_var)
        self._add_entry(model_frame, "KV Heads", self.num_kv_heads_var)
        self._add_entry(model_frame, "Intermediate Size", self.intermediate_size_var)
        self._add_entry(model_frame, "Vocab Size", self.vocab_size_var)
        self._add_entry(model_frame, "Max Seq Len", self.max_seq_len_var)
        self._add_entry(model_frame, "Activation", self.activation_var)

        train_frame = ttk.LabelFrame(self.train_tab, text="Rust Train Command")
        train_frame.pack(fill=tk.X, padx=10, pady=10)
        self._add_browse_entry(train_frame, "Corpus File", self.train_data_var, browse_dir=False)
        self._add_browse_entry(train_frame, "Output Dir", self.train_output_var, browse_dir=True)
        self._add_save_entry(train_frame, "Config JSON", self.train_config_path_var)
        self._add_entry(train_frame, "Epochs", self.train_epochs_var)
        self._add_entry(train_frame, "Batch Size", self.train_batch_size_var)
        self._add_entry(train_frame, "Learning Rate", self.train_lr_var)
        self._add_entry(train_frame, "Weight Decay", self.train_weight_decay_var)
        self._add_entry(train_frame, "Max Grad Norm", self.train_max_grad_norm_var)
        self._add_entry(train_frame, "Warmup Steps", self.train_warmup_steps_var)
        self._add_entry(train_frame, "Max Steps", self.train_max_steps_var)
        self._add_entry(train_frame, "Save Every", self.train_save_every_var)
        self._add_entry(train_frame, "Log Every", self.train_log_every_var)
        self._add_entry(train_frame, "Seed", self.train_seed_var)
        self._add_browse_entry(train_frame, "Resume Checkpoint", self.train_resume_var, browse_dir=False)

        button_row = ttk.Frame(self.train_tab)
        button_row.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_row, text="Write Config", command=self._write_model_config).pack(side=tk.LEFT, padx=5)
        self.train_button = ttk.Button(button_row, text="Run Rust Train", command=self._run_train)
        self.train_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_row, text="Clear Log", command=lambda: self._clear_text(self.train_log)).pack(side=tk.LEFT, padx=5)

        self.train_log = self._build_log(self.train_tab)

    def _build_generate_tab(self) -> None:
        frame = ttk.LabelFrame(self.generate_tab, text="Rust Generate Command")
        frame.pack(fill=tk.X, padx=10, pady=10)
        self._add_browse_entry(frame, "Model File", self.generate_model_var, browse_dir=False)
        self._add_entry(frame, "Max Tokens", self.generate_max_tokens_var)
        self._add_entry(frame, "Temperature", self.generate_temperature_var)
        self._add_entry(frame, "Top-K", self.generate_top_k_var)
        self._add_entry(frame, "Top-P", self.generate_top_p_var)
        self._add_entry(frame, "Repetition Penalty", self.generate_repetition_penalty_var)
        self._add_entry(frame, "Seed", self.generate_seed_var)

        stream_row = ttk.Frame(frame)
        stream_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(stream_row, text="Stream Output", width=18).pack(side=tk.LEFT)
        ttk.Checkbutton(stream_row, variable=self.generate_stream_var).pack(side=tk.LEFT)

        prompt_label = ttk.Label(self.generate_tab, text="Prompt")
        prompt_label.pack(anchor="w", padx=10, pady=(10, 0))
        self.prompt_text = tk.Text(self.generate_tab, height=8, width=100)
        self.prompt_text.insert("1.0", "The future of 1-bit language models is")
        self.prompt_text.pack(fill=tk.X, padx=10, pady=5)

        button_row = ttk.Frame(self.generate_tab)
        button_row.pack(fill=tk.X, padx=10, pady=10)
        self.generate_button = ttk.Button(button_row, text="Run Rust Generate", command=self._run_generate)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_row, text="Clear Log", command=lambda: self._clear_text(self.generate_log)).pack(side=tk.LEFT, padx=5)

        self.generate_log = self._build_log(self.generate_tab)

    def _build_corpus_tab(self) -> None:
        frame = ttk.LabelFrame(self.corpus_tab, text="Corpus Download")
        frame.pack(fill=tk.X, padx=10, pady=10)

        preset_row = ttk.Frame(frame)
        preset_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(preset_row, text="Preset", width=18).pack(side=tk.LEFT)
        preset_box = ttk.Combobox(
            preset_row,
            textvariable=self.corpus_preset_var,
            values=sorted(CORPUS_PRESETS),
            state="readonly",
            width=30,
        )
        preset_box.pack(side=tk.LEFT, padx=5)
        preset_box.bind("<<ComboboxSelected>>", self._sync_corpus_preset)

        self._add_entry(frame, "Source URL", self.corpus_url_var)
        self._add_save_entry(frame, "Destination", self.corpus_output_var)

        button_row = ttk.Frame(self.corpus_tab)
        button_row.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_row, text="Download Corpus", command=self._download_corpus).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_row, text="Use In Train Tab", command=self._use_corpus_for_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_row, text="Clear Log", command=lambda: self._clear_text(self.corpus_log)).pack(side=tk.LEFT, padx=5)

        self.corpus_log = self._build_log(self.corpus_tab)

    def _build_engine_tab(self, status) -> None:
        self._add_entry(self.engine_tab, "Rust CLI", self.engine_path_var)

        button_row = ttk.Frame(self.engine_tab)
        button_row.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_row, text="Refresh Status", command=self._refresh_engine_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_row, text="Clear Log", command=lambda: self._clear_text(self.engine_log)).pack(side=tk.LEFT, padx=5)

        self.engine_log = self._build_log(self.engine_tab)

    def _build_log(self, parent: ttk.Frame) -> scrolledtext.ScrolledText:
        widget = scrolledtext.ScrolledText(parent, height=18, width=120, state=tk.DISABLED)
        widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        return widget

    def _add_entry(self, parent: ttk.Frame, label: str, variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _add_browse_entry(self, parent: ttk.Frame, label: str, variable, *, browse_dir: bool) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True)
        command = (lambda: self._browse_directory(variable)) if browse_dir else (lambda: self._browse_file(variable))
        ttk.Button(row, text="Browse", command=command).pack(side=tk.LEFT, padx=5)

    def _add_save_entry(self, parent: ttk.Frame, label: str, variable) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text=label, width=18).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse", command=lambda: self._browse_save_file(variable)).pack(side=tk.LEFT, padx=5)

    def _browse_file(self, variable) -> None:
        path = filedialog.askopenfilename()
        if path:
            variable.set(path)

    def _browse_directory(self, variable) -> None:
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def _browse_save_file(self, variable) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if path:
            variable.set(path)

    def _sync_corpus_preset(self, _event=None) -> None:
        preset = self.corpus_preset_var.get()
        self.corpus_url_var.set(CORPUS_PRESETS[preset].url)
        self.corpus_output_var.set(str(REPO_ROOT / default_destination_for_preset(preset)))

    def _make_model_config(self) -> RustModelConfig:
        return RustModelConfig(
            architecture=self.architecture_var.get().strip(),
            hidden_size=self.hidden_size_var.get(),
            num_layers=self.num_layers_var.get(),
            num_attention_heads=self.num_heads_var.get(),
            num_kv_heads=self.num_kv_heads_var.get(),
            intermediate_size=self.intermediate_size_var.get(),
            vocab_size=self.vocab_size_var.get(),
            max_seq_len=self.max_seq_len_var.get(),
            activation=self.activation_var.get().strip(),
        )

    def _write_model_config(self) -> None:
        try:
            output = self._make_model_config().save_json(self.train_config_path_var.get().strip())
            self._append_log(self.train_log, f"Wrote Rust model config to {output}")
        except Exception as exc:
            LOGGER.error("Config write failed: %s", exc)
            messagebox.showerror("Config Error", str(exc))

    def _capability_state(self, command: str) -> str:
        capability = self.engine_status.capabilities.get(command)
        if capability is None:
            return "missing"
        return capability.state

    def _apply_engine_status(self, status, *, clear_log: bool) -> None:
        self.engine_status = status
        if status.binary_path:
            self.engine_path_var.set(str(status.binary_path))

        if clear_log:
            self._clear_text(self.engine_log)
        self._append_log(self.engine_log, format_engine_status(status))

        train_state = self._capability_state("train")
        if train_state == "real":
            self.train_button.configure(text="Run Rust Train", state=tk.NORMAL)
        elif train_state == "validate-only":
            self.train_button.configure(text="Validate Rust Train Inputs", state=tk.NORMAL)
        else:
            self.train_button.configure(text="Rust Train Unavailable", state=tk.DISABLED)

        generate_state = self._capability_state("generate")
        if generate_state == "real":
            self.generate_button.configure(text="Run Rust Generate", state=tk.NORMAL)
        elif generate_state == "validate-only":
            self.generate_button.configure(text="Validate Rust Generate Inputs", state=tk.NORMAL)
        else:
            self.generate_button.configure(text="Rust Generate Unavailable", state=tk.DISABLED)

    def _run_train(self) -> None:
        if self._capability_state("train") == "missing":
            LOGGER.error("Train requested while capability is missing")
            messagebox.showerror("Train Error", "Rust train is unavailable. Refresh the engine status or build the CLI.")
            return

        try:
            config_path = self._make_model_config().save_json(self.train_config_path_var.get().strip())
            job = TrainCommandConfig(
                data=self.train_data_var.get().strip(),
                output=self.train_output_var.get().strip(),
                epochs=self.train_epochs_var.get(),
                batch_size=self.train_batch_size_var.get(),
                lr=self.train_lr_var.get(),
                weight_decay=self.train_weight_decay_var.get(),
                max_grad_norm=self.train_max_grad_norm_var.get(),
                warmup_steps=self.train_warmup_steps_var.get(),
                max_steps=self.train_max_steps_var.get(),
                save_every=self.train_save_every_var.get(),
                log_every=self.train_log_every_var.get(),
                seed=int(self.train_seed_var.get()) if self.train_seed_var.get().strip() else None,
                resume=self.train_resume_var.get().strip() or None,
            )
            LOGGER.info("Starting train action for corpus %s", job.data)
        except Exception as exc:
            LOGGER.error("Train setup failed: %s", exc)
            messagebox.showerror("Train Error", str(exc))
            return

        self._clear_text(self.train_log)
        if self._capability_state("train") == "validate-only":
            self._append_log(
                self.train_log,
                "Engine capability: validate-only. This build will only validate train inputs and will not write checkpoints or model files.",
            )
        self._run_callable_in_thread(
            self.train_log,
            lambda emit: run_train_command(
                job,
                config_path=config_path,
                engine=self.engine_path_var.get().strip() or None,
                on_output=emit,
            ),
        )

    def _run_generate(self) -> None:
        if self._capability_state("generate") == "missing":
            LOGGER.error("Generate requested while capability is missing")
            messagebox.showerror(
                "Generate Error",
                "Rust generate is unavailable. Refresh the engine status or build the CLI.",
            )
            return

        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            LOGGER.error("Generate requested without a prompt")
            messagebox.showerror("Generate Error", "Prompt is required.")
            return

        try:
            job = GenerateCommandConfig(
                model=self.generate_model_var.get().strip(),
                prompt=prompt,
                max_tokens=self.generate_max_tokens_var.get(),
                temperature=self.generate_temperature_var.get(),
                top_k=self.generate_top_k_var.get(),
                top_p=self.generate_top_p_var.get(),
                repetition_penalty=self.generate_repetition_penalty_var.get(),
                seed=int(self.generate_seed_var.get()) if self.generate_seed_var.get().strip() else None,
                stream=self.generate_stream_var.get(),
            )
            LOGGER.info("Starting generate action for model %s", job.model)
        except Exception as exc:
            LOGGER.error("Generate setup failed: %s", exc)
            messagebox.showerror("Generate Error", str(exc))
            return

        self._clear_text(self.generate_log)
        if self._capability_state("generate") == "validate-only":
            self._append_log(
                self.generate_log,
                "Engine capability: validate-only. This build will only validate generate inputs and will not produce tokens.",
            )
        self._run_callable_in_thread(
            self.generate_log,
            lambda emit: run_generate_command(
                job,
                engine=self.engine_path_var.get().strip() or None,
                on_output=emit,
            ),
        )

    def _download_corpus(self) -> None:
        def worker() -> None:
            try:
                LOGGER.info("Starting corpus download for preset %s", self.corpus_preset_var.get().strip())
                url = resolve_download_url(self.corpus_preset_var.get().strip(), self.corpus_url_var.get().strip())
                path = download_corpus(url, self.corpus_output_var.get().strip())
                self._append_log(self.corpus_log, f"Downloaded corpus to {path}")
            except Exception as exc:
                LOGGER.error("Corpus download failed: %s", exc)
                self._append_log(self.corpus_log, f"Download failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _use_corpus_for_training(self) -> None:
        self.train_data_var.set(self.corpus_output_var.get().strip())
        self._append_log(self.corpus_log, f"Train tab corpus set to {self.train_data_var.get()}")

    def _refresh_engine_status(self) -> None:
        LOGGER.info("Refreshing engine status")
        status = detect_engine_status(self.engine_path_var.get().strip() or None)
        self._apply_engine_status(status, clear_log=True)

    def _run_callable_in_thread(self, widget: scrolledtext.ScrolledText, action) -> None:
        def worker() -> None:
            captured_lines: list[str] = []

            def emit(line: str) -> None:
                captured_lines.append(line)
                self._append_log(widget, line)

            try:
                returncode = action(emit)
                translated = translate_engine_output("\n".join(captured_lines))
                if translated and translated not in "\n".join(captured_lines):
                    self._append_log(widget, f"Status: {translated}")
                self._append_log(widget, f"Process exited with code {returncode}")
            except Exception as exc:
                translated = translate_engine_output(str(exc)) or str(exc)
                self._append_log(widget, f"Failed to run Rust command: {translated}")

        threading.Thread(target=worker, daemon=True).start()

    def _clear_text(self, widget: scrolledtext.ScrolledText) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.configure(state=tk.DISABLED)

    def _append_log(self, widget: scrolledtext.ScrolledText, message: str) -> None:
        if LOGGER.handlers:
            LOGGER.info(message.rstrip())

        def write() -> None:
            widget.configure(state=tk.NORMAL)
            widget.insert(tk.END, message.rstrip() + "\n")
            widget.see(tk.END)
            widget.configure(state=tk.DISABLED)

        self.root.after(0, write)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = configure_gui_logging(args.log)
    if log_path is not None:
        print(f"GUI logging to {log_path}")

    try:
        root = tk.Tk()
    except tk.TclError:
        LOGGER.error("GUI requires a display server")
        print("GUI requires a display server.")
        return 1

    OneBitLLMGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
