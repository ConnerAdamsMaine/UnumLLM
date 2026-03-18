use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use ndarray::{Array, Ix2, IxDyn};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

use onebitllm_core::backend::ComputeBackend;
use onebitllm_core::autograd::{Variable, ops};
use onebitllm_core::infer::{Sampler, SamplingConfig};
use onebitllm_core::io::custom::{ObmTensor, TensorFormat};
use onebitllm_core::io::{ModelConfig, ObmFile};
use onebitllm_core::nn::Parameter;
use onebitllm_core::optim::Optimizer;
use onebitllm_core::optim::adamw::AdamW;
use onebitllm_core::optim::scheduler::{CosineScheduler, LrScheduler, WarmupScheduler};
use onebitllm_core::quant::{PackedTernary, QuantConfig, QuantGranularity, QuantParams};
use onebitllm_core::tensor::PackedTensor;
use onebitllm_core::train::loop_::{TrainConfig, Trainer};

pub const BIGRAM_ARCHITECTURE: &str = "bigram";
pub const BIGRAM_WEIGHT_TENSOR: &str = "bigram.weight";
pub const WEIGHT_FORMAT_FP32: &str = "fp32";
pub const WEIGHT_FORMAT_TERNARY: &str = "ternary";

const META_BIGRAM_WEIGHT_GRANULARITY: &str = "bigram_weight_granularity";
const META_BIGRAM_WEIGHT_SCALES: &str = "bigram_weight_scales";

#[derive(Debug, Clone)]
pub struct LoadedBigramModel {
    pub config: ModelConfig,
    pub dense_weight: Array<f32, IxDyn>,
    pub packed_weight: Option<PackedTensor>,
}

#[derive(Debug, Clone)]
pub struct BigramEvalMetrics {
    pub pair_count: usize,
    pub avg_loss: f32,
    pub perplexity: f32,
    pub accuracy: f32,
}

#[derive(Debug, Clone)]
pub struct BigramRetentionMetrics {
    pub pair_count: usize,
    pub top1_agreement: f32,
    pub avg_kl_divergence: f32,
}

#[derive(Debug, Clone)]
pub struct BigramQuantizationMetrics {
    pub mse: f32,
    pub mean_abs_error: f32,
    pub max_abs_error: f32,
    pub exact_match_fraction: f32,
}

#[derive(Debug)]
struct SerializedBigramModel {
    config: ModelConfig,
    tensor: ObmTensor,
    quant_metrics: Option<BigramQuantizationMetrics>,
}

#[derive(Debug)]
struct BigramBatch {
    input: Array<f32, IxDyn>,
    targets: Array<f32, IxDyn>,
    source_tokens: Vec<usize>,
}

pub fn is_bigram_architecture(architecture: &str) -> bool {
    architecture.eq_ignore_ascii_case(BIGRAM_ARCHITECTURE)
}

pub fn normalize_weight_format(value: &str) -> anyhow::Result<String> {
    match value.trim().to_ascii_lowercase().as_str() {
        WEIGHT_FORMAT_FP32 => Ok(WEIGHT_FORMAT_FP32.into()),
        WEIGHT_FORMAT_TERNARY => Ok(WEIGHT_FORMAT_TERNARY.into()),
        other => bail!("unknown weight format `{other}`. Use `fp32` or `ternary`."),
    }
}

pub fn normalized_runtime_weight_format(config: &ModelConfig) -> anyhow::Result<String> {
    normalize_weight_format(&config.weight_format)
}

pub fn resolve_save_weight_format(
    train_weight_format: &str,
    save_weight_format: &str,
) -> anyhow::Result<String> {
    if save_weight_format.eq_ignore_ascii_case("same-as-train") {
        return normalize_weight_format(train_weight_format);
    }
    normalize_weight_format(save_weight_format)
}

pub fn quant_granularity(config: &ModelConfig) -> QuantGranularity {
    if config.quant_group_size == 0 {
        QuantGranularity::PerTensor
    } else {
        QuantGranularity::PerGroup(config.quant_group_size)
    }
}

pub fn require_bigram_config(config: &ModelConfig) -> anyhow::Result<()> {
    if !is_bigram_architecture(&config.architecture) {
        bail!(
            "real train/generate/quantize execution is currently implemented for architecture `{}` only; got `{}`",
            BIGRAM_ARCHITECTURE,
            config.architecture
        );
    }
    if config.vocab_size != 256 {
        bail!(
            "bigram execution currently requires `vocab_size = 256` for byte-level training/generation; got {}",
            config.vocab_size
        );
    }
    Ok(())
}

fn granularity_metadata_value(granularity: QuantGranularity) -> String {
    match granularity {
        QuantGranularity::PerTensor => "per-tensor".into(),
        QuantGranularity::PerChannel => "per-channel".into(),
        QuantGranularity::PerGroup(size) => format!("per-group:{size}"),
    }
}

fn parse_granularity_metadata(value: &str) -> anyhow::Result<QuantGranularity> {
    match value {
        "per-tensor" => Ok(QuantGranularity::PerTensor),
        "per-channel" => Ok(QuantGranularity::PerChannel),
        other => {
            if let Some(size) = other.strip_prefix("per-group:") {
                let parsed = size.parse::<usize>().with_context(|| {
                    format!("invalid serialized quantization granularity `{other}`")
                })?;
                if parsed == 0 {
                    bail!("serialized per-group quantization granularity must use group_size > 0");
                }
                Ok(QuantGranularity::PerGroup(parsed))
            } else {
                bail!("unknown serialized quantization granularity `{other}`");
            }
        }
    }
}

fn expected_scale_count(shape: &[usize], granularity: QuantGranularity) -> anyhow::Result<usize> {
    let total_elements: usize = shape.iter().product();
    Ok(match granularity {
        QuantGranularity::PerTensor => 1,
        QuantGranularity::PerChannel => shape.first().copied().ok_or_else(|| {
            anyhow::anyhow!("per-channel quantization requires a non-empty tensor shape")
        })?,
        QuantGranularity::PerGroup(group_size) => {
            if group_size == 0 {
                bail!("per-group quantization requires group_size > 0");
            }
            total_elements.div_ceil(group_size)
        }
    })
}

fn serialize_scales(scales: &[f32]) -> String {
    scales
        .iter()
        .map(|scale| format!("{scale:.9}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_scales(config: &ModelConfig) -> anyhow::Result<Vec<f32>> {
    let raw = config
        .metadata
        .get(META_BIGRAM_WEIGHT_SCALES)
        .ok_or_else(|| anyhow::anyhow!("missing `{META_BIGRAM_WEIGHT_SCALES}` metadata"))?;
    raw.split(',')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            entry
                .trim()
                .parse::<f32>()
                .with_context(|| format!("invalid serialized scale value `{entry}`"))
        })
        .collect()
}

fn parse_packed_granularity(config: &ModelConfig) -> anyhow::Result<QuantGranularity> {
    let raw = config
        .metadata
        .get(META_BIGRAM_WEIGHT_GRANULARITY)
        .map(String::as_str)
        .unwrap_or("per-tensor");
    parse_granularity_metadata(raw)
}

fn quant_config(granularity: QuantGranularity) -> QuantConfig {
    QuantConfig {
        granularity,
        use_zero_point: false,
        learnable_scale: false,
    }
}

fn dense_weight_view<'a>(
    weight: &'a Array<f32, IxDyn>,
) -> anyhow::Result<ndarray::ArrayView2<'a, f32>> {
    weight
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| anyhow::anyhow!("expected a 2D bigram weight matrix: {e}"))
}

fn quantization_metrics(
    original: &Array<f32, IxDyn>,
    quantized: &Array<f32, IxDyn>,
) -> BigramQuantizationMetrics {
    let mut mse = 0.0f32;
    let mut mean_abs_error = 0.0f32;
    let mut max_abs_error = 0.0f32;
    let mut exact_match_count = 0usize;
    let total = original.len().max(1);

    for (source, converted) in original.iter().zip(quantized.iter()) {
        let diff = (source - converted).abs();
        mse += diff * diff;
        mean_abs_error += diff;
        max_abs_error = max_abs_error.max(diff);
        if diff <= 1e-6 {
            exact_match_count += 1;
        }
    }

    BigramQuantizationMetrics {
        mse: mse / total as f32,
        mean_abs_error: mean_abs_error / total as f32,
        max_abs_error,
        exact_match_fraction: exact_match_count as f32 / total as f32,
    }
}

fn pack_tensor_from_dense(
    weight: &Array<f32, IxDyn>,
    granularity: QuantGranularity,
) -> PackedTensor {
    PackedTensor::from_ndarray(weight, &quant_config(granularity))
}

fn packed_tensor_from_obm_tensor(
    tensor: &ObmTensor,
    config: &ModelConfig,
) -> anyhow::Result<PackedTensor> {
    let granularity = parse_packed_granularity(config)?;
    let scales = parse_scales(config)?;
    let expected_scales = expected_scale_count(&tensor.shape, granularity)?;
    if scales.len() != expected_scales {
        bail!(
            "serialized packed bigram scales mismatch: expected {expected_scales}, got {}",
            scales.len()
        );
    }

    let packed = PackedTernary::from_raw_parts(
        tensor.as_packed_u64()?,
        tensor.shape.iter().product(),
    )?;
    PackedTensor::from_parts(
        packed,
        tensor.shape.clone(),
        QuantParams {
            scales,
            zero_points: Vec::new(),
            original_shape: tensor.shape.clone(),
            granularity,
        },
    )
    .map_err(|e| anyhow::anyhow!("failed to rebuild packed bigram tensor: {e}"))
}

fn serialize_bigram_model(
    config: &ModelConfig,
    weight: &Array<f32, IxDyn>,
) -> anyhow::Result<SerializedBigramModel> {
    match normalized_runtime_weight_format(config)?.as_str() {
        WEIGHT_FORMAT_FP32 => {
            let mut stored_config = config.clone();
            stored_config.metadata.remove(META_BIGRAM_WEIGHT_GRANULARITY);
            stored_config.metadata.remove(META_BIGRAM_WEIGHT_SCALES);
            Ok(SerializedBigramModel {
                config: stored_config,
                tensor: ObmTensor::from_f32(
                    BIGRAM_WEIGHT_TENSOR,
                    weight.shape().to_vec(),
                    weight
                        .as_slice()
                        .ok_or_else(|| anyhow::anyhow!("bigram weight tensor must be contiguous"))?,
                ),
                quant_metrics: None,
            })
        }
        WEIGHT_FORMAT_TERNARY => {
            let granularity = quant_granularity(config);
            let packed = pack_tensor_from_dense(weight, granularity);
            let runtime_weight = packed.to_ndarray();
            let quant_metrics = Some(quantization_metrics(weight, &runtime_weight));
            let mut stored_config = config.clone();
            stored_config.metadata.insert(
                META_BIGRAM_WEIGHT_GRANULARITY.into(),
                granularity_metadata_value(granularity),
            );
            stored_config.metadata.insert(
                META_BIGRAM_WEIGHT_SCALES.into(),
                serialize_scales(&packed.quant_params().scales),
            );
            Ok(SerializedBigramModel {
                config: stored_config,
                tensor: ObmTensor::from_packed(
                    BIGRAM_WEIGHT_TENSOR,
                    weight.shape().to_vec(),
                    packed.packed_data().raw_data(),
                ),
                quant_metrics,
            })
        }
        _ => unreachable!(),
    }
}

pub fn load_bigram_model(path: &Path) -> anyhow::Result<LoadedBigramModel> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open model file {}", path.display()))?;
    let obm = ObmFile::load(file)
        .with_context(|| format!("failed to load OBM from {}", path.display()))?;
    require_bigram_config(&obm.header.config)?;

    let tensor = obm
        .tensors
        .iter()
        .find(|tensor| tensor.name == BIGRAM_WEIGHT_TENSOR)
        .with_context(|| format!("missing tensor `{BIGRAM_WEIGHT_TENSOR}` in {}", path.display()))?;

    let (dense_weight, packed_weight) = match tensor.format {
        TensorFormat::Float32 => {
            let data = tensor
                .as_f32()
                .with_context(|| format!("tensor `{BIGRAM_WEIGHT_TENSOR}` is not stored as f32 values"))?;
            let weight = Array::from_shape_vec(IxDyn(&tensor.shape), data)
                .map_err(|e| anyhow::anyhow!("invalid `{BIGRAM_WEIGHT_TENSOR}` shape {:?}: {e}", tensor.shape))?;
            (weight, None)
        }
        TensorFormat::BitpackedTernary => {
            let packed = packed_tensor_from_obm_tensor(tensor, &obm.header.config)?;
            (packed.to_ndarray(), Some(packed))
        }
    };

    Ok(LoadedBigramModel {
        config: obm.header.config,
        dense_weight,
        packed_weight,
    })
}

pub fn load_bigram_obm(path: &Path) -> anyhow::Result<(ModelConfig, Array<f32, IxDyn>)> {
    let model = load_bigram_model(path)?;
    Ok((model.config, model.dense_weight))
}

pub fn save_bigram_obm(
    path: &Path,
    config: &ModelConfig,
    weight: &Array<f32, IxDyn>,
) -> anyhow::Result<Option<BigramQuantizationMetrics>> {
    let serialized = serialize_bigram_model(config, weight)?;
    let obm = ObmFile::new(serialized.config, vec![serialized.tensor]);
    let file = fs::File::create(path)
        .with_context(|| format!("failed to create output model {}", path.display()))?;
    obm.save(file)
        .with_context(|| format!("failed to write OBM to {}", path.display()))?;
    Ok(serialized.quant_metrics)
}

fn build_step_offsets(
    pair_count: usize,
    batch_size: usize,
    total_steps: usize,
    seed: Option<u64>,
) -> Vec<usize> {
    let steps_per_epoch = pair_count.div_ceil(batch_size);
    let mut epoch_offsets = (0..steps_per_epoch)
        .map(|step| (step * batch_size) % pair_count)
        .collect::<Vec<_>>();
    let mut offsets = Vec::with_capacity(total_steps);

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        while offsets.len() < total_steps {
            epoch_offsets.shuffle(&mut rng);
            offsets.extend(epoch_offsets.iter().copied());
        }
    } else {
        while offsets.len() < total_steps {
            offsets.extend(epoch_offsets.iter().copied());
        }
    }

    offsets.truncate(total_steps);
    offsets
}

pub fn collect_corpus_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    if path.is_file() {
        return fs::read(path)
            .with_context(|| format!("failed to read corpus file {}", path.display()));
    }
    if !path.is_dir() {
        bail!("corpus path is neither a file nor directory: {}", path.display());
    }

    let mut files = Vec::new();
    collect_files(path, &mut files)?;
    if files.is_empty() {
        bail!("corpus directory `{}` does not contain any files", path.display());
    }

    let mut corpus = Vec::new();
    for file in files {
        corpus.extend(
            fs::read(&file)
                .with_context(|| format!("failed to read corpus file {}", file.display()))?,
        );
        corpus.push(b'\n');
    }
    Ok(corpus)
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    let mut entries = fs::read_dir(path)
        .with_context(|| format!("failed to read directory {}", path.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let child = entry.path();
        if child.is_dir() {
            collect_files(&child, files)?;
        } else if child.is_file() {
            files.push(child);
        }
    }
    Ok(())
}

fn build_batch(
    tokens: &[u8],
    offset: usize,
    batch_size: usize,
    vocab_size: usize,
) -> anyhow::Result<BigramBatch> {
    let pair_count = tokens.len().saturating_sub(1);
    if pair_count == 0 {
        bail!("corpus must contain at least two bytes to build next-token pairs");
    }

    let mut input = Array::zeros(IxDyn(&[batch_size, vocab_size]));
    let mut targets = Vec::with_capacity(batch_size);
    let mut source_tokens = Vec::with_capacity(batch_size);

    for batch_idx in 0..batch_size {
        let pair_idx = (offset + batch_idx) % pair_count;
        let src = tokens[pair_idx] as usize;
        let dst = tokens[pair_idx + 1] as f32;
        input[[batch_idx, src]] = 1.0;
        targets.push(dst);
        source_tokens.push(src);
    }

    Ok(BigramBatch {
        input,
        targets: Array::from_shape_vec(IxDyn(&[batch_size]), targets)
            .map_err(|e| anyhow::anyhow!("failed to build target batch: {e}"))?,
        source_tokens,
    })
}

fn softmax_slice(logits: &[f32], temperature: f32) -> Vec<f32> {
    let safe_temperature = temperature.max(1e-6);
    let max_logit = logits
        .iter()
        .copied()
        .map(|value| value / safe_temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|value| ((value / safe_temperature) - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>().max(1e-12);
    exp.into_iter().map(|value| value / sum).collect()
}

fn argmax_slice(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn row_logits_from_dense(
    weight: &Array<f32, IxDyn>,
    source_tokens: &[usize],
) -> anyhow::Result<Array<f32, IxDyn>> {
    let weight2 = dense_weight_view(weight)?;
    let vocab_size = weight2.ncols();
    let mut rows = Vec::with_capacity(source_tokens.len() * vocab_size);

    for &token in source_tokens {
        rows.extend(weight2.row(token).iter().copied());
    }

    Array::from_shape_vec(IxDyn(&[source_tokens.len(), vocab_size]), rows)
        .map_err(|e| anyhow::anyhow!("failed to build teacher logits batch: {e}"))
}

fn teacher_target_probs(
    teacher_weight: &Array<f32, IxDyn>,
    source_tokens: &[usize],
    temperature: f32,
) -> anyhow::Result<Array<f32, IxDyn>> {
    let logits = row_logits_from_dense(teacher_weight, source_tokens)?;
    let logits2 = logits
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| anyhow::anyhow!("failed to view teacher logits as matrix: {e}"))?;
    let mut probs = Vec::with_capacity(logits.len());
    for row in logits2.rows() {
        probs.extend(softmax_slice(row.as_slice().unwrap_or(&[]), temperature));
    }
    Array::from_shape_vec(IxDyn(&[source_tokens.len(), logits2.ncols()]), probs)
        .map_err(|e| anyhow::anyhow!("failed to build teacher probability batch: {e}"))
}

pub fn evaluate_bigram_weight(
    weight: &Array<f32, IxDyn>,
    corpus: &[u8],
) -> anyhow::Result<BigramEvalMetrics> {
    let pair_count = corpus.len().saturating_sub(1);
    if pair_count == 0 {
        bail!("evaluation corpus must contain at least two bytes");
    }

    let weight2 = dense_weight_view(weight)?;
    let mut loss_sum = 0.0f32;
    let mut correct = 0usize;

    for pair_idx in 0..pair_count {
        let src = corpus[pair_idx] as usize;
        let dst = corpus[pair_idx + 1] as usize;
        let row = weight2.row(src);
        let row_slice = row
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("bigram rows must be contiguous"))?;
        let probs = softmax_slice(row_slice, 1.0);
        loss_sum -= probs[dst].max(1e-12).ln();
        if argmax_slice(row_slice) == dst {
            correct += 1;
        }
    }

    let avg_loss = loss_sum / pair_count as f32;
    Ok(BigramEvalMetrics {
        pair_count,
        avg_loss,
        perplexity: avg_loss.exp(),
        accuracy: correct as f32 / pair_count as f32,
    })
}

pub fn compare_bigram_weights(
    student_weight: &Array<f32, IxDyn>,
    teacher_weight: &Array<f32, IxDyn>,
    corpus: &[u8],
) -> anyhow::Result<BigramRetentionMetrics> {
    let pair_count = corpus.len().saturating_sub(1);
    if pair_count == 0 {
        bail!("comparison corpus must contain at least two bytes");
    }

    let student2 = dense_weight_view(student_weight)?;
    let teacher2 = dense_weight_view(teacher_weight)?;
    let mut matching_top1 = 0usize;
    let mut kl_sum = 0.0f32;

    for pair_idx in 0..pair_count {
        let src = corpus[pair_idx] as usize;
        let student_row = student2.row(src);
        let teacher_row = teacher2.row(src);
        let student_slice = student_row
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("student bigram rows must be contiguous"))?;
        let teacher_slice = teacher_row
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("teacher bigram rows must be contiguous"))?;
        let student_probs = softmax_slice(student_slice, 1.0);
        let teacher_probs = softmax_slice(teacher_slice, 1.0);

        if argmax_slice(student_slice) == argmax_slice(teacher_slice) {
            matching_top1 += 1;
        }

        for (teacher_prob, student_prob) in teacher_probs.iter().zip(student_probs.iter()) {
            if *teacher_prob > 0.0 {
                kl_sum += teacher_prob * (teacher_prob / student_prob.max(1e-12)).ln();
            }
        }
    }

    Ok(BigramRetentionMetrics {
        pair_count,
        top1_agreement: matching_top1 as f32 / pair_count as f32,
        avg_kl_divergence: kl_sum / pair_count as f32,
    })
}

pub struct BigramTrainArgs<'a> {
    pub corpus_path: &'a Path,
    pub output_dir: &'a Path,
    pub resume_path: Option<&'a Path>,
    pub teacher_model_path: Option<&'a Path>,
    pub eval_path: Option<&'a Path>,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub seed: Option<u64>,
    pub train_weight_format: &'a str,
    pub save_weight_format: &'a str,
    pub distill_alpha: f32,
    pub distill_temperature: f32,
}

pub fn train_bigram(
    config: &ModelConfig,
    args: BigramTrainArgs<'_>,
) -> anyhow::Result<PathBuf> {
    require_bigram_config(config)?;
    let train_weight_format = normalize_weight_format(args.train_weight_format)?;
    let save_weight_format =
        resolve_save_weight_format(&train_weight_format, args.save_weight_format)?;

    let corpus = collect_corpus_bytes(args.corpus_path)?;
    if corpus.len() < 2 {
        bail!(
            "corpus at {} must contain at least two bytes",
            args.corpus_path.display()
        );
    }

    let distill_alpha = args.distill_alpha.clamp(0.0, 1.0);
    let teacher_model = if let Some(path) = args.teacher_model_path {
        let teacher = load_bigram_model(path)?;
        require_bigram_config(&teacher.config)?;
        if teacher.dense_weight.shape() != [config.vocab_size, config.vocab_size] {
            bail!(
                "teacher bigram weight shape {:?} does not match expected [{}, {}]",
                teacher.dense_weight.shape(),
                config.vocab_size,
                config.vocab_size
            );
        }
        Some(teacher)
    } else {
        if distill_alpha > 0.0 {
            bail!("--distill-alpha requires --teacher-model for bigram training");
        }
        None
    };

    let eval_corpus = args
        .eval_path
        .map(collect_corpus_bytes)
        .transpose()?;

    let mut weight = if let Some(resume_path) = args.resume_path {
        let resume = load_bigram_model(resume_path)?;
        require_bigram_config(&resume.config)?;
        resume.dense_weight
    } else {
        Array::zeros(IxDyn(&[config.vocab_size, config.vocab_size]))
    };

    let mut param = Parameter::new(BIGRAM_WEIGHT_TENSOR, weight.clone());
    let mut optimizer = AdamW::new(args.lr, 0.9, 0.999, 1e-8, args.weight_decay);
    let mut trainer = Trainer::new(TrainConfig {
        grad_clip_norm: args.max_grad_norm,
        use_qat: train_weight_format == WEIGHT_FORMAT_TERNARY,
        ..TrainConfig::default()
    });

    let pair_count = corpus.len() - 1;
    let steps_per_epoch = pair_count.div_ceil(args.batch_size);
    let total_steps = if args.max_steps > 0 {
        args.max_steps
    } else {
        steps_per_epoch.saturating_mul(args.epochs.max(1))
    };
    if total_steps == 0 {
        bail!("training resolved to zero steps");
    }

    let decay_steps = total_steps.saturating_sub(args.warmup_steps).max(1);
    let scheduler = WarmupScheduler::new(
        CosineScheduler::new(args.lr, args.lr * 0.1, decay_steps),
        args.warmup_steps,
    );
    let step_offsets = build_step_offsets(pair_count, args.batch_size, total_steps, args.seed);

    fs::create_dir_all(args.output_dir)
        .with_context(|| format!("failed to create output dir {}", args.output_dir.display()))?;
    let config_path = args.output_dir.join("model_config.json");
    let final_model_path = args.output_dir.join("model.obm");

    log::info!(
        "Running real bigram training with train_weight_format={} save_weight_format={} total_steps={} seed={:?}",
        train_weight_format,
        save_weight_format,
        total_steps,
        args.seed
    );
    if teacher_model.is_some() {
        log::info!(
            "Teacher distillation enabled with alpha={:.3} temperature={:.3}",
            distill_alpha,
            args.distill_temperature
        );
    }

    for (step, batch_offset) in step_offsets.into_iter().enumerate() {
        optimizer.set_lr(scheduler.get_lr(step));
        let batch = build_batch(&corpus, batch_offset, args.batch_size, config.vocab_size)?;
        let teacher_probs = if teacher_model.is_some() && distill_alpha > 0.0 {
            Some(teacher_target_probs(
                &teacher_model.as_ref().unwrap().dense_weight,
                &batch.source_tokens,
                args.distill_temperature,
            )?)
        } else {
            None
        };

        let result = trainer
            .train_step(&mut [&mut param], &mut optimizer, |tape, vars| {
                let input_var = Variable::new(batch.input.clone(), false, tape);
                let weight_var = if train_weight_format == WEIGHT_FORMAT_TERNARY {
                    ops::quantize_unit_ste(&vars[0], 0.5)
                } else {
                    vars[0].clone()
                };
                let logits = ops::matmul(&input_var, &weight_var);
                let hard_loss = ops::cross_entropy_loss(&logits, &batch.targets);

                if let Some(teacher_probs) = &teacher_probs {
                    let distill_loss = ops::soft_target_cross_entropy_loss(
                        &logits,
                        teacher_probs,
                        args.distill_temperature,
                    );
                    if distill_alpha <= 0.0 {
                        Ok(hard_loss)
                    } else if distill_alpha >= 1.0 {
                        Ok(distill_loss)
                    } else {
                        let hard_scale = Variable::detached(Array::from_elem(
                            IxDyn(&[1]),
                            1.0 - distill_alpha,
                        ));
                        let distill_scale = Variable::detached(Array::from_elem(
                            IxDyn(&[1]),
                            distill_alpha,
                        ));
                        Ok(ops::add(
                            &ops::mul(&hard_loss, &hard_scale),
                            &ops::mul(&distill_loss, &distill_scale),
                        ))
                    }
                } else {
                    Ok(hard_loss)
                }
            })
            .map_err(|e| anyhow::anyhow!("training step {step} failed: {e}"))?;

        if args.log_every > 0 && ((step + 1) % args.log_every == 0 || step + 1 == total_steps) {
            log::info!(
                "step {}/{} loss={:.4} grad_norm={:.4} lr={:.6}",
                step + 1,
                total_steps,
                result.loss,
                result.grad_norm,
                optimizer.lr()
            );
        }

        if args.save_every > 0 && ((step + 1) % args.save_every == 0) {
            let mut checkpoint_config = config.clone();
            checkpoint_config.weight_format = save_weight_format.clone();
            checkpoint_config.training_weight_format = train_weight_format.clone();
            checkpoint_config
                .metadata
                .insert("checkpoint_step".into(), (step + 1).to_string());
            if let Some(seed) = args.seed {
                checkpoint_config
                    .metadata
                    .insert("training_seed".into(), seed.to_string());
            }
            let checkpoint_path = args
                .output_dir
                .join(format!("checkpoint-step-{:06}.obm", step + 1));
            let checkpoint_metrics = save_bigram_obm(&checkpoint_path, &checkpoint_config, &param.data)?;
            if let Some(metrics) = checkpoint_metrics {
                log::info!(
                    "checkpoint {} quantization mse={:.6} mean_abs_error={:.6} exact_match_fraction={:.4}",
                    checkpoint_path.display(),
                    metrics.mse,
                    metrics.mean_abs_error,
                    metrics.exact_match_fraction
                );
            }
        }
    }

    weight = param.data.clone();
    let mut final_config = config.clone();
    final_config.weight_format = save_weight_format;
    final_config.training_weight_format = train_weight_format;
    final_config
        .metadata
        .insert("trained_steps".into(), total_steps.to_string());
    if let Some(seed) = args.seed {
        final_config
            .metadata
            .insert("training_seed".into(), seed.to_string());
    }
    if let Some(path) = args.teacher_model_path {
        final_config
            .metadata
            .insert("teacher_model".into(), path.display().to_string());
        final_config
            .metadata
            .insert("distill_alpha".into(), format!("{distill_alpha:.4}"));
        final_config.metadata.insert(
            "distill_temperature".into(),
            format!("{:.4}", args.distill_temperature),
        );
    }
    final_config
        .save_json(fs::File::create(&config_path)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    let final_quant_metrics = save_bigram_obm(&final_model_path, &final_config, &weight)?;
    if let Some(metrics) = final_quant_metrics {
        log::info!(
            "final model quantization mse={:.6} mean_abs_error={:.6} max_abs_error={:.6} exact_match_fraction={:.4}",
            metrics.mse,
            metrics.mean_abs_error,
            metrics.max_abs_error,
            metrics.exact_match_fraction
        );
    }

    let eval_tokens = eval_corpus.as_deref().unwrap_or(&corpus);
    let deployed_model = load_bigram_model(&final_model_path)?;
    let deployed_eval = evaluate_bigram_weight(&deployed_model.dense_weight, eval_tokens)?;
    log::info!(
        "deployed bigram eval pairs={} loss={:.4} ppl={:.4} accuracy={:.4}",
        deployed_eval.pair_count,
        deployed_eval.avg_loss,
        deployed_eval.perplexity,
        deployed_eval.accuracy
    );
    if let Some(teacher) = &teacher_model {
        let retention = compare_bigram_weights(
            &deployed_model.dense_weight,
            &teacher.dense_weight,
            eval_tokens,
        )?;
        log::info!(
            "teacher retention pairs={} top1_agreement={:.4} avg_kl_divergence={:.6}",
            retention.pair_count,
            retention.top1_agreement,
            retention.avg_kl_divergence
        );
    }

    Ok(final_model_path)
}

pub fn generate_bigram_loaded(
    model: &LoadedBigramModel,
    prompt: &str,
    max_tokens: usize,
    sampling: SamplingConfig,
    stream: bool,
    backend: &dyn ComputeBackend,
) -> anyhow::Result<String> {
    let mut sampler = Sampler::new(sampling);
    let mut bytes = prompt.as_bytes().to_vec();
    let mut generated = Vec::new();
    let dense_transposed = model
        .dense_weight
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| anyhow::anyhow!("expected a 2D bigram weight matrix: {e}"))?
        .reversed_axes()
        .to_owned()
        .into_dyn();

    if bytes.is_empty() {
        bytes.push(b'\n');
    }

    for _ in 0..max_tokens {
        let token = *bytes
            .last()
            .ok_or_else(|| anyhow::anyhow!("prompt resolution failed"))? as usize;

        let logits = if let Some(packed) = &model.packed_weight {
            let mut one_hot = Array::zeros(IxDyn(&[1, model.config.vocab_size]));
            one_hot[[0, token]] = 1.0;
            backend.packed_matmul_dense_left_transposed(&one_hot, packed)?
        } else {
            let mut one_hot = Array::zeros(IxDyn(&[1, model.config.vocab_size]));
            one_hot[[0, token]] = 1.0;
            backend.dense_matmul(&one_hot, &dense_transposed)?
        };

        let token_id = sampler.sample(
            &logits
                .into_shape_with_order(IxDyn(&[model.config.vocab_size]))
                .map_err(|e| anyhow::anyhow!("failed to flatten logits: {e}"))?,
        ) as u8;
        bytes.push(token_id);
        generated.push(token_id);

        if stream {
            print!("{}", String::from_utf8_lossy(&[token_id]));
        }
    }

    if stream {
        println!();
    }

    Ok(format!(
        "{}{}",
        prompt,
        String::from_utf8_lossy(&generated)
    ))
}

pub fn quantize_bigram_model(
    input_path: &Path,
    output_path: &Path,
    target_weight_format: &str,
    granularity: QuantGranularity,
) -> anyhow::Result<Option<BigramQuantizationMetrics>> {
    let model = load_bigram_model(input_path)?;
    let mut config = model.config.clone();
    let target_weight_format = normalize_weight_format(target_weight_format)?;

    config.weight_format = target_weight_format;
    config.quant_group_size = match granularity {
        QuantGranularity::PerTensor | QuantGranularity::PerChannel => 0,
        QuantGranularity::PerGroup(group_size) => group_size,
    };

    if matches!(granularity, QuantGranularity::PerChannel) {
        config.metadata.insert(
            META_BIGRAM_WEIGHT_GRANULARITY.into(),
            granularity_metadata_value(granularity),
        );
    }

    save_bigram_obm(output_path, &config, &model.dense_weight)
}
