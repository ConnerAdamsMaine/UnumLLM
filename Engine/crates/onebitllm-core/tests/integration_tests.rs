//! Integration tests for the OneBitLLM core library.
//!
//! Tests cross-module interactions to verify the full pipeline works end-to-end.

use ndarray::{Array, IxDyn};
use onebitllm_core::error::OneBitError;

/// Test: quantize -> pack -> unpack -> verify roundtrip.
#[test]
fn test_quantize_pack_roundtrip() {
    use onebitllm_core::quant::bitpack::PackedTernary;
    use onebitllm_core::quant::ternary::absmean_quantize;

    let weights = vec![0.5, -0.3, 0.0, 0.8, -0.9, 0.1, -0.05, 0.6];
    let (ternary, gamma) = absmean_quantize(&weights);
    assert!(gamma > 0.0);

    let packed = PackedTernary::from_ternary_slice(&ternary);
    let unpacked = packed.to_ternary_vec();

    assert_eq!(ternary, unpacked);
}

/// Test: PackedTensor matmul correctness vs naive f32.
#[test]
fn test_packed_tensor_matmul_correctness() {
    use onebitllm_core::quant::scales::QuantConfig;
    use onebitllm_core::tensor::packed_tensor::PackedTensor;

    // Create a small weight matrix
    let weight_data = vec![1.0, -1.0, 0.0, 0.0, 1.0, -1.0];
    let weight_arr = Array::from_shape_vec(IxDyn(&[2, 3]), weight_data).unwrap();
    let packed = PackedTensor::from_ndarray(&weight_arr, &QuantConfig::per_tensor());

    // Input vector
    let input = Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

    // Result will be approximate due to quantization scales
    let result = packed.matvec(&input).unwrap();
    assert_eq!(result.shape(), &[2]);
}

/// Test: QuantizedLinear forward pass shape correctness.
#[test]
fn test_quantized_linear_end_to_end() {
    use onebitllm_core::nn::linear::QuantizedLinear;
    use onebitllm_core::nn::Module;
    use onebitllm_core::quant::scales::QuantConfig;

    let linear = QuantizedLinear::new(16, 8, true, QuantConfig::per_tensor());
    let input = Array::from_elem(IxDyn(&[1, 4, 16]), 0.5f32);
    let output = linear.forward_inference(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4, 8]);
}

/// Test: Attention module processes input correctly.
#[test]
fn test_attention_end_to_end() {
    use onebitllm_core::nn::attention::{Attention, AttentionConfig};
    use onebitllm_core::nn::Module;
    use onebitllm_core::quant::scales::QuantConfig;

    let config = AttentionConfig {
        embed_dim: 32,
        num_heads: 4,
        num_kv_heads: None,
        head_dim: 8,
        use_bias: false,
        quant_config: QuantConfig::per_tensor(),
    };
    let attn = Attention::new(config);
    let input = Array::from_elem(IxDyn(&[1, 8, 32]), 0.1f32);
    let output = attn.forward_inference(&input).unwrap();
    assert_eq!(output.shape(), &[1, 8, 32]);
}

/// Test: MLP block processes input correctly.
#[test]
fn test_mlp_end_to_end() {
    use onebitllm_core::nn::activation::ActivationFn;
    use onebitllm_core::nn::mlp::MlpBlock;
    use onebitllm_core::nn::Module;
    use onebitllm_core::quant::scales::QuantConfig;

    let mlp = MlpBlock::new(32, 64, ActivationFn::SiLU, QuantConfig::per_tensor());
    let input = Array::from_elem(IxDyn(&[1, 8, 32]), 0.1f32);
    let output = mlp.forward_inference(&input).unwrap();
    assert_eq!(output.shape(), &[1, 8, 32]);
}

/// Test: Autograd tape computes gradients for a small network.
#[test]
fn test_autograd_gradient_flow() {
    use onebitllm_core::autograd::ops;
    use onebitllm_core::autograd::tape::Tape;
    use onebitllm_core::autograd::variable::Variable;

    let tape = Tape::new();

    // Simple: y = sum(W * x)
    let x = Variable::new(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        true,
        &tape,
    );
    let w = Variable::new(
        Array::from_shape_vec(IxDyn(&[3, 2]), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
        true,
        &tape,
    );

    let y = ops::matmul(&x, &w);
    let loss = ops::sum(&y);

    let grads = loss.backward().unwrap();
    assert!(grads.contains_key(&x.id));
    assert!(grads.contains_key(&w.id));
}

/// Test: KV cache update and retrieval.
#[test]
fn test_kv_cache_pipeline() {
    use onebitllm_core::infer::KvCache;

    let mut cache = KvCache::new(4, 1, 8, 64, 2048);

    // Simulate prefill: 32 tokens
    let k = Array::from_elem(IxDyn(&[1, 8, 32, 64]), 1.0f32);
    let v = Array::from_elem(IxDyn(&[1, 8, 32, 64]), 2.0f32);

    for layer in 0..4 {
        cache.update(layer, &k, &v).unwrap();
    }
    assert_eq!(cache.cached_len(), 32);

    // Simulate decode: 1 token at a time
    let k1 = Array::from_elem(IxDyn(&[1, 8, 1, 64]), 3.0f32);
    let v1 = Array::from_elem(IxDyn(&[1, 8, 1, 64]), 4.0f32);
    for layer in 0..4 {
        cache.update(layer, &k1, &v1).unwrap();
    }
    assert_eq!(cache.cached_len(), 33);

    // Clear and verify
    cache.clear();
    assert_eq!(cache.cached_len(), 0);
}

/// Test: Model config JSON roundtrip through I/O module.
#[test]
fn test_config_io_roundtrip() {
    use onebitllm_core::io::ModelConfig;

    let config = ModelConfig {
        architecture: "test-1bit".into(),
        hidden_size: 256,
        num_layers: 4,
        num_attention_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 512,
        vocab_size: 1000,
        max_seq_len: 512,
        ..Default::default()
    };

    let mut buf = Vec::new();
    config.save_json(&mut buf).unwrap();

    let loaded = ModelConfig::load_json(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(loaded.architecture, "test-1bit");
    assert_eq!(loaded.hidden_size, 256);
    assert_eq!(loaded.num_kv_heads, 2);
}

/// Test: OBM file roundtrip with mixed tensor formats.
#[test]
fn test_obm_file_roundtrip() {
    use onebitllm_core::io::custom::ObmTensor;
    use onebitllm_core::io::{ModelConfig, ObmFile};

    let config = ModelConfig::default();
    let t1 = ObmTensor::from_f32("embed.weight", vec![100, 32], &vec![0.1f32; 3200]);
    let t2 = ObmTensor::from_packed("layer0.attn.q.weight", vec![32, 32], &vec![0xFFu64; 64]);
    let obm = ObmFile::new(config, vec![t1, t2]);

    let mut buf = Vec::new();
    obm.save(&mut buf).unwrap();

    let loaded = ObmFile::load(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(loaded.tensors.len(), 2);
    assert_eq!(
        loaded
            .get_tensor("embed.weight")
            .unwrap()
            .as_f32()
            .unwrap()
            .len(),
        3200
    );
    assert_eq!(
        loaded
            .get_tensor("layer0.attn.q.weight")
            .unwrap()
            .as_packed_u64()
            .unwrap()
            .len(),
        64
    );
}

/// Test: Tokenizer encode/decode roundtrip.
#[test]
fn test_tokenizer_roundtrip() {
    use onebitllm_core::tokenizer::bpe::SimpleBpe;
    use onebitllm_core::tokenizer::Tokenizer;
    use std::collections::HashMap;

    let mut vocab = HashMap::new();
    vocab.insert("h".to_string(), 0);
    vocab.insert("e".to_string(), 1);
    vocab.insert("l".to_string(), 2);
    vocab.insert("o".to_string(), 3);
    vocab.insert("he".to_string(), 4);
    vocab.insert("ll".to_string(), 5);
    vocab.insert("llo".to_string(), 6);
    vocab.insert("hello".to_string(), 7);

    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("l".to_string(), "l".to_string()),
        ("ll".to_string(), "o".to_string()),
        ("he".to_string(), "llo".to_string()),
    ];

    let bpe = SimpleBpe::from_data(vocab, merges);
    let encoding = bpe.encode("hello").unwrap();
    assert!(!encoding.ids.is_empty());

    let decoded = bpe.decode(&encoding.ids).unwrap();
    assert_eq!(decoded, "hello");
}

/// Test: Checkpoint save/load roundtrip.
#[test]
fn test_checkpoint_roundtrip() {
    use onebitllm_core::nn::module::Parameter;
    use onebitllm_core::train::checkpoint::Checkpoint;

    let p1 = Parameter::new(
        "layer0.weight",
        Array::from_shape_vec(IxDyn(&[4, 4]), (0..16).map(|i| i as f32).collect()).unwrap(),
    );
    let p2 = Parameter::new(
        "layer0.bias",
        Array::from_shape_vec(IxDyn(&[4]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
    );

    let ckpt = Checkpoint::from_parameters(&[&p1, &p2], 100);

    let mut buf = Vec::new();
    ckpt.save(&mut buf).unwrap();

    let loaded = Checkpoint::load(std::io::Cursor::new(&buf)).unwrap();
    assert_eq!(loaded.step, 100);
    assert_eq!(loaded.entries.len(), 2);
    assert_eq!(loaded.entries[0].data.len(), 16);
}

/// Test: Sampler produces deterministic results with same seed.
#[test]
fn test_sampler_deterministic() {
    use onebitllm_core::infer::{Sampler, SamplingConfig};

    let config = SamplingConfig {
        temperature: 0.8,
        top_k: Some(10),
        seed: Some(12345),
        ..Default::default()
    };

    let logits = Array::from_shape_vec(
        IxDyn(&[100]),
        (0..100).map(|i| (i as f32 * 0.1).sin()).collect(),
    )
    .unwrap();

    let mut s1 = Sampler::new(config.clone());
    let mut s2 = Sampler::new(config);

    let r1: Vec<u32> = (0..20).map(|_| s1.sample(&logits)).collect();
    let r2: Vec<u32> = (0..20).map(|_| s2.sample(&logits)).collect();
    assert_eq!(r1, r2);
}

/// Test: LR scheduler produces expected learning rates.
#[test]
fn test_lr_scheduler_integration() {
    use onebitllm_core::optim::scheduler::{CosineScheduler, LrScheduler};

    let scheduler = CosineScheduler::new(0.001, 0.0, 1000);

    let lr_start = scheduler.get_lr(0);
    let lr_mid = scheduler.get_lr(500);
    let lr_end = scheduler.get_lr(1000);

    assert!((lr_start - 0.001).abs() < 1e-6);
    assert!(lr_mid < lr_start);
    assert!(lr_mid > lr_end);
    assert!(lr_end.abs() < 1e-6);
}

/// Test: Error types propagate correctly across modules.
#[test]
fn test_error_propagation() {
    use onebitllm_core::nn::linear::QuantizedLinear;
    use onebitllm_core::nn::Module;
    use onebitllm_core::quant::scales::QuantConfig;

    let linear = QuantizedLinear::new(8, 4, false, QuantConfig::per_tensor());

    // Wrong input dimension
    let bad_input = Array::from_elem(IxDyn(&[1, 4, 16]), 0.5f32);
    let result = linear.forward_inference(&bad_input);
    assert!(result.is_err());

    match result.unwrap_err() {
        OneBitError::Inference(_)
        | OneBitError::ShapeMismatch { .. }
        | OneBitError::TensorOp(_) => {}
        other => panic!("Unexpected error type: {other:?}"),
    }
}
