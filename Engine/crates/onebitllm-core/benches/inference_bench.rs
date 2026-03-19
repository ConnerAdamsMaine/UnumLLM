use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array, IxDyn};

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for &vocab_size in &[1000, 10000, 32000] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| ((i as f32) * 0.01).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("softmax_1d", vocab_size),
            &logits,
            |b, logits| {
                b.iter(|| {
                    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    let _probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();
                    black_box(_probs)
                })
            },
        );
    }

    group.finish();
}

fn bench_kv_cache_update(c: &mut Criterion) {
    use onebitllm_core::infer::LayerKvCache;

    let mut group = c.benchmark_group("kv_cache");

    for &seq_len in &[1, 16, 64] {
        let batch = 1;
        let heads = 8;
        let dim = 64;
        let max_seq = 2048;

        group.bench_with_input(
            BenchmarkId::new("update", seq_len),
            &seq_len,
            |b, &seq_len| {
                let mut cache = LayerKvCache::new(batch, heads, dim, max_seq);
                let new_k = Array::from_elem(IxDyn(&[batch, heads, seq_len, dim]), 1.0f32);
                let new_v = Array::from_elem(IxDyn(&[batch, heads, seq_len, dim]), 1.0f32);

                b.iter(|| {
                    cache.clear();
                    cache.update(black_box(&new_k), black_box(&new_v)).unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_sampler(c: &mut Criterion) {
    use onebitllm_core::infer::{Sampler, SamplingConfig};

    let mut group = c.benchmark_group("sampler");

    for &vocab_size in &[1000, 32000] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| ((i as f32) * 0.01).sin()).collect();
        let logits_arr = Array::from_shape_vec(IxDyn(&[vocab_size]), logits).unwrap();

        group.bench_with_input(
            BenchmarkId::new("greedy", vocab_size),
            &logits_arr,
            |b, logits| {
                let config = SamplingConfig {
                    temperature: 0.0,
                    seed: Some(42),
                    ..Default::default()
                };
                let mut sampler = Sampler::new(config);
                b.iter(|| sampler.sample(black_box(logits)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("top_k_50", vocab_size),
            &logits_arr,
            |b, logits| {
                let config = SamplingConfig {
                    temperature: 0.8,
                    top_k: Some(50),
                    seed: Some(42),
                    ..Default::default()
                };
                let mut sampler = Sampler::new(config);
                b.iter(|| sampler.sample(black_box(logits)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_softmax, bench_kv_cache_update, bench_sampler);
criterion_main!(benches);
