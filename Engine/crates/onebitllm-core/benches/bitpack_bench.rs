use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use onebitllm_core::quant::bitpack::PackedTernary;
use onebitllm_core::quant::ternary::{TernaryWeight, absmean_quantize};

fn bench_pack_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitpack");

    for &size in &[64, 256, 1024, 4096] {
        let weights: Vec<TernaryWeight> = (0..size)
            .map(|i| match i % 3 {
                0 => TernaryWeight::Zero,
                1 => TernaryWeight::Pos,
                _ => TernaryWeight::Neg,
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("pack", size),
            &weights,
            |b, weights| {
                b.iter(|| PackedTernary::from_ternary_slice(black_box(weights)))
            },
        );

        let packed = PackedTernary::from_ternary_slice(&weights);
        group.bench_with_input(
            BenchmarkId::new("unpack", size),
            &packed,
            |b, packed| {
                b.iter(|| black_box(packed).to_ternary_vec())
            },
        );
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for &size in &[64, 256, 1024, 4096] {
        let f32_weights: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1) - 0.5).collect();
        let (packed, _gamma) = PackedTernary::from_f32_slice(&f32_weights);
        let activations: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("packed_dot_f32", size),
            &size,
            |b, _| {
                b.iter(|| packed.dot_f32(black_box(&activations), 1.0))
            },
        );
    }

    group.finish();
}

fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize");

    for &size in &[256, 1024, 4096, 16384] {
        let weights: Vec<f32> = (0..size)
            .map(|i| ((i as f32) * 0.123).sin())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("absmean_quantize", size),
            &weights,
            |b, weights| {
                b.iter(|| absmean_quantize(black_box(weights)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pack_unpack, bench_dot_product, bench_quantization);
criterion_main!(benches);
