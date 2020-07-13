use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use memmem::{Searcher, TwoWaySearcher};
use std::str;
use strstr::avx2::*;

fn criterion_benchmark(c: &mut Criterion) {
    let haystack = include_bytes!("../data/haystack");
    let needle = include_bytes!("../data/needle");

    let sizes = [1, 5, 10, 20, 50, 100, 1000];

    for (i, &size) in sizes.iter().enumerate() {
        let mut group = c.benchmark_group(format!("{}-byte needle", size));
        let needle = &needle[..size];

        for &size in &sizes[i..] {
            let parameter = &format!("{}-byte haystack", size);
            let haystack = &haystack[..size];

            group.bench_with_input(
                BenchmarkId::new("String::find", parameter),
                &size,
                |b, _| {
                    let haystack = str::from_utf8(haystack).unwrap();
                    let needle = str::from_utf8(needle).unwrap();
                    b.iter(|| haystack.find(needle));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("TwoWaySearcher::search_in", parameter),
                &size,
                |b, _| {
                    let searcher = TwoWaySearcher::new(needle);
                    b.iter(|| black_box(searcher.search_in(haystack)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("twoway::find_bytes", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(twoway::find_bytes(haystack, needle)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_original", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(unsafe { strstr_avx2_original(haystack, needle) }));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_rust_simple", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(unsafe { strstr_avx2_rust_simple(haystack, needle) }));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_rust_simple_2", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(unsafe { strstr_avx2_rust_simple_2(haystack, needle) }));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_rust_fast", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(unsafe { strstr_avx2_rust_fast(haystack, needle) }));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_rust_fast_2", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(strstr_avx2_rust_fast_2(haystack, needle)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("strstr_avx2_rust_aligned", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(unsafe { strstr_avx2_rust_aligned(haystack, needle) }));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("StrStrAVX2Searcher::search_in", parameter),
                &size,
                |b, _| {
                    let searcher = StrStrAVX2Searcher::new(needle);
                    b.iter(|| black_box(searcher.search_in(haystack)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("DynamicAvx2Searcher::search_in", parameter),
                &size,
                |b, _| {
                    let searcher = DynamicAvx2Searcher::new(needle.to_owned().into_boxed_slice());
                    b.iter(|| black_box(searcher.search_in(haystack)));
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
