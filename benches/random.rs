use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use memmem::{Searcher, TwoWaySearcher};
use strstr::avx2::*;

fn search(c: &mut Criterion) {
    let haystack = include_str!("../data/haystack");
    let needle = include_str!("../data/needle");

    let sizes = [1, 5, 10, 20, 50, 100, 1000];

    for (i, &size) in sizes.iter().enumerate() {
        let mut group = c.benchmark_group(format!("needle_{}_bytes", size));
        let needle = &needle[..size];

        for &size in &sizes[i..] {
            let parameter = &format!("haystack_{}_bytes", size);
            let haystack = &haystack[..size];

            group.bench_with_input(
                BenchmarkId::new("String::find", parameter),
                &size,
                |b, _| {
                    b.iter(|| haystack.find(needle));
                },
            );

            let haystack = haystack.as_bytes();
            let needle = needle.as_bytes();

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
                BenchmarkId::new("strstr_avx2_rust", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(strstr_avx2_rust(haystack, needle)));
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
                    let searcher = unsafe { DynamicAvx2Searcher::new(needle) };
                    b.iter(|| black_box(searcher.search_in(haystack)));
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, search);
criterion_main!(benches);
