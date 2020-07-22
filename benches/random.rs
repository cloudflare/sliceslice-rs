#![allow(deprecated)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use memmem::{Searcher, TwoWaySearcher};
use sliceslice::x86::avx2::{deprecated::*, *};

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

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
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
                        b.iter(|| black_box(unsafe { strstr_avx2_rust(haystack, needle) }));
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("StrStrAVX2Searcher::search_in", parameter),
                    &size,
                    |b, _| {
                        let searcher = unsafe { StrStrAVX2Searcher::new(needle) };
                        b.iter(|| black_box(unsafe { searcher.search_in(haystack) }));
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("DynamicAvx2Searcher::search_in", parameter),
                    &size,
                    |b, _| {
                        let searcher = unsafe {
                            DynamicAvx2Searcher::new(needle.to_owned().into_boxed_slice())
                        };
                        b.iter(|| black_box(unsafe { searcher.search_in(haystack) }));
                    },
                );
            }
        }

        group.finish();
    }
}

criterion_group!(benches, search);
criterion_main!(benches);
