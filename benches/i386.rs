#![allow(deprecated)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use memmem::{Searcher, TwoWaySearcher};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use sliceslice::x86::avx2::{deprecated::*, *};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
};

fn search_short_haystack(c: &mut Criterion) {
    let mut needles = BufReader::new(File::open("data/words.txt").unwrap())
        .lines()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    needles.sort_unstable_by_key(|needle| needle.len());
    let needles = needles.iter().map(String::as_str).collect::<Vec<_>>();

    let mut group = c.benchmark_group("short_haystack");

    group.bench_function("String::find", |b| {
        b.iter(|| {
            for (i, needle) in needles.iter().enumerate() {
                for haystack in &needles[i..] {
                    black_box(haystack.find(needle));
                }
            }
        });
    });

    group.bench_function("TwoWaySearcher::search_in", |b| {
        let searchers = needles
            .iter()
            .map(|needle| TwoWaySearcher::new(needle.as_bytes()))
            .collect::<Vec<_>>();

        b.iter(|| {
            for (i, searcher) in searchers.iter().enumerate() {
                for haystack in &needles[i..] {
                    black_box(searcher.search_in(haystack.as_bytes()));
                }
            }
        });
    });

    group.bench_function("twoway::find_bytes", |b| {
        b.iter(|| {
            for (i, needle) in needles.iter().enumerate() {
                for haystack in &needles[i..] {
                    black_box(twoway::find_bytes(haystack.as_bytes(), needle.as_bytes()));
                }
            }
        });
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        group.bench_function("strstr_avx2_original", |b| {
            b.iter(|| {
                for (i, needle) in needles.iter().enumerate() {
                    for haystack in &needles[i..] {
                        black_box(unsafe {
                            strstr_avx2_original(haystack.as_bytes(), needle.as_bytes())
                        });
                    }
                }
            });
        });

        group.bench_function("strstr_avx2_rust", |b| {
            b.iter(|| {
                for (i, needle) in needles.iter().enumerate() {
                    for haystack in &needles[i..] {
                        black_box(unsafe {
                            strstr_avx2_rust(haystack.as_bytes(), needle.as_bytes())
                        });
                    }
                }
            });
        });

        group.bench_function("DynamicAvx2Searcher::search_in", |b| {
            let searchers = needles
                .iter()
                .map(|&needle| unsafe {
                    DynamicAvx2Searcher::new(needle.as_bytes().to_owned().into_boxed_slice())
                })
                .collect::<Vec<_>>();

            b.iter(|| {
                for (i, searcher) in searchers.iter().enumerate() {
                    for haystack in &needles[i..] {
                        black_box(unsafe { searcher.search_in(haystack.as_bytes()) });
                    }
                }
            });
        });
    }

    group.finish();
}

fn search_long_haystack(c: &mut Criterion) {
    let haystack = fs::read("data/i386.txt").unwrap();
    let haystack = String::from_utf8_lossy(&haystack);

    let needles = BufReader::new(File::open("data/words.txt").unwrap())
        .lines()
        .map(Result::unwrap)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("long_haystack");

    group.bench_function("String::find", |b| {
        b.iter(|| {
            for needle in &needles {
                black_box(haystack.find(needle));
            }
        });
    });

    group.bench_function("TwoWaySearcher::search_in", |b| {
        let searchers = needles
            .iter()
            .map(|needle| TwoWaySearcher::new(needle.as_bytes()))
            .collect::<Vec<_>>();

        b.iter(|| {
            for searcher in &searchers {
                black_box(searcher.search_in(haystack.as_bytes()));
            }
        });
    });

    group.bench_function("twoway::find_bytes", |b| {
        b.iter(|| {
            for needle in &needles {
                black_box(twoway::find_bytes(haystack.as_bytes(), needle.as_bytes()));
            }
        });
    });

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        group.bench_function("strstr_avx2_original", |b| {
            b.iter(|| {
                for needle in &needles {
                    black_box(unsafe {
                        strstr_avx2_original(haystack.as_bytes(), needle.as_bytes())
                    });
                }
            });
        });

        group.bench_function("strstr_avx2_rust", |b| {
            b.iter(|| {
                for needle in &needles {
                    black_box(unsafe { strstr_avx2_rust(haystack.as_bytes(), needle.as_bytes()) });
                }
            });
        });

        group.bench_function("DynamicAvx2Searcher::search_in", |b| {
            let searchers = needles
                .iter()
                .map(|needle| unsafe {
                    DynamicAvx2Searcher::new(needle.as_bytes().to_owned().into_boxed_slice())
                })
                .collect::<Vec<_>>();

            b.iter(|| {
                for searcher in &searchers {
                    black_box(unsafe { searcher.search_in(haystack.as_bytes()) });
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, search_short_haystack, search_long_haystack);
criterion_main!(benches);
