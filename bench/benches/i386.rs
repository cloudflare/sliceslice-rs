use criterion::{
    black_box, criterion_group, criterion_main,
    measurement::{Measurement, WallTime},
    Criterion,
};
use criterion_linux_perf::{PerfMeasurement, PerfMode};
use memmem::{Searcher, TwoWaySearcher};
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

fn search_short_haystack<M: Measurement>(c: &mut Criterion<M>) {
    let mut needles = BufReader::new(File::open("../data/words.txt").unwrap())
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

    group.bench_function("memmem::TwoWaySearcher::search_in", |b| {
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
        use sliceslice::x86::DynamicAvx2Searcher;

        #[cfg(feature = "sse4-strstr")]
        group.bench_function("sse4_strstr::avx2_strstr_v2", |b| {
            b.iter(|| {
                for (i, needle) in needles.iter().enumerate() {
                    for haystack in &needles[i..] {
                        black_box(unsafe {
                            sse4_strstr::avx2_strstr_v2(haystack.as_bytes(), needle.as_bytes())
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

fn search_long_haystack<M: Measurement>(c: &mut Criterion<M>) {
    let haystack = include_str!("../../data/haystack");

    let needles = BufReader::new(File::open("../data/words.txt").unwrap())
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

    group.bench_function("memmem::TwoWaySearcher::search_in", |b| {
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
        use sliceslice::x86::DynamicAvx2Searcher;

        #[cfg(feature = "sse4-strstr")]
        group.bench_function("sse4_strstr::avx2_strstr_v2", |b| {
            b.iter(|| {
                for needle in &needles {
                    black_box(unsafe {
                        sse4_strstr::avx2_strstr_v2(haystack.as_bytes(), needle.as_bytes())
                    });
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

criterion_group!(
    name = i386_wall_time;
    config = Criterion::default().with_measurement(WallTime);
    targets = search_short_haystack, search_long_haystack
);
criterion_group!(
    name = i386_perf_instructions;
    config = Criterion::default().with_measurement(PerfMeasurement::new(PerfMode::Instructions));
    targets = search_short_haystack, search_long_haystack
);

criterion_main!(i386_wall_time, i386_perf_instructions);
