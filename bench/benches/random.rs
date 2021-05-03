use criterion::{
    black_box, criterion_group, criterion_main,
    measurement::{Measurement, WallTime},
    BenchmarkId, Criterion,
};
#[cfg(target_os = "linux")]
use criterion_linux_perf::{PerfMeasurement, PerfMode};
use memmem::{Searcher, TwoWaySearcher};

fn search<M: Measurement>(c: &mut Criterion<M>) {
    let haystack = include_str!("../../data/haystack");
    let needle = include_str!("../../data/needle");

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
                BenchmarkId::new("memmem::TwoWaySearcher::search_in", parameter),
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
                BenchmarkId::new("memchr::memmem::find", parameter),
                &size,
                |b, _| {
                    b.iter(|| black_box(memchr::memmem::find(haystack, needle)));
                },
            );

            group.bench_with_input(
                BenchmarkId::new("memchr::memmem::Finder::find", parameter),
                &size,
                |b, _| {
                    let finder = memchr::memmem::Finder::new(needle);
                    b.iter(|| black_box(finder.find(haystack)));
                },
            );

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use sliceslice::x86::DynamicAvx2Searcher;

                #[cfg(feature = "sse4-strstr")]
                group.bench_with_input(
                    BenchmarkId::new("sse4_strstr::avx2_strstr_v2", parameter),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            black_box(unsafe { sse4_strstr::avx2_strstr_v2(haystack, needle) })
                        });
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

criterion_group!(
    name = random_wall_time;
    config = Criterion::default().with_measurement(WallTime);
    targets = search
);

#[cfg(target_os = "linux")]
criterion_group!(
    name = random_perf_instructions;
    config = Criterion::default().with_measurement(PerfMeasurement::new(PerfMode::Instructions));
    targets = search
);

#[cfg(target_os = "linux")]
criterion_main!(random_wall_time, random_perf_instructions);

#[cfg(not(target_os = "linux"))]
criterion_main!(random_wall_time);
