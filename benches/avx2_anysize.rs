use criterion::{criterion_group, criterion_main, Criterion};
use memmem::{Searcher, TwoWaySearcher};
use std::fs::File;
use std::io::{self, BufRead, Read};
use strstr::avx2::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = File::open("./data/i386.txt").unwrap();
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer).unwrap();
    let content = String::from_utf8_lossy(&buffer);

    let file = File::open("./data/words").unwrap();
    let words: Vec<String> = io::BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok())
        .filter(|word| word.len() > 1)
        .collect();

    c.bench_function("String::find", |b| {
        b.iter(|| {
            for word in &words {
                content.find(word);
            }
        })
    });

    let twoway_words: Vec<TwoWaySearcher<'_>> = words
        .iter()
        .map(|word| TwoWaySearcher::new(word.as_bytes()))
        .collect();

    c.bench_function("TwoWaySearcher::search_in", |b| {
        b.iter(|| {
            for twoway_word in &twoway_words {
                twoway_word.search_in(content.as_bytes());
            }
        })
    });

    c.bench_function("strstr_avx2_original", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_original(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_simple(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple_2", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_simple_2(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_fast(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast_2", |b| {
        b.iter(|| {
            for word in &words {
                strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes());
            }
        })
    });

    c.bench_function("strstr_avx2_rust_aligned", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_aligned(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
