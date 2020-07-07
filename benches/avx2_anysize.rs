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
    let mut words: Vec<String> = io::BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok())
        .filter(|word| word.len() > 1)
        .collect();
    // Sort all words by length then lexicographically
    words.sort_by(|a, b| {
        if a.len() != b.len() {
            a.len().partial_cmp(&b.len()).unwrap()
        } else {
            a.partial_cmp(b).unwrap()
        }
    });
    let words = words;

    let twoway_words: Vec<TwoWaySearcher<'_>> = words
        .iter()
        .map(|word| TwoWaySearcher::new(word.as_bytes()))
        .collect();

    let avx2_words: Vec<StrStrAVX2Searcher> = words
        .iter()
        .map(|word| StrStrAVX2Searcher::new(word.as_bytes()))
        .collect();

    // Benchmarks against long haystacks

    c.bench_function("String::find with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                content.find(word);
            }
        })
    });

    c.bench_function(
        "memmem::TwoWaySearcher::search_in with long haystack",
        |b| {
            b.iter(|| {
                for twoway_word in &twoway_words {
                    twoway_word.search_in(content.as_bytes());
                }
            })
        },
    );

    c.bench_function("twoway::find_bytes with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                twoway::find_bytes(content.as_bytes(), word.as_bytes());
            }
        })
    });

    c.bench_function("strstr_avx2_original with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_original(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_simple(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple_2 with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_simple_2(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_fast(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast_2 with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes());
            }
        })
    });

    c.bench_function("strstr_avx2_rust_aligned with long haystack", |b| {
        b.iter(|| {
            for word in &words {
                unsafe {
                    strstr_avx2_rust_aligned(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("StrStrAVX2Searcher::search_in with long haystack", |b| {
        b.iter(|| {
            for avx2_word in &avx2_words {
                avx2_word.search_in(content.as_bytes());
            }
        })
    });

    // Benchmarks against short haystacks
    //
    // Since words are ordered by length, pick a word as needle
    // and use bigger words as haystacks.

    c.bench_function("String::find with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    content.find(word);
                }
            }
        })
    });

    c.bench_function(
        "memmem::TwoWaySearcher::search_in with short haystack",
        |b| {
            b.iter(|| {
                for (i, word) in twoway_words.iter().enumerate() {
                    for content in &words[(i + 1)..] {
                        word.search_in(content.as_bytes());
                    }
                }
            })
        },
    );

    c.bench_function("twoway::find_bytes with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    twoway::find_bytes(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_original with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    unsafe {
                        strstr_avx2_original(content.as_bytes(), word.as_bytes());
                    }
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    unsafe {
                        strstr_avx2_rust_simple(content.as_bytes(), word.as_bytes());
                    }
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_simple_2 with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    unsafe {
                        strstr_avx2_rust_simple_2(content.as_bytes(), word.as_bytes());
                    }
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    unsafe {
                        strstr_avx2_rust_fast(content.as_bytes(), word.as_bytes());
                    }
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_fast_2 with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes());
                }
            }
        })
    });

    c.bench_function("strstr_avx2_rust_aligned with short haystack", |b| {
        b.iter(|| {
            for (i, word) in words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    unsafe {
                        strstr_avx2_rust_aligned(content.as_bytes(), word.as_bytes());
                    }
                }
            }
        })
    });

    c.bench_function("StrStrAVX2Searcher::search_in with short haystack", |b| {
        b.iter(|| {
            for (i, word) in avx2_words.iter().enumerate() {
                for content in &words[(i + 1)..] {
                    word.search_in(content.as_bytes());
                }
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
