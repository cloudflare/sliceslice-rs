use std::io::{BufRead, BufReader};

static I386: &[u8] = include_bytes!("../data/i386.txt");
static WORDS: &[u8] = include_bytes!("../data/words.txt");

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn search(haystack: &str, needle: &str) {
    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();

    let result = find_subsequence(haystack, needle).is_some();

    cfg_if::cfg_if! {
        if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
            use sliceslice::x86::DynamicAvx2Searcher;
            let searcher = unsafe { DynamicAvx2Searcher::new(needle.to_owned().into_boxed_slice()) };
            assert_eq!(unsafe { searcher.search_in(haystack) }, result);
        } else if #[cfg(target_arch = "wasm32")] {
            use sliceslice::wasm32::Wasm32Searcher;
            let searcher = unsafe { Wasm32Searcher::new(needle) };
            assert_eq!(unsafe { searcher.search_in(haystack) }, result);
        } else if #[cfg(target_arch = "aarch64")] {
            use sliceslice::aarch64::NeonSearcher;
            let searcher = unsafe { NeonSearcher::new(needle) };
            assert_eq!(unsafe { searcher.search_in(haystack) }, result);
        } else {
            compile_error!("Unsupported architecture");
        }
    }
}

#[test]
fn search_short_haystack() {
    let mut needles = BufReader::new(WORDS)
        .lines()
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    needles.sort_unstable_by_key(|needle| needle.len());

    for (i, needle) in needles.iter().enumerate() {
        for haystack in &needles[i..] {
            search(haystack, needle);
        }
    }
}

#[test]
fn search_long_haystack() {
    let haystack = String::from_utf8_lossy(I386);

    let needles = BufReader::new(WORDS).lines().map(Result::unwrap);

    for needle in needles {
        search(&haystack, &needle);
    }
}
