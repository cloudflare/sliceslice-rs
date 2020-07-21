use memmem::{Searcher, TwoWaySearcher};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
};
use strstr::avx2::*;

fn search(haystack: &str, needle: &str) {
    let result = haystack.find(&needle).is_some();

    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();

    let searcher = TwoWaySearcher::new(needle);
    assert_eq!(searcher.search_in(haystack).is_some(), result);

    assert_eq!(twoway::find_bytes(haystack, needle).is_some(), result);

    assert_eq!(unsafe { strstr_avx2_original(haystack, needle) }, result);

    assert_eq!(strstr_avx2_rust(haystack, needle), result);

    let searcher = StrStrAVX2Searcher::new(needle);
    assert_eq!(searcher.search_in(haystack), result);

    let searcher = unsafe { DynamicAvx2Searcher::new(needle.to_owned().into_boxed_slice()) };
    assert_eq!(searcher.search_in(haystack), result);
}

#[test]
fn search_short_haystack() {
    let mut needles = BufReader::new(File::open("data/words.txt").unwrap())
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
    let haystack = fs::read("data/i386.txt").unwrap();
    let haystack = String::from_utf8_lossy(&haystack);

    let needles = BufReader::new(File::open("data/words.txt").unwrap())
        .lines()
        .map(Result::unwrap);

    for needle in needles {
        search(&haystack, &needle);
    }
}
