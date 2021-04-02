use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
};

fn search(haystack: &str, needle: &str) {
    let result = haystack.contains(&needle);

    let haystack = haystack.as_bytes();
    let needle = needle.as_bytes();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use sliceslice::x86::DynamicAvx2Searcher;

        let searcher = unsafe { DynamicAvx2Searcher::new(needle.to_owned().into_boxed_slice()) };
        assert_eq!(unsafe { searcher.search_in(haystack) }, result);
    }
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
