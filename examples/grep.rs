use memmap2::MmapOptions;
#[cfg(target_arch = "aarch64")]
use sliceslice::aarch64::NeonSearcher;
#[cfg(feature = "stdsimd")]
use sliceslice::stdsimd::StdSimdSearcher;
#[cfg(target_arch = "wasm32")]
use sliceslice::wasm32::Wasm32Searcher;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use sliceslice::x86::{Avx2Searcher, DynamicAvx2Searcher};
use std::fs::File;

#[inline(never)]
pub fn search_in_slice(backend: &str, needle: &[u8], haystack: &[u8]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if "avx2".eq_ignore_ascii_case(backend) {
        let searcher = unsafe { Avx2Searcher::new(needle) };
        return unsafe { searcher.search_in(haystack) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if "dynamicavx2".eq_ignore_ascii_case(backend) {
        let searcher = unsafe { DynamicAvx2Searcher::new(needle) };
        return unsafe { searcher.search_in(haystack) };
    }
    #[cfg(any(target_arch = "aarch64"))]
    if "neon".eq_ignore_ascii_case(backend) {
        let searcher = unsafe { NeonSearcher::new(needle) };
        return unsafe { searcher.search_in(haystack) };
    }
    #[cfg(feature = "stdsimd")]
    if "stdsimd".eq_ignore_ascii_case(backend) {
        let searcher = StdSimdSearcher::new(needle);
        return searcher.search_in(haystack);
    }
    #[cfg(any(target_arch = "wasm32"))]
    if "wasm32".eq_ignore_ascii_case(backend) {
        let searcher = unsafe { Wasm32Searcher::new(needle) };
        return unsafe { searcher.search_in(haystack) };
    }
    panic!("Invalid backend {:?}", backend);
}

fn main() {
    let usage = "./grep <backend> <needle> <file>";
    let mut args = std::env::args();
    args.next().expect(usage);
    let backend = args.next().expect(usage);
    let needle = args.next().expect(usage);
    let filename = args.next().expect(usage);
    let file = File::open(&filename).unwrap();
    let data = unsafe { MmapOptions::new().map(&file).unwrap() };
    println!(
        "Searching for {} in {:?}: {}",
        needle,
        filename,
        search_in_slice(&backend, needle.as_bytes(), &data)
    );
}
