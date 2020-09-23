//! A fast implementation of single-pattern substring search using SIMD
//! acceleration, based on the work [presented by Wojciech
//! Muła](http://0x80.pl/articles/simd-strfind.html). For a fast multi-pattern
//! substring search algorithm, see instead the [`aho-corasick`
//! crate](https://github.com/BurntSushi/aho-corasick).
//!
//! # Example
//!
//! ```
//! use sliceslice::x86::DynamicAvx2Searcher;
//!
//! let searcher = unsafe { DynamicAvx2Searcher::new(b"ipsum".to_owned().into()) };
//!
//! assert!(unsafe {
//!     searcher.search_in(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit")
//! });
//!
//! assert!(!unsafe {
//!     searcher.search_in(b"foo bar baz qux quux quuz corge grault garply waldo fred")
//! });

#![warn(missing_docs)]

/// Substring search implementations using x86 architecture features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

mod bits;
mod memcmp;

use memchr::memchr;
use seq_macro::seq;

/// Needle that can be searched for within a haystack. It is used to allow
/// specialized searcher implementations for compile-time needle lengths.
pub trait Needle: AsRef<[u8]> {
    /// Set to `true` if and only if the needle length is fixed at compile time.
    const IS_FIXED: bool = false;
}

impl<N: Needle + ?Sized> Needle for &N {
    const IS_FIXED: bool = N::IS_FIXED;
}

impl Needle for [u8] {}

seq!(N in 0..=32 {
    impl Needle for [u8; N] {
        const IS_FIXED: bool = true;
    }
});

impl Needle for Box<[u8]> {}

/// Single-byte searcher using `memchr` for faster matching.
pub struct MemchrSearcher(u8);

impl MemchrSearcher {
    /// Creates a new searcher for `needle`.
    pub fn new(needle: u8) -> Self {
        Self(needle)
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.is_empty() {
            return false;
        }

        memchr(self.0, haystack).is_some()
    }

    /// Performs a substring search for the `needle` within `haystack`.
    pub fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

#[cfg(test)]
mod tests {
    use super::MemchrSearcher;

    fn memchr_search(haystack: &[u8], needle: &[u8]) -> bool {
        MemchrSearcher::new(needle[0]).search_in(haystack)
    }

    #[test]
    fn memchr_search_same() {
        assert!(memchr_search(b"f", b"f"));
    }

    #[test]
    fn memchr_search_different() {
        assert!(!memchr_search(b"foo", b"b"));
    }

    #[test]
    fn memchr_search_prefix() {
        assert!(memchr_search(b"foobar", b"f"));
    }

    #[test]
    fn memchr_search_suffix() {
        assert!(memchr_search(b"foobar", b"r"));
    }

    #[test]
    fn memchr_search_mutiple() {
        assert!(memchr_search(b"foobarfoo", b"o"));
    }

    #[test]
    fn memchr_search_middle() {
        assert!(memchr_search(b"foobarfoo", b"b"));
    }
}
