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
//! let searcher = unsafe { DynamicAvx2Searcher::new(b"ipsum") };
//!
//! assert!(unsafe {
//!     searcher.search_in(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit")
//! });
//!
//! assert!(!unsafe {
//!     searcher.search_in(b"foo bar baz qux quux quuz corge grault garply waldo fred")
//! });
//! ```

#![warn(missing_docs)]

/// Substring search implementations using x86 architecture features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

mod bits;
mod memcmp;

use memchr::memchr;
use std::rc::Rc;
use std::sync::Arc;

/// Needle that can be searched for within a haystack. It allows specialized
/// searcher implementations for needle sizes known at compile time.
pub trait Needle {
    /// Set to `Some(<usize>)` if and only if the needle's length is known at
    /// compile time.
    const SIZE: Option<usize>;
    /// Return the slice corresponding to the needle.
    fn as_bytes(&self) -> &[u8];
}

impl<const N: usize> Needle for [u8; N] {
    const SIZE: Option<usize> = Some(N);

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

impl Needle for [u8] {
    const SIZE: Option<usize> = None;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

impl<N: Needle + ?Sized> Needle for Box<N> {
    const SIZE: Option<usize> = N::SIZE;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        (**self).as_bytes()
    }
}

impl<N: Needle + ?Sized> Needle for Rc<N> {
    const SIZE: Option<usize> = N::SIZE;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        (**self).as_bytes()
    }
}

impl<N: Needle + ?Sized> Needle for Arc<N> {
    const SIZE: Option<usize> = N::SIZE;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        (**self).as_bytes()
    }
}

impl<N: Needle + ?Sized> Needle for &N {
    const SIZE: Option<usize> = N::SIZE;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        (*self).as_bytes()
    }
}

impl Needle for Vec<u8> {
    const SIZE: Option<usize> = None;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

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
    use super::{MemchrSearcher, Needle};

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

    #[test]
    fn needle_array_size() {
        use std::rc::Rc;
        use std::sync::Arc;

        assert_eq!(<[u8; 0] as Needle>::SIZE, Some(0));

        assert_eq!(Box::<[u8; 1]>::SIZE, Some(1));

        assert_eq!(Rc::<[u8; 2]>::SIZE, Some(2));

        assert_eq!(Arc::<[u8; 3]>::SIZE, Some(3));

        assert_eq!(<&[u8; 4] as Needle>::SIZE, Some(4));
    }

    #[test]
    fn needle_slice_size() {
        use std::rc::Rc;
        use std::sync::Arc;

        assert_eq!(Box::<[u8]>::SIZE, None);

        assert_eq!(Vec::<u8>::SIZE, None);

        assert_eq!(Rc::<[u8]>::SIZE, None);

        assert_eq!(Arc::<[u8]>::SIZE, None);

        assert_eq!(<&[u8] as Needle>::SIZE, None);
    }
}
