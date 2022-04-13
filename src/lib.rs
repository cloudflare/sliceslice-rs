//! A fast implementation of single-pattern substring search using SIMD
//! acceleration, based on the work [presented by Wojciech
//! Muła](http://0x80.pl/articles/simd-strfind.html). For a fast multi-pattern
//! substring search algorithm, see instead the [`aho-corasick`
//! crate](https://github.com/BurntSushi/aho-corasick).

#![warn(missing_docs)]
// Will be stabilized in 1.61.0 with https://github.com/rust-lang/rust/pull/90621
#![cfg_attr(
    target_arch = "aarch64",
    allow(stable_features),
    feature(aarch64_target_feature)
)]
#![cfg_attr(feature = "stdsimd", feature(portable_simd))]

/// Substring search implementations using aarch64 architecture features.
#[cfg(target_arch = "aarch64")]
pub mod aarch64;

/// Substring search implementations using generic stdsimd features.
#[cfg(feature = "stdsimd")]
pub mod stdsimd;

/// Substring search implementations using x86 architecture features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

/// Substring search implementations using wasm32 architecture features.
#[cfg(target_arch = "wasm32")]
pub mod wasm32;

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

trait NeedleWithSize: Needle {
    #[inline]
    fn size(&self) -> usize {
        if let Some(size) = Self::SIZE {
            size
        } else {
            self.as_bytes().len()
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
}

impl<N: Needle + ?Sized> NeedleWithSize for N {}

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

/// Represents a generic SIMD register type.
trait Vector: Copy {
    const LANES: usize;

    type Mask;

    unsafe fn splat(a: u8) -> Self;

    unsafe fn load(a: *const u8) -> Self;

    unsafe fn lanes_eq(a: Self, b: Self) -> Self::Mask;

    unsafe fn bitwise_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;

    unsafe fn to_bitmask(a: Self::Mask) -> i32;
}

/// Hash of the first and "last" bytes in the needle for use with the SIMD
/// algorithm implemented by `Avx2Searcher::vector_search_in`. As explained, any
/// byte can be chosen to represent the "last" byte of the hash to prevent
/// worst-case attacks.
#[derive(Debug)]
struct VectorHash<V: Vector> {
    first: V,
    last: V,
}

impl<V: Vector> VectorHash<V> {
    unsafe fn new(first: u8, last: u8) -> Self {
        Self {
            first: V::splat(first),
            last: V::splat(last),
        }
    }
}

impl<T: Vector, V: Vector + From<T>> From<&VectorHash<T>> for VectorHash<V> {
    #[inline]
    fn from(hash: &VectorHash<T>) -> Self {
        Self {
            first: V::from(hash.first),
            last: V::from(hash.last),
        }
    }
}

trait Searcher<N: NeedleWithSize + ?Sized> {
    fn needle(&self) -> &N;

    fn position(&self) -> usize;

    #[multiversion::multiversion]
    #[clone(target = "[x86|x86_64]+avx2")]
    #[clone(target = "wasm32+simd128")]
    #[clone(target = "aarch64+neon")]
    unsafe fn vector_search_in_chunk<V: Vector>(
        &self,
        haystack: &[u8],
        hash: &VectorHash<V>,
        start: *const u8,
        mask: i32,
    ) -> bool {
        let first = V::load(start);
        let last = V::load(start.add(self.position()));

        let eq_first = V::lanes_eq(hash.first, first);
        let eq_last = V::lanes_eq(hash.last, last);

        let eq = V::bitwise_and(eq_first, eq_last);
        let mut eq = (V::to_bitmask(eq) & mask) as u32;

        let start = start as usize - haystack.as_ptr() as usize;
        let chunk = haystack.as_ptr().add(start + 1);
        let needle = self.needle().as_bytes().as_ptr().add(1);

        while eq != 0 {
            let chunk = chunk.add(eq.trailing_zeros() as usize);
            let equal = match N::SIZE {
                Some(0) => unreachable!(),
                Some(1) => dispatch!(memcmp::specialized::<0>(chunk, needle)),
                Some(2) => dispatch!(memcmp::specialized::<1>(chunk, needle)),
                Some(3) => dispatch!(memcmp::specialized::<2>(chunk, needle)),
                Some(4) => dispatch!(memcmp::specialized::<3>(chunk, needle)),
                Some(5) => dispatch!(memcmp::specialized::<4>(chunk, needle)),
                Some(6) => dispatch!(memcmp::specialized::<5>(chunk, needle)),
                Some(7) => dispatch!(memcmp::specialized::<6>(chunk, needle)),
                Some(8) => dispatch!(memcmp::specialized::<7>(chunk, needle)),
                Some(9) => dispatch!(memcmp::specialized::<8>(chunk, needle)),
                Some(10) => dispatch!(memcmp::specialized::<9>(chunk, needle)),
                Some(11) => dispatch!(memcmp::specialized::<10>(chunk, needle)),
                Some(12) => dispatch!(memcmp::specialized::<11>(chunk, needle)),
                Some(13) => dispatch!(memcmp::specialized::<12>(chunk, needle)),
                Some(14) => dispatch!(memcmp::specialized::<13>(chunk, needle)),
                Some(15) => dispatch!(memcmp::specialized::<14>(chunk, needle)),
                Some(16) => dispatch!(memcmp::specialized::<15>(chunk, needle)),
                _ => dispatch!(memcmp::generic(chunk, needle, self.needle().size() - 1)),
            };
            if equal {
                return true;
            }

            eq = dispatch!(bits::clear_leftmost_set(eq));
        }

        false
    }

    #[multiversion::multiversion]
    #[clone(target = "[x86|x86_64]+avx2")]
    #[clone(target = "wasm32+simd128")]
    #[clone(target = "aarch64+neon")]
    unsafe fn vector_search_in<V: Vector>(
        &self,
        haystack: &[u8],
        end: usize,
        hash: &VectorHash<V>,
    ) -> bool {
        debug_assert!(haystack.len() >= self.needle().size());

        let mut chunks = haystack[..end].chunks_exact(V::LANES);
        for chunk in &mut chunks {
            if dispatch!(self.vector_search_in_chunk(haystack, hash, chunk.as_ptr(), -1)) {
                return true;
            }
        }

        let remainder = chunks.remainder().len();
        if remainder > 0 {
            let start = haystack.as_ptr().add(end - V::LANES);
            let mask = -1 << (V::LANES - remainder);

            if dispatch!(self.vector_search_in_chunk(haystack, hash, start, mask)) {
                return true;
            }
        }

        false
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

    fn search(haystack: &[u8], needle: &[u8]) -> bool {
        let result = haystack
            .windows(needle.len())
            .any(|window| window == needle);

        for position in 0..needle.len() {
            cfg_if::cfg_if! {
                if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                    use crate::x86::{Avx2Searcher, DynamicAvx2Searcher};

                    let searcher =  unsafe { Avx2Searcher::with_position(needle, position) };
                    assert_eq!(unsafe { searcher.search_in(haystack) }, result);

                    let searcher =  unsafe { DynamicAvx2Searcher::with_position(needle, position) };
                    assert_eq!(unsafe { searcher.search_in(haystack) }, result);
                } else if #[cfg(target_arch = "wasm32")] {
                    use crate::wasm32::Wasm32Searcher;

                    let searcher = unsafe { Wasm32Searcher::with_position(needle, position) };
                    assert_eq!(unsafe { searcher.search_in(haystack) }, result);
                } else if #[cfg(target_arch = "aarch64")] {
                    use crate::aarch64::NeonSearcher;

                    let searcher = unsafe { NeonSearcher::with_position(needle, position) };
                    assert_eq!(unsafe { searcher.search_in(haystack) }, result);
                } else if #[cfg(not(feature = "stdsimd"))] {
                    compile_error!("Unsupported architecture");
                }
            }

            cfg_if::cfg_if! {
                if #[cfg(feature = "stdsimd")] {
                    use crate::stdsimd::StdSimdSearcher;

                    let searcher = StdSimdSearcher::with_position(needle, position);
                    assert_eq!(searcher.search_in(haystack), result);
                }
            }
        }

        result
    }

    #[test]
    fn search_same() {
        assert!(search(b"x", b"x"));

        assert!(search(b"xy", b"xy"));

        assert!(search(b"foo", b"foo"));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus"
        ));
    }

    #[test]
    fn search_different() {
        assert!(!search(b"x", b"y"));

        assert!(!search(b"xy", b"xz"));

        assert!(!search(b"bar", b"foo"));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"foo"
        ));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"foo"
        ));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"foo bar baz qux quux quuz corge grault garply waldo fred plugh xyzzy thud"
        ));
    }

    #[test]
    fn search_prefix() {
        assert!(search(b"xy", b"x"));

        assert!(search(b"foobar", b"foo"));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"Lorem"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"Lorem"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit"
        ));
    }

    #[test]
    fn search_suffix() {
        assert!(search(b"xy", b"y"));

        assert!(search(b"foobar", b"bar"));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"elit"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"purus"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"Aliquam iaculis fringilla mi, nec aliquet purus"
        ));
    }

    #[test]
    fn search_multiple() {
        assert!(search(b"xx", b"x"));

        assert!(search(b"xyxy", b"xy"));

        assert!(search(b"foobarfoo", b"foo"));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"it"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"conse"
        ));
    }

    #[test]
    fn search_middle() {
        assert!(search(b"xyz", b"y"));

        assert!(search(b"wxyz", b"xy"));

        assert!(search(b"foobarfoo", b"bar"));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"consectetur"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"orci"
        ));

        assert!(search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"Maecenas commodo posuere orci a consectetur"
        ));
    }
}
