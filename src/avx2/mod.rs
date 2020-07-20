#![allow(clippy::missing_safety_doc)]

mod original;
mod rust;

pub use self::{original::*, rust::*};

use crate::{bits, memchr::MemchrSearcher, memcmp::memcmp};
use std::{arch::x86_64::*, marker::PhantomData, mem};
use typenum::{Unsigned, U10, U11, U12, U13, U2, U3, U4, U5, U6, U7, U8, U9};

#[derive(Clone, Copy, Default, PartialEq)]
struct ScalarHash(usize);

impl From<&[u8]> for ScalarHash {
    #[inline]
    fn from(bytes: &[u8]) -> Self {
        bytes.iter().fold(Default::default(), |mut hash, &b| {
            hash.push(b);
            hash
        })
    }
}

impl ScalarHash {
    #[inline]
    fn push(&mut self, b: u8) {
        self.0 ^= usize::from(b);
    }

    #[inline]
    fn pop(&mut self, b: u8) {
        self.0 ^= usize::from(b);
    }
}

trait Vector: Copy {
    unsafe fn set1_epi8(a: i8) -> Self;

    unsafe fn loadu_si(a: *const Self) -> Self;

    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self;

    unsafe fn and_si(a: Self, b: Self) -> Self;

    unsafe fn movemask_epi8(a: Self) -> i32;
}

impl Vector for __m128i {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const Self) -> Self {
        _mm_loadu_si128(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        _mm_cmpeq_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        _mm_and_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm_movemask_epi8(a)
    }
}

impl Vector for __m256i {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm256_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const Self) -> Self {
        _mm256_loadu_si256(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        _mm256_cmpeq_epi8(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        _mm256_and_si256(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm256_movemask_epi8(a)
    }
}

struct VectorHash<V: Vector> {
    first: V,
    last: V,
}

impl<V: Vector> VectorHash<V> {
    #[target_feature(enable = "avx2")]
    unsafe fn new(first: u8, last: u8) -> Self {
        Self {
            first: Vector::set1_epi8(first as i8),
            last: Vector::set1_epi8(last as i8),
        }
    }
}

pub trait Size {
    #[inline]
    fn size<T>(slice: &[T]) -> usize {
        slice.len()
    }

    #[inline]
    fn is_fixed() -> bool {
        false
    }
}

pub struct AnySize;

impl Size for AnySize {}

impl<U: Unsigned> Size for U {
    #[inline]
    fn size<T>(_slice: &[T]) -> usize {
        U::to_usize()
    }

    #[inline]
    fn is_fixed() -> bool {
        true
    }
}

pub struct Avx2Searcher<S: Size = AnySize> {
    needle: Box<[u8]>,
    _size: PhantomData<S>,
    position: usize,
    scalar_hash: ScalarHash,
    sse2_hash: VectorHash<__m128i>,
    avx2_hash: VectorHash<__m256i>,
}

impl<S: Size> Avx2Searcher<S> {
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(needle: Box<[u8]>) -> Self {
        let position = S::size(&needle) - 1;
        Self::with_position(needle, position)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn with_position(needle: Box<[u8]>, position: usize) -> Self {
        assert!(needle.len() == S::size(&needle));
        assert!(S::size(&needle) >= 2);
        assert!(0 < position && position < S::size(&needle));

        let scalar_hash = ScalarHash::from(needle.as_ref());
        let sse2_hash = VectorHash::new(needle[0], needle[position]);
        let avx2_hash = VectorHash::new(needle[0], needle[position]);

        Self {
            needle,
            _size: Default::default(),
            position,
            scalar_hash,
            sse2_hash,
            avx2_hash,
        }
    }

    #[inline]
    fn scalar_search_in(&self, haystack: &[u8]) -> bool {
        debug_assert!(haystack.len() >= S::size(&self.needle));

        let mut end = S::size(&self.needle) - 1;
        let mut hash = ScalarHash::from(&haystack[..end]);

        while end < haystack.len() {
            hash.push(*unsafe { haystack.get_unchecked(end) });
            end += 1;

            let start = end - S::size(&self.needle);
            if hash == self.scalar_hash && haystack[start..end] == *self.needle {
                return true;
            }

            hash.pop(*unsafe { haystack.get_unchecked(start) });
        }

        false
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_search_in_chunk<V: Vector>(
        &self,
        haystack: &[u8],
        hash: &VectorHash<V>,
        start: *const u8,
        mask: i32,
    ) -> bool {
        let first = Vector::loadu_si(start.cast());
        let last = Vector::loadu_si(start.add(self.position).cast());

        let eq_first = Vector::cmpeq_epi8(hash.first, first);
        let eq_last = Vector::cmpeq_epi8(hash.last, last);

        let eq = Vector::and_si(eq_first, eq_last);
        let mut eq = (Vector::movemask_epi8(eq) & mask) as u32;

        if S::is_fixed() && S::size(&self.needle) <= 2 {
            return eq != 0;
        }

        let start = start as usize - haystack.as_ptr() as usize;
        while eq != 0 {
            let chunk = &haystack[start + eq.trailing_zeros() as usize..][..S::size(&self.needle)];
            if memcmp(&chunk[1..], &self.needle[1..], S::size(&self.needle) - 1) {
                return true;
            }

            eq = bits::clear_leftmost_set(eq);
        }

        false
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn vector_search_in<V: Vector>(
        &self,
        haystack: &[u8],
        hash: &VectorHash<V>,
        next: unsafe fn(&Self, &[u8]) -> bool,
    ) -> bool {
        debug_assert!(haystack.len() >= S::size(&self.needle));

        let lanes = mem::size_of::<V>();
        let end = haystack.len() - S::size(&self.needle) + 1;

        if end < lanes {
            return next(self, haystack);
        }

        let mut chunks = haystack[..end].chunks_exact(lanes);
        while let Some(chunk) = chunks.next() {
            if self.vector_search_in_chunk(haystack, hash, chunk.as_ptr(), -1) {
                return true;
            }
        }

        let remainder = chunks.remainder().len();
        if remainder > 0 {
            let start = haystack.as_ptr().add(end - lanes);
            let mask = -1 << (lanes - remainder);

            if self.vector_search_in_chunk(haystack, hash, start, mask) {
                return true;
            }
        }

        false
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sse2_search_in(&self, haystack: &[u8]) -> bool {
        self.vector_search_in(haystack, &self.sse2_hash, Self::scalar_search_in)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_search_in(&self, haystack: &[u8]) -> bool {
        self.vector_search_in(haystack, &self.avx2_hash, Self::sse2_search_in)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.len() < S::size(&self.needle) {
            return false;
        }

        self.avx2_search_in(haystack)
    }

    #[inline]
    pub fn search_in(&self, haystack: &[u8]) -> bool {
        unsafe { self.inlined_search_in(haystack) }
    }
}

pub enum DynamicAvx2Searcher {
    N0,
    N1(MemchrSearcher),
    N2(Avx2Searcher<U2>),
    N3(Avx2Searcher<U3>),
    N4(Avx2Searcher<U4>),
    N5(Avx2Searcher<U5>),
    N6(Avx2Searcher<U6>),
    N7(Avx2Searcher<U7>),
    N8(Avx2Searcher<U8>),
    N9(Avx2Searcher<U9>),
    N10(Avx2Searcher<U10>),
    N11(Avx2Searcher<U11>),
    N12(Avx2Searcher<U12>),
    N13(Avx2Searcher<U13>),
    N(Avx2Searcher),
}

impl DynamicAvx2Searcher {
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(needle: Box<[u8]>) -> Self {
        let position = needle.len() - 1;
        Self::with_position(needle, position)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn with_position(needle: Box<[u8]>, position: usize) -> Self {
        assert!(!needle.is_empty());
        assert!(position < needle.len());

        match needle.len() {
            0 => Self::N0,
            1 => Self::N1(MemchrSearcher::new(needle[0])),
            2 => Self::N2(Avx2Searcher::new(needle)),
            3 => Self::N3(Avx2Searcher::new(needle)),
            4 => Self::N4(Avx2Searcher::new(needle)),
            5 => Self::N5(Avx2Searcher::new(needle)),
            6 => Self::N6(Avx2Searcher::new(needle)),
            7 => Self::N7(Avx2Searcher::new(needle)),
            8 => Self::N8(Avx2Searcher::new(needle)),
            9 => Self::N9(Avx2Searcher::new(needle)),
            10 => Self::N10(Avx2Searcher::new(needle)),
            11 => Self::N11(Avx2Searcher::new(needle)),
            12 => Self::N12(Avx2Searcher::new(needle)),
            13 => Self::N13(Avx2Searcher::new(needle)),
            _ => Self::N(Avx2Searcher::new(needle)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        match self {
            Self::N0 => true,
            Self::N1(searcher) => searcher.inlined_search_in(haystack),
            Self::N2(searcher) => searcher.inlined_search_in(haystack),
            Self::N3(searcher) => searcher.inlined_search_in(haystack),
            Self::N4(searcher) => searcher.inlined_search_in(haystack),
            Self::N5(searcher) => searcher.inlined_search_in(haystack),
            Self::N6(searcher) => searcher.inlined_search_in(haystack),
            Self::N7(searcher) => searcher.inlined_search_in(haystack),
            Self::N8(searcher) => searcher.inlined_search_in(haystack),
            Self::N9(searcher) => searcher.inlined_search_in(haystack),
            Self::N10(searcher) => searcher.inlined_search_in(haystack),
            Self::N11(searcher) => searcher.inlined_search_in(haystack),
            Self::N12(searcher) => searcher.inlined_search_in(haystack),
            Self::N13(searcher) => searcher.inlined_search_in(haystack),
            Self::N(searcher) => searcher.inlined_search_in(haystack),
        }
    }

    #[inline]
    pub fn search_in(&self, haystack: &[u8]) -> bool {
        unsafe { self.inlined_search_in(haystack) }
    }
}

#[cfg(test)]
mod tests {
    use super::{AnySize, Avx2Searcher};
    use typenum::{Unsigned, U10, U11, U12, U13, U2, U3, U4, U5, U6, U7, U8, U9};

    fn search(haystack: &[u8], needle: &[u8]) -> bool {
        let search = |position| unsafe {
            let needle = needle.to_owned().into_boxed_slice();
            let result = Avx2Searcher::<AnySize>::with_position(needle.clone(), position)
                .search_in(haystack);

            macro_rules! search {
                ($size:ty) => {
                    if needle.len() == <$size>::to_usize() {
                        assert_eq!(
                            Avx2Searcher::<$size>::with_position(needle.clone(), position)
                                .search_in(haystack),
                            result
                        );
                    }
                };
                ($size:ty, $($sizes:ty),+) => {
                    search!($size);
                    search!($($sizes),+);
                };
            }

            search!(U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13);
            result
        };

        let result = search(1);
        for position in 2..needle.len() {
            assert_eq!(search(position), result);
        }

        result
    }

    #[test]
    fn search_same() {
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
        assert!(!search(b"ab", b"xy"));

        assert!(!search(b"bar", b"xy"));

        assert!(!search(b"bar", b"foo"));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"xy"
        ));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            b"foo"
        ));

        assert!(!search(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Maecenas commodo posuere orci a consectetur. Ut mattis turpis ut auctor consequat. Aliquam iaculis fringilla mi, nec aliquet purus",
            b"xy"
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
        assert!(search(b"xyz", b"xy"));

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
        assert!(search(b"wxy", b"xy"));

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
    fn search_mutiple() {
        assert!(search(b"xyzxyz", b"xy"));

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
