mod original;
mod rust;

pub use self::{original::*, rust::*};

use crate::{bits, memchr::MemchrSearcher, memcmp};
use std::{arch::x86_64::*, mem};

#[derive(Clone, Copy, Default, PartialEq)]
struct ScalarHash(usize);

impl From<&[u8]> for ScalarHash {
    #[inline(always)]
    fn from(bytes: &[u8]) -> Self {
        bytes.iter().fold(Default::default(), |mut hash, &b| {
            hash.push(b);
            hash
        })
    }
}

impl ScalarHash {
    #[inline(always)]
    fn push(&mut self, b: u8) {
        self.0 ^= usize::from(b);
    }

    #[inline(always)]
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
    #[inline(always)]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm_set1_epi8(a)
    }

    #[inline(always)]
    unsafe fn loadu_si(a: *const Self) -> Self {
        _mm_loadu_si128(a)
    }

    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        _mm_cmpeq_epi8(a, b)
    }

    #[inline(always)]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        _mm_and_si128(a, b)
    }

    #[inline(always)]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm_movemask_epi8(a)
    }
}

impl Vector for __m256i {
    #[inline(always)]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm256_set1_epi8(a)
    }

    #[inline(always)]
    unsafe fn loadu_si(a: *const Self) -> Self {
        _mm256_loadu_si256(a)
    }

    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        _mm256_cmpeq_epi8(a, b)
    }

    #[inline(always)]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        _mm256_and_si256(a, b)
    }

    #[inline(always)]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm256_movemask_epi8(a)
    }
}

struct VectorHash<V: Vector> {
    first: V,
    last: V,
}

impl<V: Vector> VectorHash<V> {
    fn new(first: u8, last: u8) -> Self {
        Self {
            first: unsafe { Vector::set1_epi8(first as i8) },
            last: unsafe { Vector::set1_epi8(last as i8) },
        }
    }
}

macro_rules! avx2_searcher {
    ($name:ident, $size:literal, $memcmp:path) => {
        pub struct $name {
            needle: Box<[u8]>,
            position: usize,
            scalar_hash: ScalarHash,
            sse2_hash: VectorHash<__m128i>,
            avx2_hash: VectorHash<__m256i>,
        }

        impl $name {
            pub fn new(needle: Box<[u8]>) -> Self {
                let position = needle.len() - 1;
                Self::with_position(needle, position)
            }

            pub fn with_position(needle: Box<[u8]>, position: usize) -> Self {
                assert!(!needle.is_empty());
                assert!(position < needle.len());

                let scalar_hash = ScalarHash::from(needle.as_ref());
                let sse2_hash = VectorHash::new(needle[0], needle[position]);
                let avx2_hash = VectorHash::new(needle[0], needle[position]);

                Self {
                    needle,
                    position,
                    scalar_hash,
                    sse2_hash,
                    avx2_hash,
                }
            }

            #[inline(always)]
            fn size(&self) -> usize {
                if $size > 0 {
                    $size
                } else {
                    self.needle.len()
                }
            }

            #[inline(always)]
            fn scalar_search_in(&self, haystack: &[u8]) -> bool {
                debug_assert!(haystack.len() >= self.size());

                let mut end = self.size() - 1;
                let mut hash = ScalarHash::from(&haystack[..end]);

                while end < haystack.len() {
                    hash.push(*unsafe { haystack.get_unchecked(end) });
                    end += 1;

                    let start = end - self.size();
                    if hash == self.scalar_hash && haystack[start..end] == *self.needle {
                        return true;
                    }

                    hash.pop(*unsafe { haystack.get_unchecked(start) });
                }

                false
            }

            #[inline(always)]
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

                let start = start as usize - haystack.as_ptr() as usize;
                while eq != 0 {
                    let chunk = &haystack[start + eq.trailing_zeros() as usize..];
                    if $memcmp(&chunk[1..self.size()], &self.needle[1..]) {
                        return true;
                    }

                    eq = bits::clear_leftmost_set(eq);
                }

                false
            }

            #[inline(always)]
            fn vector_search_in<V: Vector>(
                &self,
                haystack: &[u8],
                hash: &VectorHash<V>,
                next: fn(&Self, &[u8]) -> bool,
            ) -> bool {
                debug_assert!(haystack.len() >= self.size());

                let lanes = mem::size_of::<V>();
                let end = haystack.len() - self.size() + 1;

                if end < lanes {
                    return next(self, haystack);
                }

                let mut chunks = haystack[..end].chunks_exact(lanes);
                while let Some(chunk) = chunks.next() {
                    if unsafe { self.vector_search_in_chunk(haystack, hash, chunk.as_ptr(), -1) } {
                        return true;
                    }
                }

                let remainder = chunks.remainder().len();
                if remainder > 0 {
                    let start = unsafe { haystack.as_ptr().add(end - lanes) };
                    let mask = -1 << (lanes - remainder);

                    if unsafe { self.vector_search_in_chunk(haystack, hash, start, mask) } {
                        return true;
                    }
                }

                false
            }

            #[inline(always)]
            fn sse2_search_in(&self, haystack: &[u8]) -> bool {
                self.vector_search_in(haystack, &self.sse2_hash, Self::scalar_search_in)
            }

            #[inline(always)]
            fn avx2_search_in(&self, haystack: &[u8]) -> bool {
                self.vector_search_in(haystack, &self.avx2_hash, Self::sse2_search_in)
            }

            #[inline(always)]
            pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
                if haystack.len() < self.size() {
                    return false;
                }

                self.avx2_search_in(haystack)
            }

            pub fn search_in(&self, haystack: &[u8]) -> bool {
                self.inlined_search_in(haystack)
            }
        }
    };
}

avx2_searcher!(Avx2Searcher, 0, memcmp::memcmp);
avx2_searcher!(Avx2Searcher2, 2, memcmp::memcmp1);
avx2_searcher!(Avx2Searcher3, 3, memcmp::memcmp2);
avx2_searcher!(Avx2Searcher4, 4, memcmp::memcmp3);
avx2_searcher!(Avx2Searcher5, 5, memcmp::memcmp4);
avx2_searcher!(Avx2Searcher6, 6, memcmp::memcmp5);
avx2_searcher!(Avx2Searcher7, 7, memcmp::memcmp6);
avx2_searcher!(Avx2Searcher8, 8, memcmp::memcmp7);
avx2_searcher!(Avx2Searcher9, 9, memcmp::memcmp8);
avx2_searcher!(Avx2Searcher10, 10, memcmp::memcmp9);
avx2_searcher!(Avx2Searcher11, 11, memcmp::memcmp10);
avx2_searcher!(Avx2Searcher12, 12, memcmp::memcmp11);
avx2_searcher!(Avx2Searcher13, 13, memcmp::memcmp12);

pub enum DynamicAvx2Searcher {
    N0,
    N1(MemchrSearcher),
    N2(Avx2Searcher2),
    N3(Avx2Searcher3),
    N4(Avx2Searcher4),
    N5(Avx2Searcher5),
    N6(Avx2Searcher6),
    N7(Avx2Searcher7),
    N8(Avx2Searcher8),
    N9(Avx2Searcher9),
    N10(Avx2Searcher10),
    N11(Avx2Searcher11),
    N12(Avx2Searcher12),
    N13(Avx2Searcher13),
    N(Avx2Searcher),
}

impl DynamicAvx2Searcher {
    pub fn new(needle: Box<[u8]>) -> Self {
        let position = needle.len() - 1;
        Self::with_position(needle, position)
    }

    pub fn with_position(needle: Box<[u8]>, position: usize) -> Self {
        assert!(!needle.is_empty());
        assert!(position < needle.len());

        match needle.len() {
            0 => Self::N0,
            1 => Self::N1(MemchrSearcher::new(needle[0])),
            2 => Self::N2(Avx2Searcher2::new(needle)),
            3 => Self::N3(Avx2Searcher3::new(needle)),
            4 => Self::N4(Avx2Searcher4::new(needle)),
            5 => Self::N5(Avx2Searcher5::new(needle)),
            6 => Self::N6(Avx2Searcher6::new(needle)),
            7 => Self::N7(Avx2Searcher7::new(needle)),
            8 => Self::N8(Avx2Searcher8::new(needle)),
            9 => Self::N9(Avx2Searcher9::new(needle)),
            10 => Self::N10(Avx2Searcher10::new(needle)),
            11 => Self::N11(Avx2Searcher11::new(needle)),
            12 => Self::N12(Avx2Searcher12::new(needle)),
            13 => Self::N13(Avx2Searcher13::new(needle)),
            _ => Self::N(Avx2Searcher::new(needle)),
        }
    }

    #[inline(always)]
    pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
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

    pub fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

#[cfg(test)]
mod tests {
    use super::Avx2Searcher;

    fn search(haystack: &[u8], needle: &[u8]) -> bool {
        let search = |position| {
            Avx2Searcher::with_position(needle.to_owned().into_boxed_slice(), position)
                .search_in(haystack)
        };

        let result = search(0);
        for position in 1..needle.len() {
            assert_eq!(search(position), result);
        }

        result
    }

    #[test]
    fn search_same() {
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
