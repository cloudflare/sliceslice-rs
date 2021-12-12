#![allow(clippy::missing_safety_doc)]

use crate::{bits, memcmp, MemchrSearcher, Needle, NeedleWithSize};
use seq_macro::seq;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Represents an SIMD register type that is x86-specific (but could be used
/// more generically) in order to share functionality between SSE2, AVX2 and
/// possibly future implementations.
trait Vector: Copy {
    const LANES: usize;

    unsafe fn set1_epi8(a: i8) -> Self;

    unsafe fn loadu_si(a: *const u8) -> Self;

    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self;

    unsafe fn and_si(a: Self, b: Self) -> Self;

    unsafe fn movemask_epi8(a: Self) -> i32;
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct __m16i(__m128i);

impl Vector for __m16i {
    const LANES: usize = 2;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        __m16i(_mm_set1_epi8(a))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        __m16i(_mm_set1_epi16(std::ptr::read_unaligned(a as *const i16)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        __m16i(_mm_cmpeq_epi8(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        __m16i(_mm_and_si128(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm_movemask_epi8(a.0) & 0x3
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct __m32i(__m128i);

impl Vector for __m32i {
    const LANES: usize = 4;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        __m32i(_mm_set1_epi8(a))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        __m32i(_mm_set1_epi32(std::ptr::read_unaligned(a as *const i32)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        __m32i(_mm_cmpeq_epi8(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        __m32i(_mm_and_si128(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm_movemask_epi8(a.0) & 0xF
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct __m64i(__m128i);

impl Vector for __m64i {
    const LANES: usize = 8;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        __m64i(_mm_set1_epi8(a))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        __m64i(_mm_set1_epi64x(std::ptr::read_unaligned(a as *const i64)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        __m64i(_mm_cmpeq_epi8(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        __m64i(_mm_and_si128(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        _mm_movemask_epi8(a.0) & 0xFF
    }
}

impl Vector for __m128i {
    const LANES: usize = 16;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        _mm_loadu_si128(a as *const Self)
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
    const LANES: usize = 32;

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn set1_epi8(a: i8) -> Self {
        _mm256_set1_epi8(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        _mm256_loadu_si256(a as *const Self)
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

/// Hash of the first and "last" bytes in the needle for use with the SIMD
/// algorithm implemented by `Avx2Searcher::vector_search_in`. As explained, any
/// byte can be chosen to represent the "last" byte of the hash to prevent
/// worst-case attacks.
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

impl From<&VectorHash<__m128i>> for VectorHash<__m64i> {
    #[inline]
    fn from(hash: &VectorHash<__m128i>) -> Self {
        Self {
            first: __m64i(hash.first),
            last: __m64i(hash.last),
        }
    }
}

impl From<&VectorHash<__m128i>> for VectorHash<__m32i> {
    #[inline]
    fn from(hash: &VectorHash<__m128i>) -> Self {
        Self {
            first: __m32i(hash.first),
            last: __m32i(hash.last),
        }
    }
}

impl From<&VectorHash<__m128i>> for VectorHash<__m16i> {
    #[inline]
    fn from(hash: &VectorHash<__m128i>) -> Self {
        Self {
            first: __m16i(hash.first),
            last: __m16i(hash.last),
        }
    }
}

/// Single-substring searcher using an AVX2 algorithm based on the "Generic
/// SIMD" algorithm [presented by Wojciech
/// Muła](http://0x80.pl/articles/simd-strfind.html).
///
/// It is similar to the Rabin-Karp algorithm, except that the hash is not
/// rolling and is calculated for several lanes at once. It begins by picking
/// the first byte in the needle and checking at which positions in the haystack
/// it occurs. Any position where it does not can be immediately discounted as a
/// potential match.
///
/// We then repeat this idea with a second byte in the needle (where the
/// haystack is suitably offset) and take a bitwise AND to further limit the
/// possible positions the needle can match in. Any remaining positions are
/// fully evaluated using an equality comparison with the needle.
///
/// Originally, the algorithm always used the last byte for this second byte.
/// Whilst this is often the most efficient option, it is vulnerable to a
/// worst-case attack and so this implementation instead allows any byte
/// (including a random one) to be chosen.
///
/// In the case where the needle is not a multiple of the number of SIMD lanes,
/// the last chunk is made up of a partial overlap with the penultimate chunk to
/// avoid reading random memory, differing from the original implementation. In
/// this case, a mask is used to prevent performing an equality comparison on
/// the same position twice.
///
/// When the haystack is too short for an AVX2 register, a similar SSE2 fallback
/// is used instead. Finally, for very short haystacks there is a scalar
/// Rabin-Karp implementation.
pub struct Avx2Searcher<N: Needle> {
    position: usize,
    sse2_hash: VectorHash<__m128i>,
    avx2_hash: VectorHash<__m256i>,
    needle: N,
}

impl<N: Needle> Avx2Searcher<N> {
    /// Creates a new searcher for `needle`. By default, `position` is set to
    /// the last character in the needle.
    ///
    /// # Panics
    ///
    /// Panics if `needle` is empty or if the associated `SIZE` constant does
    /// not correspond to the actual size of `needle`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(needle: N) -> Self {
        // Wrapping prevents panicking on unsigned integer underflow when
        // `needle` is empty.
        let position = needle.size().wrapping_sub(1);
        Self::with_position(needle, position)
    }

    /// Same as `new` but allows additionally specifying the `position` to use.
    ///
    /// # Panics
    ///
    /// Panics if `needle` is empty, if `position` is not a valid index for
    /// `needle` or if the associated `SIZE` constant does not correspond to the
    /// actual size of `needle`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn with_position(needle: N, position: usize) -> Self {
        // Implicitly checks that the needle is not empty because position is an
        // unsized integer.
        assert!(position < needle.size());

        let bytes = needle.as_bytes();
        if let Some(size) = N::SIZE {
            assert_eq!(size, bytes.len());
        }

        let sse2_hash = VectorHash::new(bytes[0], bytes[position]);
        let avx2_hash = VectorHash::new(bytes[0], bytes[position]);

        Self {
            position,
            sse2_hash,
            avx2_hash,
            needle,
        }
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
        let first = Vector::loadu_si(start);
        let last = Vector::loadu_si(start.add(self.position));

        let eq_first = Vector::cmpeq_epi8(hash.first, first);
        let eq_last = Vector::cmpeq_epi8(hash.last, last);

        let eq = Vector::and_si(eq_first, eq_last);
        let mut eq = (Vector::movemask_epi8(eq) & mask) as u32;

        let start = start as usize - haystack.as_ptr() as usize;
        let chunk = haystack.as_ptr().add(start + 1);
        let needle = self.needle.as_bytes().as_ptr().add(1);

        while eq != 0 {
            let chunk = chunk.add(eq.trailing_zeros() as usize);
            let equal = match N::SIZE {
                Some(0) => unreachable!(),
                Some(1) => memcmp::specialized::<0>(chunk, needle),
                Some(2) => memcmp::specialized::<1>(chunk, needle),
                Some(3) => memcmp::specialized::<2>(chunk, needle),
                Some(4) => memcmp::specialized::<3>(chunk, needle),
                Some(5) => memcmp::specialized::<4>(chunk, needle),
                Some(6) => memcmp::specialized::<5>(chunk, needle),
                Some(7) => memcmp::specialized::<6>(chunk, needle),
                Some(8) => memcmp::specialized::<7>(chunk, needle),
                Some(9) => memcmp::specialized::<8>(chunk, needle),
                Some(10) => memcmp::specialized::<9>(chunk, needle),
                Some(11) => memcmp::specialized::<10>(chunk, needle),
                Some(12) => memcmp::specialized::<11>(chunk, needle),
                Some(13) => memcmp::specialized::<12>(chunk, needle),
                Some(14) => memcmp::specialized::<13>(chunk, needle),
                Some(15) => memcmp::specialized::<14>(chunk, needle),
                Some(16) => memcmp::specialized::<15>(chunk, needle),
                _ => memcmp::generic(chunk, needle, self.needle.size() - 1),
            };
            if equal {
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
        end: usize,
        hash: &VectorHash<V>,
    ) -> bool {
        debug_assert!(haystack.len() >= self.needle.size());

        let mut chunks = haystack[..end].chunks_exact(V::LANES);
        for chunk in &mut chunks {
            if self.vector_search_in_chunk(haystack, hash, chunk.as_ptr(), -1) {
                return true;
            }
        }

        let remainder = chunks.remainder().len();
        if remainder > 0 {
            let start = haystack.as_ptr().add(end - V::LANES);
            let mask = -1 << (V::LANES - remainder);

            if self.vector_search_in_chunk(haystack, hash, start, mask) {
                return true;
            }
        }

        false
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sse2_2_search_in(&self, haystack: &[u8], end: usize) -> bool {
        let hash = VectorHash::<__m16i>::from(&self.sse2_hash);
        self.vector_search_in(haystack, end, &hash)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sse2_4_search_in(&self, haystack: &[u8], end: usize) -> bool {
        let hash = VectorHash::<__m32i>::from(&self.sse2_hash);
        self.vector_search_in(haystack, end, &hash)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sse2_8_search_in(&self, haystack: &[u8], end: usize) -> bool {
        let hash = VectorHash::<__m64i>::from(&self.sse2_hash);
        self.vector_search_in(haystack, end, &hash)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sse2_16_search_in(&self, haystack: &[u8], end: usize) -> bool {
        self.vector_search_in(haystack, end, &self.sse2_hash)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_search_in(&self, haystack: &[u8], end: usize) -> bool {
        self.vector_search_in(haystack, end, &self.avx2_hash)
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.len() <= self.needle.size() {
            return haystack == self.needle.as_bytes();
        }

        let end = haystack.len() - self.needle.size() + 1;

        if end < __m16i::LANES {
            unreachable!();
        } else if end < __m32i::LANES {
            self.sse2_2_search_in(haystack, end)
        } else if end < __m64i::LANES {
            self.sse2_4_search_in(haystack, end)
        } else if end < __m128i::LANES {
            self.sse2_8_search_in(haystack, end)
        } else if end < __m256i::LANES {
            self.sse2_16_search_in(haystack, end)
        } else {
            self.avx2_search_in(haystack, end)
        }
    }

    /// Performs a substring search for the `needle` within `haystack`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

/// Single-substring searcher based on `Avx2Searcher` but with dynamic algorithm
/// selection.
///
/// It has specialized cases for zero-length needles, which are found in all
/// haystacks, and one-length needles, which uses `MemchrSearcher`. For needles
/// up to a length of thirteen it uses specialized versions of `Avx2Searcher`,
/// finally falling back to the generic version of `Avx2Searcher` for longer
/// needles.
pub enum DynamicAvx2Searcher<N: Needle> {
    /// Specialization for needles with length 0.
    N0,
    /// Specialization for needles with length 1.
    N1(MemchrSearcher),
    /// Specialization for needles with length 2.
    N2(Avx2Searcher<[u8; 2]>),
    /// Specialization for needles with length 3.
    N3(Avx2Searcher<[u8; 3]>),
    /// Specialization for needles with length 4.
    N4(Avx2Searcher<[u8; 4]>),
    /// Specialization for needles with length 5.
    N5(Avx2Searcher<[u8; 5]>),
    /// Specialization for needles with length 6.
    N6(Avx2Searcher<[u8; 6]>),
    /// Specialization for needles with length 7.
    N7(Avx2Searcher<[u8; 7]>),
    /// Specialization for needles with length 8.
    N8(Avx2Searcher<[u8; 8]>),
    /// Specialization for needles with length 9.
    N9(Avx2Searcher<[u8; 9]>),
    /// Specialization for needles with length 10.
    N10(Avx2Searcher<[u8; 10]>),
    /// Specialization for needles with length 11.
    N11(Avx2Searcher<[u8; 11]>),
    /// Specialization for needles with length 12.
    N12(Avx2Searcher<[u8; 12]>),
    /// Specialization for needles with length 13.
    N13(Avx2Searcher<[u8; 13]>),
    /// Specialization for needles with length 14.
    N14(Avx2Searcher<[u8; 14]>),
    /// Specialization for needles with length 15.
    N15(Avx2Searcher<[u8; 15]>),
    /// Specialization for needles with length 16.
    N16(Avx2Searcher<[u8; 16]>),
    /// Fallback implementation for needles of any size.
    N(Avx2Searcher<N>),
}

macro_rules! array {
    ($c:ident, $S:literal) => [seq!(N in 0..$S {
            [ #( $c#N, )* ]
    })];
}

impl<N: Needle> DynamicAvx2Searcher<N> {
    /// Creates a new searcher for `needle`. By default, `position` is set to
    /// the last character in the needle.
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(needle: N) -> Self {
        // Wrapping prevents panicking on unsigned integer underflow when
        // `needle` is empty.
        let position = needle.as_bytes().len().wrapping_sub(1);
        Self::with_position(needle, position)
    }

    /// Same as `new` but allows additionally specifying the `position` to use.
    ///
    /// # Panics
    ///
    /// When `needle` is not empty, panics if `position` is not a valid index
    /// for `needle`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn with_position(needle: N, position: usize) -> Self {
        match *needle.as_bytes() {
            [] => Self::N0,
            [c0] => {
                // Check that `position` is set correctly for consistency.
                assert_eq!(position, 0);
                Self::N1(MemchrSearcher::new(c0))
            }
            array!(c, 2) => Self::N2(Avx2Searcher::with_position(array!(c, 2), position)),
            array!(c, 3) => Self::N3(Avx2Searcher::with_position(array!(c, 3), position)),
            array!(c, 4) => Self::N4(Avx2Searcher::with_position(array!(c, 4), position)),
            array!(c, 5) => Self::N5(Avx2Searcher::with_position(array!(c, 5), position)),
            array!(c, 6) => Self::N6(Avx2Searcher::with_position(array!(c, 6), position)),
            array!(c, 7) => Self::N7(Avx2Searcher::with_position(array!(c, 7), position)),
            array!(c, 8) => Self::N8(Avx2Searcher::with_position(array!(c, 8), position)),
            array!(c, 9) => Self::N9(Avx2Searcher::with_position(array!(c, 9), position)),
            array!(c, 10) => Self::N10(Avx2Searcher::with_position(array!(c, 10), position)),
            array!(c, 11) => Self::N11(Avx2Searcher::with_position(array!(c, 11), position)),
            array!(c, 12) => Self::N12(Avx2Searcher::with_position(array!(c, 12), position)),
            array!(c, 13) => Self::N13(Avx2Searcher::with_position(array!(c, 13), position)),
            array!(c, 14) => Self::N14(Avx2Searcher::with_position(array!(c, 14), position)),
            array!(c, 15) => Self::N15(Avx2Searcher::with_position(array!(c, 15), position)),
            array!(c, 16) => Self::N16(Avx2Searcher::with_position(array!(c, 16), position)),
            _ => Self::N(Avx2Searcher::with_position(needle, position)),
        }
    }

    /// Inlined version of `search_in` for hot call sites.
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
            Self::N14(searcher) => searcher.inlined_search_in(haystack),
            Self::N15(searcher) => searcher.inlined_search_in(haystack),
            Self::N16(searcher) => searcher.inlined_search_in(haystack),
            Self::N(searcher) => searcher.inlined_search_in(haystack),
        }
    }

    /// Performs a substring search for the `needle` within `haystack`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

#[cfg(test)]
mod tests {
    use super::{Avx2Searcher, DynamicAvx2Searcher};
    use crate::Needle;

    #[test]
    #[should_panic]
    fn avx2_invalid_position() {
        unsafe { Avx2Searcher::with_position(b"foo".to_vec().into_boxed_slice(), 3) };
    }

    #[test]
    #[should_panic]
    fn dynamic_avx2_invalid_position() {
        unsafe { DynamicAvx2Searcher::with_position(b"foo".to_vec().into_boxed_slice(), 3) };
    }

    #[test]
    #[should_panic]
    fn avx2_empty_needle() {
        unsafe { Avx2Searcher::new(Box::new([])) };
    }

    #[test]
    #[should_panic]
    fn avx2_invalid_size() {
        struct Foo(&'static [u8]);

        impl Needle for Foo {
            const SIZE: Option<usize> = Some(2);

            fn as_bytes(&self) -> &[u8] {
                self.0
            }
        }

        unsafe { Avx2Searcher::new(Foo(b"foo")) };
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn size_of_avx2_searcher() {
        use std::mem::size_of;

        assert_eq!(size_of::<Avx2Searcher::<&[u8]>>(), 128);
        assert_eq!(size_of::<Avx2Searcher::<[u8; 0]>>(), 128);
        assert_eq!(size_of::<Avx2Searcher::<[u8; 16]>>(), 128);
        assert_eq!(size_of::<Avx2Searcher::<Box<[u8]>>>(), 128);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn size_of_dynamic_avx2_searcher() {
        use std::mem::size_of;

        assert_eq!(size_of::<DynamicAvx2Searcher::<&[u8]>>(), 160);
        assert_eq!(size_of::<DynamicAvx2Searcher::<[u8; 0]>>(), 160);
        assert_eq!(size_of::<DynamicAvx2Searcher::<[u8; 16]>>(), 160);
        assert_eq!(size_of::<DynamicAvx2Searcher::<Box<[u8]>>>(), 160);
    }
}
