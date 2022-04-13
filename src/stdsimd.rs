#![allow(clippy::missing_safety_doc)]

use crate::{Needle, NeedleWithSize, Searcher, Vector, VectorHash};
#[cfg(feature = "stdsimd")]
use std::simd::*;

trait ToFixedBitMask: Sized {
    fn to_fixed_bitmask(self) -> u32;
}

impl<const LANES: usize> ToFixedBitMask for Mask<i8, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Self: ToBitMask,
    <Self as ToBitMask>::BitMask: Into<u32>,
{
    #[inline]
    fn to_fixed_bitmask(self) -> u32 {
        self.to_bitmask().into()
    }
}

impl<const LANES: usize> Vector for Simd<u8, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Mask<i8, LANES>: ToFixedBitMask,
{
    const LANES: usize = LANES;
    type Mask = Mask<i8, LANES>;

    #[inline]
    unsafe fn splat(a: u8) -> Self {
        Simd::splat(a as u8)
    }

    #[inline]
    unsafe fn load(a: *const u8) -> Self {
        std::ptr::read_unaligned(a as *const Self)
    }

    #[inline]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self::Mask {
        a.lanes_eq(b)
    }

    #[inline]
    unsafe fn bitwise_and(a: Self::Mask, b: Self::Mask) -> Self::Mask {
        a & b
    }

    #[inline]
    unsafe fn to_bitmask(a: Self::Mask) -> i32 {
        std::mem::transmute(a.to_fixed_bitmask())
    }
}

type Simd2 = Simd<u8, 2>;
type Simd4 = Simd<u8, 4>;
type Simd8 = Simd<u8, 8>;
type Simd16 = Simd<u8, 16>;
type Simd32 = Simd<u8, 32>;

fn from_hash<const N1: usize, const N2: usize>(
    hash: &VectorHash<Simd<u8, N1>>,
) -> VectorHash<Simd<u8, N2>>
where
    LaneCount<N1>: SupportedLaneCount,
    Mask<i8, N1>: ToFixedBitMask,
    LaneCount<N2>: SupportedLaneCount,
    Mask<i8, N2>: ToFixedBitMask,
{
    VectorHash {
        first: Simd::splat(hash.first.as_array()[0]),
        last: Simd::splat(hash.last.as_array()[0]),
    }
}

/// Searcher for portable simd.
pub struct StdSimdSearcher<N: Needle> {
    needle: N,
    position: usize,
    simd32_hash: VectorHash<Simd32>,
}

impl<N: Needle> Searcher<N> for StdSimdSearcher<N> {
    fn needle(&self) -> &N {
        &self.needle
    }

    fn position(&self) -> usize {
        self.position
    }
}

impl<N: Needle> StdSimdSearcher<N> {
    /// Creates a new searcher for `needle`. By default, `position` is set to
    /// the last character in the needle.
    ///
    /// # Panics
    ///
    /// Panics if `needle` is empty or if the associated `SIZE` constant does
    /// not correspond to the actual size of `needle`.
    pub fn new(needle: N) -> Self {
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
    #[inline]
    pub fn with_position(needle: N, position: usize) -> Self {
        // Implicitly checks that the needle is not empty because position is an
        // unsized integer.
        assert!(position < needle.size());

        let bytes = needle.as_bytes();
        if let Some(size) = N::SIZE {
            assert_eq!(size, bytes.len());
        }

        let simd32_hash = unsafe { VectorHash::new(bytes[0], bytes[position]) };

        Self {
            position,
            simd32_hash,
            needle,
        }
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.len() <= self.needle.size() {
            return haystack == self.needle.as_bytes();
        }

        let end = haystack.len() - self.needle.size() + 1;

        if end < Simd2::LANES {
            unreachable!();
        } else if end < Simd4::LANES {
            let hash = from_hash::<32, 2>(&self.simd32_hash);
            println!("hash: {:?}", hash);
            unsafe { self.vector_search_in_default_version(haystack, end, &hash) }
        } else if end < Simd8::LANES {
            let hash = from_hash::<32, 4>(&self.simd32_hash);
            unsafe { self.vector_search_in_default_version(haystack, end, &hash) }
        } else if end < Simd16::LANES {
            let hash = from_hash::<32, 8>(&self.simd32_hash);
            unsafe { self.vector_search_in_default_version(haystack, end, &hash) }
        } else if end < Simd32::LANES {
            let hash = from_hash::<32, 16>(&self.simd32_hash);
            unsafe { self.vector_search_in_default_version(haystack, end, &hash) }
        } else {
            unsafe { self.vector_search_in_default_version(haystack, end, &self.simd32_hash) }
        }
    }

    /// Performs a substring search for the `needle` within `haystack`.
    pub fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}
/*
impl Vector for u8x32 {
    const LANES: usize = 32;

    #[inline]
    unsafe fn set1_epi8(a: i8) -> Self {
        Simd::splat(a as u8)
    }

    #[inline]
    unsafe fn loadu_si(a: *const u8) -> Self {
        std::ptr::read_unaligned(a as *const Self)
    }

    #[inline]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        a.lanes_eq(b).to_int()
    }

    #[inline]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        a | b
    }

    #[inline]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        u8x16_bitmask(a) as i32
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct v64(v128);

impl Vector for v64 {
    const LANES: usize = 8;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn set1_epi8(a: i8) -> Self {
        Self(u8x16_splat(a as u8))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        Self(u64x2_splat(std::ptr::read_unaligned(a as *const u64)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        (u8x16_bitmask(a.0) & 0xFF) as i32
    }
}

impl From<v128> for v64 {
    fn from(vector: v128) -> Self {
        Self(vector)
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct v32(v128);

impl Vector for v32 {
    const LANES: usize = 4;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn set1_epi8(a: i8) -> Self {
        Self(u8x16_splat(a as u8))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        Self(u32x4_splat(std::ptr::read_unaligned(a as *const u32)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        (u8x16_bitmask(a.0) & 0xF) as i32
    }
}

impl From<v128> for v32 {
    fn from(vector: v128) -> Self {
        Self(vector)
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct v16(v128);

impl Vector for v16 {
    const LANES: usize = 2;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn set1_epi8(a: i8) -> Self {
        Self(u8x16_splat(a as u8))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn loadu_si(a: *const u8) -> Self {
        Self(u16x8_splat(std::ptr::read_unaligned(a as *const u16)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn cmpeq_epi8(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn and_si(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn movemask_epi8(a: Self) -> i32 {
        (u8x16_bitmask(a.0) & 0x3) as i32
    }
}

impl From<v128> for v16 {
    fn from(vector: v128) -> Self {
        Self(vector)
    }
}

/// Searcher for wasm32 architecture.
pub struct Wasm32Searcher<N: Needle> {
    needle: N,
    position: usize,
    v128_hash: VectorHash<v128>,
}

impl<N: Needle> Searcher<N> for Wasm32Searcher<N> {
    fn needle(&self) -> &N {
        &self.needle
    }

    fn position(&self) -> usize {
        self.position
    }
}

impl<N: Needle> Wasm32Searcher<N> {
    /// Creates a new searcher for `needle`. By default, `position` is set to
    /// the last character in the needle.
    ///
    /// # Panics
    ///
    /// Panics if `needle` is empty or if the associated `SIZE` constant does
    /// not correspond to the actual size of `needle`.
    #[target_feature(enable = "simd128")]
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
    #[inline]
    #[target_feature(enable = "simd128")]
    pub unsafe fn with_position(needle: N, position: usize) -> Self {
        // Implicitly checks that the needle is not empty because position is an
        // unsized integer.
        assert!(position < needle.size());

        let bytes = needle.as_bytes();
        if let Some(size) = N::SIZE {
            assert_eq!(size, bytes.len());
        }

        let v128_hash = VectorHash::new(bytes[0], bytes[position]);

        Self {
            position,
            v128_hash,
            needle,
        }
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub unsafe fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.len() <= self.needle.size() {
            return haystack == self.needle.as_bytes();
        }

        let end = haystack.len() - self.needle.size() + 1;

        if end < v16::LANES {
            unreachable!();
        } else if end < v32::LANES {
            let hash = VectorHash::<v16>::from(&self.v128_hash);
            self.vector_search_in_simd128_version(haystack, end, &hash)
        } else if end < v64::LANES {
            let hash = VectorHash::<v32>::from(&self.v128_hash);
            self.vector_search_in_simd128_version(haystack, end, &hash)
        } else if end < v128::LANES {
            let hash = VectorHash::<v64>::from(&self.v128_hash);
            self.vector_search_in_simd128_version(haystack, end, &hash)
        } else {
            self.vector_search_in_simd128_version(haystack, end, &self.v128_hash)
        }
    }

    /// Performs a substring search for the `needle` within `haystack`.
    #[target_feature(enable = "simd128")]
    pub unsafe fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}
*/
