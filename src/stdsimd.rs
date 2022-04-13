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
        Simd::splat(a)
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
    unsafe fn to_bitmask(a: Self::Mask) -> u32 {
        a.to_fixed_bitmask()
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
