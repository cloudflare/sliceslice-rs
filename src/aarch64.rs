#![allow(clippy::missing_safety_doc)]

use crate::{Needle, NeedleWithSize, Searcher, Vector, VectorHash};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

static MD: [u8; 16] = [
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
];

impl Vector for uint8x16_t {
    const LANES: usize = 16;

    #[inline]
    unsafe fn splat(a: u8) -> Self {
        vdupq_n_u8(a)
    }

    #[inline]
    unsafe fn load(a: *const u8) -> Self {
        vld1q_u8(a)
    }

    #[inline]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        vceqq_u8(a, b)
    }

    #[inline]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        vandq_u8(a, b)
    }

    #[inline]
    unsafe fn to_bitmask(a: Self) -> i32 {
        let extended = vreinterpretq_u8_s8(vshrq_n_s8(vreinterpretq_s8_u8(a), 7));
        let masked = vandq_u8(vld1q_u8(MD.as_ptr()), extended);
        let maskedhi = vextq_u8(masked, masked, 8);
        vaddvq_u16(vreinterpretq_u16_u8(vzip1q_u8(masked, maskedhi))).into()
    }
}

impl Vector for uint8x8_t {
    const LANES: usize = 8;

    #[inline]
    unsafe fn splat(a: u8) -> Self {
        vdup_n_u8(a)
    }

    #[inline]
    unsafe fn load(a: *const u8) -> Self {
        vld1_u8(a)
    }

    #[inline]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        vceq_u8(a, b)
    }

    #[inline]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        vand_u8(a, b)
    }

    #[inline]
    unsafe fn to_bitmask(a: Self) -> i32 {
        vaddv_u8(vand_u8(
            vreinterpret_u8_s8(vshr_n_s8(vreinterpret_s8_u8(a), 7)),
            vld1_u8(MD.as_ptr()),
        ))
        .into()
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct uint8x4_t(uint8x8_t);

impl Vector for uint8x4_t {
    const LANES: usize = 4;

    #[inline]
    unsafe fn splat(a: u8) -> Self {
        Self(uint8x8_t::splat(a))
    }

    #[inline]
    unsafe fn load(a: *const u8) -> Self {
        Self(uint8x8_t::load(a))
    }

    #[inline]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        Self(uint8x8_t::lanes_eq(a.0, b.0))
    }

    #[inline]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        Self(uint8x8_t::bitwise_and(a.0, b.0))
    }

    #[inline]
    unsafe fn to_bitmask(a: Self) -> i32 {
        uint8x8_t::to_bitmask(a.0) & 0xF
    }
}

impl From<uint8x8_t> for uint8x4_t {
    fn from(vector: uint8x8_t) -> Self {
        Self(vector)
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
struct uint8x2_t(uint8x8_t);

impl Vector for uint8x2_t {
    const LANES: usize = 2;

    #[inline]
    unsafe fn splat(a: u8) -> Self {
        Self(uint8x8_t::splat(a))
    }

    #[inline]
    unsafe fn load(a: *const u8) -> Self {
        Self(uint8x8_t::load(a))
    }

    #[inline]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        Self(uint8x8_t::lanes_eq(a.0, b.0))
    }

    #[inline]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        Self(uint8x8_t::bitwise_and(a.0, b.0))
    }

    #[inline]
    unsafe fn to_bitmask(a: Self) -> i32 {
        uint8x8_t::to_bitmask(a.0) & 0x3
    }
}

impl From<uint8x8_t> for uint8x2_t {
    fn from(vector: uint8x8_t) -> Self {
        Self(vector)
    }
}

/// Searcher for aarch64 architecture.
pub struct NeonSearcher<N: Needle> {
    position: usize,
    neon_hash: VectorHash<uint8x16_t>,
    neon_half_hash: VectorHash<uint8x8_t>,
    needle: N,
}

impl<N: Needle> NeonSearcher<N> {
    /// Creates a new searcher for `needle`. By default, `position` is set to
    /// the last character in the needle.
    ///
    /// # Panics
    ///
    /// Panics if `needle` is empty or if the associated `SIZE` constant does
    /// not correspond to the actual size of `needle`.
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
    pub unsafe fn with_position(needle: N, position: usize) -> Self {
        // Implicitly checks that the needle is not empty because position is an
        // unsized integer.
        assert!(position < needle.size());

        let bytes = needle.as_bytes();
        if let Some(size) = N::SIZE {
            assert_eq!(size, bytes.len());
        }

        let neon_hash = VectorHash::new(bytes[0], bytes[position]);
        let neon_half_hash = VectorHash::new(bytes[0], bytes[position]);

        Self {
            position,
            neon_hash,
            neon_half_hash,
            needle,
        }
    }

    #[inline]
    unsafe fn neon_2_search_in(&self, haystack: &[u8], end: usize) -> bool {
        let hash = VectorHash::<uint8x2_t>::from(&self.neon_half_hash);
        self.vector_search_in_neon_version(haystack, end, &hash)
    }

    #[inline]
    unsafe fn neon_4_search_in(&self, haystack: &[u8], end: usize) -> bool {
        let hash = VectorHash::<uint8x4_t>::from(&self.neon_half_hash);
        self.vector_search_in_neon_version(haystack, end, &hash)
    }

    #[inline]
    unsafe fn neon_8_search_in(&self, haystack: &[u8], end: usize) -> bool {
        self.vector_search_in_neon_version(haystack, end, &self.neon_half_hash)
    }

    #[inline]
    unsafe fn neon_search_in(&self, haystack: &[u8], end: usize) -> bool {
        self.vector_search_in_neon_version(haystack, end, &self.neon_hash)
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    pub unsafe fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.len() <= self.needle.size() {
            return haystack == self.needle.as_bytes();
        }

        let end = haystack.len() - self.needle.size() + 1;

        if end < uint8x2_t::LANES {
            unreachable!();
        } else if end < uint8x4_t::LANES {
            self.neon_2_search_in(haystack, end)
        } else if end < uint8x8_t::LANES {
            self.neon_4_search_in(haystack, end)
        } else if end < uint8x16_t::LANES {
            self.neon_8_search_in(haystack, end)
        } else {
            self.neon_search_in(haystack, end)
        }
    }

    /// Performs a substring search for the `needle` within `haystack`.
    pub unsafe fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

impl<N: Needle> Searcher<N> for NeonSearcher<N> {
    #[inline(always)]
    fn needle(&self) -> &N {
        &self.needle
    }

    #[inline(always)]
    fn position(&self) -> usize {
        self.position
    }
}
