#![allow(clippy::missing_safety_doc)]

use crate::{Needle, NeedleWithSize, Searcher, Vector, VectorHash};
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

impl Vector for v128 {
    const LANES: usize = 16;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn splat(a: u8) -> Self {
        u8x16_splat(a)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load(a: *const u8) -> Self {
        std::ptr::read_unaligned(a as *const v128)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        u8x16_eq(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        v128_and(a, b)
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn to_bitmask(a: Self) -> i32 {
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
    unsafe fn splat(a: u8) -> Self {
        Self(u8x16_splat(a))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load(a: *const u8) -> Self {
        Self(u64x2_splat(std::ptr::read_unaligned(a as *const u64)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn to_bitmask(a: Self) -> i32 {
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
    unsafe fn splat(a: u8) -> Self {
        Self(u8x16_splat(a))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load(a: *const u8) -> Self {
        Self(u32x4_splat(std::ptr::read_unaligned(a as *const u32)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn to_bitmask(a: Self) -> i32 {
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
    unsafe fn splat(a: u8) -> Self {
        Self(u8x16_splat(a))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn load(a: *const u8) -> Self {
        Self(u16x8_splat(std::ptr::read_unaligned(a as *const u16)))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn lanes_eq(a: Self, b: Self) -> Self {
        Self(u8x16_eq(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn bitwise_and(a: Self, b: Self) -> Self {
        Self(v128_and(a.0, b.0))
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn to_bitmask(a: Self) -> i32 {
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
