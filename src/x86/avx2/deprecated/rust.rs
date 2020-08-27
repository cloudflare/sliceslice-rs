#![allow(clippy::cast_ptr_alignment)]

use crate::{bits::clear_leftmost_set, memcmp::*};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn strstr_avx2_rust_memcmp(
    haystack: &[u8],
    needle: &[u8],
    memcmp: unsafe fn(*const u8, *const u8, usize) -> bool,
) -> bool {
    if haystack.len() < 32 {
        return strstr_rabin_karp(haystack, needle);
    }
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut chunks = haystack[..=(haystack.len() - needle.len())].chunks_exact(32);
    while let Some(chunk) = chunks.next() {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        let block_first = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let block_last = _mm256_loadu_si256(chunk.as_ptr().add(needle.len() - 1) as *const __m256i);

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos;
            if startpos + needle.len() <= haystack.len()
                && memcmp(
                    haystack.as_ptr().add(startpos + 1),
                    needle.as_ptr().add(1),
                    needle.len() - 2,
                )
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    let chunk = chunks.remainder();
    let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
    let chunk = &haystack[i..];
    if needle.len() <= chunk.len() {
        strstr_rabin_karp(chunk, needle)
    } else {
        false
    }
}

#[inline]
fn strstr_rabin_karp(haystack: &[u8], needle: &[u8]) -> bool {
    let mut needle_sum = 0_usize;
    for &c in needle {
        needle_sum += c as usize;
    }

    let mut haystack_sum = 0_usize;
    for &c in &haystack[..needle.len() - 1] {
        haystack_sum += c as usize;
    }

    let mut i = needle.len() - 1;
    while i < haystack.len() {
        haystack_sum += *unsafe { haystack.get_unchecked(i) } as usize;
        i += 1;
        if haystack_sum == needle_sum && &haystack[(i - needle.len())..i] == needle {
            return true;
        }
        haystack_sum -= *unsafe { haystack.get_unchecked(i - needle.len()) } as usize;
    }

    false
}

/// Similar to `strstr_avx2_original` implementation, but adapted for safety to prevent reading past
/// the end of the haystack.
#[target_feature(enable = "avx2")]
pub unsafe fn strstr_avx2_rust(haystack: &[u8], needle: &[u8]) -> bool {
    match needle.len() {
        0 => true,
        1 => memchr::memchr(needle[0], haystack).is_some(),
        2 => strstr_avx2_rust_memcmp(haystack, needle, memcmp0),
        3 => strstr_avx2_rust_memcmp(haystack, needle, memcmp1),
        4 => strstr_avx2_rust_memcmp(haystack, needle, memcmp2),
        5 => strstr_avx2_rust_memcmp(haystack, needle, memcmp3),
        6 => strstr_avx2_rust_memcmp(haystack, needle, memcmp4),
        7 => strstr_avx2_rust_memcmp(haystack, needle, memcmp5),
        8 => strstr_avx2_rust_memcmp(haystack, needle, memcmp6),
        9 => strstr_avx2_rust_memcmp(haystack, needle, memcmp7),
        10 => strstr_avx2_rust_memcmp(haystack, needle, memcmp8),
        11 => strstr_avx2_rust_memcmp(haystack, needle, memcmp9),
        12 => strstr_avx2_rust_memcmp(haystack, needle, memcmp10),
        13 => strstr_avx2_rust_memcmp(haystack, needle, memcmp11),
        14 => strstr_avx2_rust_memcmp(haystack, needle, memcmp12),
        _ => strstr_avx2_rust_memcmp(haystack, needle, memcmp),
    }
}

#[cfg(test)]
mod tests {
    use super::strstr_avx2_rust;

    #[test]
    fn needle_length_3() {
        let mut input = [0; 32];

        for i in 0..=(input.len() - 3) {
            input = [b'A'; 32];
            input[i..(i + 3)].copy_from_slice(&[b'B'; 3]);
            assert_eq!(
                unsafe { strstr_avx2_rust(&input[..], b"BBB") },
                true,
                "{:?} should contain {:?}",
                &input[..],
                b"BBB"
            );
        }

        let mut input = [0; 63];

        for i in 0..=(input.len() - 3) {
            input = [b'A'; 63];
            input[i..(i + 3)].copy_from_slice(&[b'B'; 3]);
            assert_eq!(
                unsafe { strstr_avx2_rust(&input[..], b"BBB") },
                true,
                "{:?} should contain {:?}",
                &input[..],
                b"BBB"
            );
        }
    }
}
