use crate::{bits::clear_leftmost_set, memcmp::*};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn strstr_avx2_original_memcmp(
    haystack: *const u8,
    n: usize,
    needle: *const u8,
    k: usize,
    memcmp: unsafe fn(*const u8, *const u8, usize) -> bool,
) -> Option<usize> {
    let first = _mm256_set1_epi8(*needle as i8);
    let last = _mm256_set1_epi8(*needle.add(k - 1) as i8);
    let mut i = 0;
    while i < n {
        #[allow(clippy::cast_ptr_alignment)]
        let block_first = _mm256_loadu_si256(haystack.add(i) as *const __m256i);
        #[allow(clippy::cast_ptr_alignment)]
        let block_last = _mm256_loadu_si256(haystack.add(i + k - 1) as *const __m256i);

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if memcmp(haystack.add(startpos), needle.add(1), k - 2) {
                return Some(i + bitpos);
            }
            mask = clear_leftmost_set(mask);
        }
        i += 32;
    }

    None
}

/// Original version copied from the [implementation by Wojciech
/// Muła](http://0x80.pl/articles/simd-strfind.html). This version is somewhat not safe because it
/// can read past the end of the haystack slice.
#[target_feature(enable = "avx2")]
pub unsafe fn strstr_avx2_original(haystack: &[u8], needle: &[u8]) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    let result = match needle.len() {
        0 => Some(0),
        1 => memchr::memchr(needle[0], haystack),
        2 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp0,
        ),
        3 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp1,
        ),
        4 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp2,
        ),
        5 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp3,
        ),
        6 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp4,
        ),
        7 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp5,
        ),
        8 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp6,
        ),
        9 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp7,
        ),
        10 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp8,
        ),
        11 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp9,
        ),
        12 => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp10,
        ),
        _ => strstr_avx2_original_memcmp(
            haystack.as_ptr(),
            haystack.len(),
            needle.as_ptr(),
            needle.len(),
            memcmp,
        ),
    };
    if let Some(result) = result {
        result <= haystack.len() - needle.len()
    } else {
        false
    }
}
