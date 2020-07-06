use crate::bits::*;
use crate::memcmp::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub unsafe fn strstr_avx2_rust_simple(haystack: &[u8], needle: &[u8]) -> bool {
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut block_pad: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    for chunk in haystack.chunks(32) {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        let block_first = if chunk.len() == 32 {
            _mm256_loadu_si256(chunk.as_ptr() as *const __m256i)
        } else {
            block_pad[..chunk.len()].copy_from_slice(chunk);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && haystack[startpos..startpos + needle.len() - 2] == needle[1..needle.len() - 1]
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn strstr_avx2_rust_simple_2(haystack: &[u8], needle: &[u8]) -> bool {
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut block_pad: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let mut chunks = haystack.chunks_exact(32);
    while let Some(chunk) = chunks.next() {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        let block_first = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() - 2 <= haystack.len()
                && haystack[startpos..startpos + needle.len() - 2] == needle[1..needle.len() - 1]
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    let chunk = chunks.remainder();

    if chunk.len() > 0 {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        block_pad[..chunk.len()].copy_from_slice(chunk);
        let block_first = _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i);
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && haystack[startpos..startpos + needle.len() - 2] == needle[1..needle.len() - 1]
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn strstr_avx2_rust_fast_memcmp(
    haystack: &[u8],
    needle: &[u8],
    memcmp: unsafe fn(&[u8], &[u8]) -> bool,
) -> bool {
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut block_first_pad: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let mut block_last_pad: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    for chunk in haystack.chunks(32) {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        let block_first = if chunk.len() == 32 {
            _mm256_loadu_si256(chunk.as_ptr() as *const __m256i)
        } else {
            block_first_pad[..chunk.len()].copy_from_slice(chunk);
            _mm256_loadu_si256(block_first_pad.as_ptr() as *const __m256i)
        };
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_last_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_last_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && memcmp(
                    &haystack[startpos..startpos + needle.len() - 2],
                    &needle[1..needle.len() - 1],
                )
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn strstr_avx2_rust_fast(haystack: &[u8], needle: &[u8]) -> bool {
    match needle.len() {
        0 => true,
        1 => memchr::memchr(needle[0], haystack).is_some(),
        2 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp0),
        3 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp1),
        4 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp2),
        5 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp4),
        6 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp4),
        7 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp5),
        8 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp6),
        9 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp8),
        10 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp8),
        11 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp9),
        12 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp10),
        13 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp11),
        14 => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp12),
        _ => strstr_avx2_rust_fast_memcmp(haystack, needle, memcmp),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn strstr_avx2_rust_fast_2_memcmp(
    haystack: &[u8],
    needle: &[u8],
    memcmp: unsafe fn(&[u8], &[u8]) -> bool,
) -> bool {
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut block_pad: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let mut chunks = haystack.chunks_exact(32);
    while let Some(chunk) = chunks.next() {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        let block_first = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && memcmp(
                    &haystack[startpos..startpos + needle.len() - 2],
                    &needle[1..needle.len() - 1],
                )
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    let chunk = chunks.remainder();

    if chunk.len() > 0 {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        block_pad[..chunk.len()].copy_from_slice(chunk);
        let block_first = _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i);
        let block_last = if i + 31 + needle.len() <= haystack.len() {
            _mm256_loadu_si256(chunk[(needle.len() - 1)..].as_ptr() as *const __m256i)
        } else {
            let start = &haystack[(i + needle.len() - 1)..];
            block_pad[..start.len()].copy_from_slice(start);
            _mm256_loadu_si256(block_pad.as_ptr() as *const __m256i)
        };

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && memcmp(
                    &haystack[startpos..startpos + needle.len() - 2],
                    &needle[1..needle.len() - 1],
                )
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    false
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn strstr_avx2_rust_fast_2(haystack: &[u8], needle: &[u8]) -> bool {
    match needle.len() {
        0 => true,
        1 => memchr::memchr(needle[0], haystack).is_some(),
        2 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp0),
        3 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp1),
        4 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp2),
        5 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp4),
        6 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp4),
        7 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp5),
        8 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp6),
        9 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp8),
        10 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp8),
        11 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp9),
        12 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp10),
        13 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp11),
        14 => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp12),
        _ => strstr_avx2_rust_fast_2_memcmp(haystack, needle, memcmp),
    }
}

#[repr(align(32))]
struct AlignedVector([u8; 32]);

#[cfg(target_arch = "x86_64")]
pub unsafe fn strstr_avx2_rust_aligned(haystack: &[u8], needle: &[u8]) -> bool {
    let first = _mm256_set1_epi8(needle[0] as i8);
    let last = _mm256_set1_epi8(needle[needle.len() - 1] as i8);
    let mut block_pad = AlignedVector([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ]);
    for chunk in haystack.chunks(32) {
        let i = chunk.as_ptr() as usize - haystack.as_ptr() as usize;
        block_pad.0[..chunk.len()].copy_from_slice(chunk);
        let block_first = _mm256_load_si256(block_pad.0.as_ptr() as *const __m256i);
        let start = &haystack[(i + needle.len() - 1)..(i + 32 + needle.len() - 1)];
        block_pad.0[..start.len()].copy_from_slice(start);
        let block_last = _mm256_load_si256(block_pad.0.as_ptr() as *const __m256i);

        let eq_first = _mm256_cmpeq_epi8(first, block_first);
        let eq_last = _mm256_cmpeq_epi8(last, block_last);

        let mut mask = std::mem::transmute::<i32, u32>(_mm256_movemask_epi8(_mm256_and_si256(
            eq_first, eq_last,
        )));
        while mask != 0 {
            let bitpos = mask.trailing_zeros() as usize;
            let startpos = i + bitpos + 1;
            if startpos + needle.len() <= haystack.len()
                && haystack[startpos..startpos + needle.len() - 2] == needle[1..needle.len() - 1]
            {
                return true;
            }
            mask = clear_leftmost_set(mask);
        }
    }

    false
}
