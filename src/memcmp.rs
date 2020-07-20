#![allow(dead_code)]

#[inline]
pub unsafe fn memcmp0(_: &[u8], _: &[u8]) -> bool {
    true
}

#[inline]
pub unsafe fn memcmp1(left: &[u8], right: &[u8]) -> bool {
    *left == *right
}

#[inline]
pub unsafe fn memcmp2(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u16>();
    let right = right.as_ptr().cast::<u16>();
    *left == *right
}

#[inline]
pub unsafe fn memcmp3(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u32>();
    let right = right.as_ptr().cast::<u32>();
    (*left & 0x00ffffff) == (*right & 0x00ffffff)
}

#[inline]
pub unsafe fn memcmp4(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u32>();
    let right = right.as_ptr().cast::<u32>();
    *left == *right
}

#[inline]
pub unsafe fn memcmp5(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 40
}

#[inline]
pub unsafe fn memcmp6(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 48
}

#[allow(dead_code)]
#[inline]
pub unsafe fn memcmp7(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 56
}

#[inline]
pub unsafe fn memcmp8(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    *left == *right
}

#[inline]
pub unsafe fn memcmp9(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    *left_first == *right_first && *left.as_ptr().add(8) == *right.as_ptr().add(8)
}

#[inline]
pub unsafe fn memcmp10(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u16>();
    let right_second = right.as_ptr().add(8).cast::<u16>();
    *left_first == *right_first && *left_second == *right_second
}

#[inline]
pub unsafe fn memcmp11(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u32>();
    let right_second = right.as_ptr().add(8).cast::<u32>();
    *left_first == *right_first && (*left_second & 0x00ffffff) == (*right_second & 0x00ffffff)
}

#[inline]
pub unsafe fn memcmp12(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u32>();
    let right_second = right.as_ptr().add(8).cast::<u32>();
    *left_first == *right_first && *left_second == *right_second
}

#[inline]
pub unsafe fn memcmp(left: &[u8], right: &[u8], n: usize) -> bool {
    match n {
        0 => memcmp0(left, right),
        1 => memcmp1(left, right),
        2 => memcmp2(left, right),
        3 => memcmp3(left, right),
        4 => memcmp4(left, right),
        5 => memcmp5(left, right),
        6 => memcmp6(left, right),
        7 => memcmp7(left, right),
        8 => memcmp8(left, right),
        9 => memcmp9(left, right),
        10 => memcmp10(left, right),
        11 => memcmp11(left, right),
        12 => memcmp12(left, right),
        _ => left == right,
    }
}
