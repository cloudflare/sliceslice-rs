#![allow(dead_code)]

#[inline(always)]
pub(crate) unsafe fn memcmp0(_: &[u8], _: &[u8]) -> bool {
    true
}

#[inline(always)]
pub(crate) unsafe fn memcmp1(left: &[u8], right: &[u8]) -> bool {
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp2(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u16>();
    let right = right.as_ptr().cast::<u16>();
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp3(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u32>();
    let right = right.as_ptr().cast::<u32>();
    (*left & 0x00ffffff) == (*right & 0x00ffffff)
}

#[inline(always)]
pub(crate) unsafe fn memcmp4(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u32>();
    let right = right.as_ptr().cast::<u32>();
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp5(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 40
}

#[inline(always)]
pub(crate) unsafe fn memcmp6(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 48
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn memcmp7(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    (*left ^ *right).trailing_zeros() >= 56
}

#[inline(always)]
pub(crate) unsafe fn memcmp8(left: &[u8], right: &[u8]) -> bool {
    let left = left.as_ptr().cast::<u64>();
    let right = right.as_ptr().cast::<u64>();
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp9(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    *left_first == *right_first && *left.as_ptr().add(8) == *right.as_ptr().add(8)
}

#[inline(always)]
pub(crate) unsafe fn memcmp10(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u16>();
    let right_second = right.as_ptr().add(8).cast::<u16>();
    *left_first == *right_first && *left_second == *right_second
}

#[inline(always)]
pub(crate) unsafe fn memcmp11(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u32>();
    let right_second = right.as_ptr().add(8).cast::<u32>();
    *left_first == *right_first && (*left_second & 0x00ffffff) == (*right_second & 0x00ffffff)
}

#[inline(always)]
pub(crate) unsafe fn memcmp12(left: &[u8], right: &[u8]) -> bool {
    let left_first = left.as_ptr().cast::<u64>();
    let right_first = right.as_ptr().cast::<u64>();
    let left_second = left.as_ptr().add(8).cast::<u32>();
    let right_second = right.as_ptr().add(8).cast::<u32>();
    *left_first == *right_first && *left_second == *right_second
}

#[inline(always)]
pub(crate) unsafe fn memcmp(left: &[u8], right: &[u8]) -> bool {
    left == right
}
