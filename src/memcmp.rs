use std::slice;

#[inline]
pub unsafe fn generic(left: *const u8, right: *const u8, n: usize) -> bool {
    slice::from_raw_parts(left, n) == slice::from_raw_parts(right, n)
}

#[inline]
pub unsafe fn specialized<const N: usize>(left: *const u8, right: *const u8) -> bool {
    slice::from_raw_parts(left, N) == slice::from_raw_parts(right, N)
}
