use core::slice;

#[inline]
pub(crate) unsafe fn memcmp1(left: *const u8, right: *const u8) -> bool {
    *left == *right
}

#[inline]
pub(crate) unsafe fn memcmp2(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u16>() == *right.cast::<u16>()
}

#[inline]
pub(crate) unsafe fn memcmp3(left: *const u8, right: *const u8) -> bool {
    memcmp2(left, right) && memcmp1(left.add(2), right.add(2))
}

#[inline]
pub(crate) unsafe fn memcmp4(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u32>() == *right.cast::<u32>()
}

#[inline]
pub(crate) unsafe fn memcmp5(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp1(left.add(4), right.add(4))
}

#[inline]
pub(crate) unsafe fn memcmp6(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp2(left.add(4), right.add(4))
}

#[inline]
pub(crate) unsafe fn memcmp7(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp3(left.add(4), right.add(4))
}

#[inline]
pub(crate) unsafe fn memcmp8(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u64>() == *right.cast::<u64>()
}

#[inline]
pub(crate) unsafe fn memcmp9(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp1(left.add(8), right.add(8))
}

#[inline]
pub(crate) unsafe fn memcmp10(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp2(left.add(8), right.add(8))
}

#[inline]
pub(crate) unsafe fn memcmp11(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp3(left.add(8), right.add(8))
}

#[inline]
pub(crate) unsafe fn memcmp12(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp4(left.add(8), right.add(8))
}

#[inline]
pub(crate) unsafe fn memcmp(left: *const u8, right: *const u8, n: usize) -> bool {
    slice::from_raw_parts(left, n) == slice::from_raw_parts(right, n)
}

#[cfg(test)]
mod tests {
    use paste::paste;
    use seq_macro::seq;

    fn memcmp(f: unsafe fn(*const u8, *const u8) -> bool, n: usize) {
        let left = vec![b'0'; n];
        unsafe { assert!(f(left.as_ptr(), left.as_ptr())) };

        for i in 0..n {
            let mut right = left.clone();
            right[i] = b'1';
            unsafe { assert!(!f(left.as_ptr(), right.as_ptr())) };
        }
    }

    seq!(N in 1..=12 {
        paste! {
            #[test]
            fn [<memcmp N>]() {
                memcmp(super::[<memcmp N>], N);
            }
        }
    });
}
