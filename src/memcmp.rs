#![allow(dead_code)]

use std::slice;

#[inline]
pub unsafe fn memcmp0(_: *const u8, _: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 0);
    true
}

#[inline]
pub unsafe fn memcmp1(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 1);
    *left == *right
}

#[inline]
pub unsafe fn memcmp2(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 2);
    let left = left.cast::<u16>();
    let right = right.cast::<u16>();
    left.read_unaligned() == right.read_unaligned()
}

#[inline]
pub unsafe fn memcmp3(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 3);
    memcmp2(left, right, 2) && memcmp1(left.add(2), right.add(2), 1)
}

#[inline]
pub unsafe fn memcmp4(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 4);
    let left = left.cast::<u32>();
    let right = right.cast::<u32>();
    left.read_unaligned() == right.read_unaligned()
}

#[inline]
pub unsafe fn memcmp5(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 5);
    memcmp4(left, right, 4) && memcmp1(left.add(4), right.add(4), 1)
}

#[inline]
pub unsafe fn memcmp6(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 6);
    memcmp4(left, right, 4) && memcmp2(left.add(4), right.add(4), 2)
}

#[inline]
pub unsafe fn memcmp7(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 7);
    memcmp4(left, right, 4) && memcmp3(left.add(4), right.add(4), 3)
}

#[inline]
pub unsafe fn memcmp8(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 8);
    let left = left.cast::<u64>();
    let right = right.cast::<u64>();
    left.read_unaligned() == right.read_unaligned()
}

#[inline]
pub unsafe fn memcmp9(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 9);
    memcmp8(left, right, 8) && memcmp1(left.add(8), right.add(8), 1)
}

#[inline]
pub unsafe fn memcmp10(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 10);
    memcmp8(left, right, 8) && memcmp2(left.add(8), right.add(8), 2)
}

#[inline]
pub unsafe fn memcmp11(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 11);
    memcmp8(left, right, 8) && memcmp3(left.add(8), right.add(8), 3)
}

#[inline]
pub unsafe fn memcmp12(left: *const u8, right: *const u8, n: usize) -> bool {
    debug_assert_eq!(n, 12);
    memcmp8(left, right, 8) && memcmp4(left.add(8), right.add(8), 4)
}

#[inline]
pub unsafe fn memcmp(left: *const u8, right: *const u8, n: usize) -> bool {
    slice::from_raw_parts(left, n) == slice::from_raw_parts(right, n)
}

#[cfg(test)]
mod tests {
    fn memcmp(f: unsafe fn(*const u8, *const u8, usize) -> bool, n: usize) {
        let left = vec![b'0'; n];
        unsafe { assert!(f(left.as_ptr(), left.as_ptr(), n)) };
        unsafe { assert!(super::memcmp(left.as_ptr(), left.as_ptr(), n)) };

        for i in 0..n {
            let mut right = left.clone();
            right[i] = b'1';
            unsafe { assert!(!f(left.as_ptr(), right.as_ptr(), n)) };
            unsafe { assert!(!super::memcmp(left.as_ptr(), right.as_ptr(), n)) };
        }
    }

    #[test]
    fn memcmp0() {
        memcmp(super::memcmp0, 0);
    }

    #[test]
    fn memcmp1() {
        memcmp(super::memcmp1, 1);
    }

    #[test]
    fn memcmp2() {
        memcmp(super::memcmp2, 2);
    }

    #[test]
    fn memcmp3() {
        memcmp(super::memcmp3, 3);
    }

    #[test]
    fn memcmp4() {
        memcmp(super::memcmp4, 4);
    }

    #[test]
    fn memcmp5() {
        memcmp(super::memcmp5, 5);
    }

    #[test]
    fn memcmp6() {
        memcmp(super::memcmp6, 6);
    }

    #[test]
    fn memcmp7() {
        memcmp(super::memcmp7, 7);
    }

    #[test]
    fn memcmp8() {
        memcmp(super::memcmp8, 8);
    }

    #[test]
    fn memcmp9() {
        memcmp(super::memcmp9, 9);
    }

    #[test]
    fn memcmp10() {
        memcmp(super::memcmp10, 10);
    }

    #[test]
    fn memcmp11() {
        memcmp(super::memcmp11, 11);
    }

    #[test]
    fn memcmp12() {
        memcmp(super::memcmp12, 12);
    }
}
