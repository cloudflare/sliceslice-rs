use crate::Needle;
use std::slice;

#[inline]
unsafe fn memcmp0(_: *const u8, _: *const u8) -> bool {
    true
}

#[inline]
unsafe fn memcmp1(left: *const u8, right: *const u8) -> bool {
    *left == *right
}

#[inline]
unsafe fn memcmp2(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u16>() == *right.cast::<u16>()
}

#[inline]
unsafe fn memcmp3(left: *const u8, right: *const u8) -> bool {
    memcmp2(left, right) && memcmp1(left.add(2), right.add(2))
}

#[inline]
unsafe fn memcmp4(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u32>() == *right.cast::<u32>()
}

#[inline]
unsafe fn memcmp5(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp1(left.add(4), right.add(4))
}

#[inline]
unsafe fn memcmp6(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp2(left.add(4), right.add(4))
}

#[inline]
unsafe fn memcmp7(left: *const u8, right: *const u8) -> bool {
    memcmp4(left, right) && memcmp3(left.add(4), right.add(4))
}

#[inline]
unsafe fn memcmp8(left: *const u8, right: *const u8) -> bool {
    *left.cast::<u64>() == *right.cast::<u64>()
}

#[inline]
unsafe fn memcmp9(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp1(left.add(8), right.add(8))
}

#[inline]
unsafe fn memcmp10(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp2(left.add(8), right.add(8))
}

#[inline]
unsafe fn memcmp11(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp3(left.add(8), right.add(8))
}

#[inline]
unsafe fn memcmp12(left: *const u8, right: *const u8) -> bool {
    memcmp8(left, right) && memcmp4(left.add(8), right.add(8))
}

#[inline]
pub unsafe fn memcmp<N: Needle + ?Sized>(left: *const u8, right: *const u8, n: usize) -> bool {
    if N::IS_FIXED {
        match n {
            0 => return memcmp0(left, right),
            1 => return memcmp1(left, right),
            2 => return memcmp2(left, right),
            3 => return memcmp3(left, right),
            4 => return memcmp4(left, right),
            5 => return memcmp5(left, right),
            6 => return memcmp6(left, right),
            7 => return memcmp7(left, right),
            8 => return memcmp8(left, right),
            9 => return memcmp9(left, right),
            10 => return memcmp10(left, right),
            11 => return memcmp11(left, right),
            12 => return memcmp12(left, right),
            _ => {}
        }
    }

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

    seq!(N in 0..=12 {
        paste! {
            #[test]
            fn [<memcmp N>]() {
                memcmp(super::[<memcmp N>], N);
            }
        }
    });
}
