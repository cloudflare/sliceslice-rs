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
    let left = std::mem::transmute::<*const u8, *const u16>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u16>(right.as_ptr());
    *left == *right
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn memcmp3(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u32>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u32>(right.as_ptr());
    (*left & 0x00ffffff) == (*right & 0x00ffffff)
}

#[inline(always)]
pub(crate) unsafe fn memcmp4(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u32>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u32>(right.as_ptr());
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp5(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u64>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u64>(right.as_ptr());
    ((*left ^ *right) & 0x000000ffffffffff_u64) == 0
}

#[inline(always)]
pub(crate) unsafe fn memcmp6(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u64>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u64>(right.as_ptr());
    ((*left ^ *right) & 0x0000ffffffffffff_u64) == 0
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn memcmp7(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u64>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u64>(right.as_ptr());
    ((*left ^ *right) & 0x00ffffffffffffff_u64) == 0
}

#[inline(always)]
pub(crate) unsafe fn memcmp8(left: &[u8], right: &[u8]) -> bool {
    let left = std::mem::transmute::<*const u8, *const u64>(left.as_ptr());
    let right = std::mem::transmute::<*const u8, *const u64>(right.as_ptr());
    *left == *right
}

#[inline(always)]
pub(crate) unsafe fn memcmp9(left: &[u8], right: &[u8]) -> bool {
    let left_ptr = left.as_ptr();
    let right_ptr = right.as_ptr();
    let left = std::mem::transmute::<*const u8, *const u64>(left_ptr);
    let right = std::mem::transmute::<*const u8, *const u64>(right_ptr);
    *left == *right && *left_ptr.add(8) == *right_ptr.add(8)
}

#[inline(always)]
pub(crate) unsafe fn memcmp10(left: &[u8], right: &[u8]) -> bool {
    let left_ptr = left.as_ptr();
    let right_ptr = right.as_ptr();
    let left = std::mem::transmute::<*const u8, *const u64>(left_ptr);
    let right = std::mem::transmute::<*const u8, *const u64>(right_ptr);
    let left2 = std::mem::transmute::<*const u8, *const u16>(left_ptr.add(8));
    let right2 = std::mem::transmute::<*const u8, *const u16>(right_ptr.add(8));
    *left == *right && *left2 == *right2
}

#[inline(always)]
pub(crate) unsafe fn memcmp11(left: &[u8], right: &[u8]) -> bool {
    let left_ptr = left.as_ptr();
    let right_ptr = right.as_ptr();
    let left = std::mem::transmute::<*const u8, *const u64>(left_ptr);
    let right = std::mem::transmute::<*const u8, *const u64>(right_ptr);
    let left2 = std::mem::transmute::<*const u8, *const u32>(left_ptr.add(8));
    let right2 = std::mem::transmute::<*const u8, *const u32>(right_ptr.add(8));
    *left == *right && (*left2 & 0x00ffffff) == (*right2 & 0x00ffffff)
}

#[inline(always)]
pub(crate) unsafe fn memcmp12(left: &[u8], right: &[u8]) -> bool {
    let left_ptr = left.as_ptr();
    let right_ptr = right.as_ptr();
    let left = std::mem::transmute::<*const u8, *const u64>(left_ptr);
    let right = std::mem::transmute::<*const u8, *const u64>(right_ptr);
    let left2 = std::mem::transmute::<*const u8, *const u32>(left_ptr.add(8));
    let right2 = std::mem::transmute::<*const u8, *const u32>(right_ptr.add(8));
    *left == *right && *left2 == *right2
}

#[inline(always)]
pub(crate) unsafe fn memcmp(left: &[u8], right: &[u8]) -> bool {
    left == right
}
