use std::slice;

#[allow(dead_code)]
#[multiversion::multiversion]
#[clone(target = "[x86|x86_64]+avx2")]
#[clone(target = "wasm32+simd128")]
#[cfg_attr(
    all(target_arch = "aarch64", feature = "aarch64"),
    clone(target = "aarch64+neon")
)]
pub unsafe fn generic(left: *const u8, right: *const u8, n: usize) -> bool {
    slice::from_raw_parts(left, n) == slice::from_raw_parts(right, n)
}

#[allow(dead_code)]
#[multiversion::multiversion]
#[clone(target = "[x86|x86_64]+avx2")]
#[clone(target = "wasm32+simd128")]
#[cfg_attr(
    all(target_arch = "aarch64", feature = "aarch64"),
    clone(target = "aarch64+neon")
)]
pub unsafe fn specialized<const N: usize>(left: *const u8, right: *const u8) -> bool {
    slice::from_raw_parts(left, N) == slice::from_raw_parts(right, N)
}
