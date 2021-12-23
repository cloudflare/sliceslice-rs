#[allow(dead_code)]
#[inline]
#[multiversion::multiversion]
#[clone(target = "[x86|x86_64]+avx2")]
#[clone(target = "wasm32+simd128")]
pub fn clear_leftmost_set(value: u32) -> u32 {
    value & (value - 1)
}
