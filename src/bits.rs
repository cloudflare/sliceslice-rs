#[allow(dead_code)]
#[multiversion::multiversion]
#[clone(target = "[x86|x86_64]+avx2")]
#[clone(target = "wasm32+simd128")]
#[clone(target = "aarch64+neon")]
pub fn clear_leftmost_set(value: u32) -> u32 {
    value & (value - 1)
}
