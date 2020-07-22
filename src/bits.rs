#[inline]
pub fn clear_leftmost_set(value: u32) -> u32 {
    value & (value - 1)
}
