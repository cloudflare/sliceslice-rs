#![allow(clippy::missing_safety_doc)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn avx2_strstr_v2(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    match wrapper::avx2_strstr_v2(
        haystack.as_ptr().cast(),
        haystack.len() as _,
        needle.as_ptr().cast(),
        needle.len() as _,
    ) as _
    {
        usize::MAX => None,
        i => Some(i),
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod wrapper {
    #![allow(non_camel_case_types)]
    #![allow(unused)]

    include!(concat!(env!("OUT_DIR"), "/sse4_strstr.rs"));
}
