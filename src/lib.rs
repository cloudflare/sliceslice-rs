//! A fast implementation of single-pattern substring search using SIMD
//! acceleration, based on the work [presented by Wojciech
//! Muła](http://0x80.pl/articles/simd-strfind.html). For a fast multi-pattern
//! substring search algorithm, see instead the [`aho-corasick`
//! crate](https://github.com/BurntSushi/aho-corasick).
//!
//! # Example
//!
//! ```
//! use sliceslice::x86::avx2::DynamicAvx2Searcher;
//!
//! let searcher = unsafe { DynamicAvx2Searcher::new(b"ipsum".to_owned().into()) };
//!
//! assert!(unsafe {
//!     searcher.search_in(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit")
//! });
//!
//! assert!(!unsafe {
//!     searcher.search_in(b"foo bar baz qux quux quuz corge grault garply waldo fred")
//! });

#![allow(deprecated)]
#![warn(missing_docs)]

/// Substring search implementations using the `memchr` function.
pub mod memchr;

/// Substring search implementations using x86 architecture features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

mod bits;
mod memcmp;
