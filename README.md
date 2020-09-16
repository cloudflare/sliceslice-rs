# sliceslice

[![Actions](https://github.com/cloudflare/sliceslice-rs/workflows/Check/badge.svg)](https://github.com/cloudflare/sliceslice-rs/actions)
[![Crate](https://img.shields.io/crates/v/sliceslice)](https://crates.io/crates/sliceslice)
[![Docs](https://docs.rs/sliceslice/badge.svg)](https://docs.rs/sliceslice)
[![License](https://img.shields.io/crates/l/sliceslice)](LICENSE)

A fast implementation of single-pattern substring search using SIMD acceleration, based on the work [presented by Wojciech Muła](http://0x80.pl/articles/simd-strfind.html). For a fast multi-pattern substring search algorithm, see instead the [`aho-corasick` crate](https://github.com/BurntSushi/aho-corasick).

## Example

```rust
use sliceslice::x86::DynamicAvx2Searcher;

fn main() {
    let searcher = unsafe { DynamicAvx2Searcher::new(b"ipsum".to_owned().into()) };

    assert!(unsafe {
        searcher.search_in(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit")
    });

    assert!(!unsafe {
        searcher.search_in(b"foo bar baz qux quux quuz corge grault garply waldo fred")
    });
}
```

## Licensing

Licensed under the MIT license. See the [LICENSE](LICENSE) file for details.
