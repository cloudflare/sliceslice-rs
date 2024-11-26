#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sliceslice::x86::DynamicAvx2Searcher;

#[derive(Arbitrary, Debug)]
struct FuzzInput<'a> {
    needle: &'a [u8],
    haystack: &'a [u8],
    position: usize,
}

fuzz_target!(|input: FuzzInput<'_>| {
    let mut input = input;

    // This is a documented panic, so avoid it
    if !input.needle.is_empty() {
        input.position %= input.needle.len();
    }

    let result = unsafe {
        let searcher = DynamicAvx2Searcher::with_position(input.needle, input.position);
        searcher.search_in(input.haystack)
    };

    let expected = match input.needle.len() {
        0 => true,
        len => input.haystack.windows(len).any(|w| w == input.needle),
    };

    assert_eq!(result, expected);
});
