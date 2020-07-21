use memchr::memchr;

/// Single-byte searcher using `memchr` for faster matching.
pub struct MemchrSearcher(u8);

impl MemchrSearcher {
    /// Creates a new searcher for `needle`.
    pub fn new(needle: u8) -> Self {
        Self(needle)
    }

    /// Inlined version of `search_in` for hot call sites.
    #[inline]
    pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.is_empty() {
            return false;
        }

        memchr(self.0, haystack).is_some()
    }

    /// Performs a substring search for the `needle` within `haystack`.
    pub fn search_in(&self, haystack: &[u8]) -> bool {
        self.inlined_search_in(haystack)
    }
}

#[cfg(test)]
mod tests {
    use super::MemchrSearcher;

    fn search(haystack: &[u8], needle: &[u8]) -> bool {
        MemchrSearcher::new(needle[0]).search_in(haystack)
    }

    #[test]
    fn search_same() {
        assert!(search(b"f", b"f"));
    }

    #[test]
    fn search_different() {
        assert!(!search(b"foo", b"b"));
    }

    #[test]
    fn search_prefix() {
        assert!(search(b"foobar", b"f"));
    }

    #[test]
    fn search_suffix() {
        assert!(search(b"foobar", b"r"));
    }

    #[test]
    fn search_mutiple() {
        assert!(search(b"foobarfoo", b"o"));
    }

    #[test]
    fn search_middle() {
        assert!(search(b"foobarfoo", b"b"));
    }
}
