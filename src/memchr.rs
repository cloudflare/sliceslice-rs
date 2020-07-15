use memchr::memchr;

pub struct MemchrSearcher(u8);

impl MemchrSearcher {
    pub fn new(needle: u8) -> Self {
        Self(needle)
    }

    #[inline(always)]
    pub fn inlined_search_in(&self, haystack: &[u8]) -> bool {
        if haystack.is_empty() {
            return false;
        }

        memchr(self.0, haystack).is_some()
    }

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
