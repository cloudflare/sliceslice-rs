pub mod avx2;
mod bits;
mod memcmp;

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead, Read};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_strstr_anysize() {
        use crate::avx2::*;

        let mut f = File::open("./data/i386.txt").unwrap();
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer).unwrap();
        let content = String::from_utf8_lossy(&buffer);

        let file = File::open("./data/words").unwrap();
        let mut words: Vec<String> = io::BufReader::new(file)
            .lines()
            .filter_map(|line| line.ok())
            .filter(|word| word.len() > 1)
            .collect();
        // Sort all words by length then lexicographically
        words.sort_by(|a, b| {
            if a.len() != b.len() {
                a.len().partial_cmp(&b.len()).unwrap()
            } else {
                a.partial_cmp(b).unwrap()
            }
        });
        let words = words;

        for word in &words {
            let found = content.find(word).is_some();
            assert_eq!(
                unsafe { strstr_avx2_original(content.as_bytes(), word.as_bytes()) },
                found
            );
            assert_eq!(
                unsafe { strstr_avx2_rust_simple(content.as_bytes(), word.as_bytes()) },
                found
            );
            assert_eq!(
                unsafe { strstr_avx2_rust_simple_2(content.as_bytes(), word.as_bytes()) },
                found
            );
            assert_eq!(
                unsafe { strstr_avx2_rust_fast(content.as_bytes(), word.as_bytes()) },
                found
            );
            assert_eq!(
                strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes()),
                found
            );
            assert_eq!(
                unsafe { strstr_avx2_rust_aligned(content.as_bytes(), word.as_bytes()) },
                found
            );
        }

        for (i, word) in words.iter().enumerate() {
            for content in &words[(i + 1)..] {
                let found = content.find(word).is_some();
                assert_eq!(
                    unsafe { strstr_avx2_original(content.as_bytes(), word.as_bytes()) },
                    found
                );
                assert_eq!(
                    unsafe { strstr_avx2_rust_simple(content.as_bytes(), word.as_bytes()) },
                    found
                );
                assert_eq!(
                    unsafe { strstr_avx2_rust_simple_2(content.as_bytes(), word.as_bytes()) },
                    found
                );
                assert_eq!(
                    unsafe { strstr_avx2_rust_fast(content.as_bytes(), word.as_bytes()) },
                    found
                );
                assert_eq!(
                    strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes()),
                    found
                );
                assert_eq!(
                    unsafe { strstr_avx2_rust_aligned(content.as_bytes(), word.as_bytes()) },
                    found
                );
            }
        }
    }
}
