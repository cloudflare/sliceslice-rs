pub mod avx2;
mod bits;
mod memcmp;

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{self, BufRead, Read};
    use std::path::Path;

    // The output is wrapped in a Result to allow matching on errors
    // Returns an Iterator to the Reader of the lines of the file.
    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_strstr_anysize() {
        use crate::avx2::*;

        let mut f = File::open("./data/i386.txt").unwrap();
        let mut buffer = Vec::new();
        f.read_to_end(&mut buffer).unwrap();
        let content = String::from_utf8_lossy(&buffer);

        let lines = read_lines("./data/words").unwrap();
        for line in lines {
            if let Ok(word) = line {
                if word.len() > 1 {
                    let found = content.find(&word).is_some();
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
                        unsafe { strstr_avx2_rust_fast_2(content.as_bytes(), word.as_bytes()) },
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
}
