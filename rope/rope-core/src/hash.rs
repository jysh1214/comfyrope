//! SHA-256 identities. An image's id is the SHA-256 of its file; 12 hex digits are shown and accepted.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// How many hex digits of an id are shown, and used in file names. img2img.py's upload names use the same 12.
pub const SHORT: usize = 12;

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 15) as usize] as char);
    }
    s
}

pub fn sha256_bytes(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).with_context(|| format!("cannot read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Copies `src` to a new file `dst` and returns the SHA-256 and size of what was written. `src` is read once, so
/// the hash is the hash of the stored bytes even if `src` changes while this runs.
pub fn copy_hashed(src: &Path, dst: &Path) -> Result<(String, u64)> {
    let mut input = File::open(src).with_context(|| format!("cannot read {}", src.display()))?;
    let mut output = File::create_new(dst).with_context(|| format!("cannot create {}", dst.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut len = 0u64;
    loop {
        let n = input.read(&mut buf).with_context(|| format!("cannot read {}", src.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        output.write_all(&buf[..n]).with_context(|| format!("cannot write {}", dst.display()))?;
        len += n as u64;
    }
    output.sync_all()?;
    Ok((hex(&hasher.finalize()), len))
}

pub fn short(sha: &str) -> &str {
    &sha[..SHORT.min(sha.len())]
}

pub fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_abc_matches_the_standard_test_vector() {
        assert_eq!(sha256_bytes(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn copy_hashed_returns_the_hash_of_the_copy() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::write(&src, b"abc").unwrap();
        let (sha, len) = copy_hashed(&src, &dir.path().join("dst")).unwrap();
        assert_eq!((sha.as_str(), len), (sha256_bytes(b"abc").as_str(), 3));
        assert_eq!(sha256_file(&dir.path().join("dst")).unwrap(), sha);
    }
}
