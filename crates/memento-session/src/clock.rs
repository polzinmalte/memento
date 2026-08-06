//! The two date strings a session needs, in local time.
//!
//! Lives here rather than in `memento-core` so that crate stays dependency-free
//! and its formatting stays deterministic under test.

use chrono::Local;

/// `(human label for the header, filesystem stamp for the filename)`.
pub fn now_labels() -> (String, String) {
    let now = Local::now();
    (
        now.format("%Y-%m-%d %H:%M").to_string(),
        now.format("%Y-%m-%d_%H-%M-%S").to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stamp_is_safe_for_a_filename() {
        let (_, stamp) = now_labels();
        // Colons are legal on POSIX and forbidden on Windows, which is where
        // this actually runs; the format avoids them entirely.
        assert!(!stamp.contains(':'), "{stamp}");
        assert!(!stamp.contains('/'), "{stamp}");
        assert_eq!(stamp.len(), "2026-08-06_14-30-00".len(), "{stamp}");
    }

    #[test]
    fn the_label_reads_as_a_date_and_time() {
        let (label, _) = now_labels();
        assert!(label.contains('-') && label.contains(':'), "{label}");
    }

    #[test]
    fn the_stamp_survives_the_filename_sanitiser_unchanged() {
        let (_, stamp) = now_labels();
        assert_eq!(
            memento_core::transcript_filename(&stamp),
            format!("memento_{stamp}.md"),
            "the stamp must already be filename-safe, not rely on being cleaned"
        );
    }
}
