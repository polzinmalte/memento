//! Markdown export. Ported from listnr `Session/MeetingSession.swift`, same
//! line shape so transcripts from either tool read alike.
//!
//! Takes the timestamp already formatted rather than reaching for a clock or a
//! date library: it keeps this crate dependency-free, and it makes the output
//! deterministic under test, which a function whose whole job is formatting
//! ought to be.

use crate::transcript::{TranscriptLine, format_clock};

/// Render a finished transcript.
///
/// `recorded_label` is a human date for the header; `title` names the meeting.
pub fn render_markdown(title: &str, recorded_label: &str, lines: &[TranscriptLine]) -> String {
    let mut out = format!("# {title}\n\n_Recorded {recorded_label}_\n\n");
    if lines.is_empty() {
        out.push_str("_No speech was transcribed in this session._\n");
        return out;
    }
    for line in lines {
        out.push_str(&format!(
            "- **[{}] {}:** {}\n",
            format_clock(line.start_seconds),
            line.speaker,
            line.text
        ));
    }
    out
}

/// `memento_2026-08-06_14-30-00.md`. `stamp` is the caller's already-formatted
/// `yyyy-MM-dd_HH-mm-ss`; anything unsuitable for a filename is replaced, since
/// this string reaches the filesystem.
pub fn transcript_filename(stamp: &str) -> String {
    let safe: String = stamp
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    format!("memento_{safe}.md")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines() -> Vec<TranscriptLine> {
        vec![
            TranscriptLine::new("You", 1.0, "Guten Morgen."),
            TranscriptLine::new("Speaker 1", 63.4, "Die Tests laufen noch."),
        ]
    }

    #[test]
    fn renders_the_listnr_line_shape() {
        let md = render_markdown("memento meeting", "2026-08-06 14:30", &lines());
        assert!(md.starts_with("# memento meeting\n\n_Recorded 2026-08-06 14:30_\n\n"));
        assert!(md.contains("- **[00:01] You:** Guten Morgen.\n"));
        assert!(md.contains("- **[01:03] Speaker 1:** Die Tests laufen noch.\n"));
    }

    /// An empty transcript is a real outcome — a muted mic, a silent call — and
    /// a file that says nothing at all looks like the export broke.
    #[test]
    fn an_empty_transcript_says_so() {
        let md = render_markdown("memento meeting", "2026-08-06 14:30", &[]);
        assert!(md.contains("No speech was transcribed"));
    }

    #[test]
    fn filename_carries_the_stamp() {
        assert_eq!(
            transcript_filename("2026-08-06_14-30-00"),
            "memento_2026-08-06_14-30-00.md"
        );
    }

    /// The stamp reaches the filesystem, so separators must not survive it.
    #[test]
    fn filename_neutralises_path_characters() {
        let name = transcript_filename("../../etc/passwd");
        assert!(!name.contains('/'));
        assert!(!name.contains(".."), "{name}");
        assert_eq!(name, "memento_------etc-passwd.md");
    }

    #[test]
    fn german_text_survives_intact() {
        let md = render_markdown(
            "memento meeting",
            "heute",
            &[TranscriptLine::new(
                "You",
                0.0,
                "Können wir uns über die Größe abstimmen?",
            )],
        );
        assert!(md.contains("Können wir uns über die Größe abstimmen?"));
    }
}
