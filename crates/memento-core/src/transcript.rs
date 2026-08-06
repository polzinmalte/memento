//! Transcript lines, the two-lane merge, and the shared-clock offset.
//! Ported from listnr's `LaneTranscriber.swift` (`TranscriptMerger`) and the
//! `laneOffsets` half of `DualCaptureSession.swift`, which is pure arithmetic
//! and so belongs here rather than in the platform crate.

#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptLine {
    pub speaker: String,
    pub start_seconds: f64,
    pub text: String,
}

impl TranscriptLine {
    pub fn new(speaker: impl Into<String>, start_seconds: f64, text: impl Into<String>) -> Self {
        Self {
            speaker: speaker.into(),
            start_seconds,
            text: text.into(),
        }
    }
}

/// `mm:ss` for transcript display.
pub fn format_clock(seconds: f64) -> String {
    let total = seconds.floor().max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

/// Sort a set of lanes onto one chronological transcript.
pub fn merge(lanes: Vec<Vec<TranscriptLine>>) -> Vec<TranscriptLine> {
    let mut all: Vec<TranscriptLine> = lanes.into_iter().flatten().collect();
    all.sort_by(|a, b| a.start_seconds.total_cmp(&b.start_seconds));
    all
}

/// Shift a lane's timestamps onto the shared session clock.
pub fn shift(lines: Vec<TranscriptLine>, offset: f64) -> Vec<TranscriptLine> {
    if offset == 0.0 {
        return lines;
    }
    lines
        .into_iter()
        .map(|l| TranscriptLine {
            start_seconds: (l.start_seconds + offset).max(0.0),
            ..l
        })
        .collect()
}

/// Clock time of each lane's first sample, when known. The lanes start
/// milliseconds apart, so their sample-count timestamps are not directly
/// comparable; these anchor both onto one session timeline.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaneAnchors {
    pub mic_first_sample_time: Option<f64>,
    pub system_first_sample_time: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneOffsets {
    pub mic: f64,
    pub system: f64,
    /// True when the anchors were discarded as implausible; the caller should
    /// tell the user the lanes were merged without alignment.
    pub skew_rejected: bool,
}

/// A real start gap is tens to hundreds of milliseconds. Anything larger means
/// the two clocks are not comparable on this system, and shifting by a bogus
/// amount is worse than not shifting at all.
const MAX_PLAUSIBLE_SKEW_SECONDS: f64 = 5.0;

impl LaneAnchors {
    /// Seconds to add to each lane's timestamps to place them on a shared clock
    /// whose zero is whichever lane started first. `(0, 0)` when the measurement
    /// is missing or implausible.
    pub fn offsets(self) -> LaneOffsets {
        let (Some(m), Some(s)) = (self.mic_first_sample_time, self.system_first_sample_time) else {
            return LaneOffsets {
                mic: 0.0,
                system: 0.0,
                skew_rejected: false,
            };
        };
        if (m - s).abs() > MAX_PLAUSIBLE_SKEW_SECONDS {
            return LaneOffsets {
                mic: 0.0,
                system: 0.0,
                skew_rejected: true,
            };
        }
        let origin = m.min(s);
        LaneOffsets {
            mic: m - origin,
            system: s - origin,
            skew_rejected: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(speaker: &str, t: f64) -> TranscriptLine {
        TranscriptLine::new(speaker, t, "x")
    }

    fn anchors(mic: Option<f64>, system: Option<f64>) -> LaneAnchors {
        LaneAnchors {
            mic_first_sample_time: mic,
            system_first_sample_time: system,
        }
    }

    #[test]
    fn shift_by_zero_is_identity() {
        let lines = vec![line("You", 1.0), line("You", 2.0)];
        let shifted = shift(lines, 0.0);
        assert_eq!(
            shifted.iter().map(|l| l.start_seconds).collect::<Vec<_>>(),
            [1.0, 2.0]
        );
    }

    #[test]
    fn shift_moves_timestamps() {
        let shifted = shift(vec![line("Others", 1.5)], 0.25);
        assert!((shifted[0].start_seconds - 1.75).abs() < 1e-9);
    }

    #[test]
    fn shift_clamps_at_zero() {
        let shifted = shift(vec![line("Others", 0.1)], -0.5);
        assert_eq!(shifted[0].start_seconds, 0.0);
    }

    #[test]
    fn merge_interleaves_lanes_by_time() {
        let you = vec![line("You", 0.5), line("You", 4.0)];
        let others = vec![line("Speaker 1", 2.0), line("Speaker 2", 3.0)];
        let merged = merge(vec![you, others]);
        assert_eq!(
            merged
                .iter()
                .map(|l| l.speaker.as_str())
                .collect::<Vec<_>>(),
            ["You", "Speaker 1", "Speaker 2", "You"]
        );
    }

    #[test]
    fn missing_anchors_mean_no_shift() {
        let o = anchors(None, Some(100.0)).offsets();
        assert_eq!(o.mic, 0.0);
        assert_eq!(o.system, 0.0);
    }

    #[test]
    fn implausible_skew_means_no_shift() {
        // Clocks that disagree by more than 5 s are not comparable.
        let o = anchors(Some(100.0), Some(107.0)).offsets();
        assert_eq!(o.mic, 0.0);
        assert_eq!(o.system, 0.0);
        assert!(o.skew_rejected);
    }

    #[test]
    fn earlier_lane_becomes_origin() {
        // Mic started first: mic lines stay, system lines move forward by the gap.
        let o = anchors(Some(100.0), Some(100.2)).offsets();
        assert_eq!(o.mic, 0.0);
        assert!((o.system - 0.2).abs() < 1e-9);

        let r = anchors(Some(100.25), Some(100.0)).offsets();
        assert!((r.mic - 0.25).abs() < 1e-9);
        assert_eq!(r.system, 0.0);
    }

    #[test]
    fn clock_formats_as_mm_ss() {
        assert_eq!(format_clock(0.0), "00:00");
        assert_eq!(format_clock(9.9), "00:09");
        assert_eq!(format_clock(61.0), "01:01");
        assert_eq!(format_clock(-5.0), "00:00");
        assert_eq!(format_clock(3600.0), "60:00");
    }
}
