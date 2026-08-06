//! How loud speech has to be before memento treats it as speech.
//! Ported from listnr `Sources/ListnrCore/Session/MicSensitivity.swift`.
//!
//! The detection thresholds are absolute RMS values, and microphone levels vary
//! by more than an order of magnitude across hardware and gain settings. A
//! built-in laptop microphone at default input volume can peak around 0.017 —
//! below the 0.018 floor that decides whether a segment is worth transcribing at
//! all. The result is a transcript with minutes missing from the middle and no
//! obvious cause, because the audio never reached Whisper.
//!
//! Scaling every threshold by one factor keeps their relationships intact — the
//! system lane stays stricter than the microphone lane, and the segment floor
//! stays above the speech threshold — while moving the whole set to where a
//! given microphone actually sits.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MicSensitivity {
    /// Quiet microphone, soft speaker, or a mic placed far away.
    High,
    #[default]
    Normal,
    /// Hot microphone or a noisy room, where `Normal` picks up keyboard and fans.
    Low,
}

impl MicSensitivity {
    pub const ALL: [MicSensitivity; 3] = [Self::High, Self::Normal, Self::Low];

    /// Multiplier applied to every threshold. Lower means more is heard.
    pub fn multiplier(self) -> f32 {
        match self {
            Self::High => 0.5,
            Self::Normal => 1.0,
            Self::Low => 1.6,
        }
    }

    pub fn summary(self) -> &'static str {
        match self {
            Self::High => "high (quiet mic, hears more)",
            Self::Normal => "normal",
            Self::Low => "low (hot mic or noisy room, hears less)",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "high" => Some(Self::High),
            "normal" => Some(Self::Normal),
            "low" => Some(Self::Low),
            _ => None,
        }
    }
}

/// The speech-detection thresholds for one lane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneThresholds {
    /// Frame RMS above which the segmenter considers a frame voiced.
    pub speech_threshold: f32,
    /// Whole-segment RMS below which a finished utterance is discarded.
    pub min_segment_rms: f32,
}

/// Base thresholds at `Normal`, and how they scale.
///
/// The system lane is stricter than the microphone lane on purpose: silence
/// there was making Whisper hallucinate "Thank you".
pub mod lane_tuning {
    use super::{LaneThresholds, MicSensitivity};

    pub const YOU_BASE: LaneThresholds = LaneThresholds {
        speech_threshold: 0.014,
        min_segment_rms: 0.018,
    };
    pub const OTHERS_BASE: LaneThresholds = LaneThresholds {
        speech_threshold: 0.022,
        min_segment_rms: 0.025,
    };

    pub fn you(sensitivity: MicSensitivity) -> LaneThresholds {
        scale(YOU_BASE, sensitivity)
    }

    pub fn others(sensitivity: MicSensitivity) -> LaneThresholds {
        scale(OTHERS_BASE, sensitivity)
    }

    fn scale(base: LaneThresholds, sensitivity: MicSensitivity) -> LaneThresholds {
        LaneThresholds {
            speech_threshold: base.speech_threshold * sensitivity.multiplier(),
            min_segment_rms: base.min_segment_rms * sensitivity.multiplier(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::lane_tuning::{self, OTHERS_BASE, YOU_BASE};
    use super::*;

    type LaneFn = fn(MicSensitivity) -> LaneThresholds;
    const LANES: [LaneFn; 2] = [lane_tuning::you, lane_tuning::others];

    #[test]
    fn parses_every_level() {
        assert_eq!(MicSensitivity::parse("high"), Some(MicSensitivity::High));
        assert_eq!(
            MicSensitivity::parse("normal"),
            Some(MicSensitivity::Normal)
        );
        assert_eq!(MicSensitivity::parse("low"), Some(MicSensitivity::Low));
    }

    #[test]
    fn parse_is_case_and_whitespace_insensitive() {
        assert_eq!(MicSensitivity::parse("  HIGH "), Some(MicSensitivity::High));
        assert_eq!(
            MicSensitivity::parse("Normal"),
            Some(MicSensitivity::Normal)
        );
    }

    #[test]
    fn unknown_level_is_rejected() {
        assert_eq!(MicSensitivity::parse("loud"), None);
        assert_eq!(MicSensitivity::parse(""), None);
    }

    #[test]
    fn normal_is_unity() {
        // Existing behaviour must not move for anyone who does not opt in.
        assert_eq!(MicSensitivity::Normal.multiplier(), 1.0);
        assert_eq!(lane_tuning::you(MicSensitivity::Normal), YOU_BASE);
        assert_eq!(lane_tuning::others(MicSensitivity::Normal), OTHERS_BASE);
    }

    #[test]
    fn higher_sensitivity_hears_more() {
        assert!(MicSensitivity::High.multiplier() < MicSensitivity::Normal.multiplier());
        assert!(MicSensitivity::Low.multiplier() > MicSensitivity::Normal.multiplier());
    }

    #[test]
    fn thresholds_are_ordered_across_levels() {
        for lane in LANES {
            assert!(
                lane(MicSensitivity::High).speech_threshold
                    < lane(MicSensitivity::Normal).speech_threshold
            );
            assert!(
                lane(MicSensitivity::Normal).speech_threshold
                    < lane(MicSensitivity::Low).speech_threshold
            );
            assert!(
                lane(MicSensitivity::High).min_segment_rms
                    < lane(MicSensitivity::Normal).min_segment_rms
            );
            assert!(
                lane(MicSensitivity::Normal).min_segment_rms
                    < lane(MicSensitivity::Low).min_segment_rms
            );
        }
    }

    /// Scaling must not invert the relationships the thresholds encode: the
    /// segment floor sits above the per-frame speech threshold, so a lane cannot
    /// start an utterance it would then always discard.
    #[test]
    fn segment_floor_stays_above_the_speech_threshold_at_every_level() {
        for level in MicSensitivity::ALL {
            for lane in LANES {
                assert!(
                    lane(level).min_segment_rms > lane(level).speech_threshold,
                    "{level:?}"
                );
            }
        }
    }

    /// The system lane is deliberately stricter than the microphone lane,
    /// because silence there made Whisper hallucinate. Scaling must preserve it.
    #[test]
    fn system_lane_stays_stricter_than_the_mic_lane() {
        for level in MicSensitivity::ALL {
            assert!(
                lane_tuning::others(level).speech_threshold
                    > lane_tuning::you(level).speech_threshold,
                "{level:?}"
            );
            assert!(
                lane_tuning::others(level).min_segment_rms
                    > lane_tuning::you(level).min_segment_rms,
                "{level:?}"
            );
        }
    }

    /// The case that prompted all of this: a microphone peaking at 0.017.
    #[test]
    fn high_sensitivity_admits_a_quiet_mic_that_normal_rejects() {
        let observed_peak: f32 = 0.017;
        assert!(
            observed_peak < lane_tuning::you(MicSensitivity::Normal).min_segment_rms,
            "the reported peak should be below the normal floor"
        );
        assert!(
            observed_peak > lane_tuning::you(MicSensitivity::High).min_segment_rms,
            "high sensitivity must actually admit it"
        );
    }

    #[test]
    fn every_level_has_a_human_summary() {
        for level in MicSensitivity::ALL {
            assert!(!level.summary().is_empty());
        }
    }
}
