//! Per-script decoding and acceptance thresholds. Ported from listnr
//! `Sources/ListnrCore/Transcription/DecodeTuning.swift`.
//!
//! The invariant worth stating, because it is easy to break: the decoder's
//! thresholds must be **at least as permissive** as the acceptance gate. If the
//! decoder discards a candidate the gate would have accepted, relaxing the gate
//! cannot recover it — the text is already gone. The tests below enforce it.

use crate::language::ScriptClass;

/// The decoder-side fallback triggers for one script.
///
/// These decide when the decoder re-runs a window at a higher temperature
/// instead of keeping what it has. listnr passes them to WhisperKit; memento
/// passes them to whisper.cpp, which is why they live here as plain numbers
/// rather than in either backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeThresholds {
    pub compression_ratio_threshold: f32,
    pub log_prob_threshold: f32,
    pub first_token_log_prob_threshold: f32,
    pub no_speech_threshold: f32,
    /// How many times to re-decode at a raised temperature before giving up.
    ///
    /// This is the *only* mechanism that turns a hallucinated window into a real
    /// one, so scripts that trip the checks more often need more of them. It is
    /// also not free: each fallback is another full decode, which matters more
    /// on memento's CPU-only backend than it did on the Neural Engine. Raising
    /// the thresholds and the retry count together is deliberate — the relaxed
    /// thresholds mean fewer windows fail at all, which buys back the time the
    /// extra retries cost.
    pub temperature_fallback_count: usize,
    pub temperature_increment_on_fallback: f32,
}

/// Thresholds for accepting one decoded segment.
///
/// The no-speech and log-probability tests are combined with AND, matching the
/// reference Whisper implementation: either signal alone fires often enough on
/// quiet or accented speech that OR would cut real content. Degenerate
/// repetition is an independent signal, so the compression ratio stands on its
/// own — and it stays bounded on every script, because the relaxed non-Latin
/// gates would otherwise trade missing text for gibberish.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentConfidence {
    pub max_no_speech_prob: f32,
    pub min_avg_log_prob: f32,
    pub max_compression_ratio: f32,
}

impl Default for SegmentConfidence {
    /// The shipped Latin defaults.
    fn default() -> Self {
        Self {
            max_no_speech_prob: 0.7,
            min_avg_log_prob: -0.9,
            max_compression_ratio: 2.6,
        }
    }
}

impl SegmentConfidence {
    pub fn accepts(&self, no_speech_prob: f32, avg_log_prob: f32, compression_ratio: f32) -> bool {
        if compression_ratio > self.max_compression_ratio {
            return false;
        }
        !(no_speech_prob > self.max_no_speech_prob && avg_log_prob < self.min_avg_log_prob)
    }
}

/// Decoder-side thresholds, matching [`confidence`]: the compression trigger
/// follows the output text, everything else the input.
pub fn decoding(script: ScriptClass, translating: bool) -> DecodeThresholds {
    let mut thresholds = base_decoding(script);
    if translating {
        thresholds.compression_ratio_threshold =
            base_decoding(ScriptClass::Latin).compression_ratio_threshold;
    }
    thresholds
}

/// Latin numbers are listnr's shipped ones, unchanged on purpose: English works
/// today, and the port must not move it.
fn base_decoding(script: ScriptClass) -> DecodeThresholds {
    match script {
        ScriptClass::Latin => DecodeThresholds {
            compression_ratio_threshold: 2.2,
            log_prob_threshold: -0.8,
            first_token_log_prob_threshold: -1.2,
            no_speech_threshold: 0.45,
            temperature_fallback_count: 1,
            temperature_increment_on_fallback: 0.2,
        },
        ScriptClass::Cjk => DecodeThresholds {
            compression_ratio_threshold: 3.4,
            log_prob_threshold: -1.2,
            first_token_log_prob_threshold: -1.8,
            no_speech_threshold: 0.6,
            temperature_fallback_count: 3,
            temperature_increment_on_fallback: 0.2,
        },
        ScriptClass::Indic | ScriptClass::Unknown => DecodeThresholds {
            compression_ratio_threshold: 4.0,
            log_prob_threshold: -1.5,
            first_token_log_prob_threshold: -2.0,
            no_speech_threshold: 0.6,
            temperature_fallback_count: 3,
            temperature_increment_on_fallback: 0.2,
        },
    }
}

/// The acceptance gate for one script, optionally translating to English.
///
/// Translation splits the two signals apart, because they measure different
/// things:
///
/// - `compression_ratio` is measured on the **output** text, which in translate
///   mode is English. The relaxed Indic and CJK ceilings exist because those
///   *scripts* compress well; English output does not, so keeping a relaxed
///   ceiling here would admit repetition loops for free. It reverts to Latin.
/// - `avg_log_prob` and `no_speech_prob` describe how hard the **input** audio
///   and task were, and translating out of a low-resource language is if
///   anything harder than transcribing it. Those keep the input script's floors.
pub fn confidence(script: ScriptClass, translating: bool) -> SegmentConfidence {
    let mut gate = base_confidence(script);
    if translating {
        gate.max_compression_ratio = base_confidence(ScriptClass::Latin).max_compression_ratio;
    }
    gate
}

fn base_confidence(script: ScriptClass) -> SegmentConfidence {
    match script {
        ScriptClass::Latin => SegmentConfidence::default(),
        ScriptClass::Cjk => SegmentConfidence {
            max_no_speech_prob: 0.8,
            min_avg_log_prob: -1.3,
            max_compression_ratio: 3.6,
        },
        ScriptClass::Indic | ScriptClass::Unknown => SegmentConfidence {
            max_no_speech_prob: 0.8,
            min_avg_log_prob: -1.6,
            max_compression_ratio: 4.2,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SCRIPTS: [ScriptClass; 4] = [
        ScriptClass::Latin,
        ScriptClass::Indic,
        ScriptClass::Cjk,
        ScriptClass::Unknown,
    ];

    /// The regression this whole design exists for: real Bangla, decoded
    /// correctly, scored the way Bengali always scores, was thrown away by the
    /// English thresholds and returned as empty text with nothing logged.
    #[test]
    fn english_gate_rejects_typical_healthy_bangla_scores() {
        let english = confidence(ScriptClass::Latin, false);
        assert!(!english.accepts(0.3, -1.25, 3.1));
    }

    #[test]
    fn indic_gate_accepts_the_same_scores() {
        let indic = confidence(ScriptClass::Indic, false);
        assert!(indic.accepts(0.3, -1.25, 3.1));
    }

    #[test]
    fn latin_gate_is_unchanged_from_the_shipped_defaults() {
        // English works today. The port must not move it.
        assert_eq!(
            confidence(ScriptClass::Latin, false),
            SegmentConfidence::default()
        );
    }

    #[test]
    fn non_latin_gates_are_strictly_more_permissive() {
        let latin = confidence(ScriptClass::Latin, false);
        for script in [ScriptClass::Indic, ScriptClass::Cjk, ScriptClass::Unknown] {
            let gate = confidence(script, false);
            assert!(
                gate.max_compression_ratio > latin.max_compression_ratio,
                "{script:?}"
            );
            assert!(gate.min_avg_log_prob < latin.min_avg_log_prob, "{script:?}");
        }
    }

    /// Permissive is not unbounded. Degenerate repetition must still be caught by
    /// the compression ratio on every script, or relaxing the gate would simply
    /// trade missing text for gibberish.
    #[test]
    fn every_gate_still_rejects_severe_repetition() {
        for script in ALL_SCRIPTS {
            assert!(
                !confidence(script, false).accepts(0.1, -0.2, 8.0),
                "{script:?} accepted an obvious repetition loop"
            );
        }
    }

    #[test]
    fn unknown_script_is_as_permissive_as_the_widest_known_script() {
        // Auto mode cannot know the script yet, so it must not be the path that
        // silently drops text; it takes the widest gate.
        let unknown = confidence(ScriptClass::Unknown, false);
        let indic = confidence(ScriptClass::Indic, false);
        assert!(unknown.max_compression_ratio >= indic.max_compression_ratio);
        assert!(unknown.min_avg_log_prob <= indic.min_avg_log_prob);
    }

    #[test]
    fn decode_thresholds_track_the_acceptance_gate() {
        // The decoder's own fallback triggers must be at least as permissive as
        // the gate, otherwise the decoder discards a candidate the gate would
        // have accepted and no amount of gate-relaxing can recover it.
        for script in ALL_SCRIPTS {
            for translating in [false, true] {
                let decode = decoding(script, translating);
                let gate = confidence(script, translating);
                assert!(
                    decode.compression_ratio_threshold <= gate.max_compression_ratio,
                    "{script:?} translating={translating}"
                );
                assert!(
                    decode.log_prob_threshold >= gate.min_avg_log_prob,
                    "{script:?} translating={translating}"
                );
            }
        }
    }

    #[test]
    fn non_latin_gets_more_temperature_fallbacks() {
        // Re-decoding at a higher temperature is the only mechanism that turns a
        // hallucinated window into a real one. Non-Latin trips the checks more
        // often, so it needs more than the single retry English gets.
        let latin = decoding(ScriptClass::Latin, false);
        for script in [ScriptClass::Indic, ScriptClass::Cjk, ScriptClass::Unknown] {
            assert!(
                decoding(script, false).temperature_fallback_count
                    > latin.temperature_fallback_count,
                "{script:?}"
            );
        }
    }

    #[test]
    fn latin_decode_thresholds_are_unchanged() {
        let latin = decoding(ScriptClass::Latin, false);
        assert_eq!(latin.compression_ratio_threshold, 2.2);
        assert_eq!(latin.log_prob_threshold, -0.8);
        assert_eq!(latin.first_token_log_prob_threshold, -1.2);
        assert_eq!(latin.no_speech_threshold, 0.45);
        assert_eq!(latin.temperature_fallback_count, 1);
    }

    /// Translating reverts only the compression ceiling, because that one is
    /// measured on English output while the others describe the input audio.
    #[test]
    fn translating_reverts_compression_but_keeps_the_input_floors() {
        let indic = confidence(ScriptClass::Indic, false);
        let translating = confidence(ScriptClass::Indic, true);
        assert_eq!(
            translating.max_compression_ratio,
            confidence(ScriptClass::Latin, false).max_compression_ratio
        );
        assert_eq!(translating.min_avg_log_prob, indic.min_avg_log_prob);
        assert_eq!(translating.max_no_speech_prob, indic.max_no_speech_prob);
    }
}
