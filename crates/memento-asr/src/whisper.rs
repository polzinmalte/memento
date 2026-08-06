//! whisper.cpp transcription, gated by the ported per-script thresholds.
//!
//! Replaces listnr's `WhisperKitTranscriber.swift`. The decisions that file
//! makes — which decoder thresholds to use, how to score a segment, what counts
//! as non-speech — all live in `memento-core` and are shared verbatim; what is
//! here is only the whisper.cpp binding and the confidence arithmetic that
//! WhisperKit reported for free but whisper.cpp does not.

use memento_core::decode_tuning::{self, DecodeThresholds, SegmentConfidence};
use memento_core::{Error, LanguageMode, ScriptClass, Transcriber, transcript_text};
use std::io::Write;
use std::path::{Path, PathBuf};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSegment,
    WhisperState,
};

pub struct WhisperTranscriber {
    model_id: String,
    model_path: PathBuf,
    language: LanguageMode,
    translate: bool,
    n_threads: i32,
    state: Option<WhisperState>,
}

impl WhisperTranscriber {
    pub fn new(
        model_id: impl Into<String>,
        model_path: impl Into<PathBuf>,
        language: LanguageMode,
        translate: bool,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_path: model_path.into(),
            language,
            translate,
            // Physical cores, not logical. Whisper's GEMMs are memory-bandwidth
            // bound, so hyperthreads add contention rather than throughput.
            n_threads: default_thread_count(),
            state: None,
        }
    }

    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads.max(1);
        self
    }

    fn script(&self) -> ScriptClass {
        self.language.script_class()
    }

    /// Built without borrowing `self`, so the caller can hold `&mut self.state`
    /// at the same time. The language codes are `'static`, which is what makes
    /// that possible.
    fn params(&self) -> FullParams<'static, 'static> {
        let thresholds: DecodeThresholds = decode_tuning::decoding(self.script(), self.translate);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.n_threads);
        params.set_translate(self.translate);
        params.set_language(self.language.whisper_language());

        // The per-script decoder fallbacks. Without these, German and every
        // other non-English language is decoded against English-tuned triggers.
        params.set_entropy_thold(thresholds.compression_ratio_threshold);
        params.set_logprob_thold(thresholds.log_prob_threshold);
        params.set_no_speech_thold(thresholds.no_speech_threshold);
        params.set_temperature_inc(thresholds.temperature_increment_on_fallback);

        // Segments come from the VAD already trimmed to one utterance, so
        // whisper's own context carry-over only invites it to continue a
        // sentence it never heard.
        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_suppress_blank(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params
    }
}

impl Transcriber for WhisperTranscriber {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn warm_up(&mut self) -> Result<(), Error> {
        if self.state.is_some() {
            return Ok(());
        }
        if !self.model_path.exists() {
            return Err(format!("model file not found: {}", self.model_path.display()).into());
        }

        let path = self
            .model_path
            .to_str()
            .ok_or_else(|| -> Error { "model path is not valid UTF-8".into() })?;
        // `WhisperState` holds its own reference to the context internals, so
        // the context does not need to be kept alive separately here.
        let context = WhisperContext::new_with_params(path, WhisperContextParameters::default())?;
        self.state = Some(context.create_state()?);
        Ok(())
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String, Error> {
        let script = self.script();
        let translating = self.translate;
        let params = self.params();
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| -> Error { "transcriber not loaded; call warm_up() first".into() })?;

        state.full(params, audio)?;

        let gate = decode_tuning::confidence(script, translating);
        let mut kept = String::new();
        for i in 0..state.full_n_segments() {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };
            let Ok(raw) = segment.to_str_lossy() else {
                continue;
            };
            let text = transcript_text::clean(&raw);
            if text.is_empty() {
                continue;
            }
            if !accepts_segment(&segment, &text, &gate) {
                continue;
            }
            if !kept.is_empty() {
                kept.push(' ');
            }
            kept.push_str(&text);
        }

        // The shape-based guard runs on the assembled text, not per segment: a
        // loop often spans segment boundaries.
        let assembled = transcript_text::clean(&kept);
        if transcript_text::is_non_speech(&assembled) {
            return Ok(String::new());
        }
        Ok(assembled)
    }
}

/// Whisper's own acceptance signals, which WhisperKit surfaces directly but
/// whisper.cpp leaves to the caller to derive.
fn accepts_segment(segment: &WhisperSegment<'_>, text: &str, gate: &SegmentConfidence) -> bool {
    let avg_log_prob = average_token_log_prob(segment).unwrap_or(0.0);
    gate.accepts(
        segment.no_speech_probability(),
        avg_log_prob,
        compression_ratio(text),
    )
}

/// Mean per-token log probability. whisper.cpp reports it per token, and the
/// threshold ported from listnr is defined on the average over the segment.
fn average_token_log_prob(segment: &WhisperSegment<'_>) -> Option<f32> {
    let count = segment.n_tokens();
    if count <= 0 {
        return None;
    }
    let mut total = 0.0f32;
    let mut used = 0;
    for i in 0..count {
        if let Some(token) = segment.get_token(i) {
            total += token.token_data().plog;
            used += 1;
        }
    }
    (used > 0).then(|| total / used as f32)
}

/// Text length over zlib-compressed length — the definition the reference
/// Whisper implementation uses, and therefore the one the ported ceilings were
/// calibrated against. A degenerate loop compresses far better than speech.
fn compression_ratio(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    if encoder.write_all(text.as_bytes()).is_err() {
        return 0.0;
    }
    match encoder.finish() {
        Ok(compressed) if !compressed.is_empty() => text.len() as f32 / compressed.len() as f32,
        _ => 0.0,
    }
}

/// Physical cores where the platform reports them, minus nothing: whisper is the
/// only meaningful load while a meeting is being transcribed.
fn default_thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1) as i32)
        .unwrap_or(4)
}

/// whisper.cpp writes `GGML_FILE_MAGIC` (`0x67676d6c`) as a little-endian u32,
/// so on disk the first four bytes read `lmgg`, not `ggml`. Getting that
/// backwards rejects every real model.
const GGML_MAGIC_LE: &[u8; 4] = b"lmgg";
/// Newer conversions ship as GGUF, whose magic *is* stored as ASCII.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// True when the file looks like a complete model rather than a truncated
/// download or an HTML error page saved under a `.bin` name.
pub fn looks_like_ggml_model(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).is_ok() && (&magic == GGML_MAGIC_LE || &magic == GGUF_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repetition_compresses_far_better_than_speech() {
        // The property the compression ceiling relies on. Both strings are the
        // same length, so only their redundancy differs.
        let loop_text = "ja ja ja ja ja ja ja ja ja ja ja ja ja ja ja ja ja ja";
        let speech = "wir sollten das Release auf Freitag verschieben und danach";
        assert!(
            compression_ratio(loop_text) > compression_ratio(speech),
            "loop {} vs speech {}",
            compression_ratio(loop_text),
            compression_ratio(speech)
        );
    }

    #[test]
    fn a_german_sentence_passes_the_latin_gate() {
        // German is Latin-script, so it is gated by the strict English numbers.
        // That is correct — the relaxed gates exist for Indic and CJK — but it
        // means an ordinary German sentence must not look degenerate.
        let gate = decode_tuning::confidence(ScriptClass::Latin, false);
        let sentence =
            "wir sollten das Release auf Freitag verschieben und danach das Changelog schreiben";
        assert!(gate.accepts(0.1, -0.4, compression_ratio(sentence)));
    }

    #[test]
    fn a_german_repetition_loop_fails_the_latin_gate() {
        let gate = decode_tuning::confidence(ScriptClass::Latin, false);
        let looped = "genau genau genau genau genau genau genau genau genau genau genau genau";
        assert!(!gate.accepts(0.1, -0.4, compression_ratio(looped)));
    }

    /// The compression ceilings ported from listnr were calibrated against the
    /// reference Whisper implementation, which measures against **zlib**. Not
    /// every DEFLATE implementation agrees on short repetitive input: flate2's
    /// default miniz_oxide backend emits 29 bytes for the string below where
    /// zlib emits 17, which reads as a ratio of 2.45 instead of 4.18 — under the
    /// 2.6 Latin ceiling, so the loop would be accepted as speech.
    ///
    /// This pins the backend. If it fails, the gate has silently gone permissive.
    #[test]
    fn matches_reference_zlib() {
        let looped = "genau genau genau genau genau genau genau genau genau genau genau genau";
        let ratio = compression_ratio(looped);
        assert!(
            (ratio - 4.18).abs() < 0.05,
            "expected the reference zlib ratio of ~4.18, got {ratio} — \
             is flate2 still using the zlib-rs backend?"
        );
    }

    #[test]
    fn empty_text_has_no_ratio() {
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn thread_count_is_at_least_one() {
        assert!(default_thread_count() >= 1);
    }

    #[test]
    fn a_missing_model_is_not_mistaken_for_a_real_one() {
        assert!(!looks_like_ggml_model(Path::new("/nonexistent/model.bin")));
    }

    /// Verified against the real first bytes of `ggml-small-q5_1.bin`. The magic
    /// is a little-endian u32, so it reads `lmgg` on disk; checking for `ggml`
    /// rejects every genuine model and reports it as a corrupt download.
    #[test]
    fn recognises_real_model_magic_and_rejects_an_error_page() {
        let dir = std::env::temp_dir().join("memento-magic-test");
        std::fs::create_dir_all(&dir).unwrap();

        let good = dir.join("good.bin");
        std::fs::write(&good, b"lmgg\x99\xca\x00\x00").unwrap();
        assert!(looks_like_ggml_model(&good));

        let gguf = dir.join("gguf.bin");
        std::fs::write(&gguf, b"GGUF\x03\x00\x00\x00").unwrap();
        assert!(looks_like_ggml_model(&gguf));

        // What a proxy or an expired link actually hands back.
        let html = dir.join("bad.bin");
        std::fs::write(&html, b"<!DOCTYPE html><html>404").unwrap();
        assert!(!looks_like_ggml_model(&html));

        let truncated = dir.join("tiny.bin");
        std::fs::write(&truncated, b"lm").unwrap();
        assert!(!looks_like_ggml_model(&truncated));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transcribing_before_warm_up_errors_rather_than_panics() {
        let mut t = WhisperTranscriber::new("small", "/nonexistent.bin", LanguageMode::De, false);
        assert!(t.transcribe(&[0.0; 16_000]).is_err());
    }

    #[test]
    fn warm_up_reports_a_missing_model_file() {
        let mut t = WhisperTranscriber::new("small", "/nonexistent.bin", LanguageMode::De, false);
        let err = t.warm_up().unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }
}
