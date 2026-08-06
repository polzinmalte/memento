//! Turning diarization spans into speaker-labelled transcript lines.
//! Ported from the `DiarizedTranscriber` half of listnr
//! `Sources/ListnrCore/Speakers/RemoteDiarizer.swift`.
//!
//! The diarizer itself is a platform concern (`memento-diar` wraps sherpa-onnx);
//! everything here is arithmetic on spans and so lives where it can be tested
//! without models.

use crate::audio_math;
use crate::transcript::TranscriptLine;
use crate::{DiarizedSpan, Transcriber};

#[derive(Debug, Clone, Copy)]
pub struct SpanOptions {
    /// Diarization boundaries land on the vowel, not the consonant before it, so
    /// each slice is widened backwards or the first word arrives decapitated.
    pub pad_before_seconds: f64,
    pub pad_after_seconds: f64,
    /// Floor below which a span is not worth transcribing.
    pub min_span_rms: f32,
}

impl Default for SpanOptions {
    fn default() -> Self {
        Self {
            pad_before_seconds: 0.35,
            pad_after_seconds: 0.20,
            min_span_rms: 0.005,
        }
    }
}

/// Cut lane audio by diarization spans and transcribe each, yielding
/// multi-speaker lines.
pub fn transcribe_spans(
    audio: &[f32],
    sample_rate: f64,
    spans: &[DiarizedSpan],
    transcriber: &mut dyn Transcriber,
    options: SpanOptions,
) -> Vec<TranscriptLine> {
    let pad_before = (options.pad_before_seconds * sample_rate) as usize;
    let pad_after = (options.pad_after_seconds * sample_rate) as usize;
    let mut lines = Vec::new();

    for span in spans {
        let raw_start = ((span.start_seconds * sample_rate).max(0.0) as usize).min(audio.len());
        let raw_end = ((span.end_seconds * sample_rate).max(0.0) as usize).min(audio.len());
        if raw_end <= raw_start {
            continue;
        }
        let start = raw_start.saturating_sub(pad_before);
        let end = (raw_end + pad_after).min(audio.len());

        // Measure loudness on the span the diarizer actually found, but
        // transcribe the padded slice. Measuring the padded slice would penalise
        // exactly what the padding is for: the leading silence drags the average
        // down, so a real but quiet utterance falls under the floor precisely
        // because it was widened.
        if audio_math::rms(&audio[raw_start..raw_end]) < options.min_span_rms {
            continue;
        }

        match transcriber.transcribe(&audio[start..end]) {
            Ok(text) if !text.is_empty() => {
                lines.push(TranscriptLine::new(&span.label, span.start_seconds, text));
            }
            // A failed span loses one utterance; the rest of the meeting stands.
            _ => continue,
        }
    }

    lines
}

/// Map the diarizer's zero-based cluster ids onto the names people read.
pub fn speaker_label(cluster_id: i32) -> String {
    format!("Speaker {}", cluster_id + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::testing::{SAMPLE_RATE, silence, tone};
    use std::sync::{Arc, Mutex};

    /// Records the length of every slice it is handed, which is how the padding
    /// behaviour is observed.
    struct SliceRecorder {
        lengths: Arc<Mutex<Vec<usize>>>,
        text: String,
    }

    impl Transcriber for SliceRecorder {
        fn model_id(&self) -> &str {
            "recorder"
        }
        fn warm_up(&mut self) -> Result<(), Error> {
            Ok(())
        }
        fn transcribe(&mut self, audio: &[f32]) -> Result<String, Error> {
            self.lengths.lock().unwrap().push(audio.len());
            Ok(self.text.clone())
        }
    }

    fn recorder(text: &str) -> (SliceRecorder, Arc<Mutex<Vec<usize>>>) {
        let lengths = Arc::new(Mutex::new(Vec::new()));
        (
            SliceRecorder {
                lengths: Arc::clone(&lengths),
                text: text.into(),
            },
            lengths,
        )
    }

    fn span(label: &str, start: f64, end: f64) -> DiarizedSpan {
        DiarizedSpan {
            label: label.into(),
            start_seconds: start,
            end_seconds: end,
        }
    }

    /// Loud speech from 1.0s to 3.0s, silence either side.
    fn two_second_utterance() -> Vec<f32> {
        [silence(1.0), tone(2.0, 0.1), silence(1.0)].concat()
    }

    #[test]
    fn each_span_becomes_a_labelled_line() {
        let audio = two_second_utterance();
        let (mut fake, _) = recorder("hallo");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 1.0, 3.0)],
            &mut fake,
            SpanOptions::default(),
        );
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].speaker, "Speaker 1");
        assert_eq!(lines[0].text, "hallo");
        // The line is stamped at the span's real start, not the padded one.
        assert_eq!(lines[0].start_seconds, 1.0);
    }

    #[test]
    fn the_transcribed_slice_is_padded_on_both_sides() {
        let audio = two_second_utterance();
        let (mut fake, lengths) = recorder("hallo");
        let options = SpanOptions::default();
        transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 1.0, 3.0)],
            &mut fake,
            options,
        );
        let expected =
            ((2.0 + options.pad_before_seconds + options.pad_after_seconds) * SAMPLE_RATE) as usize;
        assert_eq!(lengths.lock().unwrap()[0], expected);
    }

    /// The subtlety the Swift original calls out: the loudness floor is measured
    /// on the diarizer's span, not the padded slice. Padding a short utterance
    /// with silence drags its average down, so measuring the padded slice would
    /// reject quiet speech *because* it was widened.
    #[test]
    fn the_rms_floor_ignores_the_padding() {
        // 0.25s of quiet speech. Its own RMS clears the 0.005 floor; averaged
        // together with the 0.55s of padding silence it does not.
        let options = SpanOptions::default();
        let (start_s, end_s) = (1.0, 1.25);
        let audio = [
            silence(start_s),
            tone(end_s - start_s, 0.0065),
            silence(1.0),
        ]
        .concat();

        let core = |s: f64, e: f64| ((s * SAMPLE_RATE) as usize)..((e * SAMPLE_RATE) as usize);
        let core_rms = audio_math::rms(&audio[core(start_s, end_s)]);
        let padded_rms = audio_math::rms(
            &audio[core(
                start_s - options.pad_before_seconds,
                end_s + options.pad_after_seconds,
            )],
        );
        assert!(
            core_rms >= options.min_span_rms && padded_rms < options.min_span_rms,
            "test premise: core {core_rms} must clear {} and padded {padded_rms} must not",
            options.min_span_rms
        );

        let (mut fake, _) = recorder("leise");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", start_s, end_s)],
            &mut fake,
            options,
        );
        assert_eq!(lines.len(), 1, "quiet speech must survive being padded");
    }

    #[test]
    fn silent_spans_never_reach_the_transcriber() {
        let audio = silence(4.0);
        let (mut fake, lengths) = recorder("hallucination");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 1.0, 3.0)],
            &mut fake,
            SpanOptions::default(),
        );
        assert!(lines.is_empty());
        assert!(lengths.lock().unwrap().is_empty(), "silence must be gated");
    }

    #[test]
    fn padding_is_clamped_to_the_audio() {
        // A span at the very start and one running to the very end: padding must
        // not index outside the buffer.
        let audio = tone(2.0, 0.1);
        let (mut fake, lengths) = recorder("x");
        transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 0.0, 0.5), span("Speaker 2", 1.5, 2.0)],
            &mut fake,
            SpanOptions::default(),
        );
        for length in lengths.lock().unwrap().iter() {
            assert!(*length <= audio.len());
        }
    }

    #[test]
    fn spans_past_the_end_of_the_audio_are_skipped() {
        let audio = two_second_utterance();
        let (mut fake, _) = recorder("x");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 99.0, 100.0)],
            &mut fake,
            SpanOptions::default(),
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn inverted_and_empty_spans_are_skipped() {
        let audio = two_second_utterance();
        let (mut fake, _) = recorder("x");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 3.0, 1.0), span("Speaker 2", 2.0, 2.0)],
            &mut fake,
            SpanOptions::default(),
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn empty_transcriptions_produce_no_line() {
        let audio = two_second_utterance();
        let (mut fake, _) = recorder("");
        let lines = transcribe_spans(
            &audio,
            SAMPLE_RATE,
            &[span("Speaker 1", 1.0, 3.0)],
            &mut fake,
            SpanOptions::default(),
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn cluster_ids_become_one_based_speaker_names() {
        assert_eq!(speaker_label(0), "Speaker 1");
        assert_eq!(speaker_label(3), "Speaker 4");
    }
}
