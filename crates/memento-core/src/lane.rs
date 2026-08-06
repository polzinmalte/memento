//! Per-lane pipeline: captured audio → VAD → Whisper → transcript lines.
//! Ported from listnr `Sources/ListnrCore/Transcription/LaneTranscriber.swift`.
//!
//! Lanes stay separate (You vs Others); see the "never mix before ASR" rule.
//!
//! Two properties this type exists to guarantee:
//!
//! 1. **Order.** Audio is segmented in capture order. One thread per stage,
//!    each draining a FIFO queue, is what makes that true — the Swift original
//!    warns that spawning a task per buffer shuffles the VAD's input and
//!    corrupts utterances, and the same hazard exists with a thread pool here.
//! 2. **No head-of-line blocking.** Whisper takes hundreds of milliseconds to
//!    seconds per segment. Running it inline with segmentation would stall the
//!    VAD and let audio pile up without bound, so ASR is a separate stage behind
//!    a bounded queue that drops oldest under sustained overload.

use crate::audio_math;
use crate::queue::BoundedQueue;
use crate::transcript::{TranscriptLine, format_clock};
use crate::vad::{EnergyVad, EnergyVadConfig, Segment};
use crate::{Error, Transcriber};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Buffers arrive roughly every 85 ms, so this is ~17 s of slack. The VAD stage
/// is cheap and should never fall this far behind; dropping oldest is a safety
/// valve, not an expected path.
const AUDIO_QUEUE_CAPACITY: usize = 200;
/// Each segment holds at most `max_speech_seconds` of audio, so bounding the
/// count bounds worst-case memory while Whisper catches up.
const SEGMENT_QUEUE_CAPACITY: usize = 32;

/// Called for each finished line as it is produced, for a live view. The Swift
/// original writes these straight to stderr; memento hands them to the UI.
pub type LineSink = Arc<dyn Fn(&TranscriptLine) + Send + Sync>;
/// Called when a segment fails to transcribe. Failures are per-segment and the
/// lane keeps running, so these must surface somewhere rather than be swallowed.
pub type ErrorSink = Arc<dyn Fn(&str, &Error) + Send + Sync>;

pub struct LaneConfig {
    pub speaker_label: String,
    pub speech_threshold: f32,
    pub min_segment_rms: f32,
    pub on_line: Option<LineSink>,
    pub on_error: Option<ErrorSink>,
}

impl LaneConfig {
    pub fn new(
        speaker_label: impl Into<String>,
        speech_threshold: f32,
        min_segment_rms: f32,
    ) -> Self {
        Self {
            speaker_label: speaker_label.into(),
            speech_threshold,
            min_segment_rms,
            on_line: None,
            on_error: None,
        }
    }
}

pub struct LaneTranscriber {
    pub speaker_label: String,
    audio: Arc<BoundedQueue<Vec<f32>>>,
    segments: Arc<BoundedQueue<Segment>>,
    vad_handle: Option<JoinHandle<()>>,
    asr_handle: Option<JoinHandle<Vec<TranscriptLine>>>,
}

impl LaneTranscriber {
    pub fn start(cfg: LaneConfig, mut transcriber: Box<dyn Transcriber>) -> Self {
        let audio: Arc<BoundedQueue<Vec<f32>>> = Arc::new(BoundedQueue::new(AUDIO_QUEUE_CAPACITY));
        let segments: Arc<BoundedQueue<Segment>> =
            Arc::new(BoundedQueue::new(SEGMENT_QUEUE_CAPACITY));

        let vad_handle = {
            let audio = Arc::clone(&audio);
            let segments = Arc::clone(&segments);
            let speech_threshold = cfg.speech_threshold;
            thread::spawn(move || {
                // Owned solely by this thread, which is why EnergyVad needs no
                // synchronisation of its own.
                let mut segmenter = EnergyVad::new(EnergyVadConfig {
                    speech_threshold,
                    min_speech_seconds: 0.45,
                    hangover_seconds: 0.60,
                    preroll_seconds: 0.45,
                    ..EnergyVadConfig::default()
                });
                while let Some(chunk) = audio.pop() {
                    for segment in segmenter.push(&chunk) {
                        segments.push(segment);
                    }
                }
                // Queue finished: emit whatever the tail of the session left buffered.
                for segment in segmenter.flush() {
                    segments.push(segment);
                }
                segments.close();
            })
        };

        let asr_handle = {
            let segments = Arc::clone(&segments);
            let label = cfg.speaker_label.clone();
            let min_segment_rms = cfg.min_segment_rms;
            let on_line = cfg.on_line.clone();
            let on_error = cfg.on_error.clone();
            thread::spawn(move || {
                let mut lines: Vec<TranscriptLine> = Vec::new();
                while let Some(segment) = segments.pop() {
                    if audio_math::rms(&segment.samples) < min_segment_rms {
                        continue;
                    }
                    match transcriber.transcribe(&segment.samples) {
                        Ok(text) if text.is_empty() => continue,
                        Ok(text) => {
                            let line = TranscriptLine::new(&label, segment.start_seconds, text);
                            if let Some(sink) = &on_line {
                                sink(&line);
                            } else {
                                eprintln!(
                                    "[{}] {:<8} {}",
                                    format_clock(line.start_seconds),
                                    line.speaker,
                                    line.text
                                );
                            }
                            lines.push(line);
                        }
                        Err(e) => {
                            if let Some(sink) = &on_error {
                                sink(&label, &e);
                            } else {
                                eprintln!("transcription error ({label}): {e}");
                            }
                        }
                    }
                }
                lines
            })
        };

        Self {
            speaker_label: cfg.speaker_label,
            audio,
            segments,
            vad_handle: Some(vad_handle),
            asr_handle: Some(asr_handle),
        }
    }

    /// Hand audio to the pipeline. Safe to call from the capture callback:
    /// returns immediately and never blocks on segmentation or ASR.
    pub fn push(&self, chunk: Vec<f32>) {
        self.audio.push(chunk);
    }

    /// Hand audio to the pipeline, waiting rather than dropping when it is
    /// behind. Only for sources that can be slowed down — replaying a file,
    /// tests — where a drop would corrupt the transcript instead of protecting
    /// it. A live capture callback must use [`Self::push`]: blocking there
    /// stalls the audio thread and loses samples at the driver instead.
    pub fn push_blocking(&self, chunk: Vec<f32>) {
        self.audio.push_blocking(chunk);
    }

    /// Close the pipeline and drain both stages. Returns the lane's lines in
    /// timestamp order. Safe to call after [`Self::abandon`], and safe to call
    /// twice — the second call returns nothing.
    pub fn finish(&mut self) -> Vec<TranscriptLine> {
        self.audio.close();
        if let Some(h) = self.vad_handle.take() {
            let _ = h.join();
        }
        // The VAD thread closes the segment queue on its way out; if it never
        // ran (abandoned before start) this makes sure ASR still terminates.
        self.segments.close();
        let mut lines = match self.asr_handle.take() {
            Some(h) => h.join().unwrap_or_default(),
            None => Vec::new(),
        };
        lines.sort_by(|a, b| a.start_seconds.total_cmp(&b.start_seconds));
        lines
    }

    /// Tear the pipeline down without draining it, for sessions that fail or are
    /// cancelled before `finish()`. Idempotent, and harmless after a normal
    /// `finish()`.
    pub fn abandon(&self) {
        self.audio.close_and_clear();
        self.segments.close_and_clear();
    }

    /// Buffers and segments lost to overload. Reported once at the end so a
    /// struggling machine says so instead of silently losing speech.
    pub fn drop_report(&self) -> Option<String> {
        let audio_lost = self.audio.drops();
        let segments_lost = self.segments.drops();
        if audio_lost == 0 && segments_lost == 0 {
            return None;
        }
        let mut note = format!("{}: dropped", self.speaker_label);
        if audio_lost > 0 {
            note += &format!(" {audio_lost} audio buffer(s)");
        }
        if audio_lost > 0 && segments_lost > 0 {
            note += " and";
        }
        if segments_lost > 0 {
            note += &format!(" {segments_lost} speech segment(s)");
        }
        note += ", transcription could not keep up. Try a smaller model.";
        Some(note)
    }
}

impl Drop for LaneTranscriber {
    /// A lane dropped without `finish()` must not leave two threads parked on a
    /// queue that nobody will ever close.
    fn drop(&mut self) {
        self.abandon();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{FakeTranscriber, silence, tone};

    fn lane_with(fake: FakeTranscriber, min_segment_rms: f32) -> LaneTranscriber {
        LaneTranscriber::start(
            LaneConfig {
                on_line: Some(Arc::new(|_| {})), // keep test output quiet
                ..LaneConfig::new("You", 0.01, min_segment_rms)
            },
            Box::new(fake),
        )
    }

    fn push_all(audio: &[f32], lane: &LaneTranscriber) {
        for chunk in audio.chunks(1024) {
            lane.push(chunk.to_vec());
        }
    }

    fn one_utterance() -> Vec<f32> {
        [silence(1.0), tone(0.5, 0.1), silence(1.5)].concat()
    }

    #[test]
    fn utterance_flows_through_pipeline() {
        let mut lane = lane_with(FakeTranscriber::new(&["hello"]), 0.01);
        push_all(&one_utterance(), &lane);
        let lines = lane.finish();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].speaker, "You");
        assert_eq!(lines[0].text, "hello");
    }

    #[test]
    fn lines_keep_capture_order() {
        let mut lane = LaneTranscriber::start(
            LaneConfig {
                on_line: Some(Arc::new(|_| {})),
                ..LaneConfig::new("Others", 0.01, 0.01)
            },
            Box::new(FakeTranscriber::new(&["first", "second"])),
        );
        push_all(
            &[
                silence(1.0),
                tone(0.5, 0.1),
                silence(2.0),
                tone(0.5, 0.1),
                silence(1.5),
            ]
            .concat(),
            &lane,
        );
        let lines = lane.finish();
        assert_eq!(
            lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(),
            ["first", "second"],
            "FIFO through both stages"
        );
        assert!(lines[0].start_seconds < lines[1].start_seconds);
    }

    #[test]
    fn empty_transcription_produces_no_line() {
        let mut lane = lane_with(FakeTranscriber::new(&[""]), 0.01);
        push_all(&one_utterance(), &lane);
        assert!(lane.finish().is_empty());
    }

    #[test]
    fn quiet_segment_gated_by_rms_floor() {
        let fake = FakeTranscriber::new(&["hello"]);
        let calls = fake.call_counter();
        // Impossibly high floor: nothing may reach the transcriber.
        let mut lane = lane_with(fake, 0.5);
        push_all(&one_utterance(), &lane);
        assert!(lane.finish().is_empty());
        assert_eq!(calls(), 0, "segment must be dropped before the transcriber");
    }

    #[test]
    fn transcriber_errors_do_not_kill_the_lane() {
        let mut lane = lane_with(FakeTranscriber::failing(), 0.01);
        push_all(&one_utterance(), &lane);
        assert!(lane.finish().is_empty(), "the error is reported, not fatal");
        // Reaching here at all is the assertion: finish() must not hang or panic.
    }

    #[test]
    fn abandon_tears_down_without_hanging() {
        let mut lane = lane_with(FakeTranscriber::new(&["hello"]), 0.01);
        push_all(&[silence(1.0), tone(0.5, 0.1)].concat(), &lane);
        lane.abandon();
        // After abandon, finish() must return promptly instead of joining threads
        // parked on a queue nobody will close. Double teardown must also be safe.
        let _ = lane.finish();
        lane.abandon();
    }

    #[test]
    fn finish_twice_is_safe() {
        let mut lane = lane_with(FakeTranscriber::new(&["hello"]), 0.01);
        push_all(&one_utterance(), &lane);
        assert_eq!(lane.finish().len(), 1);
        assert!(lane.finish().is_empty());
    }

    #[test]
    fn no_drops_means_no_report() {
        let mut lane = lane_with(FakeTranscriber::new(&["hello"]), 0.01);
        push_all(&one_utterance(), &lane);
        let _ = lane.finish();
        assert_eq!(lane.drop_report(), None);
    }
}
