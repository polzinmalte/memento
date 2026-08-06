//! Platform-independent core of memento: the two-lane transcription pipeline.
//!
//! Ported from [listnr](https://github.com/rokib16x/listnr) (MIT, © 2026
//! Rokibul Hasan), whose macOS implementation this crate follows closely —
//! same thresholds, same segmentation rules, same tests. Everything Apple-only
//! (ScreenCaptureKit, AVAudioEngine, WhisperKit, SpeakerKit) lives behind the
//! [`Transcriber`] and [`Diarizer`] traits and is supplied by the platform
//! crates. See `LICENSE-listnr`.
//!
//! Nothing here touches the filesystem, the network, or an audio device, so it
//! builds and tests on any host — including a macOS machine developing the
//! Windows lanes.

pub mod audio_math;
pub mod decode_tuning;
pub mod export;
pub mod lane;
pub mod language;
pub mod queue;
pub mod recorder;
pub mod sensitivity;
pub mod speakers;
pub mod stop_state;
pub mod testing;
pub mod transcript;
pub mod transcript_text;
pub mod vad;

pub use decode_tuning::{DecodeThresholds, SegmentConfidence};
pub use export::{render_markdown, transcript_filename};
pub use lane::{LaneConfig, LaneTranscriber};
pub use language::{LanguageMode, ScriptClass};
pub use recorder::LaneRecorder;
pub use sensitivity::{LaneThresholds, MicSensitivity, lane_tuning};
pub use speakers::{SpanOptions, speaker_label, transcribe_spans};
pub use stop_state::SessionStopState;
pub use transcript::{LaneAnchors, LaneOffsets, TranscriptLine, format_clock, merge, shift};
pub use vad::{EnergyVad, EnergyVadConfig, Segment};

/// Errors cross thread boundaries and come from wildly different backends
/// (whisper.cpp, ONNX Runtime, WASAPI), so the core keeps them opaque rather
/// than pretending to enumerate them.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Speech-to-text for one segment of 16 kHz mono audio.
///
/// Implementations are owned by a single lane's ASR thread and called strictly
/// in order, so they may hold mutable model state without locking. The shared
/// CPU work queue that memento uses to stop two lanes oversubscribing the CPU
/// is itself an implementation of this trait, which is why it stays this simple.
pub trait Transcriber: Send {
    fn model_id(&self) -> &str;
    /// Load the model. Called once before capture starts, because doing it
    /// lazily makes the first utterance of a meeting arrive minutes late.
    fn warm_up(&mut self) -> Result<(), Error>;
    fn transcribe(&mut self, audio: &[f32]) -> Result<String, Error>;
}

/// A stretch of audio attributed to one speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizedSpan {
    pub label: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

/// Speaker diarization over a whole lane, run once at the end of a session.
pub trait Diarizer: Send {
    fn warm_up(&mut self) -> Result<(), Error>;
    fn diarize(&mut self, audio: &[f32]) -> Result<Vec<DiarizedSpan>, Error>;
}
