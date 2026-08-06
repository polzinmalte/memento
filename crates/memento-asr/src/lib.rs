//! CPU speech-to-text for memento, via whisper.cpp.
//!
//! Replaces listnr's WhisperKit backend, which is Core ML and Apple-only. The
//! decisions that backend encodes are not re-made here: the per-script decoder
//! thresholds, the acceptance gate and the text rules all come from
//! `memento-core` unchanged.

pub mod download;
pub mod models;
pub mod shared;
pub mod whisper;

pub use models::{Model, preferred_model_id};
pub use shared::SharedAsr;
pub use whisper::WhisperTranscriber;
