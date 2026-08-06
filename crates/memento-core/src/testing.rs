//! Synthetic audio and a fake transcriber, so the pipeline's ordering and
//! teardown guarantees are testable without models or hardware.
//!
//! Public (not `#[cfg(test)]`) because the platform crates use the same helpers
//! to exercise capture against a known signal.

use crate::{Error, Transcriber};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const SAMPLE_RATE: f64 = 16_000.0;

/// Alternating-sign samples of the given amplitude, so RMS == amplitude exactly.
/// Every VAD threshold assertion depends on that identity.
pub fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
    let count = (seconds * SAMPLE_RATE) as usize;
    (0..count)
        .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
        .collect()
}

pub fn silence(seconds: f64) -> Vec<f32> {
    vec![0.0; (seconds * SAMPLE_RATE) as usize]
}

/// Returns `texts[n]` for the n-th call, repeating the last entry.
pub struct FakeTranscriber {
    texts: Vec<String>,
    calls: Arc<AtomicUsize>,
    fail: bool,
}

impl FakeTranscriber {
    pub fn new(texts: &[&str]) -> Self {
        Self {
            texts: texts.iter().map(|s| s.to_string()).collect(),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }
    }

    /// Fails every call, for the "a bad segment must not kill the lane" case.
    pub fn failing() -> Self {
        Self {
            fail: true,
            ..Self::new(&[""])
        }
    }

    /// A handle that survives the move into the lane's ASR thread.
    pub fn call_counter(&self) -> impl Fn() -> usize + use<> {
        let calls = Arc::clone(&self.calls);
        move || calls.load(Ordering::SeqCst)
    }
}

impl Transcriber for FakeTranscriber {
    fn model_id(&self) -> &str {
        "fake"
    }

    fn warm_up(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn transcribe(&mut self, _audio: &[f32]) -> Result<String, Error> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err("fake transcriber failure".into());
        }
        Ok(self.texts[n.min(self.texts.len() - 1)].clone())
    }
}
