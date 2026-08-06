//! Where a lane's audio comes from.
//!
//! The session does not know whether it is reading a microphone, one
//! application's output, or a file. That indirection is what lets the whole
//! session path — lanes, VAD, ASR, diarization, merge, export — be exercised on
//! a machine with no Windows audio stack at all.

use memento_core::audio_math;
use std::path::Path;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// The microphone: you.
    You,
    /// System audio: everyone else.
    Others,
}

/// A lane's audio, delivered as 16 kHz mono f32 chunks.
pub trait CaptureSource: Send {
    /// Begin capturing. Returns the clock reading of the first sample, when the
    /// source can measure one — the two lanes start milliseconds apart and this
    /// is what places them on a shared timeline.
    fn start(&mut self) -> Result<Option<f64>, Error>;

    /// The next chunk, or `None` when the source is finished. Blocking is fine;
    /// the session runs this on its own thread.
    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Error>;

    fn stop(&mut self);
}

/// Replays a WAV. Used for testing the session end to end, and for developing
/// the UI on a machine that cannot capture Windows audio.
pub struct FileSource {
    samples: Vec<f32>,
    cursor: usize,
    chunk_samples: usize,
    /// Seconds this lane starts after the other, for exercising lane alignment.
    start_offset: Option<f64>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let wav = audio_math::parse_wav(&std::fs::read(path)?)?;
        if wav.sample_rate != 16_000 || wav.channels != 1 {
            return Err(format!(
                "expected 16 kHz mono, got {} Hz {}ch",
                wav.sample_rate, wav.channels
            )
            .into());
        }
        Ok(Self::from_samples(wav.samples))
    }

    pub fn from_samples(samples: Vec<f32>) -> Self {
        Self {
            samples,
            cursor: 0,
            // Roughly what WASAPI hands over per callback.
            chunk_samples: 1024,
            start_offset: None,
        }
    }

    /// An empty lane, for a session where only one side has audio.
    pub fn silent() -> Self {
        Self::from_samples(Vec::new())
    }

    pub fn with_start_offset(mut self, seconds: f64) -> Self {
        self.start_offset = Some(seconds);
        self
    }
}

impl CaptureSource for FileSource {
    fn start(&mut self) -> Result<Option<f64>, Error> {
        self.cursor = 0;
        Ok(self.start_offset)
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Error> {
        if self.cursor >= self.samples.len() {
            return Ok(None);
        }
        let end = (self.cursor + self.chunk_samples).min(self.samples.len());
        let chunk = self.samples[self.cursor..end].to_vec();
        self.cursor = end;
        Ok(Some(chunk))
    }

    fn stop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use memento_core::testing::{silence, tone};

    #[test]
    fn a_file_source_delivers_every_sample_once() {
        let audio = [silence(0.5), tone(0.5, 0.1)].concat();
        let mut source = FileSource::from_samples(audio.clone());
        source.start().unwrap();

        let mut got = Vec::new();
        while let Some(chunk) = source.next_chunk().unwrap() {
            got.extend(chunk);
        }
        assert_eq!(got, audio);
        assert_eq!(
            source.next_chunk().unwrap(),
            None,
            "and then stays finished"
        );
    }

    #[test]
    fn a_silent_source_finishes_immediately() {
        let mut source = FileSource::silent();
        source.start().unwrap();
        assert_eq!(source.next_chunk().unwrap(), None);
    }

    #[test]
    fn start_offsets_reach_the_lane_alignment() {
        let mut source = FileSource::from_samples(vec![0.0; 10]).with_start_offset(100.25);
        assert_eq!(source.start().unwrap(), Some(100.25));
    }

    #[test]
    fn restarting_replays_from_the_beginning() {
        let mut source = FileSource::from_samples(vec![0.5; 100]);
        source.start().unwrap();
        while source.next_chunk().unwrap().is_some() {}
        source.start().unwrap();
        assert!(source.next_chunk().unwrap().is_some());
    }

    #[test]
    fn a_wrongly_formatted_file_is_rejected_with_a_useful_message() {
        let path = std::env::temp_dir().join("memento-source-test.wav");
        std::fs::write(&path, audio_math::wav_bytes(&[0.1; 100], 44_100)).unwrap();
        let err = match FileSource::open(&path) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 44.1 kHz file must be rejected"),
        };
        assert!(err.contains("16 kHz mono"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
