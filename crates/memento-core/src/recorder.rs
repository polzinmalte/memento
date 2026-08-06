//! Streams a lane's audio to a WAV file as it arrives.
//!
//! listnr keeps both lanes in RAM and documents the cost: roughly 460 MB an
//! hour. Diarization needs the whole remote lane at the end of the session, so
//! the audio has to survive somehow — but it does not have to be resident.
//! Writing it out as it arrives keeps memory flat regardless of meeting length,
//! and diarization reads it back once at the end.
//!
//! The header is written up front with placeholder sizes and patched on
//! [`finalize`](LaneRecorder::finalize), the usual approach for a stream whose
//! length is unknown until it ends.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Byte offsets of the two length fields in a canonical 44-byte WAV header.
const RIFF_SIZE_OFFSET: u64 = 4;
const DATA_SIZE_OFFSET: u64 = 40;
const HEADER_BYTES: u32 = 44;

pub struct LaneRecorder {
    writer: BufWriter<File>,
    path: PathBuf,
    samples_written: u32,
    sample_rate: u32,
}

impl LaneRecorder {
    pub fn create(path: impl Into<PathBuf>, sample_rate: u32) -> Result<Self, Error> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut writer = BufWriter::new(File::create(&path)?);
        // Placeholder sizes; patched by finalize().
        writer.write_all(&header(sample_rate, 0))?;
        Ok(Self {
            writer,
            path,
            samples_written: 0,
            sample_rate,
        })
    }

    pub fn write(&mut self, samples: &[f32]) -> Result<(), Error> {
        for s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            self.writer
                .write_all(&((clamped * 32767.0) as i16).to_le_bytes())?;
        }
        self.samples_written += samples.len() as u32;
        Ok(())
    }

    pub fn seconds_written(&self) -> f64 {
        self.samples_written as f64 / self.sample_rate as f64
    }

    /// Patch the header and close. Returns the finished file's path.
    ///
    /// A recorder dropped without this leaves a header claiming zero samples —
    /// readable, but empty — so a crashed session yields a file that is honest
    /// about being unusable rather than one that silently truncates.
    pub fn finalize(mut self) -> Result<PathBuf, Error> {
        let data_size = self.samples_written * 2;
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(RIFF_SIZE_OFFSET))?;
        file.write_all(&(36 + data_size).to_le_bytes())?;
        file.seek(SeekFrom::Start(DATA_SIZE_OFFSET))?;
        file.write_all(&data_size.to_le_bytes())?;
        file.flush()?;
        Ok(self.path)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn header(sample_rate: u32, data_size: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_math::parse_wav;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join("memento-recorder-tests")
            .join(name)
    }

    #[test]
    fn written_audio_reads_back_identically() {
        let path = temp_path("roundtrip.wav");
        let mut recorder = LaneRecorder::create(&path, 16_000).unwrap();
        let original = vec![0.0, 0.5, -0.5, 0.25];
        recorder.write(&original).unwrap();
        let finished = recorder.finalize().unwrap();

        let parsed = parse_wav(&std::fs::read(&finished).unwrap()).unwrap();
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples.len(), original.len());
        for (got, want) in parsed.samples.iter().zip(&original) {
            assert!((got - want).abs() < 1e-4);
        }
        let _ = std::fs::remove_file(&finished);
    }

    /// Audio arrives in many small buffers; the header must account for all of
    /// them, not just the last write.
    #[test]
    fn many_small_writes_are_all_accounted_for() {
        let path = temp_path("chunked.wav");
        let mut recorder = LaneRecorder::create(&path, 16_000).unwrap();
        for _ in 0..100 {
            recorder.write(&[0.1; 160]).unwrap();
        }
        assert!((recorder.seconds_written() - 1.0).abs() < 1e-9);
        let finished = recorder.finalize().unwrap();

        let parsed = parse_wav(&std::fs::read(&finished).unwrap()).unwrap();
        assert_eq!(parsed.samples.len(), 16_000);
        let _ = std::fs::remove_file(&finished);
    }

    #[test]
    fn an_empty_recording_is_still_a_valid_wav() {
        let path = temp_path("empty.wav");
        let recorder = LaneRecorder::create(&path, 16_000).unwrap();
        let finished = recorder.finalize().unwrap();
        let parsed = parse_wav(&std::fs::read(&finished).unwrap()).unwrap();
        assert!(parsed.samples.is_empty());
        let _ = std::fs::remove_file(&finished);
    }

    /// A session that crashes leaves the file unpatched. It must still parse —
    /// as empty — rather than claiming a length the file does not have.
    #[test]
    fn an_unfinalized_file_parses_as_empty_rather_than_lying() {
        let path = temp_path("crashed.wav");
        {
            let mut recorder = LaneRecorder::create(&path, 16_000).unwrap();
            recorder.write(&[0.5; 1000]).unwrap();
            // Dropped without finalize, as a crash would.
        }
        let parsed = parse_wav(&std::fs::read(&path).unwrap()).unwrap();
        assert!(parsed.samples.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn samples_are_clamped_to_the_representable_range() {
        let path = temp_path("clamped.wav");
        let mut recorder = LaneRecorder::create(&path, 16_000).unwrap();
        recorder.write(&[9.0, -9.0]).unwrap();
        let finished = recorder.finalize().unwrap();
        let parsed = parse_wav(&std::fs::read(&finished).unwrap()).unwrap();
        assert!(parsed.samples[0] > 0.99 && parsed.samples[1] < -0.99);
        let _ = std::fs::remove_file(&finished);
    }
}
