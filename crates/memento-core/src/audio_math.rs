//! Ported from listnr `Sources/ListnrCore/Capture/AudioMath.swift`.
//!
//! The `HostClock` half of the original is Mach-specific and does not come
//! along; memento derives its session clock from `std::time::Instant` plus a
//! per-lane sample count instead (see `memento-audio`).

/// Root mean square of a buffer. Accumulates in `f64` because a minute of
/// `f32` addition loses enough precision to move a threshold decision.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

/// Peak absolute sample value. Cheap clipping check.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

/// 16-bit mono WAV bytes. Used for the on-disk lane spill and for the capture
/// spike; diarization reads these back at the end of a session.
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let data_size = samples.len() as u32 * bytes_per_sample;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes());
    out.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        out.extend_from_slice(&((clamped * 32767.0) as i16).to_le_bytes());
    }
    out
}

/// A parsed 16-bit PCM WAV.
#[derive(Debug, Clone, PartialEq)]
pub struct Wav {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Parse a 16-bit PCM WAV. Deliberately minimal — it reads what
/// [`wav_bytes`] writes, plus the common variations real files have: extra
/// chunks before `data`, and an `fmt ` chunk longer than 16 bytes.
///
/// Diarization reads each lane's spill file back from disk at the end of a
/// session, so this is on the main path, not just a test helper.
pub fn parse_wav(bytes: &[u8]) -> Result<Wav, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut cursor = 12;
    let mut format: Option<(u16, u16, u32)> = None; // (audio_format, channels, rate)

    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let body = cursor + 8;
        if body + size > bytes.len() {
            return Err(format!(
                "chunk {} runs past the end of the file",
                show_id(id)
            ));
        }

        match id {
            b"fmt " => {
                if size < 16 {
                    return Err("fmt chunk is too short".into());
                }
                format = Some((
                    u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap()),
                    u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap()),
                    u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()),
                ));
            }
            b"data" => {
                let Some((audio_format, channels, sample_rate)) = format else {
                    return Err("data chunk appeared before fmt".into());
                };
                if audio_format != 1 {
                    return Err(format!(
                        "unsupported WAV format {audio_format}; expected 16-bit PCM"
                    ));
                }
                let samples = bytes[body..body + size]
                    .chunks_exact(2)
                    .map(|p| i16::from_le_bytes([p[0], p[1]]) as f32 / 32768.0)
                    .collect();
                return Ok(Wav {
                    samples,
                    sample_rate,
                    channels,
                });
            }
            _ => {}
        }
        // Chunks are word-aligned, so an odd size is followed by a pad byte.
        cursor = body + size + (size % 2);
    }

    Err("no data chunk".into())
}

fn show_id(id: &[u8]) -> String {
    String::from_utf8_lossy(id).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn rms_of_alternating_signal_is_its_amplitude() {
        // The property every VAD test leans on: alternating ±a has RMS a.
        let signal: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        assert!((rms(&signal) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn peak_ignores_sign() {
        assert_eq!(peak(&[0.1, -0.9, 0.3]), 0.9);
    }

    /// Ported from `WAVWriterTests.swift`: header shape and sample encoding.
    #[test]
    fn wav_header_is_well_formed() {
        let bytes = wav_bytes(&[0.0, 1.0, -1.0], 16_000);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(bytes.len(), 44 + 3 * 2);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            16_000
        );
        assert_eq!(i16::from_le_bytes(bytes[44..46].try_into().unwrap()), 0);
        assert_eq!(i16::from_le_bytes(bytes[46..48].try_into().unwrap()), 32767);
        assert_eq!(
            i16::from_le_bytes(bytes[48..50].try_into().unwrap()),
            -32767
        );
    }

    #[test]
    fn wav_clamps_out_of_range_samples() {
        let bytes = wav_bytes(&[9.0, -9.0], 16_000);
        assert_eq!(i16::from_le_bytes(bytes[44..46].try_into().unwrap()), 32767);
        assert_eq!(
            i16::from_le_bytes(bytes[46..48].try_into().unwrap()),
            -32767
        );
    }

    #[test]
    fn wav_round_trips_through_the_parser() {
        let original = vec![0.0, 0.5, -0.5, 0.25];
        let parsed = parse_wav(&wav_bytes(&original, 16_000)).unwrap();
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        for (got, want) in parsed.samples.iter().zip(&original) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
    }

    #[test]
    fn parser_skips_unknown_chunks_before_data() {
        // Real files (afconvert, ffmpeg) put LIST/fact chunks ahead of data.
        let mut bytes = wav_bytes(&[0.5], 16_000);
        let mut extra = b"LIST".to_vec();
        extra.extend_from_slice(&4u32.to_le_bytes());
        extra.extend_from_slice(b"INFO");
        bytes.splice(36..36, extra);
        let parsed = parse_wav(&bytes).unwrap();
        assert_eq!(parsed.samples.len(), 1);
    }

    #[test]
    fn parser_rejects_non_wav_input() {
        assert!(parse_wav(b"not a wav at all").is_err());
        assert!(parse_wav(&[]).is_err());
    }

    #[test]
    fn parser_rejects_a_truncated_data_chunk() {
        let mut bytes = wav_bytes(&[0.5, 0.5, 0.5], 16_000);
        bytes.truncate(bytes.len() - 3);
        assert!(parse_wav(&bytes).is_err());
    }
}
