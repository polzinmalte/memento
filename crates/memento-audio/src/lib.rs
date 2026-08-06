//! Windows audio capture for memento: the mic lane and the system lane.
//!
//! Replaces listnr's `Capture/` directory, which is built on AVAudioEngine and
//! ScreenCaptureKit. The pure parts of that directory (RMS, WAV writing, the
//! lane clock offset) live in `memento-core` instead, so this crate is only the
//! platform layer.

pub mod resample;

#[cfg(windows)]
pub mod mic;
#[cfg(windows)]
pub mod process_loopback;
#[cfg(windows)]
pub mod sessions;

#[cfg(windows)]
pub use mic::MicCapture;
#[cfg(windows)]
pub use process_loopback::{
    CAPTURE_CHANNELS, CAPTURE_SAMPLE_RATE, CaptureChunk, ProcessLoopbackCapture,
};
#[cfg(windows)]
pub use sessions::{AudioSource, list_audio_sources};

pub use resample::Decimator;

/// Interleaved stereo to mono, which is what the whole pipeline downstream of
/// capture expects. Averaging rather than taking one channel: a participant
/// panned to one side would otherwise be transcribed at half level or lost.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_input_is_unchanged() {
        assert_eq!(downmix_to_mono(&[0.1, 0.2], 1), vec![0.1, 0.2]);
    }

    #[test]
    fn stereo_averages_both_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
    }

    #[test]
    fn a_hard_panned_speaker_survives_at_half_level() {
        // Taking channel 0 instead of averaging would drop this entirely.
        assert_eq!(downmix_to_mono(&[0.0, 0.8], 2), vec![0.4]);
    }

    #[test]
    fn trailing_partial_frame_is_dropped() {
        assert_eq!(downmix_to_mono(&[1.0, 1.0, 0.5], 2), vec![1.0]);
    }
}
