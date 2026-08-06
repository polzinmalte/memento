//! 48 kHz → 16 kHz downsampling.
//!
//! listnr uses `AVAudioConverter` and its `SystemAudioCapture` explicitly warns
//! against naive decimation at a 3:1 ratio: dropping two samples in three folds
//! everything above 8 kHz back into the speech band as aliasing, and consonants
//! are exactly what lives up there. Whisper then hears a lisp it cannot undo.
//!
//! So this low-passes before it decimates. A windowed-sinc FIR is enough — the
//! ratio is a fixed integer 3, which is the easy case.

/// Filter half-length in output samples. 32 taps either side of centre at the
/// 16 kHz rate is ~64 dB of stopband rejection, which is far below anything the
/// microphone contributes, and cheap enough to run per buffer.
const HALF_TAPS: usize = 32 * 3;

/// Integer-ratio decimating FIR. Holds the tail of the previous buffer so that
/// audio arriving in chunks is filtered identically to the same audio arriving
/// whole — without that, every buffer boundary is a small discontinuity, and
/// there are twelve of them a second.
pub struct Decimator {
    taps: Vec<f32>,
    history: Vec<f32>,
    factor: usize,
}

impl Decimator {
    /// `factor` is the integer ratio, 3 for 48 kHz → 16 kHz.
    pub fn new(factor: usize) -> Self {
        assert!(factor >= 1, "decimation factor must be at least 1");
        Self {
            taps: sinc_lowpass(HALF_TAPS, factor),
            history: vec![0.0; 2 * HALF_TAPS],
            factor,
        }
    }

    pub fn for_rates(input_rate: u32, output_rate: u32) -> Option<Self> {
        (input_rate.is_multiple_of(output_rate) && output_rate > 0)
            .then(|| Self::new((input_rate / output_rate) as usize))
    }

    /// Filter and decimate one buffer. Output length is roughly
    /// `input.len() / factor`; the exact split across buffers depends on the
    /// carried history, which is the point.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.factor == 1 {
            return input.to_vec();
        }
        self.history.extend_from_slice(input);

        let span = 2 * HALF_TAPS + 1;
        let mut out = Vec::with_capacity(input.len() / self.factor + 1);
        let mut centre = HALF_TAPS;
        while centre + HALF_TAPS < self.history.len() {
            let window = &self.history[centre - HALF_TAPS..centre + HALF_TAPS + 1];
            out.push(window.iter().zip(&self.taps).map(|(s, t)| s * t).sum());
            centre += self.factor;
        }

        // Keep exactly the tail the next call needs to continue seamlessly.
        let consumed = centre - HALF_TAPS;
        self.history.drain(..consumed.min(self.history.len()));
        if self.history.len() < span {
            // Nothing to do; the next buffer will top it up.
        }
        out
    }
}

/// Windowed-sinc low-pass with its cutoff at the output Nyquist, i.e. at
/// `1 / (2 * factor)` of the input rate. Blackman window: its sidelobes are low
/// enough that the stopband stays under the noise floor of any real microphone.
fn sinc_lowpass(half_taps: usize, factor: usize) -> Vec<f32> {
    let n = 2 * half_taps + 1;
    let cutoff = 0.5 / factor as f64;
    let mut taps = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 - half_taps as f64;
        let sinc = if x == 0.0 {
            2.0 * cutoff
        } else {
            (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
        };
        let w = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
        let blackman = 0.42 - 0.5 * w.cos() + 0.08 * (2.0 * w).cos();
        taps.push((sinc * blackman) as f32);
    }
    // Unity DC gain, so the filter cannot change the level the VAD thresholds
    // are calibrated against.
    let sum: f32 = taps.iter().sum();
    taps.iter_mut().for_each(|t| *t /= sum);
    taps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: f32, seconds: f32, rate: f32) -> Vec<f32> {
        let n = (seconds * rate) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / rate).sin() * 0.5)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        memento_core::audio_math::rms(x)
    }

    #[test]
    fn output_rate_is_the_input_rate_over_the_factor() {
        let mut d = Decimator::new(3);
        let out = d.process(&sine(440.0, 1.0, 48_000.0));
        // Allow for the filter's startup transient, which eats a few samples.
        assert!(
            (out.len() as i64 - 16_000).abs() < 100,
            "expected ~16000 samples, got {}",
            out.len()
        );
    }

    /// A tone well inside the 8 kHz passband must come through at its own level.
    #[test]
    fn passband_audio_survives_at_the_same_level() {
        let input = sine(440.0, 0.5, 48_000.0);
        let out = Decimator::new(3).process(&input);
        let (before, after) = (rms(&input), rms(&out[200..]));
        assert!(
            (before - after).abs() / before < 0.05,
            "level moved from {before} to {after}"
        );
    }

    /// The reason this module exists rather than a `step_by(3)`. A 15 kHz tone
    /// is above the 8 kHz output Nyquist and must be rejected, not folded back
    /// down into the speech band as a phantom 1 kHz.
    #[test]
    fn out_of_band_audio_is_rejected_not_aliased() {
        let input = sine(15_000.0, 0.5, 48_000.0);
        let filtered = Decimator::new(3).process(&input);
        let naive: Vec<f32> = input.iter().step_by(3).copied().collect();

        let filtered_rms = rms(&filtered[200..]);
        let naive_rms = rms(&naive[200..]);
        assert!(
            filtered_rms < naive_rms / 10.0,
            "aliasing not suppressed: filtered {filtered_rms} vs naive {naive_rms}"
        );
    }

    /// Audio arrives in buffers, so chunked input must give the same answer as
    /// the whole signal at once — otherwise every buffer boundary is a click.
    #[test]
    fn chunked_input_matches_whole_input() {
        let input = sine(1000.0, 0.5, 48_000.0);
        let whole = Decimator::new(3).process(&input);

        let mut chunked_d = Decimator::new(3);
        let mut chunked = Vec::new();
        for chunk in input.chunks(1024) {
            chunked.extend(chunked_d.process(chunk));
        }

        assert_eq!(whole.len(), chunked.len());
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn silence_stays_silent() {
        let out = Decimator::new(3).process(&vec![0.0; 4800]);
        assert!(out.iter().all(|s| s.abs() < 1e-9));
    }

    #[test]
    fn factor_of_one_is_a_passthrough() {
        let input = sine(440.0, 0.1, 16_000.0);
        assert_eq!(Decimator::new(1).process(&input), input);
    }

    #[test]
    fn only_integer_ratios_are_accepted() {
        assert!(Decimator::for_rates(48_000, 16_000).is_some());
        assert!(Decimator::for_rates(44_100, 16_000).is_none());
    }
}
