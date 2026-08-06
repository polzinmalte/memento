//! Energy-based speech segmenter. Ported from listnr
//! `Sources/ListnrCore/Transcription/EnergyVAD.swift`, constants unchanged.
//!
//! Keeps a short **pre-roll** ring so the first syllable after a pause is not
//! clipped (the VAD only fires once energy crosses the threshold, and by then
//! the onset consonant is often already gone).
//!
//! Owned by exactly one consumer thread in [`crate::lane::LaneTranscriber`], so
//! it is a plain `struct` with no interior synchronisation.

use crate::audio_math;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Segment {
    pub start_seconds: f64,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct EnergyVadConfig {
    pub sample_rate: f64,
    pub speech_threshold: f32,
    pub min_speech_seconds: f64,
    pub max_speech_seconds: f64,
    pub hangover_seconds: f64,
    pub preroll_seconds: f64,
    pub min_voiced_seconds: f64,
    pub min_voiced_run_seconds: f64,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000.0,
            speech_threshold: 0.010,
            min_speech_seconds: 0.35,
            max_speech_seconds: 12.0,
            hangover_seconds: 0.55,
            preroll_seconds: 0.45,
            min_voiced_seconds: 0.15,
            min_voiced_run_seconds: 0.09,
        }
    }
}

pub struct EnergyVad {
    cfg: EnergyVadConfig,
    frame_samples: usize,
    preroll_samples: usize,
    min_voiced_samples: usize,
    min_voiced_run_samples: usize,

    absolute_frame: usize,
    in_speech: bool,
    speech_start_frame: usize,
    hangover_left: i64,
    utterance: Vec<f32>,
    /// Frames actually above threshold, as opposed to pre-roll and hangover
    /// padding. See `min_voiced_seconds`.
    voiced_samples: usize,
    /// Longest *contiguous* voiced stretch, which is what separates a syllable
    /// from a click.
    longest_voiced_run: usize,
    current_voiced_run: usize,
    carry: VecDeque<f32>,
    /// Rolling history used as lookback when speech starts.
    preroll: VecDeque<f32>,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new(EnergyVadConfig::default())
    }
}

impl EnergyVad {
    pub fn new(cfg: EnergyVadConfig) -> Self {
        let frame_samples = ((cfg.sample_rate * 0.03) as usize).max(1);
        Self {
            frame_samples,
            preroll_samples: ((cfg.sample_rate * cfg.preroll_seconds) as usize).max(frame_samples),
            min_voiced_samples: (cfg.sample_rate * cfg.min_voiced_seconds) as usize,
            min_voiced_run_samples: (cfg.sample_rate * cfg.min_voiced_run_seconds) as usize,
            cfg,
            absolute_frame: 0,
            in_speech: false,
            speech_start_frame: 0,
            hangover_left: 0,
            utterance: Vec::new(),
            voiced_samples: 0,
            longest_voiced_run: 0,
            current_voiced_run: 0,
            carry: VecDeque::new(),
            preroll: VecDeque::new(),
        }
    }

    pub fn push(&mut self, chunk: &[f32]) -> Vec<Segment> {
        self.carry.extend(chunk.iter().copied());
        let mut out = Vec::new();

        while self.carry.len() >= self.frame_samples {
            let frame: Vec<f32> = self.carry.drain(..self.frame_samples).collect();
            if let Some(seg) = self.process_frame(&frame) {
                out.push(seg);
            }
            self.absolute_frame += self.frame_samples;
        }
        out
    }

    pub fn flush(&mut self) -> Vec<Segment> {
        let mut out = Vec::new();
        if self.in_speech
            && let Some(seg) = self.finish_utterance()
        {
            out.push(seg);
        }
        self.reset_utterance_state();
        self.carry.clear();
        // flush means end of stream, so the lookback ring goes too.
        self.preroll.clear();
        out
    }

    fn hangover_frames(&self) -> i64 {
        ((self.cfg.hangover_seconds * self.cfg.sample_rate / self.frame_samples as f64) as i64)
            .max(1)
    }

    fn process_frame(&mut self, frame: &[f32]) -> Option<Segment> {
        let speaking = audio_math::rms(frame) >= self.cfg.speech_threshold;

        if !self.in_speech {
            self.append_preroll(frame);
            if speaking {
                self.in_speech = true;
                // The ring already ends with this frame, so it *is* the
                // utterance so far. Appending `frame` again here duplicated the
                // onset, which both stuttered the audio Whisper saw and made the
                // buffer longer than the span its timestamp claimed.
                self.utterance = self.preroll.drain(..).collect();
                // The buffer ends with this frame, so its first sample sits
                // `utterance.len()` back from the end of it.
                self.speech_start_frame =
                    (self.absolute_frame + frame.len()).saturating_sub(self.utterance.len());
                self.hangover_left = self.hangover_frames();
                self.voiced_samples = frame.len();
                self.current_voiced_run = frame.len();
                self.longest_voiced_run = frame.len();
            }
            return None;
        }

        self.utterance.extend_from_slice(frame);
        if speaking {
            self.voiced_samples += frame.len();
            self.current_voiced_run += frame.len();
            self.longest_voiced_run = self.longest_voiced_run.max(self.current_voiced_run);
            self.hangover_left = self.hangover_frames();
        } else {
            self.current_voiced_run = 0;
            self.hangover_left -= 1;
        }

        let duration = self.utterance.len() as f64 / self.cfg.sample_rate;
        if self.hangover_left <= 0 || duration >= self.cfg.max_speech_seconds {
            return self.finish_utterance();
        }
        None
    }

    fn append_preroll(&mut self, frame: &[f32]) {
        self.preroll.extend(frame.iter().copied());
        while self.preroll.len() > self.preroll_samples {
            self.preroll.pop_front();
        }
    }

    fn finish_utterance(&mut self) -> Option<Segment> {
        let result = self.finish_utterance_inner();
        self.reset_utterance_state();
        result
    }

    fn finish_utterance_inner(&mut self) -> Option<Segment> {
        if self.utterance.is_empty() {
            return None;
        }

        // Three floors, each rejecting something the others cannot.
        //
        // `min_speech_seconds` bounds the *buffer*; Whisper needs some context.
        //
        // `min_voiced_seconds` bounds how much of that buffer was above
        // threshold. Without it the pre-roll defeats the buffer floor entirely:
        // one frame over threshold yields preroll + frame + hangover ≈ 1 s, so a
        // keyboard clack always passed as speech and Whisper duly hallucinated.
        //
        // `min_voiced_run_seconds` bounds the longest *contiguous* voiced
        // stretch, because total duration alone cannot tell one syllable from a
        // burst of typing: six 30 ms key clicks inside the hangover window sum to
        // the same 180 ms as a short word. Speech has a continuous syllable;
        // clicks do not.
        let duration = self.utterance.len() as f64 / self.cfg.sample_rate;
        if duration < self.cfg.min_speech_seconds
            || self.voiced_samples < self.min_voiced_samples
            || self.longest_voiced_run < self.min_voiced_run_samples
        {
            return None;
        }

        Some(Segment {
            start_seconds: self.speech_start_frame as f64 / self.cfg.sample_rate,
            samples: std::mem::take(&mut self.utterance),
        })
    }

    fn reset_utterance_state(&mut self) {
        self.in_speech = false;
        self.hangover_left = 0;
        self.voiced_samples = 0;
        self.longest_voiced_run = 0;
        self.current_voiced_run = 0;
        self.utterance.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{SAMPLE_RATE, silence, tone};

    /// Push audio through a fresh default VAD in capture-sized chunks.
    fn segments_with(audio: &[f32], flush: bool, mut vad: EnergyVad) -> Vec<Segment> {
        let mut out = Vec::new();
        for chunk in audio.chunks(1024) {
            out.extend(vad.push(chunk));
        }
        if flush {
            out.extend(vad.flush());
        }
        out
    }

    fn segments(audio: &[f32]) -> Vec<Segment> {
        segments_with(audio, true, EnergyVad::default())
    }

    #[test]
    fn silence_produces_nothing() {
        assert!(segments(&silence(3.0)).is_empty());
    }

    #[test]
    fn quiet_audio_below_threshold_ignored() {
        let audio = [silence(1.0), tone(1.0, 0.005), silence(1.0)].concat();
        assert!(segments(&audio).is_empty());
    }

    #[test]
    fn single_word_detected() {
        let audio = [silence(1.0), tone(0.5, 0.1), silence(1.5)].concat();
        assert_eq!(segments(&audio).len(), 1);
    }

    #[test]
    fn tiny_blip_rejected() {
        // One 30 ms frame over threshold. Pre-roll + hangover pad the buffer to
        // ~1 s, which used to pass the duration floor on its own.
        let audio = [silence(1.0), tone(0.03, 0.1), silence(1.5)].concat();
        assert!(segments(&audio).is_empty());
    }

    #[test]
    fn click_train_rejected() {
        // Six 30 ms clicks inside one hangover window: 180 ms of total voiced
        // time, which passes the voiced-duration floor, but no contiguous run
        // longer than a click. Speech has a continuous syllable; typing does not.
        let mut audio = silence(1.0);
        for _ in 0..6 {
            audio.extend(tone(0.03, 0.1));
            audio.extend(silence(0.06));
        }
        audio.extend(silence(1.5));
        assert!(segments(&audio).is_empty());
    }

    #[test]
    fn short_voiced_below_floor_rejected() {
        // 100 ms voiced: run floor (90 ms) passes, voiced floor (150 ms) fails.
        let audio = [silence(1.0), tone(0.10, 0.1), silence(1.5)].concat();
        assert!(segments(&audio).is_empty());
    }

    #[test]
    fn short_word_accepted() {
        // 250 ms clears both voiced floors.
        let audio = [silence(1.0), tone(0.25, 0.1), silence(1.5)].concat();
        assert_eq!(segments(&audio).len(), 1);
    }

    #[test]
    fn two_utterances_separated_by_pause() {
        let audio = [
            silence(1.0),
            tone(0.5, 0.1),
            silence(2.0),
            tone(0.5, 0.1),
            silence(1.5),
        ]
        .concat();
        let segs = segments(&audio);
        assert_eq!(segs.len(), 2);
        assert!(segs[0].start_seconds < segs[1].start_seconds);
    }

    #[test]
    fn preroll_included_in_segment() {
        let audio = [silence(1.0), tone(0.5, 0.1), silence(1.5)].concat();
        let segs = segments(&audio);
        let seg = segs.first().expect("expected a segment");
        // The segment carries lookback audio from before the trigger frame, so
        // it must be longer than the voiced part alone.
        assert!(seg.samples.len() as f64 / SAMPLE_RATE > 0.5 + 0.3);
        // And its start is credited before the audible onset at t=1.0.
        assert!(seg.start_seconds < 1.0);
        assert!(seg.start_seconds > 0.4);
    }

    /// The invariant that ties audio and timestamps together: a segment must be
    /// exactly the source audio at the position it claims to start at. This
    /// catches both duplicated samples and an off-by-a-frame timestamp.
    #[test]
    fn segment_is_a_contiguous_slice_of_the_source_at_its_timestamp() {
        // A signal where every sample is distinguishable, so a duplicated or
        // shifted region cannot accidentally compare equal.
        let total = (4.0 * SAMPLE_RATE) as usize;
        let mut source = vec![0.0f32; total];
        let lo = (1.0 * SAMPLE_RATE) as usize;
        let hi = (1.8 * SAMPLE_RATE) as usize;
        for (offset, sample) in source[lo..hi].iter_mut().enumerate() {
            let envelope = 0.08 + 0.04 * offset as f32 / (hi - lo) as f32;
            *sample = if (lo + offset).is_multiple_of(2) {
                envelope
            } else {
                -envelope
            };
        }

        let segs = segments(&source);
        assert_eq!(segs.len(), 1);
        let seg = &segs[0];

        let start = (seg.start_seconds * SAMPLE_RATE).round() as usize;
        assert!(start + seg.samples.len() <= source.len());
        assert_eq!(
            seg.samples,
            source[start..start + seg.samples.len()],
            "segment audio must match the source at its own timestamp"
        );
    }

    #[test]
    fn consecutive_segments_do_not_overlap() {
        let audio = [
            silence(1.0),
            tone(0.5, 0.1),
            silence(2.0),
            tone(0.5, 0.1),
            silence(1.5),
        ]
        .concat();
        let segs = segments(&audio);
        assert_eq!(segs.len(), 2);
        let first_end = segs[0].start_seconds + segs[0].samples.len() as f64 / SAMPLE_RATE;
        assert!(
            first_end <= segs[1].start_seconds + 1e-9,
            "segments must not overlap in time"
        );
    }

    #[test]
    fn long_utterance_split_at_max_length() {
        let audio = [silence(1.0), tone(30.0, 0.1), silence(1.5)].concat();
        let segs = segments(&audio);
        assert!(segs.len() >= 2);
        for seg in &segs {
            assert!(seg.samples.len() as f64 / SAMPLE_RATE <= 12.1);
        }
    }

    #[test]
    fn flush_emits_trailing_utterance() {
        // Speech running right up to the end of the stream, with no trailing
        // silence to trigger the hangover: only flush() can emit it.
        let audio = [silence(1.0), tone(0.5, 0.1)].concat();
        assert!(segments_with(&audio, false, EnergyVad::default()).is_empty());
        assert_eq!(segments_with(&audio, true, EnergyVad::default()).len(), 1);
    }
}
