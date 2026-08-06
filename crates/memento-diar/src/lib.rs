//! Speaker diarization for the system-audio lane, via sherpa-onnx.
//!
//! Replaces listnr's SpeakerKit wrapper (`RemoteDiarizer.swift`). SpeakerKit is
//! a Pyannote wrapper, and this uses the same Pyannote segmentation model
//! exported to ONNX, so the two should behave comparably.
//!
//! Runs once at the end of a session over the whole remote lane, which is why
//! listnr can only name remote speakers after the meeting — and why memento
//! inherits that. Live diarization would need a different model entirely.

pub mod models;

use memento_core::{DiarizedSpan, Diarizer, Error, speakers::speaker_label};
use sherpa_rs::diarize::{Diarize, DiarizeConfig};
use std::path::PathBuf;

/// Spans shorter than this are dropped. Below roughly a third of a second there
/// is not enough voice for the embedding to be meaningful, and the cluster it
/// lands in is close to arbitrary.
const MIN_SPAN_SECONDS: f64 = 0.35;

/// Diarization needs at least this much audio to say anything useful.
const MIN_AUDIO_SECONDS: f64 = 0.5;

pub struct SherpaDiarizer {
    segmentation_path: PathBuf,
    embedding_path: PathBuf,
    /// How many remote speakers to expect. `None` lets the clustering decide by
    /// distance threshold instead, which is the honest default when the number
    /// of people on a call is not known ahead of time.
    speaker_hint: Option<i32>,
    sample_rate: f64,
    diarize: Option<Diarize>,
}

impl SherpaDiarizer {
    pub fn new(segmentation_path: impl Into<PathBuf>, embedding_path: impl Into<PathBuf>) -> Self {
        Self {
            segmentation_path: segmentation_path.into(),
            embedding_path: embedding_path.into(),
            speaker_hint: None,
            sample_rate: 16_000.0,
            diarize: None,
        }
    }

    /// Fix the number of speakers when it is known. Worth setting: clustering by
    /// threshold alone tends to split one person across two labels when their
    /// level changes partway through a call.
    pub fn with_speaker_hint(mut self, speakers: Option<i32>) -> Self {
        self.speaker_hint = speakers.filter(|n| *n > 0);
        self
    }

    fn config(&self) -> DiarizeConfig {
        DiarizeConfig {
            // sherpa-onnx picks between the two: `fast-clustering.cc` uses
            // `num_clusters` only when it is **> 0**, and falls back to the
            // distance threshold otherwise.
            //
            // The -1 is load-bearing. Passing `None` here does not mean "no
            // hint" — sherpa-rs defaults it to `4`, so an unhinted two-person
            // call would be split into four speakers.
            num_clusters: Some(self.speaker_hint.unwrap_or(-1)),
            threshold: Some(0.5),
            min_duration_on: Some(MIN_SPAN_SECONDS as f32),
            // Silence shorter than this does not split a span. Keep it small:
            // people take the floor from each other in well under half a second,
            // and anything larger merges a real turn change into one span, which
            // then gets attributed wholesale to whoever spoke longer.
            min_duration_off: Some(0.1),
            provider: None,
            debug: false,
        }
    }
}

impl Diarizer for SherpaDiarizer {
    fn warm_up(&mut self) -> Result<(), Error> {
        if self.diarize.is_some() {
            return Ok(());
        }
        for path in [&self.segmentation_path, &self.embedding_path] {
            if !path.exists() {
                return Err(format!("diarization model not found: {}", path.display()).into());
            }
        }
        let diarize = Diarize::new(&self.segmentation_path, &self.embedding_path, self.config())
            .map_err(|e| -> Error { format!("could not load diarization models: {e}").into() })?;
        self.diarize = Some(diarize);
        Ok(())
    }

    fn diarize(&mut self, audio: &[f32]) -> Result<Vec<DiarizedSpan>, Error> {
        if (audio.len() as f64) < MIN_AUDIO_SECONDS * self.sample_rate {
            return Ok(Vec::new());
        }
        self.warm_up()?;
        let engine = self
            .diarize
            .as_mut()
            .ok_or_else(|| -> Error { "diarizer not loaded".into() })?;

        let segments = engine
            .compute(audio.to_vec(), None)
            .map_err(|e| -> Error { format!("diarization failed: {e}").into() })?;

        let mut spans: Vec<DiarizedSpan> = segments
            .into_iter()
            .filter(|s| (s.end - s.start) as f64 >= MIN_SPAN_SECONDS)
            .map(|s| DiarizedSpan {
                label: speaker_label(s.speaker),
                start_seconds: s.start as f64,
                end_seconds: s.end as f64,
            })
            .collect();
        spans.sort_by(|a, b| a.start_seconds.total_cmp(&b.start_seconds));
        Ok(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memento_core::testing::silence;

    #[test]
    fn too_little_audio_yields_no_spans_without_loading_models() {
        // Short input must short-circuit before touching the model files, so a
        // session with almost no remote audio does not fail on a missing model.
        let mut d = SherpaDiarizer::new("/nonexistent-seg.onnx", "/nonexistent-emb.onnx");
        assert_eq!(d.diarize(&silence(0.2)).unwrap().len(), 0);
    }

    #[test]
    fn missing_models_are_reported_clearly() {
        let mut d = SherpaDiarizer::new("/nonexistent-seg.onnx", "/nonexistent-emb.onnx");
        let err = d.warm_up().unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn a_speaker_hint_pins_the_cluster_count() {
        let hinted = SherpaDiarizer::new("a", "b").with_speaker_hint(Some(2));
        assert_eq!(hinted.config().num_clusters, Some(2));
    }

    /// Without a hint the cluster count must be non-positive, so sherpa falls
    /// back to the distance threshold. `None` would *not* do that: sherpa-rs
    /// turns it into 4, which would split a two-person call four ways.
    #[test]
    fn no_hint_means_threshold_clustering_not_four_speakers() {
        let unhinted = SherpaDiarizer::new("a", "b");
        let clusters = unhinted.config().num_clusters.expect("must be explicit");
        assert!(
            clusters <= 0,
            "expected a non-positive count to select threshold clustering, got {clusters}"
        );
        assert!(unhinted.config().threshold.is_some());
    }

    #[test]
    fn a_nonsense_speaker_hint_falls_back_to_the_threshold() {
        let d = SherpaDiarizer::new("a", "b").with_speaker_hint(Some(0));
        assert!(d.config().num_clusters.unwrap() <= 0);
    }
}
