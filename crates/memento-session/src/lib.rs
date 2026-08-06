//! One meeting, start to finish.
//!
//! Replaces listnr's `LiveSessionRunner.swift`. The shape is the same: two
//! lanes that are never mixed, live lines from both while the meeting runs,
//! then diarization over the remote lane at the end to split "Others" into
//! Speaker 1..N, then a merged Markdown transcript.
//!
//! Capture sits behind [`CaptureSource`] so the session can be driven from a
//! WAV file as well as from Windows audio devices. That is not only a test
//! affordance: it is the only way the whole session path can be exercised on a
//! machine that is not Windows.

pub mod clock;
pub mod source;

use memento_asr::{WhisperTranscriber, download, models as asr_models};
use memento_core::{
    Diarizer, LaneConfig, LaneRecorder, LaneTranscriber, LanguageMode, MicSensitivity, SpanOptions,
    Transcriber, TranscriptLine, audio_math, lane_tuning, speakers, transcript,
};
use memento_diar::{SherpaDiarizer, models as diar_models};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub use source::{CaptureSource, FileSource, Lane};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Lane A, the microphone. Named as listnr names it, because the label reaches
/// the transcript.
pub const YOU: &str = "You";
/// Lane B before diarization splits it.
pub const OTHERS: &str = "Others";

pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub language: LanguageMode,
    /// `None` takes the language's default, which for German is the turbo model.
    pub model_id: Option<String>,
    pub sensitivity: MicSensitivity,
    /// Expected number of remote speakers, when known.
    pub speakers: Option<i32>,
    pub diarize: bool,
    /// Where the Markdown transcript lands.
    pub output_dir: PathBuf,
    /// Where lane audio is spilled while the session runs.
    pub work_dir: PathBuf,
    /// Keep the lane WAVs after the transcript is written.
    pub keep_audio: bool,
}

impl SessionConfig {
    pub fn resolved_model_id(&self) -> String {
        self.model_id
            .clone()
            .unwrap_or_else(|| asr_models::preferred_model_id(self.language, false).to_string())
    }
}

/// Progress the caller can show. Deliberately not tied to any UI toolkit.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Model download or load, before capture starts.
    Preparing(String),
    DownloadProgress {
        done: u64,
        total: Option<u64>,
    },
    /// Capture is live.
    Started,
    /// A finished transcript line, for the running view.
    Line(TranscriptLine),
    /// Signal level per lane, for the meters.
    Level {
        lane: Lane,
        rms: f32,
    },
    /// Something the user should know but which did not stop the session.
    Warning(String),
    /// Post-session work.
    Finalizing(String),
}

pub type EventSink = Arc<dyn Fn(SessionEvent) + Send + Sync>;

pub struct SessionOutcome {
    pub lines: Vec<TranscriptLine>,
    pub transcript_path: PathBuf,
    pub markdown: String,
    pub audio_paths: Vec<PathBuf>,
}

/// Cancel handle. Cloneable so a UI thread can stop a session running elsewhere.
#[derive(Clone, Default)]
pub struct StopHandle {
    stop: Arc<AtomicBool>,
}

impl StopHandle {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
    pub fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

/// Run one meeting to completion. Blocks until the sources stop or `stop` fires.
///
/// `recorded_label` and `stamp` are pre-formatted date strings; this crate does
/// not pick a clock or a timezone for the caller.
pub fn run_session(
    config: &SessionConfig,
    mut mic: Box<dyn CaptureSource>,
    mut system: Box<dyn CaptureSource>,
    stop: StopHandle,
    events: EventSink,
    recorded_label: &str,
    stamp: &str,
) -> Result<SessionOutcome, Error> {
    // --- models, before capture -----------------------------------------
    // Loading lazily would make the opening minutes of a meeting vanish behind
    // a 574 MB download.
    let model_id = config.resolved_model_id();
    if let Some(warning) = asr_models::compatibility_warning(&model_id, config.language, false) {
        events(SessionEvent::Warning(warning));
    }
    events(SessionEvent::Preparing(format!("model {model_id}")));

    let model_dir = download::default_model_dir();
    let events_for_progress = Arc::clone(&events);
    let asr_path = download::ensure_model_id(&model_id, &model_dir, &mut |done, total| {
        events_for_progress(SessionEvent::DownloadProgress { done, total });
    })?;

    // One transcriber per lane, both behind a single work queue: two whisper
    // contexts decoding at once would ask for twice the cores that exist.
    let mut lane_a_asr = WhisperTranscriber::new(&model_id, &asr_path, config.language, false);
    let mut lane_b_asr = WhisperTranscriber::new(&model_id, &asr_path, config.language, false);
    events(SessionEvent::Preparing("loading model".into()));
    lane_a_asr.warm_up()?;
    lane_b_asr.warm_up()?;

    let mut pool = memento_asr::shared::AsrPool::new(vec![
        Box::new(lane_a_asr) as Box<dyn Transcriber>,
        Box::new(lane_b_asr),
    ]);

    // --- lanes ------------------------------------------------------------
    let you_tuning = lane_tuning::you(config.sensitivity);
    let others_tuning = lane_tuning::others(config.sensitivity);

    let make_sink = |events: &EventSink| {
        let events = Arc::clone(events);
        Arc::new(move |line: &TranscriptLine| events(SessionEvent::Line(line.clone())))
            as memento_core::lane::LineSink
    };
    let make_error_sink = |events: &EventSink| {
        let events = Arc::clone(events);
        Arc::new(move |lane: &str, e: &Error| events(SessionEvent::Warning(format!("{lane}: {e}"))))
            as memento_core::lane::ErrorSink
    };

    let mut lane_a = LaneTranscriber::start(
        LaneConfig {
            on_line: Some(make_sink(&events)),
            on_error: Some(make_error_sink(&events)),
            ..LaneConfig::new(YOU, you_tuning.speech_threshold, you_tuning.min_segment_rms)
        },
        Box::new(pool.handle(0)),
    );
    let mut lane_b = LaneTranscriber::start(
        LaneConfig {
            on_line: Some(make_sink(&events)),
            on_error: Some(make_error_sink(&events)),
            ..LaneConfig::new(
                OTHERS,
                others_tuning.speech_threshold,
                others_tuning.min_segment_rms,
            )
        },
        Box::new(pool.handle(1)),
    );

    // Lane B is spilled to disk because diarization needs the whole lane at the
    // end; holding an hour of it costs ~230 MB of RAM for no reason.
    std::fs::create_dir_all(&config.work_dir)?;
    let mut recorder = LaneRecorder::create(config.work_dir.join("others.wav"), SAMPLE_RATE)?;

    // --- capture ----------------------------------------------------------
    events(SessionEvent::Started);
    let mic_anchor = mic.start()?;
    let system_anchor = system.start()?;

    let mut mic_done = false;
    let mut system_done = false;
    while !stop.is_stopped() && !(mic_done && system_done) {
        match mic.next_chunk()? {
            Some(chunk) => {
                events(SessionEvent::Level {
                    lane: Lane::You,
                    rms: audio_math::rms(&chunk),
                });
                lane_a.push(chunk);
            }
            None => mic_done = true,
        }
        match system.next_chunk()? {
            Some(chunk) => {
                events(SessionEvent::Level {
                    lane: Lane::Others,
                    rms: audio_math::rms(&chunk),
                });
                recorder.write(&chunk)?;
                lane_b.push(chunk);
            }
            None => system_done = true,
        }
    }

    mic.stop();
    system.stop();

    // --- drain ------------------------------------------------------------
    events(SessionEvent::Finalizing("transcribing the tail".into()));
    let you_lines = lane_a.finish();
    let live_others_lines = lane_b.finish();
    for report in [lane_a.drop_report(), lane_b.drop_report()]
        .into_iter()
        .flatten()
    {
        events(SessionEvent::Warning(report));
    }

    let others_wav = recorder.finalize()?;

    // --- diarize the remote lane ------------------------------------------
    let others_lines = if config.diarize {
        match diarize_lane(config, &others_wav, &model_id, &asr_path, &events) {
            Ok(lines) if !lines.is_empty() => lines,
            Ok(_) => {
                // listnr's fallback chain: diarized spans, else the live lines.
                events(SessionEvent::Warning(
                    "diarization found no speakers; keeping the undivided Others lane".into(),
                ));
                live_others_lines
            }
            Err(e) => {
                events(SessionEvent::Warning(format!(
                    "diarization failed ({e}); keeping the undivided Others lane"
                )));
                live_others_lines
            }
        }
    } else {
        live_others_lines
    };

    pool.shutdown();

    // --- merge and write ---------------------------------------------------
    let offsets = transcript::LaneAnchors {
        mic_first_sample_time: mic_anchor,
        system_first_sample_time: system_anchor,
    }
    .offsets();
    if offsets.skew_rejected {
        events(SessionEvent::Warning(
            "the two lanes' clocks disagree by more than 5 s; merging without alignment".into(),
        ));
    }
    let lines = memento_core::merge(vec![
        memento_core::shift(you_lines, offsets.mic),
        memento_core::shift(others_lines, offsets.system),
    ]);

    let markdown = memento_core::render_markdown("memento meeting", recorded_label, &lines);
    std::fs::create_dir_all(&config.output_dir)?;
    let transcript_path = config
        .output_dir
        .join(memento_core::transcript_filename(stamp));
    std::fs::write(&transcript_path, &markdown)?;

    let mut audio_paths = Vec::new();
    if config.keep_audio {
        audio_paths.push(others_wav);
    } else {
        let _ = std::fs::remove_file(&others_wav);
    }

    Ok(SessionOutcome {
        lines,
        transcript_path,
        markdown,
        audio_paths,
    })
}

fn diarize_lane(
    config: &SessionConfig,
    wav_path: &Path,
    model_id: &str,
    asr_path: &Path,
    events: &EventSink,
) -> Result<Vec<TranscriptLine>, Error> {
    events(SessionEvent::Finalizing(
        "separating remote speakers".into(),
    ));

    let dir = download::default_model_dir();
    let events_for_progress = Arc::clone(events);
    let (segmentation, embedding) = diar_models::ensure_all(&dir, &mut |done, total| {
        events_for_progress(SessionEvent::DownloadProgress { done, total });
    })?;

    let audio = audio_math::parse_wav(&std::fs::read(wav_path)?)?;
    let mut diarizer =
        SherpaDiarizer::new(&segmentation, &embedding).with_speaker_hint(config.speakers);
    let spans = diarizer.diarize(&audio.samples)?;
    if spans.is_empty() {
        return Ok(Vec::new());
    }

    events(SessionEvent::Finalizing(format!(
        "transcribing {} speaker turn(s)",
        spans.len()
    )));
    let mut transcriber = WhisperTranscriber::new(model_id, asr_path, config.language, false);
    transcriber.warm_up()?;
    Ok(speakers::transcribe_spans(
        &audio.samples,
        SAMPLE_RATE as f64,
        &spans,
        &mut transcriber,
        SpanOptions::default(),
    ))
}
