//! Run a WAV through the real lane pipeline: VAD → whisper.cpp → transcript.
//!
//! This is how the pipeline is verified without a meeting, and — while
//! development happens on a Mac — without Windows capture. Everything from the
//! VAD onwards is the exact code a live session runs; only the audio source
//! differs.
//!
//! Usage:  transcribe-file <audio.wav> [--model <id>] [--lang <code>] [--threads <n>]
//!
//! The WAV must be 16 kHz mono 16-bit. On macOS:
//!   say -v Anna -o out.aiff "…" && afconvert -f WAVE -d LEI16@16000 -c 1 out.aiff out.wav

use memento_asr::{download, models, whisper::WhisperTranscriber};
use memento_core::{LanguageMode, Transcriber, audio_math, lane::LaneConfig, sensitivity};
use std::time::Instant;

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!(
            "usage: transcribe-file <audio.wav> [--model <id>] [--lang <code>] [--threads <n>]"
        );
        eprintln!("\nmodels:");
        for m in models::REGISTRY {
            eprintln!("  {:<16} {:>5} MB  {}", m.id, m.size_mb, m.note);
        }
        std::process::exit(2);
    };

    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let language = match flag("--lang") {
        Some(code) => LanguageMode::parse(&code)
            .ok_or_else(|| -> Error { format!("unknown language '{code}'").into() })?,
        None => LanguageMode::De,
    };
    let model_id =
        flag("--model").unwrap_or_else(|| models::preferred_model_id(language, false).to_string());
    let threads: Option<i32> = flag("--threads").and_then(|t| t.parse().ok());

    if let Some(warning) = models::compatibility_warning(&model_id, language, false) {
        eprintln!("! {warning}\n");
    }

    // --- audio ---------------------------------------------------------
    let wav = audio_math::parse_wav(&std::fs::read(path)?)?;
    if wav.sample_rate != 16_000 || wav.channels != 1 {
        return Err(format!(
            "expected 16 kHz mono, got {} Hz {}ch — convert with: \
             afconvert -f WAVE -d LEI16@16000 -c 1 in.wav out.wav",
            wav.sample_rate, wav.channels
        )
        .into());
    }
    let audio_seconds = wav.samples.len() as f64 / 16_000.0;
    println!(
        "{path}: {audio_seconds:.1}s, rms {:.4}, peak {:.4}",
        audio_math::rms(&wav.samples),
        audio_math::peak(&wav.samples)
    );

    // --- model ---------------------------------------------------------
    let dir = download::default_model_dir();
    println!("model {model_id} (lang {})", language.as_str());
    let mut last_percent = u64::MAX;
    let model_path = download::ensure_model_id(&model_id, &dir, &mut |done, total| {
        if let Some(total) = total.filter(|t| *t > 0) {
            let percent = done * 100 / total;
            if percent != last_percent && percent.is_multiple_of(5) {
                last_percent = percent;
                eprint!("\r  downloading… {percent}%");
            }
        }
    })?;
    if last_percent != u64::MAX {
        eprintln!("\r  downloaded.        ");
    }

    let mut transcriber = WhisperTranscriber::new(&model_id, &model_path, language, false);
    if let Some(n) = threads {
        transcriber = transcriber.with_threads(n);
    }
    let load_start = Instant::now();
    transcriber.warm_up()?;
    println!("loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    // --- the real pipeline ---------------------------------------------
    // Same lane the mic uses in a live session, same thresholds.
    let tuning = sensitivity::lane_tuning::you(memento_core::MicSensitivity::Normal);
    let mut lane = memento_core::LaneTranscriber::start(
        LaneConfig::new("You", tuning.speech_threshold, tuning.min_segment_rms),
        Box::new(transcriber),
    );

    let start = Instant::now();
    // Capture-sized chunks so the VAD sees what it would live, but pushed with
    // backpressure: a file replays far faster than realtime, so the non-blocking
    // path would overflow the lane queue and silently drop speech — which then
    // reads as the model mistranscribing rather than as audio it never saw.
    for chunk in wav.samples.chunks(1024) {
        lane.push_blocking(chunk.to_vec());
    }
    let lines = lane.finish();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n-- transcript --");
    for line in &lines {
        println!(
            "[{}] {}: {}",
            memento_core::format_clock(line.start_seconds),
            line.speaker,
            line.text
        );
    }
    println!("--------");

    // The number that decides whether a model is usable live. Above 1.0 the lane
    // falls behind and its bounded queue starts dropping speech.
    let realtime_factor = elapsed / audio_seconds.max(f64::EPSILON);
    println!(
        "\n{} segment(s) in {elapsed:.1}s for {audio_seconds:.1}s of audio \
         — {realtime_factor:.2}x realtime ({})",
        lines.len(),
        if realtime_factor <= 1.0 {
            "keeps up"
        } else {
            "TOO SLOW to stay live"
        }
    );
    if let Some(report) = lane.drop_report() {
        println!("! {report}");
    }
    Ok(())
}
