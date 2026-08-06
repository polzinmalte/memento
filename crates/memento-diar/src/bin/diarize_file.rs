//! Diarize a WAV and transcribe each speaker's spans — the "Others" lane's
//! end-of-session path, runnable without a meeting.
//!
//! Usage:  diarize-file <audio.wav> [--speakers <n>] [--model <id>] [--lang <code>]
//!
//! Two-speaker German test audio on macOS:
//!   say -v Anna  -o a.aiff "Guten Morgen, wie ist der Stand?"
//!   say -v Markus -o b.aiff "Die Tests laufen noch, wir brauchen bis Freitag."
//!   … concatenate, then afconvert -f WAVE -d LEI16@16000 -c 1 in.aiff out.wav

use memento_asr::{download, models as asr_models, whisper::WhisperTranscriber};
use memento_core::{
    Diarizer, LanguageMode, SpanOptions, Transcriber, audio_math, format_clock, speakers,
};
use memento_diar::{SherpaDiarizer, models as diar_models};
use std::time::Instant;

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!(
            "usage: diarize-file <audio.wav> [--speakers <n>] [--model <id>] [--lang <code>]"
        );
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
        flag("--model").unwrap_or_else(|| asr_models::preferred_model_id(language, false).into());
    let speakers_hint: Option<i32> = flag("--speakers").and_then(|s| s.parse().ok());

    let wav = audio_math::parse_wav(&std::fs::read(path)?)?;
    if wav.sample_rate != 16_000 || wav.channels != 1 {
        return Err(format!(
            "expected 16 kHz mono, got {} Hz {}ch",
            wav.sample_rate, wav.channels
        )
        .into());
    }
    let audio_seconds = wav.samples.len() as f64 / 16_000.0;
    println!("{path}: {audio_seconds:.1}s");

    // --- models --------------------------------------------------------
    let dir = download::default_model_dir();
    let mut announced = false;
    let mut note = |done: u64, total: Option<u64>| {
        if !announced {
            announced = true;
            eprintln!("  downloading models…");
        }
        let _ = (done, total);
    };
    let (segmentation, embedding) = diar_models::ensure_all(&dir, &mut note)?;
    let asr_path = download::ensure_model_id(&model_id, &dir, &mut note)?;

    // --- diarize -------------------------------------------------------
    let mut diarizer =
        SherpaDiarizer::new(&segmentation, &embedding).with_speaker_hint(speakers_hint);
    let start = Instant::now();
    let spans = diarizer.diarize(&wav.samples)?;
    println!(
        "diarized in {:.1}s: {} span(s), {} speaker(s)",
        start.elapsed().as_secs_f64(),
        spans.len(),
        {
            let mut labels: Vec<&str> = spans.iter().map(|s| s.label.as_str()).collect();
            labels.sort_unstable();
            labels.dedup();
            labels.len()
        }
    );
    for span in &spans {
        println!(
            "  {:<10} {:>6.2}s → {:>6.2}s",
            span.label, span.start_seconds, span.end_seconds
        );
    }

    // --- transcribe each span ------------------------------------------
    let mut transcriber = WhisperTranscriber::new(&model_id, &asr_path, language, false);
    transcriber.warm_up()?;
    let lines = speakers::transcribe_spans(
        &wav.samples,
        16_000.0,
        &spans,
        &mut transcriber,
        SpanOptions::default(),
    );

    println!("\n-- transcript --");
    for line in &lines {
        println!(
            "[{}] {}: {}",
            format_clock(line.start_seconds),
            line.speaker,
            line.text
        );
    }
    println!("--------");
    Ok(())
}
