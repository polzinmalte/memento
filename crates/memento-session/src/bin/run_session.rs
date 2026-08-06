//! Run a whole session from two WAV files — the complete meeting path without
//! a meeting, and without Windows.
//!
//! Usage:  run-session --mic you.wav --system others.wav [--speakers n] [--model id]
//!
//! Either lane may be omitted, which stands in for a muted mic or a silent call.

use memento_core::MicSensitivity;
use memento_session::{
    CaptureSource, FileSource, SessionConfig, SessionEvent, StopHandle, clock, run_session,
};
use std::path::PathBuf;
use std::sync::Arc;

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    if args.is_empty() {
        eprintln!(
            "usage: run-session --mic you.wav --system others.wav [--speakers n] [--model id]"
        );
        std::process::exit(2);
    }

    let mic: Box<dyn CaptureSource> = match flag("--mic") {
        Some(path) => Box::new(FileSource::open(&path)?),
        None => Box::new(FileSource::silent()),
    };
    let system: Box<dyn CaptureSource> = match flag("--system") {
        Some(path) => Box::new(FileSource::open(&path)?),
        None => Box::new(FileSource::silent()),
    };

    let work_dir = std::env::temp_dir().join("memento-run-session");
    let config = SessionConfig {
        language: memento_core::LanguageMode::De,
        model_id: flag("--model"),
        sensitivity: MicSensitivity::Normal,
        speakers: flag("--speakers").and_then(|s| s.parse().ok()),
        diarize: true,
        output_dir: flag("--out")
            .map(PathBuf::from)
            .unwrap_or_else(|| work_dir.clone()),
        work_dir,
        keep_audio: args.iter().any(|a| a == "--keep-audio"),
    };

    let events: memento_session::EventSink = Arc::new(|event| match event {
        SessionEvent::Preparing(what) => eprintln!("… {what}"),
        SessionEvent::Started => eprintln!("… capturing"),
        SessionEvent::Line(line) => eprintln!(
            "[{}] {:<10} {}",
            memento_core::format_clock(line.start_seconds),
            line.speaker,
            line.text
        ),
        SessionEvent::Warning(w) => eprintln!("! {w}"),
        SessionEvent::Finalizing(what) => eprintln!("… {what}"),
        // Levels drive meters in the UI; far too noisy for a terminal.
        SessionEvent::Level { .. } | SessionEvent::DownloadProgress { .. } => {}
    });

    let (label, stamp) = clock::now_labels();
    let outcome = run_session(
        &config,
        mic,
        system,
        StopHandle::new(),
        events,
        &label,
        &stamp,
    )?;

    println!("\n{}", outcome.markdown);
    println!("written to {}", outcome.transcript_path.display());
    Ok(())
}
