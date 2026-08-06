//! Phase 0 of the port plan: prove per-process WASAPI loopback works before
//! anything is built on top of it.
//!
//! Usage:  loopback-spike <pid> [seconds] [out.wav]
//!
//! Find the PID with `tasklist | findstr Teams` (or Zoom, chrome, ms-teams).
//! Pick the *parent* process — the capture includes its whole tree, which is
//! what makes it work at all, since meeting apps render audio from children.
//!
//! The acceptance test from the plan: play a video in your browser and music in
//! Spotify at the same time, target only the browser, and confirm the resulting
//! WAV contains only the video. That is the whole point of per-process capture,
//! and the reason a meeting recording will not also transcribe your music.

#[cfg(not(windows))]
fn main() {
    eprintln!("loopback-spike is Windows-only; WASAPI process loopback has no equivalent here.");
    std::process::exit(1);
}

#[cfg(windows)]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use memento_audio::{
        CAPTURE_CHANNELS, CAPTURE_SAMPLE_RATE, ProcessLoopbackCapture, downmix_to_mono,
    };
    use memento_core::audio_math;
    use std::time::Duration;

    let args: Vec<String> = std::env::args().collect();
    let Some(pid) = args.get(1).and_then(|a| a.parse::<u32>().ok()) else {
        eprintln!("usage: loopback-spike <pid> [seconds] [out.wav]");
        std::process::exit(2);
    };
    let seconds: u64 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(10);
    let out_path = args.get(3).cloned().unwrap_or_else(|| "spike.wav".into());

    println!("capturing pid {pid} (and its children) for {seconds}s…");

    let mut mono: Vec<f32> = Vec::new();
    let mut packets = 0usize;
    ProcessLoopbackCapture::new().run(pid, Some(Duration::from_secs(seconds)), |chunk| {
        packets += 1;
        mono.extend(downmix_to_mono(&chunk.samples, CAPTURE_CHANNELS as usize));
    })?;

    let captured_seconds = mono.len() as f64 / CAPTURE_SAMPLE_RATE as f64;
    let rms = audio_math::rms(&mono);
    let peak = audio_math::peak(&mono);
    println!("{packets} packets, {captured_seconds:.2}s, rms {rms:.5}, peak {peak:.5}");

    if peak == 0.0 {
        // The single most likely failure, and it is not obvious from a silent
        // WAV: the process tree rendered nothing the whole time.
        eprintln!("! captured only silence — is that pid actually playing audio?");
        eprintln!("  Try the parent process of the meeting app, or check it is not muted.");
    }

    // Note this writes at the capture rate; resampling to the 16 kHz the ASR
    // pipeline wants is phase 2's job, deliberately not shortcut here.
    std::fs::write(&out_path, audio_math::wav_bytes(&mono, CAPTURE_SAMPLE_RATE))?;
    println!("wrote {out_path}");
    Ok(())
}
