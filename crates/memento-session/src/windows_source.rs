//! The Windows captures, as lanes the session can pull from.
//!
//! Each capture in `memento-audio` is a blocking loop that calls you back; the
//! session pulls one chunk at a time. [`ThreadedSource`] is the adapter, and
//! this module is only the three ways to build one — plus the format conversion
//! each capture needs, which differs: the mic and the whole-device loopback
//! already deliver what the lanes want, while process loopback hands over raw
//! 48 kHz stereo.

#![cfg(windows)]

use crate::SAMPLE_RATE;
use crate::source::{ChunkSender, Error, ThreadedSource};
use memento_audio::{
    CAPTURE_CHANNELS, CAPTURE_SAMPLE_RATE, Decimator, DeviceLoopbackCapture, MicCapture,
    ProcessLoopbackCapture, downmix_to_mono,
};
use std::thread::JoinHandle;

/// Lane A. Whatever Windows calls the default input device.
pub fn microphone() -> ThreadedSource {
    ThreadedSource::new(Box::new(|chunks: ChunkSender| {
        let capture = MicCapture::new();
        let worker = capture.clone();
        // MicCapture already downmixes and resamples, because it is the one
        // capture whose device format is negotiable.
        let thread = std::thread::spawn(move || {
            let result = worker.run(None, |chunk| {
                let _ = chunks.send(Ok(chunk.samples));
            });
            if let Err(e) = result {
                let _ = chunks.send(Err(e.to_string()));
            }
        });
        Ok(stopper(move || capture.stop(), thread))
    }))
}

/// Lane B, the good case: one application's audio and nothing else, which is
/// what keeps the music playing in another window out of the transcript.
pub fn application(pid: u32) -> ThreadedSource {
    ThreadedSource::new(Box::new(move |chunks: ChunkSender| {
        // Process loopback asserts its format rather than negotiating one, so
        // the conversion the mic does internally has to happen here. Decimating
        // without the low-pass in `Decimator` would fold consonants back into
        // the speech band, which Whisper hears as a lisp.
        let mut decimator =
            Decimator::for_rates(CAPTURE_SAMPLE_RATE, SAMPLE_RATE).ok_or_else(|| -> Error {
                format!("cannot resample {CAPTURE_SAMPLE_RATE} Hz capture to {SAMPLE_RATE} Hz")
                    .into()
            })?;

        let capture = ProcessLoopbackCapture::new();
        let worker = capture.clone();
        let thread = std::thread::spawn(move || {
            let result = worker.run(pid, None, |chunk| {
                let mono = downmix_to_mono(&chunk.samples, CAPTURE_CHANNELS as usize);
                let resampled = decimator.process(&mono);
                if !resampled.is_empty() {
                    let _ = chunks.send(Ok(resampled));
                }
            });
            if let Err(e) = result {
                let _ = chunks.send(Err(e.to_string()));
            }
        });
        Ok(stopper(move || capture.stop(), thread))
    }))
}

/// Lane B's fallback: everything the output device is playing, for when there
/// is no application to point at yet.
pub fn whole_device() -> ThreadedSource {
    ThreadedSource::new(Box::new(|chunks: ChunkSender| {
        let capture = DeviceLoopbackCapture::new();
        let worker = capture.clone();
        let thread = std::thread::spawn(move || {
            let result = worker.run(None, |chunk| {
                let _ = chunks.send(Ok(chunk.samples));
            });
            if let Err(e) = result {
                let _ = chunks.send(Err(e.to_string()));
            }
        });
        Ok(stopper(move || capture.stop(), thread))
    }))
}

/// Signal the capture, then wait for its thread. Joining matters: the WASAPI
/// client is released on the way out, and a session started immediately after
/// this one would otherwise be opening a device the last one still holds.
fn stopper(
    signal: impl FnOnce() + Send + 'static,
    thread: JoinHandle<()>,
) -> Box<dyn FnOnce() + Send> {
    Box::new(move || {
        signal();
        let _ = thread.join();
    })
}
