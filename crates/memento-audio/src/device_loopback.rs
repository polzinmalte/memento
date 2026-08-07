//! Lane B's fallback: everything the default output device is playing.
//!
//! [`crate::ProcessLoopbackCapture`] is the one to prefer — capturing only the
//! meeting app is what stops the transcript filling up with your music. But it
//! needs a PID, and there is none to offer when nothing is playing audio yet,
//! which is exactly the state the app is in while the user is setting up. This
//! is the whole-device answer for that case.
//!
//! Unlike process loopback this is an ordinary render endpoint opened with
//! `AUDCLNT_STREAMFLAGS_LOOPBACK`, so the device dictates the format and it is
//! polled rather than event-driven: a loopback client is not signalled while
//! the device is idle, so waiting on an event would simply never wake.

#![cfg(windows)]

use crate::{CaptureChunk, Decimator, downmix_to_mono};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
    IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator, WAVE_FORMAT_PCM,
    eConsole, eRender,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// What the rest of the pipeline consumes.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// About the default device period, which is what an event-driven client would
/// have been woken at anyway. Short enough that the lane stays responsive to
/// `stop`, long enough not to spin a core or flood the UI with level updates.
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const BUFFER_DURATION_HNS: i64 = 10_000_000;
/// WAVE_FORMAT_IEEE_FLOAT. Not exported by the `windows` crate outside the
/// KernelStreaming feature.
const WAVE_FORMAT_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

#[derive(Clone, Default)]
pub struct DeviceLoopbackCapture {
    stop: Arc<AtomicBool>,
}

impl DeviceLoopbackCapture {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ends the capture at the next poll, from any thread.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Capture the default output device, delivering 16 kHz mono f32.
    ///
    /// Blocks on the calling thread until stopped or `duration` elapses.
    pub fn run(
        &self,
        duration: Option<Duration>,
        mut on_chunk: impl FnMut(CaptureChunk),
    ) -> Result<(), Error> {
        let stop = Arc::clone(&self.stop);
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
            let _com = ComGuard;

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            // eRender even though this captures: loopback taps what the output
            // device is playing.
            let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

            // Loopback has to take the render endpoint's own mix format;
            // asking for a different one fails initialization outright.
            let format_ptr = client.GetMixFormat()?;
            // WAVEFORMATEX is `#[repr(packed)]`, so every field has to be copied
            // into a local before it is used — taking a reference to one (which
            // `format!` and comparisons do implicitly) is undefined behaviour.
            let format = *format_ptr;
            let tag = format.wFormatTag;
            let bits = format.wBitsPerSample;
            let channels = format.nChannels as usize;
            let source_rate = format.nSamplesPerSec;
            let is_float =
                tag == WAVE_FORMAT_FLOAT || (tag == WAVE_FORMAT_EXTENSIBLE && bits == 32);
            let is_pcm16 = tag == WAVE_FORMAT_PCM as u16 && bits == 16;

            let init = client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                BUFFER_DURATION_HNS,
                0,
                format_ptr,
                None,
            );
            CoTaskMemFree(Some(format_ptr.cast()));
            init?;

            if !(is_float || is_pcm16) {
                return Err(
                    format!("unsupported output device format: tag {tag} at {bits} bits").into(),
                );
            }

            let mut decimator =
                Decimator::for_rates(source_rate, TARGET_SAMPLE_RATE).ok_or_else(|| -> Error {
                    format!(
                        "the output device runs at {source_rate} Hz, which is not a whole \
                         multiple of {TARGET_SAMPLE_RATE} Hz; pick the meeting application \
                         instead of whole-device capture"
                    )
                    .into()
                })?;

            let capture: IAudioCaptureClient = client.GetService()?;
            client.Start()?;

            let started = Instant::now();
            // Samples handed to the caller so far. This, not wall clock, defines
            // the lane's timeline — see the padding below.
            let mut samples_emitted: u64 = 0;

            while !stop.load(Ordering::Relaxed) {
                if let Some(limit) = duration
                    && started.elapsed() >= limit
                {
                    break;
                }
                std::thread::sleep(POLL_INTERVAL);

                loop {
                    if capture.GetNextPacketSize()? == 0 {
                        break;
                    }
                    let mut data: *mut u8 = std::ptr::null_mut();
                    let mut frames: u32 = 0;
                    let mut flags: u32 = 0;
                    capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;

                    let sample_count = frames as usize * channels;
                    let interleaved: Vec<f32> = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0
                    {
                        // Windows is entitled to hand back a garbage pointer with
                        // the SILENT flag set; the contract is to treat it as zeros.
                        vec![0.0; sample_count]
                    } else if is_float {
                        std::slice::from_raw_parts(data.cast::<f32>(), sample_count).to_vec()
                    } else {
                        std::slice::from_raw_parts(data.cast::<i16>(), sample_count)
                            .iter()
                            .map(|s| *s as f32 / 32768.0)
                            .collect()
                    };
                    capture.ReleaseBuffer(frames)?;

                    let mono = downmix_to_mono(&interleaved, channels);
                    let resampled = decimator.process(&mono);
                    if !resampled.is_empty() {
                        samples_emitted += resampled.len() as u64;
                        on_chunk(CaptureChunk {
                            start_seconds: (samples_emitted - resampled.len() as u64) as f64
                                / TARGET_SAMPLE_RATE as f64,
                            samples: resampled,
                        });
                    }
                }

                // A device playing nothing delivers no packets at all, so
                // without this the lane's timeline compresses: ten silent
                // minutes would occupy zero seconds, and every timestamp after
                // them would be wrong by ten minutes when merged against the
                // mic lane.
                let expected = (started.elapsed().as_secs_f64() * TARGET_SAMPLE_RATE as f64) as u64;
                if let Some(missing) = expected.checked_sub(samples_emitted)
                    && missing > 0
                {
                    on_chunk(CaptureChunk {
                        samples: vec![0.0f32; missing as usize],
                        start_seconds: samples_emitted as f64 / TARGET_SAMPLE_RATE as f64,
                    });
                    samples_emitted += missing;
                }
            }

            client.Stop()?;
        }
        Ok(())
    }
}

struct ComGuard;

impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}
