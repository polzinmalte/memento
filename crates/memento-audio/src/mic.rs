//! Lane A: microphone capture via WASAPI shared mode.
//!
//! Replaces listnr's `MicCapture.swift` (AVAudioEngine). Unlike the system lane
//! this is an ordinary capture client, so the mix format is negotiable and the
//! device tells us what it is rather than us asserting it.

#![cfg(windows)]

use crate::{CaptureChunk, Decimator, downmix_to_mono};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator, WAVE_FORMAT_PCM,
    eCapture, eConsole,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::PCWSTR;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// What the rest of the pipeline consumes.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

const WAIT_TIMEOUT_MS: u32 = 20;
const BUFFER_DURATION_HNS: i64 = 10_000_000;
/// WAVE_FORMAT_IEEE_FLOAT. Not exported by the `windows` crate outside the
/// KernelStreaming feature.
const WAVE_FORMAT_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

#[derive(Clone, Default)]
pub struct MicCapture {
    stop: Arc<AtomicBool>,
}

impl MicCapture {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Capture the default input device, delivering 16 kHz mono f32.
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
            // eConsole, not eCommunications: the latter is whatever Windows
            // thinks is the "chat" device, which is often not the one the user
            // picked for the meeting.
            let device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)?;
            let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;

            // The mic client, unlike process loopback, has a mix format to ask
            // for — so take the device's own rather than asserting one and
            // hoping the driver agrees.
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

            let init = client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                BUFFER_DURATION_HNS,
                0,
                format_ptr,
                None,
            );
            CoTaskMemFree(Some(format_ptr.cast()));
            init?;

            if !is_float && !(tag == WAVE_FORMAT_PCM as u16 && bits == 16) {
                return Err(
                    format!("unsupported microphone format: tag {tag} at {bits} bits").into(),
                );
            }

            let mut decimator =
                Decimator::for_rates(source_rate, TARGET_SAMPLE_RATE).ok_or_else(|| -> Error {
                    format!(
                        "microphone runs at {source_rate} Hz, which is not a whole multiple of \
                         {TARGET_SAMPLE_RATE} Hz"
                    )
                    .into()
                })?;

            let ready = CreateEventW(None, false, false, PCWSTR::null())?;
            client.SetEventHandle(ready)?;
            let capture: IAudioCaptureClient = client.GetService()?;
            client.Start()?;

            let started = Instant::now();
            let mut frames_emitted: u64 = 0;

            while !stop.load(Ordering::Relaxed) {
                if let Some(limit) = duration
                    && started.elapsed() >= limit
                {
                    break;
                }
                let _ = WaitForSingleObject(ready, WAIT_TIMEOUT_MS);

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
                        on_chunk(CaptureChunk {
                            samples: resampled.clone(),
                            start_seconds: frames_emitted as f64 / TARGET_SAMPLE_RATE as f64,
                        });
                        frames_emitted += resampled.len() as u64;
                    }
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
