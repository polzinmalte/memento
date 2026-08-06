//! Per-process WASAPI loopback capture.
//!
//! This is the capability macOS listnr does not have: instead of capturing
//! everything the default output device is playing, capture only the audio
//! rendered by one process **and its children**. A meeting app renders from
//! child processes (Teams, Zoom and any Chromium-based client all do), so
//! targeting the bare PID captures silence — `INCLUDE_TARGET_PROCESS_TREE` is
//! the mode that works.
//!
//! Nothing here is visible to the application being captured. WASAPI loopback is
//! a passive read of the render endpoint; there is no bot, no injected device,
//! and no recording indicator in the meeting UI.
//!
//! No Rust crate wraps `ActivateAudioInterfaceAsync` process loopback, so this
//! is written directly against the `windows` crate.

#![cfg(windows)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, S_OK, WAIT_OBJECT_0};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    AUDCLNT_STREAMFLAGS_LOOPBACK, AUDIOCLIENT_ACTIVATION_PARAMS, AUDIOCLIENT_ACTIVATION_PARAMS_0,
    AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS,
    ActivateAudioInterfaceAsync, IActivateAudioInterfaceAsyncOperation,
    IActivateAudioInterfaceCompletionHandler, IActivateAudioInterfaceCompletionHandler_Impl,
    IAudioCaptureClient, IAudioClient, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
    VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, WAVEFORMATEX,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForSingleObject};
use windows::core::{IUnknown, Interface, PCWSTR, Ref, implement};

/// The device the capture format is fixed at. A process-loopback client has no
/// mix format to query (`GetMixFormat` is not available on it), so the format is
/// asserted rather than negotiated, and 48 kHz stereo float is what the Windows
/// audio engine mixes at.
pub const CAPTURE_SAMPLE_RATE: u32 = 48_000;
pub const CAPTURE_CHANNELS: u16 = 2;

/// 20 ms. WASAPI's event only fires when the target actually renders, so a wait
/// this short doubles as the tick that drives silence padding.
const WAIT_TIMEOUT_MS: u32 = 20;
/// 1 s of 48 kHz stereo, in 100 ns units, as `Initialize` wants it.
const BUFFER_DURATION_HNS: i64 = 10_000_000;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// A chunk of interleaved f32 samples plus where it sits on the session clock.
/// The offset is what makes silence padding observable to the caller.
pub struct CaptureChunk {
    pub samples: Vec<f32>,
    pub start_seconds: f64,
}

/// PROPVARIANT carrying a VT_BLOB, which is how `ActivateAudioInterfaceAsync`
/// takes its activation parameters.
///
/// Hand-rolled because the `windows` crate models `PROPVARIANT` as an opaque
/// type with no VT_BLOB constructor. The layout is the documented one: a 8-byte
/// header, then the union, whose BLOB arm is `{ ULONG cbSize; BYTE* pBlobData }`.
#[repr(C)]
struct BlobPropVariant {
    vt: u16,
    reserved1: u16,
    reserved2: u16,
    reserved3: u16,
    cb_size: u32,
    _padding: u32,
    p_blob_data: *mut u8,
}

const VT_BLOB: u16 = 65;

impl BlobPropVariant {
    fn new(blob: &mut AUDIOCLIENT_ACTIVATION_PARAMS) -> Self {
        Self {
            vt: VT_BLOB,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            cb_size: size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
            _padding: 0,
            p_blob_data: (blob as *mut AUDIOCLIENT_ACTIVATION_PARAMS).cast(),
        }
    }
}

/// `ActivateAudioInterfaceAsync` completes on a COM thread, so the caller waits
/// on an event this handler sets.
#[implement(IActivateAudioInterfaceCompletionHandler)]
struct ActivationHandler {
    done: HANDLE,
}

impl IActivateAudioInterfaceCompletionHandler_Impl for ActivationHandler_Impl {
    fn ActivateCompleted(
        &self,
        _operation: Ref<'_, IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        // The result itself is read by the waiter via GetActivateResult; this
        // only needs to unblock it. Failing to signal here would hang the
        // session start forever, so the error is deliberately swallowed.
        unsafe {
            let _ = SetEvent(self.done);
        }
        Ok(())
    }
}

/// Handle for a capture that is about to run, or is running on another thread.
/// Cloning it is how the UI thread gets something it can call `stop()` on.
#[derive(Clone, Default)]
pub struct ProcessLoopbackCapture {
    stop: Arc<AtomicBool>,
}

impl ProcessLoopbackCapture {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ends the capture at the next tick, from any thread.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Capture everything `pid` and its children render, calling `on_chunk` with
    /// interleaved 48 kHz stereo f32.
    ///
    /// Blocks until [`stop`](Self::stop) is called or `duration` elapses. Runs on
    /// the calling thread, which must not be the UI thread.
    pub fn run(
        &self,
        pid: u32,
        duration: Option<Duration>,
        mut on_chunk: impl FnMut(CaptureChunk),
    ) -> Result<(), Error> {
        let stop = Arc::clone(&self.stop);
        unsafe {
            // MTA: the activation completion handler is called on a pool thread.
            CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
            let _com = ComGuard;

            let client = activate_process_loopback_client(pid)?;

            let format = WAVEFORMATEX {
                // WAVE_FORMAT_IEEE_FLOAT. Spelled out rather than imported:
                // the `windows` crate only exposes it under the KernelStreaming
                // feature, which is a lot of surface for one constant.
                wFormatTag: 3,
                nChannels: CAPTURE_CHANNELS,
                nSamplesPerSec: CAPTURE_SAMPLE_RATE,
                nAvgBytesPerSec: CAPTURE_SAMPLE_RATE * CAPTURE_CHANNELS as u32 * 4,
                nBlockAlign: CAPTURE_CHANNELS * 4,
                wBitsPerSample: 32,
                cbSize: 0,
            };

            // Loopback + event callback, shared mode, and a periodicity of 0:
            // the combination process loopback requires. A device-loopback
            // client would instead take its format from GetMixFormat.
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                BUFFER_DURATION_HNS,
                0,
                &format,
                None,
            )?;

            let ready = CreateEventW(None, false, false, PCWSTR::null())?;
            client.SetEventHandle(ready)?;
            let capture: IAudioCaptureClient = client.GetService()?;
            client.Start()?;

            let started = Instant::now();
            let frame_rate = CAPTURE_SAMPLE_RATE as f64;
            // Frames handed to the caller so far. This, not wall clock, defines
            // the lane's timeline — see `pad_silence`.
            let mut frames_emitted: u64 = 0;

            while !stop.load(Ordering::Relaxed) {
                if let Some(limit) = duration
                    && started.elapsed() >= limit
                {
                    break;
                }

                // A timeout rather than INFINITE: a silent target renders
                // nothing and signals nothing, and the loop still has to run to
                // notice `stop` and to emit padding.
                let _ = WaitForSingleObject(ready, WAIT_TIMEOUT_MS);

                loop {
                    let packet_frames = capture.GetNextPacketSize()?;
                    if packet_frames == 0 {
                        break;
                    }

                    let mut data: *mut u8 = std::ptr::null_mut();
                    let mut frames: u32 = 0;
                    let mut flags: u32 = 0;
                    capture.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;

                    let sample_count = frames as usize * CAPTURE_CHANNELS as usize;
                    let samples = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                        // Windows is entitled to hand back a garbage pointer with
                        // the SILENT flag set; the contract is to treat it as zeros.
                        vec![0.0f32; sample_count]
                    } else {
                        std::slice::from_raw_parts(data.cast::<f32>(), sample_count).to_vec()
                    };
                    capture.ReleaseBuffer(frames)?;

                    on_chunk(CaptureChunk {
                        samples,
                        start_seconds: frames_emitted as f64 / frame_rate,
                    });
                    frames_emitted += frames as u64;
                }

                // The gap that device loopback does not have. A silent process
                // delivers no packets at all, so without this the lane's
                // timeline compresses: ten silent minutes would occupy zero
                // seconds, and every timestamp after them would be wrong by ten
                // minutes when merged against the mic lane.
                let expected = (started.elapsed().as_secs_f64() * frame_rate) as u64;
                if let Some(missing) = expected.checked_sub(frames_emitted)
                    && missing > 0
                {
                    let sample_count = missing as usize * CAPTURE_CHANNELS as usize;
                    on_chunk(CaptureChunk {
                        samples: vec![0.0f32; sample_count],
                        start_seconds: frames_emitted as f64 / frame_rate,
                    });
                    frames_emitted += missing;
                }
            }

            client.Stop()?;
        }
        Ok(())
    }
}

/// Activate an `IAudioClient` bound to one process tree's render stream.
unsafe fn activate_process_loopback_client(pid: u32) -> Result<IAudioClient, Error> {
    unsafe {
        let mut params = AUDIOCLIENT_ACTIVATION_PARAMS {
            ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
                ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                    TargetProcessId: pid,
                    // Tree, not the bare PID: meeting apps render from a child
                    // process, so the bare PID captures nothing at all.
                    ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
                },
            },
        };
        let propvariant = BlobPropVariant::new(&mut params);

        let done = CreateEventW(None, false, false, PCWSTR::null())?;
        let handler: IActivateAudioInterfaceCompletionHandler = ActivationHandler { done }.into();

        let operation: IActivateAudioInterfaceAsyncOperation = ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some((&propvariant as *const BlobPropVariant).cast()),
            &handler,
        )?;

        if WaitForSingleObject(done, INFINITE) != WAIT_OBJECT_0 {
            return Err("waiting for audio interface activation failed".into());
        }

        let mut activate_result = S_OK;
        let mut interface: Option<IUnknown> = None;
        operation.GetActivateResult(&mut activate_result, &mut interface)?;
        activate_result.ok()?;

        interface
            .ok_or_else(|| -> Error { "process loopback activation returned no interface".into() })?
            .cast::<IAudioClient>()
            .map_err(Into::into)
    }
}

struct ComGuard;

impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}
