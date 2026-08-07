//! Where a lane's audio comes from.
//!
//! The session does not know whether it is reading a microphone, one
//! application's output, or a file. That indirection is what lets the whole
//! session path — lanes, VAD, ASR, diarization, merge, export — be exercised on
//! a machine with no Windows audio stack at all.

use memento_core::audio_math;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// The microphone: you.
    You,
    /// System audio: everyone else.
    Others,
}

/// A lane's audio, delivered as 16 kHz mono f32 chunks.
pub trait CaptureSource: Send {
    /// Begin capturing. Returns the clock reading of the first sample, when the
    /// source can measure one — the two lanes start milliseconds apart and this
    /// is what places them on a shared timeline.
    fn start(&mut self) -> Result<Option<f64>, Error>;

    /// The next chunk, or `None` when the source is finished. Blocking is fine;
    /// the session runs this on its own thread.
    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Error>;

    fn stop(&mut self);
}

/// Chunks on their way from a capture thread to the session, 16 kHz mono. An
/// `Err` is the capture giving up: whatever it says is the last thing the lane
/// has to say.
pub type ChunkSender = Sender<Result<Vec<f32>, String>>;

/// Ends a running capture, waiting no longer than the given deadline for its
/// thread to unwind.
pub type Stopper = Box<dyn FnOnce(Duration) + Send>;

/// Starts a capture, handing it the sender to feed. Called once, by
/// [`ThreadedSource::start`].
pub type Starter = Box<dyn FnOnce(ChunkSender) -> Result<Stopper, Error> + Send>;

/// Told about a lane that died mid-session, so the UI can say so.
pub type WarningSink = Arc<dyn Fn(String) + Send + Sync>;

/// How long [`ThreadedSource::start`] waits for the first buffer before
/// deciding the lane cannot be placed on the shared timeline.
const ANCHOR_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a running lane may deliver nothing before it is treated as dead.
///
/// Capture delivers about every 10 ms — a device that has stopped rendering
/// still produces silence — so a gap this long is a broken lane, not a quiet
/// one. Waiting forever instead is worse than it sounds: the session pulls the
/// two lanes in turn on one thread, so a lane that never answers also stops the
/// session noticing that the user pressed Stop.
const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(10);

/// How long [`ThreadedSource::stop`] waits for the capture thread to unwind
/// before leaving it to finish on its own. The session has a transcript to
/// write; it must not be held up by a device driver that will not let go.
const STOP_DEADLINE: Duration = Duration::from_secs(2);

/// A [`CaptureSource`] fed by a capture thread.
///
/// The Windows captures are blocking callback loops — you hand them a closure
/// and they run until stopped — while the session pulls one chunk at a time.
/// This is the adapter between the two, and it is also where a lane's anchor is
/// measured: the clock is read when the first buffer actually arrives, not when
/// `start` was called, because the two lanes' devices take different amounts of
/// time to produce their first audio and that difference is exactly what the
/// merge has to correct for.
pub struct ThreadedSource {
    starter: Option<Starter>,
    stopper: Option<Stopper>,
    chunks: Option<Receiver<Result<Vec<f32>, String>>>,
    /// The buffer `start` took off the channel to measure the anchor with.
    pending: Option<Vec<f32>>,
    warning: Option<WarningSink>,
    anchor_timeout: Duration,
    inactivity_timeout: Duration,
    stop_deadline: Duration,
}

impl ThreadedSource {
    pub fn new(starter: Starter) -> Self {
        Self {
            starter: Some(starter),
            stopper: None,
            chunks: None,
            pending: None,
            warning: None,
            anchor_timeout: ANCHOR_TIMEOUT,
            inactivity_timeout: INACTIVITY_TIMEOUT,
            stop_deadline: STOP_DEADLINE,
        }
    }

    /// Where to report a capture that dies after the meeting has begun.
    pub fn on_warning(mut self, sink: WarningSink) -> Self {
        self.warning = Some(sink);
        self
    }

    /// Shortens the wait for the first buffer. For tests, which must not sit
    /// out the real timeout.
    pub fn with_anchor_timeout(mut self, timeout: Duration) -> Self {
        self.anchor_timeout = timeout;
        self
    }

    /// Shortens how long a silent lane is tolerated. For tests — an untestable
    /// timeout is how a hang ships.
    pub fn with_inactivity_timeout(mut self, timeout: Duration) -> Self {
        self.inactivity_timeout = timeout;
        self
    }

    /// Shortens the wait for the capture thread to unwind. For tests.
    pub fn with_stop_deadline(mut self, deadline: Duration) -> Self {
        self.stop_deadline = deadline;
        self
    }

    fn warn(&self, message: String) {
        if let Some(sink) = &self.warning {
            sink(message);
        }
    }
}

impl CaptureSource for ThreadedSource {
    fn start(&mut self) -> Result<Option<f64>, Error> {
        let starter = self
            .starter
            .take()
            .ok_or_else(|| -> Error { "this capture has already been started".into() })?;
        let (sender, receiver) = std::sync::mpsc::channel();
        let stopper = starter(sender)?;

        let anchor = match receiver.recv_timeout(self.anchor_timeout) {
            Ok(Ok(first)) => {
                self.pending = Some(first);
                Some(now_seconds())
            }
            // A device that cannot open at all: nothing has been recorded yet,
            // so this is worth failing the session for — the user needs to know
            // why Start did nothing.
            Ok(Err(e)) => {
                stopper(self.stop_deadline);
                return Err(e.into());
            }
            // Silence so far, or a lane that finished immediately. Neither is an
            // error: the session runs on, unaligned, which `LaneAnchors` treats
            // as the honest answer rather than guessing an offset.
            Err(_) => None,
        };

        self.stopper = Some(stopper);
        self.chunks = Some(receiver);
        Ok(anchor)
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Error> {
        if let Some(first) = self.pending.take() {
            return Ok(Some(first));
        }
        let Some(chunks) = &self.chunks else {
            return Ok(None);
        };
        let outcome = chunks.recv_timeout(self.inactivity_timeout);
        match outcome {
            Ok(Ok(samples)) => Ok(Some(samples)),
            // The lane ends, the session does not. A microphone unplugged forty
            // minutes in must not cost the forty minutes.
            Ok(Err(e)) => {
                self.warn(e);
                Ok(None)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                self.warn(format!(
                    "a capture device stopped delivering audio for {} s and was dropped; \
                     the transcript keeps everything recorded up to then",
                    self.inactivity_timeout.as_secs()
                ));
                Ok(None)
            }
            Err(_) => Ok(None),
        }
    }

    fn stop(&mut self) {
        if let Some(stopper) = self.stopper.take() {
            stopper(self.stop_deadline);
        }
        self.chunks = None;
        // Including the buffer `start` was holding: a stopped source is
        // finished, and one that still answers with audio afterwards is a trap
        // for anyone who assumes `None` means done.
        self.pending = None;
    }
}

/// Seconds since the Unix epoch. Only differences between the two lanes are
/// used, so the epoch is arbitrary — but it has to be the same one for both,
/// which rules out `Instant`.
fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Replays a WAV. Used for testing the session end to end, and for developing
/// the UI on a machine that cannot capture Windows audio.
pub struct FileSource {
    samples: Vec<f32>,
    cursor: usize,
    chunk_samples: usize,
    /// Seconds this lane starts after the other, for exercising lane alignment.
    start_offset: Option<f64>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let wav = audio_math::parse_wav(&std::fs::read(path)?)?;
        if wav.sample_rate != 16_000 || wav.channels != 1 {
            return Err(format!(
                "expected 16 kHz mono, got {} Hz {}ch",
                wav.sample_rate, wav.channels
            )
            .into());
        }
        Ok(Self::from_samples(wav.samples))
    }

    pub fn from_samples(samples: Vec<f32>) -> Self {
        Self {
            samples,
            cursor: 0,
            // Roughly what WASAPI hands over per callback.
            chunk_samples: 1024,
            start_offset: None,
        }
    }

    /// An empty lane, for a session where only one side has audio.
    pub fn silent() -> Self {
        Self::from_samples(Vec::new())
    }

    pub fn with_start_offset(mut self, seconds: f64) -> Self {
        self.start_offset = Some(seconds);
        self
    }
}

impl CaptureSource for FileSource {
    fn start(&mut self) -> Result<Option<f64>, Error> {
        self.cursor = 0;
        Ok(self.start_offset)
    }

    fn next_chunk(&mut self) -> Result<Option<Vec<f32>>, Error> {
        if self.cursor >= self.samples.len() {
            return Ok(None);
        }
        let end = (self.cursor + self.chunk_samples).min(self.samples.len());
        let chunk = self.samples[self.cursor..end].to_vec();
        self.cursor = end;
        Ok(Some(chunk))
    }

    fn stop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use memento_core::testing::{silence, tone};
    use std::time::Instant;

    #[test]
    fn a_file_source_delivers_every_sample_once() {
        let audio = [silence(0.5), tone(0.5, 0.1)].concat();
        let mut source = FileSource::from_samples(audio.clone());
        source.start().unwrap();

        let mut got = Vec::new();
        while let Some(chunk) = source.next_chunk().unwrap() {
            got.extend(chunk);
        }
        assert_eq!(got, audio);
        assert_eq!(
            source.next_chunk().unwrap(),
            None,
            "and then stays finished"
        );
    }

    #[test]
    fn a_silent_source_finishes_immediately() {
        let mut source = FileSource::silent();
        source.start().unwrap();
        assert_eq!(source.next_chunk().unwrap(), None);
    }

    #[test]
    fn start_offsets_reach_the_lane_alignment() {
        let mut source = FileSource::from_samples(vec![0.0; 10]).with_start_offset(100.25);
        assert_eq!(source.start().unwrap(), Some(100.25));
    }

    #[test]
    fn restarting_replays_from_the_beginning() {
        let mut source = FileSource::from_samples(vec![0.5; 100]);
        source.start().unwrap();
        while source.next_chunk().unwrap().is_some() {}
        source.start().unwrap();
        assert!(source.next_chunk().unwrap().is_some());
    }

    /// A stand-in for a Windows capture: sends what it is told to, then ends.
    /// `stopped` records that the source really did shut the producer down.
    fn producer(
        script: Vec<Result<Vec<f32>, String>>,
        stopped: Arc<std::sync::atomic::AtomicBool>,
    ) -> ThreadedSource {
        ThreadedSource::new(Box::new(move |tx: ChunkSender| {
            let thread = std::thread::spawn(move || {
                for item in script {
                    if tx.send(item).is_err() {
                        break;
                    }
                }
            });
            Ok(Box::new(move |_deadline: Duration| {
                stopped.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = thread.join();
            }) as Stopper)
        }))
        .with_anchor_timeout(Duration::from_millis(250))
        .with_inactivity_timeout(Duration::from_millis(250))
    }

    fn flag() -> Arc<std::sync::atomic::AtomicBool> {
        Arc::new(std::sync::atomic::AtomicBool::new(false))
    }

    #[test]
    fn every_captured_buffer_reaches_the_session_in_order() {
        // Including the first one, which `start` takes off the channel to read
        // the clock and must hand back rather than swallow.
        let script = vec![Ok(vec![0.1]), Ok(vec![0.2]), Ok(vec![0.3])];
        let mut source = producer(script, flag());
        source.start().unwrap();

        let mut got = Vec::new();
        while let Some(chunk) = source.next_chunk().unwrap() {
            got.extend(chunk);
        }
        assert_eq!(got, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn the_anchor_is_the_clock_when_the_first_buffer_arrived() {
        let mut source = producer(vec![Ok(vec![0.1])], flag());
        let anchor = source.start().unwrap().expect("a lane with audio anchors");
        assert!((anchor - now_seconds()).abs() < 5.0, "{anchor}");
    }

    /// Nothing has been recorded yet, so a device that will not open is worth
    /// failing on — otherwise Start appears to work and records silence.
    #[test]
    fn a_capture_that_cannot_start_fails_the_session_and_stops_the_producer() {
        let stopped = flag();
        let mut source = producer(vec![Err("no microphone".into())], Arc::clone(&stopped));
        let err = source.start().unwrap_err().to_string();
        assert!(err.contains("no microphone"), "{err}");
        assert!(stopped.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// The opposite case: the device opened but is silent so far. That is an
    /// ordinary meeting that has not started talking yet, not a failure.
    #[test]
    fn a_silent_lane_still_starts_just_without_an_anchor() {
        let mut source = producer(Vec::new(), flag());
        assert_eq!(source.start().unwrap(), None);
    }

    /// The regression this whole indirection exists to prevent: losing a
    /// recorded meeting because a device disappeared during it.
    #[test]
    fn a_device_lost_mid_meeting_ends_the_lane_not_the_session() {
        let warnings: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let collected = Arc::clone(&warnings);
        let script = vec![Ok(vec![0.1]), Ok(vec![0.2]), Err("device removed".into())];
        let mut source = producer(script, flag())
            .on_warning(Arc::new(move |w| collected.lock().unwrap().push(w)));
        source.start().unwrap();

        assert_eq!(source.next_chunk().unwrap(), Some(vec![0.1]));
        assert_eq!(source.next_chunk().unwrap(), Some(vec![0.2]));
        assert_eq!(source.next_chunk().unwrap(), None, "the lane ends");
        assert_eq!(warnings.lock().unwrap().as_slice(), ["device removed"]);
    }

    /// The session pulls both lanes in turn on one thread, so a lane that never
    /// answers does not just lose its own audio — it stops the session noticing
    /// that Stop was pressed, and the meeting is never written out at all.
    #[test]
    fn a_lane_that_goes_quiet_ends_instead_of_wedging_the_session() {
        let warnings: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let collected = Arc::clone(&warnings);

        // Sends one buffer, then never speaks again — a device that stopped
        // rendering without saying so.
        let mut source = ThreadedSource::new(Box::new(|tx: ChunkSender| {
            std::thread::spawn(move || {
                let _ = tx.send(Ok(vec![0.1]));
                std::thread::sleep(Duration::from_secs(30));
            });
            Ok(Box::new(|_deadline: Duration| {}) as Stopper)
        }))
        .with_anchor_timeout(Duration::from_millis(250))
        .with_inactivity_timeout(Duration::from_millis(150))
        .on_warning(Arc::new(move |w| collected.lock().unwrap().push(w)));

        source.start().unwrap();
        assert_eq!(source.next_chunk().unwrap(), Some(vec![0.1]));

        let waited = Instant::now();
        assert_eq!(source.next_chunk().unwrap(), None, "the lane must give up");
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "and give up quickly"
        );
        assert!(
            warnings.lock().unwrap()[0].contains("stopped delivering"),
            "{:?}",
            warnings.lock().unwrap()
        );
    }

    /// Same reasoning one step later: the transcript is written after the
    /// captures are told to stop, so a driver that will not let go must not be
    /// able to hold the finished meeting hostage.
    #[test]
    fn stopping_gives_up_on_a_capture_that_will_not_unwind() {
        let mut source = ThreadedSource::new(Box::new(|tx: ChunkSender| {
            std::thread::spawn(move || {
                let _ = tx.send(Ok(vec![0.1]));
                std::thread::sleep(Duration::from_secs(30));
            });
            Ok(Box::new(|deadline: Duration| std::thread::sleep(deadline)) as Stopper)
        }))
        .with_anchor_timeout(Duration::from_millis(250))
        .with_stop_deadline(Duration::from_millis(100));

        source.start().unwrap();
        let waited = Instant::now();
        source.stop();
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "must not block on it"
        );
    }

    #[test]
    fn stopping_shuts_the_producer_down() {
        let stopped = flag();
        let mut source = producer(vec![Ok(vec![0.1])], Arc::clone(&stopped));
        source.start().unwrap();
        source.stop();
        assert!(stopped.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(
            source.next_chunk().unwrap(),
            None,
            "a stopped source is finished, buffered chunk and all"
        );
    }

    #[test]
    fn a_wrongly_formatted_file_is_rejected_with_a_useful_message() {
        let path = std::env::temp_dir().join("memento-source-test.wav");
        std::fs::write(&path, audio_math::wav_bytes(&[0.1; 100], 44_100)).unwrap();
        let err = match FileSource::open(&path) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a 44.1 kHz file must be rejected"),
        };
        assert!(err.contains("16 kHz mono"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
