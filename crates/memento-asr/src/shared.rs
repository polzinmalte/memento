//! One CPU work queue shared by both lanes.
//!
//! The reason this exists rather than each lane simply owning a transcriber:
//! whisper.cpp sizes its thread pool to the machine, so two lanes decoding at
//! once ask for twice the cores that exist. The result is not half speed each —
//! it is worse, because the two decodes fight over the same memory bandwidth and
//! both fall behind, and a lane that falls behind loses speech to its bounded
//! queue rather than merely running late.
//!
//! Serialising the decodes keeps each one running at full width. The lanes stay
//! logically independent — separate models, separate state, separate transcript
//! lines — they just do not run at the same instant.
//!
//! Each lane holds a [`SharedAsr`], which implements [`Transcriber`] by handing
//! the work to the pool and blocking. From the lane's point of view nothing has
//! changed, which is why `memento-core` needed no modification to support this.

use memento_core::queue::BoundedQueue;
use memento_core::{Error, Transcriber};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

enum Request {
    WarmUp,
    Transcribe(Vec<f32>),
}

struct Job {
    lane: usize,
    request: Request,
    reply: SyncSender<Result<String, Error>>,
}

pub struct AsrPool {
    queue: Arc<BoundedQueue<Job>>,
    worker: Option<JoinHandle<()>>,
    model_ids: Vec<String>,
}

impl AsrPool {
    /// One transcriber per lane, in lane order. They stay warm for the whole
    /// session; only their execution is serialised.
    pub fn new(mut transcribers: Vec<Box<dyn Transcriber>>) -> Self {
        let model_ids = transcribers
            .iter()
            .map(|t| t.model_id().to_string())
            .collect();
        // Every lane blocks on its own reply, so at most one job per lane is ever
        // in flight and this can never drop.
        let queue = Arc::new(BoundedQueue::new(transcribers.len().max(1) * 2));

        let worker = {
            let queue: Arc<BoundedQueue<Job>> = Arc::clone(&queue);
            thread::spawn(move || {
                while let Some(job) = queue.pop() {
                    let result = match transcribers.get_mut(job.lane) {
                        Some(t) => match job.request {
                            Request::WarmUp => t.warm_up().map(|()| String::new()),
                            Request::Transcribe(audio) => t.transcribe(&audio),
                        },
                        None => Err(format!("no transcriber for lane {}", job.lane).into()),
                    };
                    // A disconnected receiver means the lane was abandoned
                    // mid-decode, which is normal on cancel.
                    let _ = job.reply.send(result);
                }
            })
        };

        Self {
            queue,
            worker: Some(worker),
            model_ids,
        }
    }

    /// A handle for one lane. Panics if `lane` has no transcriber, since that is
    /// a wiring mistake rather than a runtime condition.
    pub fn handle(&self, lane: usize) -> SharedAsr {
        assert!(
            lane < self.model_ids.len(),
            "lane {lane} has no transcriber"
        );
        SharedAsr {
            lane,
            model_id: self.model_ids[lane].clone(),
            queue: Arc::clone(&self.queue),
        }
    }

    pub fn lanes(&self) -> usize {
        self.model_ids.len()
    }

    /// Stop the worker and drain. Idempotent.
    pub fn shutdown(&mut self) {
        self.queue.close();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for AsrPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// One lane's view of the shared pool.
pub struct SharedAsr {
    lane: usize,
    model_id: String,
    queue: Arc<BoundedQueue<Job>>,
}

impl SharedAsr {
    fn submit(&self, request: Request) -> Result<String, Error> {
        let (reply, receiver): (_, Receiver<Result<String, Error>>) = sync_channel(1);
        let accepted = self.queue.push(Job {
            lane: self.lane,
            request,
            reply,
        });
        if !accepted {
            return Err("transcription pool is shut down".into());
        }
        receiver
            .recv()
            .map_err(|_| -> Error { "transcription pool stopped before answering".into() })?
    }
}

impl Transcriber for SharedAsr {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn warm_up(&mut self) -> Result<(), Error> {
        self.submit(Request::WarmUp).map(|_| ())
    }

    fn transcribe(&mut self, audio: &[f32]) -> Result<String, Error> {
        self.submit(Request::Transcribe(audio.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memento_core::testing::FakeTranscriber;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records how many transcriptions are in flight at once, which is the
    /// property the pool exists to bound.
    struct ConcurrencyProbe {
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    impl Transcriber for ConcurrencyProbe {
        fn model_id(&self) -> &str {
            "probe"
        }
        fn warm_up(&mut self) -> Result<(), Error> {
            Ok(())
        }
        fn transcribe(&mut self, _audio: &[f32]) -> Result<String, Error> {
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(std::time::Duration::from_millis(20));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok("ok".into())
        }
    }

    #[test]
    fn each_lane_gets_its_own_transcriber() {
        let pool = AsrPool::new(vec![
            Box::new(FakeTranscriber::new(&["from lane 0"])),
            Box::new(FakeTranscriber::new(&["from lane 1"])),
        ]);
        let mut a = pool.handle(0);
        let mut b = pool.handle(1);
        assert_eq!(a.transcribe(&[0.0]).unwrap(), "from lane 0");
        assert_eq!(b.transcribe(&[0.0]).unwrap(), "from lane 1");
    }

    /// The whole point: two lanes submitting at once must not decode at once.
    #[test]
    fn decodes_never_overlap_across_lanes() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let pool = AsrPool::new(vec![
            Box::new(ConcurrencyProbe {
                active: Arc::clone(&active),
                peak: Arc::clone(&peak),
            }),
            Box::new(ConcurrencyProbe {
                active: Arc::clone(&active),
                peak: Arc::clone(&peak),
            }),
        ]);

        let workers: Vec<_> = (0..2)
            .map(|lane| {
                let mut handle = pool.handle(lane);
                thread::spawn(move || {
                    for _ in 0..4 {
                        handle.transcribe(&[0.0; 16]).unwrap();
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }

        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "two lanes decoded at once; they would oversubscribe the CPU"
        );
    }

    #[test]
    fn work_is_served_in_submission_order() {
        let pool = AsrPool::new(vec![Box::new(FakeTranscriber::new(&[
            "first", "second", "third",
        ]))]);
        let mut handle = pool.handle(0);
        let got: Vec<String> = (0..3).map(|_| handle.transcribe(&[0.0]).unwrap()).collect();
        assert_eq!(got, ["first", "second", "third"]);
    }

    #[test]
    fn warm_up_reaches_the_underlying_transcriber() {
        let pool = AsrPool::new(vec![Box::new(FakeTranscriber::new(&["x"]))]);
        assert!(pool.handle(0).warm_up().is_ok());
    }

    #[test]
    fn errors_propagate_back_to_the_lane() {
        let pool = AsrPool::new(vec![Box::new(FakeTranscriber::failing())]);
        assert!(pool.handle(0).transcribe(&[0.0]).is_err());
    }

    /// A lane must get an error rather than block forever when the pool is gone.
    #[test]
    fn submitting_after_shutdown_errors_instead_of_hanging() {
        let mut pool = AsrPool::new(vec![Box::new(FakeTranscriber::new(&["x"]))]);
        let mut handle = pool.handle(0);
        pool.shutdown();
        assert!(handle.transcribe(&[0.0]).is_err());
    }

    #[test]
    fn shutdown_is_idempotent() {
        let mut pool = AsrPool::new(vec![Box::new(FakeTranscriber::new(&["x"]))]);
        pool.shutdown();
        pool.shutdown();
    }

    #[test]
    fn model_ids_are_visible_per_lane() {
        let pool = AsrPool::new(vec![
            Box::new(FakeTranscriber::new(&["a"])),
            Box::new(FakeTranscriber::new(&["b"])),
        ]);
        assert_eq!(pool.lanes(), 2);
        assert_eq!(pool.handle(1).model_id(), "fake");
    }
}
