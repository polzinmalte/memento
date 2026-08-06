//! The stop-versus-cancel decision for one session. Ported from listnr
//! `Sources/ListnrCore/Session/SessionStopState.swift`.
//!
//! This lives apart from the session runner because the distinction it encodes
//! is the one that broke in listnr `0.1.0-beta`, and the runner itself cannot be
//! exercised without a microphone. Keeping the rule here makes it testable.
//!
//! The rule: a **graceful stop** ends capture and keeps the transcript, while a
//! **cancel** throws the session away. The only exception is a stop that arrives
//! before capture has begun, where there is no transcript to keep and the only
//! thing running is a model download, so the work is aborted instead.

use std::sync::Mutex;

#[derive(Default)]
pub struct SessionStopState {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    stop_requested: bool,
    cancel_requested: bool,
    capture_started: bool,
}

impl SessionStopState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request a graceful finish.
    ///
    /// Returns `true` when the caller should also abort the underlying work,
    /// which is the case only before capture started. Aborting after that point
    /// is what discarded finished meetings.
    pub fn request_stop(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner.stop_requested = true;
        !inner.capture_started
    }

    /// Request an abort. The transcript is discarded.
    pub fn request_cancel(&self) {
        self.inner.lock().unwrap().cancel_requested = true;
    }

    /// Called once capture is live, after which a graceful stop must not cancel.
    pub fn mark_capture_started(&self) {
        self.inner.lock().unwrap().capture_started = true;
    }

    /// True when the capture loop should exit, for either reason.
    pub fn should_finish(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.stop_requested || inner.cancel_requested
    }

    /// True only for an abort. A graceful stop never sets this, which is what
    /// lets finalization run.
    pub fn is_cancel(&self) -> bool {
        self.inner.lock().unwrap().cancel_requested
    }

    pub fn capture_has_started(&self) -> bool {
        self.inner.lock().unwrap().capture_started
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_is_idle() {
        let s = SessionStopState::new();
        assert!(!s.should_finish());
        assert!(!s.is_cancel());
        assert!(!s.capture_has_started());
    }

    #[test]
    fn stop_before_capture_also_aborts() {
        // Nothing to keep yet, and the only thing running is a model download.
        let s = SessionStopState::new();
        assert!(s.request_stop(), "caller should abort the pending work");
        assert!(s.should_finish());
        assert!(!s.is_cancel());
    }

    /// The regression this type exists for: a graceful stop after capture began
    /// must not cancel, because cancelling discarded the finished meeting.
    #[test]
    fn stop_after_capture_must_not_abort() {
        let s = SessionStopState::new();
        s.mark_capture_started();
        assert!(
            !s.request_stop(),
            "cancelling here would discard the transcript"
        );
        assert!(s.should_finish());
        assert!(!s.is_cancel(), "finalization must still run");
    }

    #[test]
    fn cancel_discards_regardless_of_capture() {
        let s = SessionStopState::new();
        s.mark_capture_started();
        s.request_cancel();
        assert!(s.should_finish());
        assert!(s.is_cancel());
    }

    #[test]
    fn stop_then_cancel_becomes_a_cancel() {
        let s = SessionStopState::new();
        s.mark_capture_started();
        s.request_stop();
        s.request_cancel();
        assert!(s.is_cancel());
    }

    #[test]
    fn repeated_requests_are_idempotent() {
        let s = SessionStopState::new();
        s.mark_capture_started();
        assert!(!s.request_stop());
        assert!(!s.request_stop());
        assert!(s.should_finish());
    }
}
