//! Bounded drop-oldest queue.
//!
//! Stands in for Swift's `AsyncStream(bufferingPolicy: .bufferingNewest(n))`,
//! which listnr relies on for the two properties that make the lane pipeline
//! correct: strict FIFO between stages, and a hard memory bound with an
//! explicit drop count when transcription cannot keep up. Dropping the *oldest*
//! is what keeps the live transcript current rather than lagging further behind
//! with every overloaded second.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

pub struct BoundedQueue<T> {
    inner: Mutex<Inner<T>>,
    ready: Condvar,
    /// Signalled when an item leaves, so `push_blocking` can wait for room.
    drained: Condvar,
    capacity: usize,
}

struct Inner<T> {
    items: VecDeque<T>,
    closed: bool,
    drops: usize,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "a zero-capacity queue drops everything");
        Self {
            inner: Mutex::new(Inner {
                items: VecDeque::new(),
                closed: false,
                drops: 0,
            }),
            ready: Condvar::new(),
            drained: Condvar::new(),
            capacity,
        }
    }

    /// Never blocks — safe to call from an audio callback, which is the whole
    /// point. Returns false if the item was dropped (queue full or closed).
    pub fn push(&self, item: T) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return false;
        }
        let dropped = if inner.items.len() == self.capacity {
            inner.items.pop_front();
            inner.drops += 1;
            true
        } else {
            false
        };
        inner.items.push_back(item);
        drop(inner);
        self.ready.notify_one();
        !dropped
    }

    /// Push, waiting for room instead of dropping. For sources that can be
    /// slowed down — replaying a file, feeding tests — where dropping would
    /// corrupt the result rather than protect it. Never use this from a live
    /// audio callback: blocking there stalls the capture thread and loses audio
    /// at the driver instead, which is worse. Returns false only if closed.
    pub fn push_blocking(&self, item: T) -> bool {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if inner.closed {
                return false;
            }
            if inner.items.len() < self.capacity {
                inner.items.push_back(item);
                drop(inner);
                self.ready.notify_one();
                return true;
            }
            inner = self.drained.wait(inner).unwrap();
        }
    }

    /// Blocks until an item is available or the queue is closed and drained.
    pub fn pop(&self) -> Option<T> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(item) = inner.items.pop_front() {
                drop(inner);
                self.drained.notify_one();
                return Some(item);
            }
            if inner.closed {
                return None;
            }
            inner = self.ready.wait(inner).unwrap();
        }
    }

    /// Idempotent. Wakes every blocked consumer; already-queued items still
    /// drain, so a normal shutdown loses nothing.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        drop(inner);
        self.ready.notify_all();
        // A blocked producer must not wait for room that will never come.
        self.drained.notify_all();
    }

    /// Close *and* discard anything still queued, for abandoning a session.
    pub fn close_and_clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        inner.items.clear();
        drop(inner);
        self.ready.notify_all();
        self.drained.notify_all();
    }

    pub fn drops(&self) -> usize {
        self.inner.lock().unwrap().drops
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn preserves_fifo_order() {
        let q = BoundedQueue::new(8);
        for i in 0..5 {
            assert!(q.push(i));
        }
        q.close();
        let got: Vec<i32> = std::iter::from_fn(|| q.pop()).collect();
        assert_eq!(got, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn drops_oldest_when_full_and_counts_it() {
        let q = BoundedQueue::new(3);
        for i in 0..5 {
            q.push(i);
        }
        q.close();
        let got: Vec<i32> = std::iter::from_fn(|| q.pop()).collect();
        assert_eq!(got, [2, 3, 4], "the newest items survive");
        assert_eq!(q.drops(), 2);
    }

    #[test]
    fn pop_returns_none_once_closed_and_drained() {
        let q = BoundedQueue::new(2);
        q.push(1);
        q.close();
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn close_unblocks_a_waiting_consumer() {
        let q = Arc::new(BoundedQueue::<i32>::new(2));
        let consumer = {
            let q = Arc::clone(&q);
            thread::spawn(move || q.pop())
        };
        q.close();
        assert_eq!(consumer.join().unwrap(), None, "must not hang");
    }

    #[test]
    fn close_is_idempotent_and_rejects_later_pushes() {
        let q = BoundedQueue::new(2);
        q.close();
        q.close();
        assert!(!q.push(1));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn push_blocking_waits_for_room_instead_of_dropping() {
        let q = Arc::new(BoundedQueue::new(2));
        for i in 0..2 {
            q.push(i);
        }
        let producer = {
            let q = Arc::clone(&q);
            thread::spawn(move || q.push_blocking(99))
        };
        // Make room. Until this happens the producer must still be parked.
        thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(q.pop(), Some(0));
        assert!(producer.join().unwrap());
        assert_eq!(q.drops(), 0, "blocking push must never drop");

        q.close();
        let rest: Vec<i32> = std::iter::from_fn(|| q.pop()).collect();
        assert_eq!(rest, [1, 99]);
    }

    #[test]
    fn push_blocking_gives_up_when_the_queue_closes() {
        let q = Arc::new(BoundedQueue::new(1));
        q.push(0);
        let producer = {
            let q = Arc::clone(&q);
            thread::spawn(move || q.push_blocking(1))
        };
        thread::sleep(std::time::Duration::from_millis(20));
        q.close();
        assert!(!producer.join().unwrap(), "must not hang forever");
    }

    #[test]
    fn close_and_clear_discards_queued_items() {
        let q = BoundedQueue::new(4);
        q.push(1);
        q.push(2);
        q.close_and_clear();
        assert_eq!(q.pop(), None);
    }
}
