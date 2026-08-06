# memento

Local meeting transcriber for Windows. Captures your microphone and one
application's audio as two separate lanes, transcribes both on-device, and
labels You plus remote Speaker 1..N. Nothing leaves your machine, and nothing
joins the call.

## Why it is invisible to the meeting

memento does not join meetings as a participant or bot. It reads audio through
**WASAPI loopback** — a passive read of what Windows is already rendering to
your speakers. The meeting application is not involved and cannot tell: no extra
attendee, no recording indicator, no notification.

It goes one better than a normal recorder by targeting a **single process tree**
(`AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`), so it captures Teams without
also capturing Spotify, notification sounds, or a video in another tab.

## Status

The whole pipeline works end to end — lanes, transcription, diarization, merge,
Markdown export, and the desktop app. The one unfinished join is between the
Windows capture backends and the session, which is exactly the part that cannot
be tested without a Windows machine.

| crate | what it is | state |
|---|---|---|
| `memento-core` | lanes, VAD, transcript rules, decode tuning, merge, export — no OS deps | working, 130 tests |
| `memento-audio` | WASAPI mic + per-process loopback | written; compiles for Windows, **unverified on hardware** |
| `memento-asr` | whisper.cpp via `whisper-rs`, CPU | working end-to-end, 32 tests |
| `memento-diar` | speaker diarization via `sherpa-rs` | working end-to-end, 8 tests |
| `memento-session` | one meeting: lanes, diarization, merge, export | working end-to-end, 8 tests |
| `memento-app` (`src-tauri/`) | the desktop app | builds; capture not yet wired |

## Language and models

German is the primary language, and that decides the default model. The obvious
CPU-friendly pick, `small`, is noticeably weaker on German, so the default is
`large-v3-turbo`: large-v3's full encoder — where the multilingual accuracy
lives — with a four-layer decoder, so it costs about what `medium` costs and
transcribes German better.

Measured on an Apple Silicon Mac, CPU only, on 16 s of German speech:

| model | size | realtime factor | notes |
|---|---|---|---|
| `small` | 190 MB | 0.07x | "Integrations-Tests", "Kangelog" |
| `large-v3-turbo` | 574 MB | 0.27x | "Integrationstests", cleaner punctuation |

Two caveats on those numbers, both of which make them optimistic:

- The audio is macOS TTS (`say -v Anna`) — clean, no room noise, no crosstalk.
  Real meeting audio is harder.
- A Windows laptop CPU is several times slower than this Mac. A 0.27x factor
  here could land near 1.0x there, which is the point at which the lane's
  bounded queue starts dropping speech. `transcribe-file` prints the factor and
  says plainly whether it keeps up; run it on the target machine before trusting
  a model choice.

## Building

`memento-core` builds and tests anywhere:

```
cargo test -p memento-core
```

The Windows lanes can be **type-checked** from macOS or Linux, which is enough to
develop them, but they cannot be linked or run there:

```
rustup target add x86_64-pc-windows-msvc
cargo check -p memento-audio --target x86_64-pc-windows-msvc --all-targets
```

CI covers the gap: the `windows` job on a `windows-latest` runner is the only
place the capture code is actually compiled *and linked* against the real Win32
import libraries, which `cargo check --target` never does. It still cannot prove
capture works — GitHub's runners have no audio device.

### The capture spike

Before anything is built on top of the loopback layer, it has to be proven on a
real machine:

```
cargo run --bin loopback-spike -- <pid> 10 spike.wav
```

Find the PID with `tasklist | findstr Teams`, and pick the parent process — the
capture includes its whole tree, which is what makes it work at all.

The acceptance test: play a video in your browser and music in Spotify at the
same time, target only the browser, and confirm `spike.wav` contains only the
video.

### Transcribing a file

Everything from the VAD onwards can be exercised without Windows and without a
meeting — this runs the real lane pipeline, just from a WAV instead of a capture
device:

```
cargo run --release --bin transcribe-file -- speech.wav --model large-v3-turbo --lang de
```

The WAV must be 16 kHz mono. To make German test audio on macOS:

```
say -v Anna -o de.aiff "Guten Morgen zusammen…"
afconvert -f WAVE -d LEI16@16000 -c 1 de.aiff de.wav
```

## Diarization

Pyannote segmentation plus WeSpeaker VoxCeleb embeddings, both ONNX through
sherpa-onnx — the same Pyannote lineage listnr reaches via SpeakerKit. Runs once
at the end of a session over the remote lane, so remote speakers can only be
named after the meeting. Speaker embeddings model voice timbre rather than
words, so an English-trained model separates German speakers fine.

```
cargo run --release --bin diarize-file -- two-speakers.wav --speakers 2
```

On a synthetic two-speaker German conversation (17.7 s, two macOS voices
alternating), diarized in 2.0 s:

- **Segmentation — when the speaker changes — is solid.** Boundaries came out at
  3.64 / 9.08 / 12.84 s against actual turn changes at 3.64 / 9.08 / 12.88 s,
  and were identical across every run.
- **Clustering — who is who — is not verified, and on this material it is not
  stable.** Subtracting one LSB from every sample (an inaudible change, about
  −90 dBFS) flips the result from two speakers to one. So is round-tripping the
  same audio through a 16-bit WAV, which is what the session's lane spill does.

That instability means no accuracy number can be read off this fixture — a run
that looks mostly right is as likely to be luck as skill. Two TTS voices
evidently sit close enough together in embedding space that the clustering is on
a knife edge; real voices are much further apart, and should be far more stable.
**Diarization accuracy is therefore an open question until it is run on a real
recorded call.**

One related bug is fixed: with no `--speakers` hint, sherpa-rs defaults the
cluster count to `4`, so a two-person call would have been split four ways. The
code now passes `-1` explicitly, which is what selects threshold-based
clustering. Separately, `--speakers 2` did *not* visibly pin the speaker count in
testing, despite `fast-clustering.cc` appearing to honour a positive count —
unresolved, and worth chasing with real audio.

## The app

```
cargo tauri dev          # or: cargo run -p memento-app
```

Source picker, live level meters per lane, a transcript that fills in as the
meeting runs, and a stop button that saves. Everything under it is
`memento-session`, which is driven by a `CaptureSource` — a microphone, one
application's output, or a WAV file.

**One gap, deliberately left visible.** The Windows capture backends are written
and compile, but they are not yet adapted to the `CaptureSource` trait, so
pressing Start reports that plainly instead of recording silence. Wiring that
adapter is the last step, and it is the step that cannot be tested without a
Windows machine.

The whole path below capture is already exercisable:

```
cargo run --release --bin run-session -- --mic you.wav --system others.wav --speakers 2
```

That runs the real lanes, the real models, diarization, the merge and the
Markdown export — the same code a live session runs, with the audio coming from
files.

## Development notes

`sherpa-rs` ships a prebuilt `libonnxruntime` but sets no rpath, so on macOS the
diarization binaries need it pointed out explicitly:

```
DYLD_LIBRARY_PATH=target/release ./target/release/diarize-file …
```

Not an issue on Windows, where the DLL sits next to the executable.

## Credit

memento is derived from [listnr](https://github.com/rokib16x/listnr) by Rokibul
Hasan (MIT). listnr is the macOS original: it established the two-lane design,
the "never mix before ASR" rule, the energy VAD and its three rejection floors,
the repetition guard, and the sensitivity scaling — all of which `memento-core`
ports directly, tests included.

None of listnr's platform layer could come along, because it is built entirely
on Apple frameworks that have no Windows equivalent: ScreenCaptureKit,
AVAudioEngine, WhisperKit (Core ML) and SpeakerKit. Those are replaced with
WASAPI, whisper.cpp and sherpa-onnx respectively.

listnr's MIT license is included as `LICENSE-listnr`.
