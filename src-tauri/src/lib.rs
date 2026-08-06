//! The desktop app: source picker, live meters and transcript, stop button.
//!
//! Everything here is glue. The session logic lives in `memento-session`, and
//! this crate's job is to expose it as Tauri commands and to turn
//! [`SessionEvent`]s into events the web view can render.

use memento_core::{LanguageMode, MicSensitivity};
use memento_session::{SessionConfig, SessionEvent, StopHandle, clock, run_session};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

/// The two lanes a session captures from.
type LanePair = (
    Box<dyn memento_session::CaptureSource>,
    Box<dyn memento_session::CaptureSource>,
);

/// One selectable audio source in the picker.
#[derive(Debug, Clone, Serialize)]
pub struct SourceOption {
    pub pid: u32,
    pub label: String,
    pub detail: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StartRequest {
    /// Which process tree to capture. `None` means the whole output device.
    pub pid: Option<u32>,
    pub language: String,
    pub model_id: Option<String>,
    pub sensitivity: String,
    pub speakers: Option<i32>,
    pub diarize: bool,
    pub output_dir: Option<String>,
}

#[derive(Default)]
pub struct AppState {
    running: Mutex<Option<StopHandle>>,
}

/// Applications currently playing audio.
///
/// Empty is a normal answer, not an error: nothing is playing yet. The UI says
/// so and offers the whole-device fallback rather than showing a failure.
#[tauri::command]
fn list_sources() -> Result<Vec<SourceOption>, String> {
    #[cfg(windows)]
    {
        memento_audio::list_audio_sources()
            .map(|sources| {
                sources
                    .into_iter()
                    .map(|s| SourceOption {
                        pid: s.pid,
                        label: s.display_name,
                        detail: format!("{} · pid {}", s.process_name, s.pid),
                    })
                    .collect()
            })
            .map_err(|e| e.to_string())
    }
    #[cfg(not(windows))]
    {
        // Per-process capture is a WASAPI feature; on the development machine
        // there is nothing to enumerate. Say so plainly rather than pretending.
        Ok(Vec::new())
    }
}

#[tauri::command]
fn list_models() -> Vec<serde_json::Value> {
    memento_asr::models::REGISTRY
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "sizeMb": m.size_mb,
                "multilingual": m.multilingual,
                "note": m.note,
            })
        })
        .collect()
}

#[tauri::command]
fn default_model_for(language: String) -> String {
    let language = LanguageMode::parse(&language).unwrap_or(LanguageMode::De);
    memento_asr::models::preferred_model_id(language, false).to_string()
}

#[tauri::command]
fn is_capture_supported() -> bool {
    cfg!(windows)
}

#[tauri::command]
fn start_session(
    app: AppHandle,
    state: State<'_, AppState>,
    request: StartRequest,
) -> Result<(), String> {
    {
        let running = state.running.lock().unwrap();
        if running.is_some() {
            return Err("a session is already running".into());
        }
    }

    let stop = StopHandle::new();
    *state.running.lock().unwrap() = Some(stop.clone());

    let language = LanguageMode::parse(&request.language).unwrap_or(LanguageMode::De);
    let sensitivity = MicSensitivity::parse(&request.sensitivity).unwrap_or(MicSensitivity::Normal);
    let work_dir = std::env::temp_dir().join("memento").join("session");
    let output_dir = request
        .output_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_output_dir);

    let config = SessionConfig {
        language,
        model_id: request.model_id,
        sensitivity,
        speakers: request.speakers,
        diarize: request.diarize,
        output_dir,
        work_dir,
        keep_audio: false,
    };

    let emitter = app.clone();
    let events: memento_session::EventSink = Arc::new(move |event| {
        let _ = emitter.emit("session", describe(&event));
    });

    let app_for_thread = app.clone();
    let pid = request.pid;
    // Capture blocks; the UI thread must stay free for the stop button.
    std::thread::spawn(move || {
        let (mic, system) = match build_sources(pid) {
            Ok(pair) => pair,
            Err(e) => {
                let _ = app_for_thread.emit(
                    "session",
                    serde_json::json!({ "kind": "error", "message": e }),
                );
                clear_running(&app_for_thread);
                return;
            }
        };

        let (label, stamp) = clock::now_labels();
        let result = run_session(&config, mic, system, stop, events, &label, &stamp);
        match result {
            Ok(outcome) => {
                let _ = app_for_thread.emit(
                    "session",
                    serde_json::json!({
                        "kind": "finished",
                        "path": outcome.transcript_path.to_string_lossy(),
                        "lines": outcome.lines.len(),
                        "markdown": outcome.markdown,
                    }),
                );
            }
            Err(e) => {
                let _ = app_for_thread.emit(
                    "session",
                    serde_json::json!({ "kind": "error", "message": e.to_string() }),
                );
            }
        }
        clear_running(&app_for_thread);
    });

    Ok(())
}

#[tauri::command]
fn stop_session(state: State<'_, AppState>) -> Result<(), String> {
    match state.running.lock().unwrap().as_ref() {
        Some(handle) => {
            // A graceful stop, never a cancel: the transcript is kept. Cancelling
            // after capture began is what discarded finished meetings in listnr.
            handle.stop();
            Ok(())
        }
        None => Err("no session is running".into()),
    }
}

fn clear_running(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        *state.running.lock().unwrap() = None;
    }
}

#[cfg(windows)]
fn build_sources(pid: Option<u32>) -> Result<LanePair, String> {
    let _ = pid;
    // The Windows capture sources are written but not yet wired through the
    // CaptureSource trait; that adapter is the next step and is deliberately not
    // faked here, because a UI that silently records nothing is worse than one
    // that refuses.
    Err("Windows capture is not wired to the session yet".into())
}

#[cfg(not(windows))]
fn build_sources(_pid: Option<u32>) -> Result<LanePair, String> {
    Err("audio capture requires Windows; this build can only replay files".into())
}

fn default_output_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    } else {
        std::env::var_os("HOME").map(PathBuf::from)
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("Documents")
        .join("memento")
}

/// Session events as the web view wants them.
fn describe(event: &SessionEvent) -> serde_json::Value {
    match event {
        SessionEvent::Preparing(what) => {
            serde_json::json!({ "kind": "preparing", "message": what })
        }
        SessionEvent::DownloadProgress { done, total } => serde_json::json!({
            "kind": "download",
            "done": done,
            "total": total,
        }),
        SessionEvent::Started => serde_json::json!({ "kind": "started" }),
        SessionEvent::Line(line) => serde_json::json!({
            "kind": "line",
            "speaker": line.speaker,
            "clock": memento_core::format_clock(line.start_seconds),
            "text": line.text,
        }),
        SessionEvent::Level { lane, rms } => serde_json::json!({
            "kind": "level",
            "lane": match lane {
                memento_session::Lane::You => "you",
                memento_session::Lane::Others => "others",
            },
            "rms": rms,
        }),
        SessionEvent::Warning(message) => {
            serde_json::json!({ "kind": "warning", "message": message })
        }
        SessionEvent::Finalizing(what) => {
            serde_json::json!({ "kind": "finalizing", "message": what })
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            list_sources,
            list_models,
            default_model_for,
            is_capture_supported,
            start_session,
            stop_session,
        ])
        .run(tauri::generate_context!())
        .expect("error while running memento");
}

#[cfg(test)]
mod tests {
    // Not `use super::*`: `#[tauri::command]` generates a macro per command, and
    // a glob import collides with every one of them.
    use super::{SessionEvent, default_model_for, default_output_dir, describe, list_models};
    use memento_session::Lane;

    #[test]
    fn the_model_list_reaches_the_ui_intact() {
        let models = list_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m["id"] == "large-v3-turbo"));
    }

    #[test]
    fn german_gets_the_multilingual_default() {
        assert_eq!(default_model_for("de".into()), "large-v3-turbo");
    }

    #[test]
    fn an_unknown_language_falls_back_to_german_rather_than_english() {
        // German is the primary language; defaulting to an English-only model
        // would silently transcribe German as English words.
        let id = default_model_for("klingon".into());
        assert!(memento_asr::models::find(&id).unwrap().multilingual);
    }

    #[test]
    fn session_events_serialise_with_a_kind_the_ui_can_switch_on() {
        let cases = [
            SessionEvent::Started,
            SessionEvent::Warning("careful".into()),
            SessionEvent::Level {
                lane: Lane::You,
                rms: 0.1,
            },
            SessionEvent::Line(memento_core::TranscriptLine::new("You", 61.0, "hallo")),
        ];
        for event in cases {
            let json = describe(&event);
            assert!(json["kind"].is_string(), "{json}");
        }
    }

    #[test]
    fn a_line_event_carries_a_preformatted_clock() {
        let json = describe(&SessionEvent::Line(memento_core::TranscriptLine::new(
            "Speaker 1",
            61.0,
            "hallo",
        )));
        assert_eq!(json["clock"], "01:01");
        assert_eq!(json["speaker"], "Speaker 1");
    }

    #[test]
    fn the_output_directory_is_under_the_users_documents() {
        let dir = default_output_dir();
        assert!(dir.ends_with("Documents/memento") || dir.ends_with("Documents\\memento"));
    }
}
