//! Which applications are currently playing audio.
//!
//! This is the source picker: the user chooses "Microsoft Teams", not a PID.
//! macOS listnr has no equivalent because ScreenCaptureKit captures everything
//! at once; on Windows the whole point is picking one.

#![cfg(windows)]

use std::collections::HashMap;
use std::path::Path;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Media::Audio::{
    AudioSessionStateExpired, IAudioSessionControl2, IAudioSessionManager2, IMMDeviceEnumerator,
    MMDeviceEnumerator, eConsole, eRender,
};
use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::core::Interface;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSource {
    pub pid: u32,
    /// Executable name, e.g. `ms-teams.exe`.
    pub process_name: String,
    /// What to show the user — the friendly name where Windows has one.
    pub display_name: String,
}

/// Processes currently rendering audio on the default output device.
///
/// Sessions are deduplicated by PID: a browser or Teams opens several audio
/// sessions at once, and listing each one separately would offer the user four
/// identical-looking "Chrome" entries.
pub fn list_audio_sources() -> Result<Vec<AudioSource>, Error> {
    let mut by_pid: HashMap<u32, AudioSource> = HashMap::new();

    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let manager: IAudioSessionManager2 = device.Activate(CLSCTX_ALL, None)?;
        let sessions = manager.GetSessionEnumerator()?;

        for index in 0..sessions.GetCount()? {
            let Ok(control) = sessions.GetSession(index) else {
                continue;
            };
            let Ok(control2) = control.cast::<IAudioSessionControl2>() else {
                continue;
            };
            // Expired sessions belong to processes that have already gone.
            //
            // Compared with `==`, not matched: in a pattern, the constant's name
            // binds as a fresh variable and matches every `Ok(_)`, which silently
            // skips every session and leaves the picker permanently empty.
            if control2.GetState() == Ok(AudioSessionStateExpired) {
                continue;
            }
            let Ok(pid) = control2.GetProcessId() else {
                continue;
            };
            // PID 0 is the system mix, not something a user picks.
            if pid == 0 || control2.IsSystemSoundsSession().is_ok() {
                continue;
            }

            let process_name = process_name(pid).unwrap_or_else(|| format!("pid {pid}"));
            let display_name = control2
                .GetDisplayName()
                .ok()
                .and_then(|n| n.to_string().ok())
                .filter(|n| !n.trim().is_empty())
                // Windows leaves the display name empty for most applications,
                // so the executable is the usual answer, not the fallback.
                .unwrap_or_else(|| friendly_from_process(&process_name));

            by_pid.entry(pid).or_insert(AudioSource {
                pid,
                process_name,
                display_name,
            });
        }
    }

    let mut sources: Vec<AudioSource> = by_pid.into_values().collect();
    sources.sort_by_key(|s| s.display_name.to_lowercase());
    Ok(sources)
}

fn process_name(pid: u32) -> Option<String> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buffer = [0u16; 260];
        let mut length = buffer.len() as u32;
        let result = QueryFullProcessImageNameW(
            handle,
            windows::Win32::System::Threading::PROCESS_NAME_FORMAT(0),
            windows::core::PWSTR(buffer.as_mut_ptr()),
            &mut length,
        );
        let _ = CloseHandle(handle);
        result.ok()?;

        let full = String::from_utf16_lossy(&buffer[..length as usize]);
        Path::new(&full)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    }
}

/// `ms-teams.exe` → `Ms Teams`. Crude, but it beats showing a filename, and the
/// user is picking from a short list they already recognise.
fn friendly_from_process(process_name: &str) -> String {
    let stem = process_name
        .strip_suffix(".exe")
        .unwrap_or(process_name)
        .replace(['-', '_'], " ");
    let mut out = String::with_capacity(stem.len());
    for (i, word) in stem.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    if out.is_empty() {
        process_name.to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_names_become_readable_labels() {
        assert_eq!(friendly_from_process("ms-teams.exe"), "Ms Teams");
        assert_eq!(friendly_from_process("chrome.exe"), "Chrome");
        assert_eq!(friendly_from_process("Zoom.exe"), "Zoom");
        assert_eq!(friendly_from_process("some_app.exe"), "Some App");
    }

    #[test]
    fn a_name_without_an_extension_still_works() {
        assert_eq!(friendly_from_process("slack"), "Slack");
    }

    #[test]
    fn an_empty_name_falls_back_rather_than_vanishing() {
        assert_eq!(friendly_from_process(""), "");
        assert_eq!(friendly_from_process(".exe"), ".exe");
    }
}
