//! Model download, with progress.
//!
//! Ported in spirit from listnr's `DownloadProgressPrinter.swift`. The reason it
//! exists at all: loading the model lazily on the first utterance makes the
//! opening minute of a meeting vanish while a 574 MB file comes down.

use crate::models::{self, Model};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Bytes downloaded and the total, when the server declares one.
pub type ProgressSink<'a> = &'a mut dyn FnMut(u64, Option<u64>);

/// Where models live. `%LOCALAPPDATA%\memento\models` on Windows, the XDG-ish
/// equivalent elsewhere so the crate stays testable on the dev machine.
pub fn default_model_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("memento")
        .join("models")
}

/// Path a model occupies, downloaded or not.
pub fn model_path(model: &Model, dir: &Path) -> PathBuf {
    dir.join(model.file)
}

/// Fetch `model` into `dir` unless it is already there. Returns its path.
///
/// Downloads to a temporary name and renames on success, so an interrupted
/// download can never be mistaken for a usable model on the next run.
pub fn ensure_model(
    model: &Model,
    dir: &Path,
    progress: ProgressSink<'_>,
) -> Result<PathBuf, Error> {
    let final_path = model_path(model, dir);
    if final_path.exists() && crate::whisper::looks_like_ggml_model(&final_path) {
        return Ok(final_path);
    }

    fs::create_dir_all(dir)?;
    let partial = final_path.with_extension("partial");

    let response = ureq::get(&model.url()).call()?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    let mut reader = response.into_body().into_reader();
    let mut file = fs::File::create(&partial)?;
    let mut buffer = vec![0u8; 1 << 16];
    let mut downloaded = 0u64;

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        progress(downloaded, total);
    }
    file.flush()?;
    drop(file);

    if !crate::whisper::looks_like_ggml_model(&partial) {
        // Most often an HTML error page saved under a .bin name, which would
        // otherwise fail much later inside whisper.cpp with nothing useful said.
        let _ = fs::remove_file(&partial);
        return Err(format!("downloaded file is not a ggml model: {}", model.url()).into());
    }

    fs::rename(&partial, &final_path)?;
    Ok(final_path)
}

/// Resolve a model id and make sure it is on disk.
pub fn ensure_model_id(id: &str, dir: &Path, progress: ProgressSink<'_>) -> Result<PathBuf, Error> {
    let model = models::find(id).ok_or_else(|| -> Error {
        let known: Vec<&str> = models::REGISTRY.iter().map(|m| m.id).collect();
        format!("unknown model '{id}'; known models: {}", known.join(", ")).into()
    })?;
    ensure_model(model, dir, progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dir_is_under_a_memento_folder() {
        let dir = default_model_dir();
        assert!(
            dir.ends_with("memento/models") || dir.ends_with("memento\\models"),
            "{dir:?}"
        );
    }

    #[test]
    fn model_path_uses_the_upstream_file_name() {
        let model = models::find("large-v3-turbo").unwrap();
        let path = model_path(model, Path::new("/models"));
        assert!(path.ends_with("ggml-large-v3-turbo-q5_0.bin"));
    }

    #[test]
    fn unknown_model_id_lists_the_known_ones() {
        let dir = std::env::temp_dir();
        let err = ensure_model_id("whisper-enormous", &dir, &mut |_, _| {})
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown model"));
        assert!(err.contains("large-v3-turbo"), "{err}");
    }
}
