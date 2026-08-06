//! The two ONNX models diarization needs, and fetching them.
//!
//! Diarization is two stages: a **segmentation** model says when speech is
//! happening and how it overlaps, and an **embedding** model turns each stretch
//! into a voice fingerprint that gets clustered. sherpa-onnx wants both.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type ProgressSink<'a> = &'a mut dyn FnMut(u64, Option<u64>);

pub struct DiarModel {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub approx_mb: u32,
}

/// The same Pyannote lineage listnr reaches through SpeakerKit, exported to
/// ONNX — so the diarization quality should be comparable rather than merely
/// similar in spirit.
pub const SEGMENTATION: DiarModel = DiarModel {
    name: "pyannote-segmentation-3.0",
    file: "pyannote-segmentation-3-0.onnx",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
    approx_mb: 6,
};

/// Speaker embeddings are near enough language-independent — they model voice
/// timbre, not words — so an English-trained VoxCeleb model separates German
/// speakers perfectly well.
pub const EMBEDDING: DiarModel = DiarModel {
    name: "wespeaker-voxceleb-CAM++",
    file: "wespeaker_en_voxceleb_CAM++.onnx",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
    approx_mb: 28,
};

/// Where the diarization models live, alongside the whisper ones.
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

/// ONNX files begin with a protobuf field header; byte 0 is always 0x08. It is a
/// weak check, but it is enough to catch the failure that actually happens — an
/// HTML error page or a redirect body saved under a `.onnx` name.
pub fn looks_like_onnx(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut first = [0u8; 1];
    file.read_exact(&mut first).is_ok()
        && first[0] == 0x08
        && fs::metadata(path).map(|m| m.len() > 1_000).unwrap_or(false)
}

/// Fetch `model` into `dir` unless it is already there. Downloads to a temporary
/// name and renames on success, so an interrupted download is never mistaken for
/// a usable model.
pub fn ensure(model: &DiarModel, dir: &Path, progress: ProgressSink<'_>) -> Result<PathBuf, Error> {
    let final_path = dir.join(model.file);
    if final_path.exists() && looks_like_onnx(&final_path) {
        return Ok(final_path);
    }

    fs::create_dir_all(dir)?;
    let partial = final_path.with_extension("partial");

    let response = ureq::get(model.url).call()?;
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

    if !looks_like_onnx(&partial) {
        let _ = fs::remove_file(&partial);
        return Err(format!("downloaded file is not an ONNX model: {}", model.url).into());
    }

    fs::rename(&partial, &final_path)?;
    Ok(final_path)
}

/// Both models, in the order [`crate::SherpaDiarizer`] wants them.
pub fn ensure_all(dir: &Path, progress: ProgressSink<'_>) -> Result<(PathBuf, PathBuf), Error> {
    let segmentation = ensure(&SEGMENTATION, dir, progress)?;
    let embedding = ensure(&EMBEDDING, dir, progress)?;
    Ok((segmentation, embedding))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_models_are_named_and_sized() {
        for model in [&SEGMENTATION, &EMBEDDING] {
            assert!(model.url.starts_with("https://"));
            assert!(model.file.ends_with(".onnx"));
            assert!(model.approx_mb > 0);
        }
    }

    #[test]
    fn models_share_the_directory_with_the_whisper_ones() {
        let dir = default_model_dir();
        assert!(
            dir.ends_with("memento/models") || dir.ends_with("memento\\models"),
            "{dir:?}"
        );
    }

    #[test]
    fn an_html_error_page_is_not_taken_for_a_model() {
        let dir = std::env::temp_dir().join("memento-onnx-test");
        fs::create_dir_all(&dir).unwrap();

        let html = dir.join("bad.onnx");
        fs::write(&html, b"<!DOCTYPE html><html>404 not found</html>").unwrap();
        assert!(!looks_like_onnx(&html));

        // Right magic byte but far too small to be a model.
        let stub = dir.join("stub.onnx");
        fs::write(&stub, [0x08u8, 0x01]).unwrap();
        assert!(!looks_like_onnx(&stub));

        let plausible = dir.join("good.onnx");
        let mut body = vec![0x08u8];
        body.extend(std::iter::repeat_n(0x00, 2_000));
        fs::write(&plausible, body).unwrap();
        assert!(looks_like_onnx(&plausible));

        let _ = fs::remove_dir_all(&dir);
    }
}
