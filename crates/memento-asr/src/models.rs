//! The ggml model registry.
//!
//! Replaces listnr's `TranscriptionModel.swift`, which names WhisperKit Core ML
//! model IDs. memento runs whisper.cpp, so the entries are ggml files from the
//! upstream Hugging Face repo instead.
//!
//! Everything here is sized for **CPU inference**, which is a much tighter
//! budget than the Neural Engine listnr targets. A model that cannot keep up
//! with a live conversation does not merely run late: the lane pipeline's
//! bounded queue drops its backlog, so speech is lost outright. Quality that
//! arrives too slowly is worse than slightly lower quality that keeps up.

use memento_core::LanguageMode;

const BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub id: &'static str,
    /// File name in the upstream repo, which is also the on-disk name.
    pub file: &'static str,
    pub size_mb: u32,
    pub multilingual: bool,
    /// Whisper's `translate` task. The turbo distillations were trained for
    /// transcription only and quietly return the source language instead of
    /// English, which is worse than failing.
    pub supports_translation: bool,
    pub note: &'static str,
}

impl Model {
    pub fn url(&self) -> String {
        format!("{BASE_URL}/{}", self.file)
    }
}

/// Ordered roughly by cost. `large-v3-turbo` is the interesting one for
/// memento's default language: it pairs large-v3's full encoder — where the
/// multilingual accuracy actually lives — with a four-layer decoder, so it costs
/// about what `medium` costs while transcribing German substantially better.
pub const REGISTRY: &[Model] = &[
    Model {
        id: "base.en",
        file: "ggml-base.en.bin",
        size_mb: 148,
        multilingual: false,
        supports_translation: false,
        note: "fastest; English only",
    },
    Model {
        id: "small",
        file: "ggml-small-q5_1.bin",
        size_mb: 190,
        multilingual: true,
        supports_translation: true,
        note: "fast fallback for slower CPUs; noticeably weaker on German",
    },
    Model {
        id: "medium",
        file: "ggml-medium-q5_0.bin",
        size_mb: 539,
        multilingual: true,
        supports_translation: true,
        note: "solid multilingual; slower decode than turbo for similar cost",
    },
    Model {
        id: "large-v3-turbo",
        file: "ggml-large-v3-turbo-q5_0.bin",
        size_mb: 574,
        multilingual: true,
        supports_translation: false,
        note: "best German per CPU second; transcription only",
    },
    Model {
        id: "large-v3",
        file: "ggml-large-v3-q5_0.bin",
        size_mb: 1080,
        multilingual: true,
        supports_translation: true,
        note: "highest quality; likely too slow to stay live on CPU",
    },
];

/// The model used when translation is requested and the language default cannot
/// translate.
pub const DEFAULT_TRANSLATION_MODEL: &str = "medium";

pub fn find(id: &str) -> Option<&'static Model> {
    REGISTRY.iter().find(|m| m.id == id)
}

pub fn translation_capable() -> impl Iterator<Item = &'static Model> {
    REGISTRY.iter().filter(|m| m.supports_translation)
}

/// The default model for a language.
///
/// German is memento's primary language, so it gets the best multilingual model
/// that still has a chance of keeping up on CPU rather than the small default
/// that suits English.
pub fn preferred_model_id(language: LanguageMode, translating: bool) -> &'static str {
    if translating {
        // The turbo default cannot translate, and picking it here would return
        // German text from a request for English with no error at all.
        return DEFAULT_TRANSLATION_MODEL;
    }
    match language {
        // An English-only model is both faster and more accurate on English than
        // any multilingual one of the same size.
        LanguageMode::En => "base.en",
        // Everything else is multilingual. Turbo is the pick for German, and the
        // same reasoning holds for the other non-English languages.
        _ => "large-v3-turbo",
    }
}

/// A warning when the chosen model cannot serve the chosen language, or `None`
/// when the pair is fine.
///
/// This combination fails in silence, and in the most confusing way available:
/// an English-only model set to German does not error, it transcribes German
/// speech *into English words*. The output looks like a bad transcript rather
/// than a misconfiguration.
pub fn compatibility_warning(
    model_id: &str,
    language: LanguageMode,
    translating: bool,
) -> Option<String> {
    if translating && language == LanguageMode::En {
        return Some(
            "translation does nothing with language 'en' — Whisper only translates into English. \
             Set the language actually being spoken, or turn translation off."
                .into(),
        );
    }

    let model = find(model_id)?;

    if translating && !model.supports_translation {
        return Some(format!(
            "{model_id} cannot translate; it returns the original language instead. \
             Use '{DEFAULT_TRANSLATION_MODEL}'."
        ));
    }
    if !model.multilingual && language != LanguageMode::En {
        return Some(format!(
            "{model_id} is English-only, so {} speech will be transcribed as English words. \
             Use 'large-v3-turbo' instead.",
            language.as_str()
        ));
    }
    if model.multilingual && language == LanguageMode::En {
        return Some(format!(
            "{model_id} is multilingual; for English, 'base.en' is smaller, faster and more accurate."
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registry_entry_is_findable_and_distinct() {
        for model in REGISTRY {
            assert_eq!(find(model.id), Some(model));
        }
        let mut ids: Vec<&str> = REGISTRY.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate model id");
    }

    #[test]
    fn urls_point_at_the_upstream_ggml_repo() {
        for model in REGISTRY {
            let url = model.url();
            assert!(url.starts_with("https://huggingface.co/ggerganov/whisper.cpp/"));
            assert!(url.ends_with(".bin"), "{url}");
        }
    }

    /// German is the language memento is actually used in, so its default must
    /// be a multilingual model — the English-only default would silently
    /// transcribe German into English words.
    #[test]
    fn german_default_is_multilingual() {
        let id = preferred_model_id(LanguageMode::De, false);
        let model = find(id).expect("german default must exist");
        assert!(
            model.multilingual,
            "{id} would transcribe German as English"
        );
    }

    #[test]
    fn german_default_beats_small_on_quality() {
        // The plan's original CPU-driven default was `small`; German is the
        // reason it is not the default any more.
        let id = preferred_model_id(LanguageMode::De, false);
        assert_ne!(id, "small");
        let chosen = find(id).unwrap();
        let small = find("small").unwrap();
        assert!(
            chosen.size_mb > small.size_mb,
            "a bigger encoder is the point"
        );
    }

    #[test]
    fn english_default_is_the_english_only_model() {
        assert_eq!(preferred_model_id(LanguageMode::En, false), "base.en");
    }

    /// The failure this guards is silent: a turbo model asked to translate
    /// returns the source language instead of English.
    #[test]
    fn translation_never_defaults_to_a_model_that_cannot_translate() {
        for language in LanguageMode::ALL {
            let id = preferred_model_id(language, true);
            assert!(
                find(id).unwrap().supports_translation,
                "{language:?} would silently not translate"
            );
        }
    }

    #[test]
    fn english_only_model_with_german_warns() {
        let warning = compatibility_warning("base.en", LanguageMode::De, false);
        assert!(warning.is_some_and(|w| w.contains("English-only")));
    }

    #[test]
    fn the_german_default_pairing_is_clean() {
        let id = preferred_model_id(LanguageMode::De, false);
        assert_eq!(compatibility_warning(id, LanguageMode::De, false), None);
    }

    #[test]
    fn translating_with_a_turbo_model_warns() {
        let warning = compatibility_warning("large-v3-turbo", LanguageMode::De, true);
        assert!(warning.is_some_and(|w| w.contains("cannot translate")));
    }

    #[test]
    fn translating_english_into_english_warns() {
        assert!(compatibility_warning("medium", LanguageMode::En, true).is_some());
    }

    #[test]
    fn at_least_one_model_can_translate() {
        assert!(translation_capable().next().is_some());
        assert!(
            find(DEFAULT_TRANSLATION_MODEL)
                .unwrap()
                .supports_translation
        );
    }
}
