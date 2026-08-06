//! Language selection and the writing system it implies. Ported from the
//! `ScriptClass` half of listnr `DecodeTuning.swift` and `LanguageMode` from
//! `SessionOptions.swift`.
//!
//! Model selection (listnr's `preferredModelID`) deliberately does *not* come
//! along: it names WhisperKit Core ML model IDs, and memento runs ggml models
//! through whisper.cpp. That mapping lives in `memento-asr` with the registry it
//! refers to.

/// The writing system a language uses, as far as Whisper's confidence signals
/// are concerned.
///
/// This exists because those signals are **not comparable across scripts**, and
/// treating them as if they were is what made Bangla and Hindi look broken while
/// English looked perfect:
///
/// - `avg_log_prob` is an average over *tokens*. Whisper's BPE vocabulary is
///   English-centric, so Bengali and Devanagari fall back to byte-level pieces
///   and cost several times more tokens per word, each individually less
///   confident. Correct Bangla routinely averages around −1.2, well below the
///   −0.9 that means "probably a hallucination" in English.
/// - `compression_ratio` is the text's length over its compressed length. Indic
///   and Han text is three bytes per character drawn from a small repertoire, so
///   it compresses far better than English *before* any repetition exists. An
///   ordinary Bangla sentence lands near 3.0, past the 2.6 that flags a
///   degenerate loop in English.
///
/// So the same numbers that correctly reject English silence-hallucinations
/// reject real speech in every other script. The thresholds are keyed on script
/// instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptClass {
    Latin,
    /// Indic abugidas: Bengali, Devanagari, Tamil, Telugu, and relatives.
    Indic,
    /// Han, Kana, and Hangul: no word spacing, few characters per word.
    Cjk,
    /// Not yet known (auto-detect before its first detection) or not one of the
    /// families above. Takes the widest gate, because the failure mode of
    /// guessing narrow is silently dropping a whole language.
    Unknown,
}

const LATIN_CODES: &[&str] = &[
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "da", "no", "nb", "nn", "fi", "pl", "cs", "sk",
    "sl", "hr", "hu", "ro", "tr", "id", "ms", "vi", "ca", "gl", "af", "sw", "tl", "et", "lv", "lt",
    "cy", "is", "eu", "la",
];

const INDIC_CODES: &[&str] = &[
    "bn", "hi", "ta", "te", "mr", "gu", "pa", "ne", "kn", "ml", "si", "as", "or", "sa", "mai", "bh",
];

const CJK_CODES: &[&str] = &["ja", "zh", "ko", "yue"];

impl ScriptClass {
    /// Classify a Whisper language code. Accepts the tagged forms that come back
    /// from language detection too (`zh-Hans`, `en_US`).
    pub fn from_language_code(code: Option<&str>) -> Self {
        let Some(code) = code else {
            return Self::Unknown;
        };
        let lowered = code.to_lowercase();
        let base = lowered.split(['-', '_']).next().unwrap_or("");

        if LATIN_CODES.contains(&base) {
            Self::Latin
        } else if INDIC_CODES.contains(&base) {
            Self::Indic
        } else if CJK_CODES.contains(&base) {
            Self::Cjk
        } else {
            Self::Unknown
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LanguageMode {
    #[default]
    En,
    Auto,
    Bn,
    Hi,
    Es,
    Fr,
    De,
    Ja,
    Zh,
}

impl LanguageMode {
    pub const ALL: [LanguageMode; 9] = [
        Self::En,
        Self::Auto,
        Self::Bn,
        Self::Hi,
        Self::Es,
        Self::Fr,
        Self::De,
        Self::Ja,
        Self::Zh,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Auto => "auto",
            Self::Bn => "bn",
            Self::Hi => "hi",
            Self::Es => "es",
            Self::Fr => "fr",
            Self::De => "de",
            Self::Ja => "ja",
            Self::Zh => "zh",
        }
    }

    /// The code to hand Whisper, or `None` to let it detect.
    pub fn whisper_language(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            other => Some(other.as_str()),
        }
    }

    /// Which script family this language writes in. Drives the decoding and
    /// acceptance thresholds; see [`crate::decode_tuning`] for why they cannot
    /// be shared.
    pub fn script_class(self) -> ScriptClass {
        ScriptClass::from_language_code(self.whisper_language())
    }

    pub fn parse(raw: &str) -> Option<Self> {
        let s = raw.trim().to_lowercase();
        if s == "multi" {
            return Some(Self::Auto);
        }
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn script(code: &str) -> ScriptClass {
        ScriptClass::from_language_code(Some(code))
    }

    #[test]
    fn latin_languages() {
        for code in ["en", "es", "fr", "de", "it", "pt", "nl"] {
            assert_eq!(
                script(code),
                ScriptClass::Latin,
                "expected {code} to be latin"
            );
        }
    }

    #[test]
    fn indic_languages() {
        for code in ["bn", "hi", "ta", "te", "mr", "gu", "pa", "ne"] {
            assert_eq!(
                script(code),
                ScriptClass::Indic,
                "expected {code} to be indic"
            );
        }
    }

    #[test]
    fn cjk_languages() {
        for code in ["ja", "zh", "ko", "yue"] {
            assert_eq!(script(code), ScriptClass::Cjk, "expected {code} to be cjk");
        }
    }

    #[test]
    fn no_language_is_unknown() {
        // Auto mode before the first detection lands here.
        assert_eq!(ScriptClass::from_language_code(None), ScriptClass::Unknown);
    }

    #[test]
    fn unrecognised_code_is_unknown() {
        assert_eq!(script("xx"), ScriptClass::Unknown);
    }

    #[test]
    fn code_is_case_and_region_insensitive() {
        assert_eq!(script("BN"), ScriptClass::Indic);
        assert_eq!(script("zh-Hans"), ScriptClass::Cjk);
        assert_eq!(script("en_US"), ScriptClass::Latin);
    }

    #[test]
    fn language_mode_maps_to_script() {
        assert_eq!(LanguageMode::En.script_class(), ScriptClass::Latin);
        assert_eq!(LanguageMode::Bn.script_class(), ScriptClass::Indic);
        assert_eq!(LanguageMode::Hi.script_class(), ScriptClass::Indic);
        assert_eq!(LanguageMode::Ja.script_class(), ScriptClass::Cjk);
        assert_eq!(LanguageMode::Zh.script_class(), ScriptClass::Cjk);
        assert_eq!(LanguageMode::Auto.script_class(), ScriptClass::Unknown);
    }

    #[test]
    fn parse_accepts_codes_and_the_multi_alias() {
        assert_eq!(LanguageMode::parse("bn"), Some(LanguageMode::Bn));
        assert_eq!(LanguageMode::parse("  AUTO "), Some(LanguageMode::Auto));
        assert_eq!(LanguageMode::parse("multi"), Some(LanguageMode::Auto));
        assert_eq!(LanguageMode::parse("klingon"), None);
    }

    #[test]
    fn auto_leaves_the_language_to_whisper() {
        assert_eq!(LanguageMode::Auto.whisper_language(), None);
        assert_eq!(LanguageMode::Bn.whisper_language(), Some("bn"));
    }
}
