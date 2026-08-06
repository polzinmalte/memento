//! Cleanup and non-speech detection for Whisper output. Ported from listnr
//! `Sources/ListnrCore/Transcription/TranscriptText.swift`.
//!
//! The guiding principle is that **text alone cannot tell you whether Whisper
//! heard something.** "Okay" is both the single most common silence
//! hallucination and a completely ordinary thing to say in a meeting. Deciding
//! from the string means choosing which of the two to break. So the confidence
//! signals Whisper reports per segment do that job (see `memento-asr`), and what
//! remains here is limited to output that is not speech in any language:
//! bracketed stage directions, decoder tokens, and training-data bleed from
//! captioned video.
//!
//! The Swift original uses `NSRegularExpression`; the patterns are simple
//! delimiter scans and a suffix test, so they are hand-rolled here rather than
//! pulling a regex engine into an otherwise dependency-free crate.

/// Non-speech annotations Whisper emits, matched against the *contents* of a
/// bracket, paren, or asterisk pair. Matching the content rather than the
/// delimiter is what lets real speech keep its punctuation: "(v2)" and
/// "*really*" survive, "(laughs)" and "[MUSIC]" do not.
const ANNOTATION_KEYWORDS: &[&str] = &[
    "music",
    "applause",
    "applauds",
    "silence",
    "blank audio",
    "blank_audio",
    "inaudible",
    "unintelligible",
    "indistinct",
    "crosstalk",
    "cross talk",
    "laugh",
    "laughs",
    "laughter",
    "laughing",
    "chuckles",
    "sighs",
    "sigh",
    "coughs",
    "cough",
    "clears throat",
    "throat clearing",
    "noise",
    "background noise",
    "static",
    "beep",
    "beeping",
    "ringing",
    "footsteps",
    "no audio",
    "no speech",
    "speaking foreign language",
    "foreign",
    "sound effect",
    "instrumental",
    "singing",
    "humming",
    "breathing",
];

/// Phrases that are captioned-video artifacts rather than meeting speech.
/// A person on a call does not say these; Whisper says them when it has nothing
/// to transcribe. Matched only as the entire output.
const CAPTION_ARTIFACTS: &[&str] = &[
    "thanks for watching",
    "thank you for watching",
    "thanks for watching!",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "don't forget to subscribe",
    "see you in the next video",
    "see you next time",
    "subtitles by the amara.org community",
    "transcription by castingwords",
    "www.mooji.org",
];

#[derive(Clone, Copy, PartialEq)]
enum Delimiter {
    Brackets,
    Parens,
    Asterisks,
}

impl Delimiter {
    fn pair(self) -> (char, char) {
        match self {
            Self::Brackets => ('[', ']'),
            Self::Parens => ('(', ')'),
            Self::Asterisks => ('*', '*'),
        }
    }
}

/// Punctuation and symbols, for trimming token edges. Deliberately an explicit
/// set rather than "not alphanumeric": the latter also strips Indic combining
/// marks, which are part of the word.
fn is_punct_or_symbol(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '…' | '“'
                | '”'
                | '‘'
                | '’'
                | '—'
                | '–'
                | '«'
                | '»'
                | '।'
                | '॥'
                | '。'
                | '，'
                | '！'
                | '？'
                | '、'
                | '；'
                | '：'
                | '「'
                | '」'
        )
}

/// Strip decoder tokens and non-speech annotations, then normalise spacing.
pub fn clean(text: &str) -> String {
    // Decoder tokens like <|startoftranscript|> are never speech.
    let mut out = replace_delimited(text, '<', '>', |inner| {
        // The regex was `<\|[^|]*\|>`, so only pipe-wrapped content counts.
        (inner.starts_with('|')
            && inner.ends_with('|')
            && inner.len() >= 2
            && !inner[1..inner.len() - 1].contains('|'))
        .then(|| " ".to_string())
    });

    for delimiter in [Delimiter::Brackets, Delimiter::Parens, Delimiter::Asterisks] {
        let (open, close) = delimiter.pair();
        out = replace_delimited(&out, open, close, |inner| {
            is_annotation(inner, delimiter).then(|| " ".to_string())
        });
    }

    out = strip_trailing_digit_artifact(&out);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whisper sometimes tacks digits onto a trailing word: "Thank you.000". The
/// original's regex requires a *letter* before the period, so genuine decimals
/// are untouched: "19.99" and "version 3.14" keep their digits.
fn strip_trailing_digit_artifact(text: &str) -> String {
    let trimmed = text.trim_end();
    let digits: String = trimmed
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.len() < 2 {
        return text.to_string();
    }
    let before_digits = &trimmed[..trimmed.len() - digits.len()];
    let Some(head) = before_digits.strip_suffix('.') else {
        return text.to_string();
    };
    match head.chars().next_back() {
        Some(c) if c.is_alphabetic() => head.to_string(),
        _ => text.to_string(),
    }
}

/// True when the text carries no speech content at all.
///
/// Note what is *not* here: "okay", "thanks", "bye", "you", "uh", and a minimum
/// length. Those were all rejected by an earlier version of listnr, and each one
/// threw away real speech. The length rule was the worst, because it counted
/// characters: Chinese 好的 ("okay") and Japanese はい ("yes") are two
/// characters, so the most common replies in two supported languages were
/// discarded before they could ever be printed.
pub fn is_non_speech(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Nothing but punctuation or symbols. `is_alphanumeric` here includes
    // ideographs and Indic scripts, and digits are allowed so a bare "42"
    // survives.
    if !trimmed.chars().any(char::is_alphanumeric) {
        return true;
    }

    let normalised = trimmed
        .to_lowercase()
        .trim_matches(|c| ".!?,…\"' ".contains(c))
        .to_string();

    if CAPTION_ARTIFACTS.contains(&normalised.as_str()) {
        return true;
    }
    // A whole segment that is only a stage direction, brackets already gone.
    if ANNOTATION_KEYWORDS.contains(&normalised.as_str()) {
        return true;
    }
    is_repetition_loop(trimmed)
}

/// True when the text is a decoder loop rather than speech.
///
/// The non-Latin confidence gates are deliberately permissive (see the decode
/// tuning in `memento-asr`), because the English ones were rejecting correct
/// Bangla and Hindi outright. This is the counterweight for what that lets
/// through.
///
/// It reads the *shape* of the output rather than matching a keyword list, so it
/// works the same in every language, including the ones nobody has written a
/// keyword list for. The two thresholds are set where emphasis stops and
/// degeneracy starts: three repeats of a word is a person insisting, four is the
/// decoder stuck. Two occurrences of a phrase is a callback, three is a loop.
pub fn is_repetition_loop(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(normalised_token)
        .filter(|t| !t.is_empty())
        .collect();

    if has_degenerate_run(&words, 4, 3) {
        return true;
    }

    // Han, Kana, and Hangul are not word-spaced, so the whole utterance arrives
    // as one or two tokens and the pass above cannot see the loop. Character
    // granularity is the only view that can, and the limits go up to match:
    // characters repeat far more readily than words do.
    if words.len() <= 2 {
        let characters: Vec<String> = text
            .chars()
            .filter(|c| !c.is_whitespace() && !is_punct_or_symbol(*c))
            .map(String::from)
            .collect();
        if characters.len() >= 8 && has_degenerate_run(&characters, 6, 4) {
            return true;
        }
    }

    false
}

/// Lowercase and shed surrounding punctuation, so "YES!" and "yes," are the same
/// token for repetition purposes.
fn normalised_token(token: &str) -> String {
    token
        .to_lowercase()
        .trim_matches(is_punct_or_symbol)
        .to_string()
}

/// True when `tokens` contains either one token repeated `same_token_limit`
/// times in a row, or an n-gram (n of 2..=4) repeated `gram_repeat_limit` times
/// back to back. Only *consecutive* repetition counts: a word recurring
/// throughout a sentence is normal speech, a word recurring immediately is not.
fn has_degenerate_run(
    tokens: &[String],
    same_token_limit: usize,
    gram_repeat_limit: usize,
) -> bool {
    if tokens.len() < 2 {
        return false;
    }

    let mut run = 1;
    for i in 1..tokens.len() {
        if tokens[i] == tokens[i - 1] {
            run += 1;
            if run >= same_token_limit {
                return true;
            }
        } else {
            run = 1;
        }
    }

    for gram_length in 2..=4usize {
        let span = gram_length * gram_repeat_limit;
        if tokens.len() < span {
            continue;
        }
        for start in 0..=(tokens.len() - span) {
            let gram = &tokens[start..start + gram_length];
            let mut repeats = 1;
            let mut cursor = start + gram_length;
            while cursor + gram_length <= tokens.len()
                && &tokens[cursor..cursor + gram_length] == gram
            {
                repeats += 1;
                if repeats >= gram_repeat_limit {
                    return true;
                }
                cursor += gram_length;
            }
        }
    }

    false
}

fn is_annotation(inner: &str, delimiter: Delimiter) -> bool {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lowered = trimmed.to_lowercase();
    if ANNOTATION_KEYWORDS.iter().any(|k| lowered.contains(k)) {
        return true;
    }

    // Whisper writes its markers in caps: [MUSIC], [BLANK_AUDIO], [SILENCE].
    // Only inside square brackets, where real speech essentially never goes.
    // Parenthesised acronyms in real speech, like "(API)", must survive.
    delimiter == Delimiter::Brackets
        && trimmed.chars().count() <= 24
        && trimmed == trimmed.to_uppercase()
        && trimmed.chars().any(char::is_alphabetic)
}

/// Rewrite `open`…`close` pairs whose inner content `transform` opts into.
/// Returning `None` leaves that pair untouched. Scans left to right and does not
/// nest, matching the non-greedy `[^close]*` regexes it replaces.
fn replace_delimited(
    text: &str,
    open: char,
    close: char,
    transform: impl Fn(&str) -> Option<String>,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find(open) {
        let after_open = &rest[start + open.len_utf8()..];
        let Some(end) = after_open.find(close) else {
            break;
        };
        let inner = &after_open[..end];
        out.push_str(&rest[..start]);
        match transform(inner) {
            Some(replacement) => out.push_str(&replacement),
            None => {
                out.push(open);
                out.push_str(inner);
                out.push(close);
            }
        }
        rest = &after_open[end + close.len_utf8()..];
    }

    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // MARK: clean()

    #[test]
    fn decoder_tokens_stripped() {
        assert_eq!(clean("<|startoftranscript|> Hello <|endoftext|>"), "Hello");
        assert_eq!(clean("<|nospeech|>"), "");
    }

    #[test]
    fn bracket_annotations_removed() {
        assert_eq!(clean("[MUSIC]"), "");
        assert_eq!(clean("[BLANK_AUDIO]"), "");
        assert_eq!(clean("[music] hello [applause]"), "hello");
    }

    #[test]
    fn uppercase_marker_only_inside_brackets() {
        assert_eq!(clean("[NOISE MARKER] hi"), "hi");
        // A parenthesised acronym is real speech and must survive.
        assert_eq!(
            clean("we call the (API) directly"),
            "we call the (API) directly"
        );
    }

    #[test]
    fn paren_and_asterisk_annotations_removed_only_when_annotations() {
        assert_eq!(clean("(laughs) I agree"), "I agree");
        assert_eq!(clean("*sighs* fine"), "fine");
        // Content that is not an annotation keeps its delimiters.
        assert_eq!(clean("the API (v2) ships"), "the API (v2) ships");
        assert_eq!(clean("I *really* mean it"), "I *really* mean it");
    }

    #[test]
    fn trailing_digit_artifact_cleaned_but_decimals_survive() {
        assert_eq!(clean("Thank you.000"), "Thank you");
        assert_eq!(clean("version 3.14"), "version 3.14");
        assert_eq!(clean("that costs $19.99"), "that costs $19.99");
        assert_eq!(clean("19.99"), "19.99");
    }

    #[test]
    fn whitespace_normalised() {
        assert_eq!(clean("  a   b  "), "a b");
        assert_eq!(clean("(laughs)   and (background noise) then"), "and then");
    }

    // MARK: is_non_speech() — what must be rejected

    #[test]
    fn empty_and_punctuation_only_are_non_speech() {
        for s in ["", "   ", "...", "!!!", "?!."] {
            assert!(is_non_speech(s), "{s:?}");
        }
    }

    #[test]
    fn caption_artifacts_are_non_speech() {
        for s in [
            "thanks for watching",
            "Thanks for watching!",
            "Please subscribe",
            "Subtitles by the Amara.org community",
            "see you in the next video",
        ] {
            assert!(is_non_speech(s), "{s:?}");
        }
    }

    #[test]
    fn bare_annotation_keyword_is_non_speech() {
        // A whole segment that is only a stage direction, brackets already gone.
        assert!(is_non_speech("music"));
        assert!(is_non_speech("laughter"));
    }

    // MARK: is_non_speech() — what must survive

    #[test]
    fn ordinary_short_replies_survive() {
        // Every one of these was wrongly rejected by the old text blocklist.
        for phrase in [
            "okay",
            "ok",
            "thanks",
            "thank you",
            "bye",
            "you",
            "uh",
            "um",
            "hmm",
            "the",
            "a",
        ] {
            assert!(!is_non_speech(phrase), "{phrase} is real speech");
        }
    }

    #[test]
    fn short_non_latin_replies_survive() {
        // The old two-character length rule deleted the most common replies in
        // the languages memento advertises.
        for phrase in ["好的", "はい", "да", "হ্যাঁ", "हाँ"] {
            assert!(!is_non_speech(phrase), "{phrase}");
        }
    }

    #[test]
    fn bare_number_survives() {
        assert!(!is_non_speech("42"));
    }

    #[test]
    fn ordinary_sentences_survive() {
        assert!(!is_non_speech("hello world"));
        assert!(!is_non_speech("let's ship the API v2 on Friday"));
        assert!(!is_non_speech("thanks, that helps a lot"));
    }

    #[test]
    fn clean_then_detect_pipeline() {
        // The two functions as the ASR layer composes them.
        assert!(is_non_speech(&clean("[MUSIC] (laughs) ")));
        let kept = clean("<|startoftranscript|> okay, sounds good <|endoftext|>");
        assert!(!is_non_speech(&kept));
        assert_eq!(kept, "okay, sounds good");
    }

    // MARK: repetition guard — loops that must be caught

    #[test]
    fn same_word_repeated_many_times() {
        assert!(is_repetition_loop("okay okay okay okay okay"));
    }

    #[test]
    fn repeated_phrase() {
        assert!(is_repetition_loop("I think I think I think"));
    }

    #[test]
    fn repeated_longer_phrase() {
        assert!(is_repetition_loop(
            "we need to ship we need to ship we need to ship"
        ));
    }

    #[test]
    fn repetition_is_case_and_punctuation_insensitive() {
        assert!(is_repetition_loop("Yes. yes, YES! yes"));
    }

    #[test]
    fn repetition_in_non_latin_script() {
        // The exact failure mode the relaxed Indic gate would otherwise admit.
        assert!(is_repetition_loop("আচ্ছা আচ্ছা আচ্ছা আচ্ছা"));
    }

    #[test]
    fn repetition_in_spaceless_script() {
        // Han text arrives as one whitespace-delimited token, so the word-level
        // pass cannot see the loop and the character pass has to.
        assert!(is_repetition_loop("好的好的好的好的好的好的"));
    }

    #[test]
    fn loop_buried_after_real_speech() {
        assert!(is_repetition_loop(
            "so the plan is to ship on friday friday friday friday friday"
        ));
    }

    // MARK: repetition guard — real speech that must survive

    #[test]
    fn ordinary_sentence_survives() {
        assert!(!is_repetition_loop(
            "let us ship the beta on friday and write the changelog after"
        ));
    }

    #[test]
    fn emphatic_doubling_survives() {
        // "very very good" is ordinary speech in most languages, and doubling
        // for emphasis is grammatical in Bangla and Hindi.
        assert!(!is_repetition_loop("that is very very good"));
        assert!(!is_repetition_loop("খুব খুব ভালো"));
    }

    #[test]
    fn tripled_word_survives() {
        // Three is still plausibly a human being emphatic; four is a decoder loop.
        assert!(!is_repetition_loop("no no no"));
    }

    #[test]
    fn repeated_phrase_pair_survives() {
        // Two occurrences of a phrase is a callback, not a loop.
        assert!(!is_repetition_loop("I think I think we agree"));
    }

    #[test]
    fn short_text_survives() {
        for s in ["okay", "", "hello there"] {
            assert!(!is_repetition_loop(s), "{s:?}");
        }
    }

    #[test]
    fn short_non_latin_reply_survives() {
        // The bug this codebase already fixed once: two-character replies are
        // the most common thing said in these languages.
        for s in ["好的", "はい", "আচ্ছা"] {
            assert!(!is_repetition_loop(s), "{s}");
        }
    }

    #[test]
    fn naturally_repetitive_counting_survives() {
        assert!(!is_repetition_loop("one two three four five six"));
    }

    // MARK: wiring

    #[test]
    fn repetition_loop_counts_as_non_speech() {
        assert!(is_non_speech("okay okay okay okay okay"));
    }

    #[test]
    fn non_speech_still_accepts_ordinary_speech() {
        assert!(!is_non_speech("we should ship on friday"));
    }
}
