use std::sync::{OnceLock, RwLock};

#[derive(Debug, Clone, Copy, Default)]
pub struct SecurityNormalizationOptions {
    pub decode_obfuscation: bool,
}

static NORMALIZER_OPTIONS: OnceLock<RwLock<SecurityNormalizationOptions>> = OnceLock::new();

fn current_options() -> SecurityNormalizationOptions {
    NORMALIZER_OPTIONS
        .get_or_init(|| RwLock::new(SecurityNormalizationOptions::default()))
        .read()
        .map(|guard| *guard)
        .unwrap_or_default()
}

pub fn set_security_normalization_enabled(enabled: bool) {
    if let Ok(mut guard) = NORMALIZER_OPTIONS
        .get_or_init(|| RwLock::new(SecurityNormalizationOptions::default()))
        .write()
    {
        guard.decode_obfuscation = enabled;
    }
}

pub fn normalize_text_with_options(input: &str, options: SecurityNormalizationOptions) -> String {
    let preprocessed = if options.decode_obfuscation {
        decode_obfuscated_text(input)
    } else {
        input.to_string()
    };

    preprocessed
        .chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
}

pub fn normalize_text(input: &str) -> String {
    normalize_text_with_options(input, current_options())
}

fn decode_obfuscated_text(input: &str) -> String {
    let html_decoded = decode_html_entities(input);
    let percent_decoded = decode_percent_encoding(&html_decoded);
    normalize_compatibility_chars(&percent_decoded)
}

fn decode_html_entities(input: &str) -> String {
    let mut out = input
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'");

    out = decode_numeric_entities(&out, "&#x", 16);
    decode_numeric_entities(&out, "&#", 10)
}

fn decode_numeric_entities(input: &str, prefix: &str, radix: u32) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '&' {
            let mut candidate = String::from("&");
            while let Some(&next) = chars.peek() {
                candidate.push(next);
                chars.next();
                if next == ';' || candidate.len() > 12 {
                    break;
                }
            }
            if candidate.starts_with(prefix) && candidate.ends_with(';') {
                let body = &candidate[prefix.len()..candidate.len() - 1];
                if let Ok(value) = u32::from_str_radix(body, radix) {
                    if let Some(decoded) = char::from_u32(value) {
                        out.push(decoded);
                        continue;
                    }
                }
            }
            out.push_str(&candidate);
        } else {
            out.push(ch);
        }
    }
    out
}

fn decode_percent_encoding(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 5 < bytes.len() && (bytes[i + 1] == b'u' || bytes[i + 1] == b'U') {
                if let Ok(value) = u16::from_str_radix(&input[i + 2..i + 6], 16) {
                    if let Some(ch) = char::from_u32(value as u32) {
                        out.push(ch);
                        i += 6;
                        continue;
                    }
                }
            }
            if i + 2 < bytes.len() {
                if let Ok(value) = u8::from_str_radix(&input[i + 1..i + 3], 16) {
                    out.push(value as char);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn normalize_compatibility_chars(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '\u{FF10}'..='\u{FF19}' => char::from_u32((c as u32) - 0xFF10 + b'0' as u32).unwrap_or(c),
            '\u{FF21}'..='\u{FF3A}' => char::from_u32((c as u32) - 0xFF21 + b'A' as u32).unwrap_or(c),
            '\u{FF41}'..='\u{FF5A}' => char::from_u32((c as u32) - 0xFF41 + b'a' as u32).unwrap_or(c),
            _ => c,
        })
        .collect()
}
