#[derive(Debug, Clone, Copy, Default)]
pub struct SecurityNormalizationOptions {
    pub decode_obfuscation: bool,
}

pub fn set_security_normalization_enabled(_enabled: bool) {
    // Compatibility shim kept for older call sites.
    // Normalization options are now expected to be passed explicitly.
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
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
}

pub fn normalize_text(input: &str) -> String {
    normalize_text_with_options(input, SecurityNormalizationOptions::default())
}

pub fn decode_obfuscated_text(input: &str) -> String {
    let mut current = input.to_string();
    for _ in 0..4 {
        let html_decoded = decode_html_entities(&current);
        let percent_decoded = decode_percent_encoding(&html_decoded);
        let normalized = normalize_compatibility_chars(&percent_decoded);
        let without_nulls = remove_null_bytes(&normalized);
        let cleaned = remove_inline_comments(&without_nulls);
        if cleaned == current {
            break;
        }
        current = cleaned;
    }
    current
}

fn remove_null_bytes(input: &str) -> String {
    input.chars().filter(|c| *c != '\0').collect()
}

fn remove_inline_comments(input: &str) -> String {
    let mut out = input.replace("/**/", "");
    out = out.replace("/* */", "");
    out = out.replace("--", " ");
    out = out.replace('#', " ");
    out
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
                let number = &candidate[prefix.len()..candidate.len() - 1];
                if let Ok(value) = u32::from_str_radix(number, radix) {
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
    let mut out = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = bytes[i + 1] as char;
            let lo = bytes[i + 2] as char;
            let hex = [hi, lo].iter().collect::<String>();
            if let Ok(value) = u8::from_str_radix(&hex, 16) {
                out.push(value);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn normalize_compatibility_chars(input: &str) -> String {
    input
        .replace('＇', "'")
        .replace('％', "%")
        .replace('／', "/")
        .replace('＜', "<")
        .replace('＞', ">")
        .replace('＝', "=")
}
