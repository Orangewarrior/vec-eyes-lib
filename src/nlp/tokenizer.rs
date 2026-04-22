pub fn tokenize(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut prev_is_upper = false;

    for c in input.chars() {
        if !c.is_alphanumeric() {
            flush_token(&mut current, &mut tokens);
            prev_is_upper = false;
            continue;
        }

        let is_upper = c.is_uppercase();
        if !current.is_empty() && prev_is_upper != is_upper {
            flush_token(&mut current, &mut tokens);
        }

        // normalize_text already applies to_lowercase, so just push the char
        current.push(c);
        prev_is_upper = is_upper;
    }

    flush_token(&mut current, &mut tokens);
    tokens
}

fn flush_token(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        // Preserve all tokens including single-char ones for security analysis
        // Symbols like <, >, | and single digits can be critical in malware/syscall detection
        let token = current.clone();
        current.clear();
        tokens.push(token);
    }
}
