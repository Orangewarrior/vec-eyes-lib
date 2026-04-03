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

        current.push(c.to_ascii_lowercase());
        prev_is_upper = is_upper;
    }

    flush_token(&mut current, &mut tokens);
    tokens
}

fn flush_token(current: &mut String, tokens: &mut Vec<String>) {
    if current.len() >= 2 {
        tokens.push(std::mem::take(current));
    } else {
        current.clear();
    }
}
