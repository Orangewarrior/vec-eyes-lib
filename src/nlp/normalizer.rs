pub fn normalize_text(input: &str) -> String {
    input
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() { c } else { ' ' })
        .collect::<String>()
}
