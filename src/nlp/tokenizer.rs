pub fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in input.split_whitespace() {
        if !part.is_empty() {
            out.push(part.to_string());
        }
    }
    out
}
