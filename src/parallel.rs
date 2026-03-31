use rayon::ThreadPoolBuilder;

pub(crate) fn install_pool<T, F>(threads: Option<usize>, job: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match threads {
        Some(n) if n > 1 => {
            match ThreadPoolBuilder::new().num_threads(n).build() {
                Ok(pool) => pool.install(job),
                Err(_) => job(),
            }
        }
        _ => job(),
    }
}
