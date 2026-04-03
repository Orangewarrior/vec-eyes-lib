use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

static GLOBAL_POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();

fn cached_pool(threads: usize) -> Option<Arc<ThreadPool>> {
    let cache = GLOBAL_POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().ok()?;
    if let Some(pool) = guard.get(&threads) {
        return Some(Arc::clone(pool));
    }
    let pool = Arc::new(ThreadPoolBuilder::new().num_threads(threads).build().ok()?);
    guard.insert(threads, Arc::clone(&pool));
    Some(pool)
}

pub(crate) fn install_pool<T, F>(threads: Option<usize>, job: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    match threads {
        Some(n) if n > 1 => match cached_pool(n) {
            Some(pool) => pool.install(job),
            None => job(),
        },
        _ => job(),
    }
}
