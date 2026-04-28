use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

static GLOBAL_POOLS: OnceLock<RwLock<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();

const MAX_POOL_CACHE: usize = 32;

fn cached_pool(threads: usize) -> Option<Arc<ThreadPool>> {
    let cache = GLOBAL_POOLS.get_or_init(|| RwLock::new(HashMap::new()));

    // Fast path: pool already exists — shared read, no contention.
    {
        let guard = cache.read().ok()?;
        if let Some(pool) = guard.get(&threads) {
            return Some(Arc::clone(pool));
        }
    }

    // Slow path: create the pool under an exclusive write lock.
    let mut guard = cache.write().ok()?;
    // Double-check: another thread may have inserted while we were waiting.
    if let Some(pool) = guard.get(&threads) {
        return Some(Arc::clone(pool));
    }
    if guard.len() >= MAX_POOL_CACHE {
        if let Some(first_key) = guard.keys().next().copied() {
            guard.remove(&first_key);
        }
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
