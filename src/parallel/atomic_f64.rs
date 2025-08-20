use std::sync::atomic::{AtomicU64, Ordering};

// Helper struct for atomic f64 operations
pub struct AtomicF64 {
    inner: AtomicU64,
}

impl AtomicF64 {
    pub fn new(val: f64) -> Self {
        AtomicF64 {
            inner: AtomicU64::new(val.to_bits()),
        }
    }

    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.inner.load(ordering))
    }

    pub fn store(&self, val: f64, ordering: Ordering) {
        self.inner.store(val.to_bits(), ordering);
    }

    pub fn compare_exchange_weak(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        match self
            .inner
            .compare_exchange_weak(current.to_bits(), new.to_bits(), success, failure)
        {
            Ok(v) => Ok(f64::from_bits(v)),
            Err(v) => Err(f64::from_bits(v)),
        }
    }
}
