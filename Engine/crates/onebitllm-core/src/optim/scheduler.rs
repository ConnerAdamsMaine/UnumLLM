/// Learning rate scheduler trait and common implementations.

/// A learning rate scheduler that adjusts the LR based on the current step.
pub trait LrScheduler {
    /// Compute the learning rate for the given step.
    fn get_lr(&self, step: usize) -> f32;

    /// Return the total number of steps this scheduler is configured for.
    fn total_steps(&self) -> usize;
}

/// Cosine annealing scheduler.
///
/// Decays the learning rate following a cosine curve from `lr_max` to `lr_min`
/// over `total_steps` steps.
pub struct CosineScheduler {
    lr_max: f32,
    lr_min: f32,
    total: usize,
}

impl CosineScheduler {
    /// Create a new cosine scheduler.
    pub fn new(lr_max: f32, lr_min: f32, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            total: total_steps,
        }
    }
}

impl LrScheduler for CosineScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total {
            return self.lr_min;
        }
        let progress = step as f32 / self.total as f32;
        let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
        self.lr_min + (self.lr_max - self.lr_min) * cosine
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

/// Linear decay scheduler.
///
/// Linearly decays the learning rate from `lr_start` to `lr_end`
/// over `total_steps` steps.
pub struct LinearScheduler {
    lr_start: f32,
    lr_end: f32,
    total: usize,
}

impl LinearScheduler {
    /// Create a new linear scheduler.
    pub fn new(lr_start: f32, lr_end: f32, total_steps: usize) -> Self {
        Self {
            lr_start,
            lr_end,
            total: total_steps,
        }
    }
}

impl LrScheduler for LinearScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total {
            return self.lr_end;
        }
        let progress = step as f32 / self.total as f32;
        self.lr_start + (self.lr_end - self.lr_start) * progress
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

/// Warmup scheduler that wraps another scheduler.
///
/// During the first `warmup_steps`, the learning rate linearly increases
/// from 0 to the inner scheduler's LR at step 0. After warmup, delegates
/// to the inner scheduler (with step offset by warmup_steps).
pub struct WarmupScheduler<S: LrScheduler> {
    inner: S,
    warmup_steps: usize,
}

impl<S: LrScheduler> WarmupScheduler<S> {
    /// Create a warmup wrapper around an existing scheduler.
    pub fn new(inner: S, warmup_steps: usize) -> Self {
        Self {
            inner,
            warmup_steps,
        }
    }
}

impl<S: LrScheduler> LrScheduler for WarmupScheduler<S> {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup from 0 to inner.get_lr(0)
            let target = self.inner.get_lr(0);
            target * (step as f32 + 1.0) / self.warmup_steps as f32
        } else {
            self.inner.get_lr(step - self.warmup_steps)
        }
    }

    fn total_steps(&self) -> usize {
        self.warmup_steps + self.inner.total_steps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_scheduler_endpoints() {
        let sched = CosineScheduler::new(1.0, 0.0, 100);
        // At step 0, should be at lr_max
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-6);
        // At step 100 (past end), should be lr_min
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_scheduler_midpoint() {
        let sched = CosineScheduler::new(1.0, 0.0, 100);
        // At step 50 (halfway), cosine should give 0.5
        let lr = sched.get_lr(50);
        assert!((lr - 0.5).abs() < 1e-5, "Midpoint should be ~0.5, got {lr}");
    }

    #[test]
    fn test_cosine_scheduler_curve() {
        let sched = CosineScheduler::new(1.0, 0.0, 1000);
        let mut prev = sched.get_lr(0);
        // Cosine should be monotonically decreasing
        for step in 1..=1000 {
            let lr = sched.get_lr(step);
            assert!(lr <= prev + 1e-6, "Cosine should decrease: step={step}, lr={lr}, prev={prev}");
            prev = lr;
        }
    }

    #[test]
    fn test_cosine_scheduler_with_min_lr() {
        let sched = CosineScheduler::new(1.0, 0.1, 100);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((sched.get_lr(100) - 0.1).abs() < 1e-6);
        // All values should be in [0.1, 1.0]
        for step in 0..=100 {
            let lr = sched.get_lr(step);
            assert!(lr >= 0.1 - 1e-6 && lr <= 1.0 + 1e-6,
                "LR out of range at step {step}: {lr}");
        }
    }

    #[test]
    fn test_linear_scheduler() {
        let sched = LinearScheduler::new(1.0, 0.0, 100);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-6);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-5);
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_scheduler_monotonic() {
        let sched = LinearScheduler::new(1.0, 0.0, 100);
        let mut prev = sched.get_lr(0);
        for step in 1..=100 {
            let lr = sched.get_lr(step);
            assert!(lr <= prev + 1e-6);
            prev = lr;
        }
    }

    #[test]
    fn test_warmup_scheduler() {
        let inner = CosineScheduler::new(1.0, 0.0, 100);
        let sched = WarmupScheduler::new(inner, 10);

        // During warmup (steps 0..10): linear from 0 to 1.0
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-6); // step 0 -> (0+1)/10 * 1.0
        assert!((sched.get_lr(4) - 0.5).abs() < 1e-6); // step 4 -> 5/10 * 1.0
        assert!((sched.get_lr(9) - 1.0).abs() < 1e-6); // step 9 -> 10/10 * 1.0

        // After warmup: delegates to cosine starting from step 0
        let lr_10 = sched.get_lr(10);
        assert!((lr_10 - 1.0).abs() < 1e-5, "After warmup should start at cosine(0), got {lr_10}");

        // Total steps
        assert_eq!(sched.total_steps(), 110);
    }

    #[test]
    fn test_warmup_linear_combination() {
        let inner = LinearScheduler::new(0.01, 0.001, 90);
        let sched = WarmupScheduler::new(inner, 10);

        // Step 0: warmup, should be 1/10 * 0.01 = 0.001
        assert!((sched.get_lr(0) - 0.001).abs() < 1e-6);

        // Step 10: start of inner scheduler, should be 0.01
        assert!((sched.get_lr(10) - 0.01).abs() < 1e-6);

        assert_eq!(sched.total_steps(), 100);
    }
}
