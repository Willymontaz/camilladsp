// CamillaDSP - A flexible tool for processing audio
// Copyright (C) 2026 Henrik Enquist
//
// This file is part of CamillaDSP.
//
// CamillaDSP is free software; you can redistribute it and/or modify it
// under the terms of either:
//
// a) the GNU General Public License version 3,
//    or
// b) the Mozilla Public License Version 2.0.
//
// You should have received copies of the GNU General Public License and the
// Mozilla Public License along with this program. If not, see
// <https://www.gnu.org/licenses/> and <https://www.mozilla.org/MPL/2.0/>.

// Simple per-sample one-pole smoothing for hot-swap of TubeStage parameters
// like drive, bias and makeup gain. Avoids audible clicks when parameters
// change at runtime.

use crate::PrcFmt;

#[derive(Clone, Copy, Debug)]
pub struct RampedValue {
    target: PrcFmt,
    current: PrcFmt,
    coeff: PrcFmt,
}

impl RampedValue {
    /// Creates a smoother whose `current` and `target` start at `initial`.
    /// `tau_seconds` is the time constant of the one-pole smoother and `fs`
    /// is the rate at which `next()` will be called (the oversampled rate
    /// in our use case).
    pub fn new(initial: PrcFmt, tau_seconds: PrcFmt, fs: usize) -> Self {
        let coeff = (-1.0 / (tau_seconds * (fs as PrcFmt))).exp();
        Self {
            target: initial,
            current: initial,
            coeff,
        }
    }

    pub fn set_target(&mut self, target: PrcFmt) {
        self.target = target;
    }

    /// Update `tau` and `fs` while preserving the current value (used when the
    /// effective oversampled sample rate changes).
    pub fn retune(&mut self, tau_seconds: PrcFmt, fs: usize) {
        self.coeff = (-1.0 / (tau_seconds * (fs as PrcFmt))).exp();
    }

    /// Advance the smoother by one sample and return the new value.
    /// Renamed from `next` to avoid clashing with `Iterator::next`.
    #[inline]
    pub fn tick(&mut self) -> PrcFmt {
        self.current = self.coeff * self.current + (1.0 - self.coeff) * self.target;
        self.current
    }

    pub fn target(&self) -> PrcFmt {
        self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_converges_to_target() {
        let mut r = RampedValue::new(0.0, 0.005, 48_000);
        r.set_target(1.0);
        for _ in 0..10_000 {
            r.tick();
        }
        let v = r.tick();
        assert!((v - 1.0).abs() < 1e-6, "ramp did not converge: {v}");
    }

    #[test]
    fn ramp_no_overshoot() {
        let mut r = RampedValue::new(0.0, 0.005, 48_000);
        r.set_target(1.0);
        let mut prev = 0.0;
        for _ in 0..1000 {
            let v = r.tick();
            assert!(v >= prev - 1e-12);
            assert!(v <= 1.0 + 1e-12);
            prev = v;
        }
    }
}
