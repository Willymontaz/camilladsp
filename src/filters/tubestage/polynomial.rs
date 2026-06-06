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

// Asymmetric polynomial waveshaper used for the "audiophile color pack" mode of
// the `TubeStage` filter. The transfer function is
//   y(x) = (drive * x + bias) + k2 (drive * x + bias)^2
//                              + k3 (drive * x + bias)^3
//                              + k4 (drive * x + bias)^4
// The constant DC offset introduced by the bias and the even terms is removed
// later by the output DC blocker in the parent `TubeStage`.

use crate::PrcFmt;
use crate::config;
use crate::filters::tubestage::ramp::RampedValue;
use crate::utils::decibels::db_to_linear;

/// One-pole smoother time constant for parameters that influence the audio
/// amplitude. Short enough to follow the user's intent, long enough to not click.
const SMOOTH_TAU_S: PrcFmt = 0.020;

#[derive(Clone, Debug)]
pub struct PolynomialModel {
    pub drive_lin: RampedValue,
    pub bias: RampedValue,
    pub k2: PrcFmt,
    pub k3: PrcFmt,
    pub k4: PrcFmt,
}

impl PolynomialModel {
    pub fn from_config(p: &config::PolynomialModelParameters, oversampled_fs: usize) -> Self {
        Self {
            drive_lin: RampedValue::new(db_to_linear(p.drive_db()), SMOOTH_TAU_S, oversampled_fs),
            bias: RampedValue::new(p.bias(), SMOOTH_TAU_S, oversampled_fs),
            k2: p.k2(),
            k3: p.k3(),
            k4: p.k4(),
        }
    }

    pub fn update(&mut self, p: &config::PolynomialModelParameters, oversampled_fs: usize) {
        // Update targets only -- the per-sample ramp will move smoothly to them.
        self.drive_lin.set_target(db_to_linear(p.drive_db()));
        self.bias.set_target(p.bias());
        self.drive_lin.retune(SMOOTH_TAU_S, oversampled_fs);
        self.bias.retune(SMOOTH_TAU_S, oversampled_fs);
        // Polynomial coefficients have small audible discontinuity on swap; we
        // accept that and leave them un-ramped to keep the math straightforward.
        self.k2 = p.k2();
        self.k3 = p.k3();
        self.k4 = p.k4();
    }

    #[inline]
    pub fn process_sample(&mut self, x: PrcFmt) -> PrcFmt {
        let drive = self.drive_lin.tick();
        let bias = self.bias.tick();
        let d = drive * x + bias;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d3 * d;
        d + self.k2 * d2 + self.k3 * d3 + self.k4 * d4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(drive_lin: PrcFmt, bias: PrcFmt, k2: PrcFmt, k3: PrcFmt, k4: PrcFmt) -> PolynomialModel {
        let cfg = config::PolynomialModelParameters {
            drive_db: None,
            bias: Some(bias),
            k2: Some(k2),
            k3: Some(k3),
            k4: Some(k4),
        };
        let mut m = PolynomialModel::from_config(&cfg, 48000);
        // Force-snap the ramped values to the requested ones for deterministic
        // tests (avoids interaction with the smoothing transient).
        m.drive_lin = RampedValue::new(drive_lin, 1e-6, 48000);
        m.bias = RampedValue::new(bias, 1e-6, 48000);
        m
    }

    #[test]
    fn polynomial_zero_input_with_bias() {
        let mut model = make_model(1.0, 0.05, 0.15, 0.02, 0.0);
        let y = model.process_sample(0.0);
        let expected = 0.05 + 0.15 * 0.05 * 0.05 + 0.02 * 0.05 * 0.05 * 0.05;
        assert!((y - expected).abs() < 1e-9, "got {y}, expected {expected}");
    }

    #[test]
    fn polynomial_asymmetric_creates_dc_offset() {
        let mut model = make_model(1.0, 0.0, 0.5, 0.0, 0.0);
        let fs: PrcFmt = 48000.0;
        let cycles: PrcFmt = 100.0;
        let n: usize = 4800;
        let freq: PrcFmt = cycles * fs / (n as PrcFmt);
        let two_pi = 2.0 * std::f64::consts::PI as PrcFmt;
        let amp: PrcFmt = 0.5;
        let mut sum = 0.0;
        for i in 0..n {
            let x = amp * (two_pi * freq * (i as PrcFmt) / fs).sin();
            sum += model.process_sample(x);
        }
        let mean = sum / (n as PrcFmt);
        let expected = 0.5 * amp * amp / 2.0;
        assert!(
            (mean - expected).abs() < 1e-3,
            "DC offset {mean} differs from expected {expected}"
        );
    }

    #[test]
    fn polynomial_drive_change_is_smooth() {
        // Verify that updating the drive does not introduce a sample-level jump.
        let mut model = make_model(1.0, 0.0, 0.0, 0.0, 0.0);
        // Use the default 20 ms ramp.
        let cfg = config::PolynomialModelParameters {
            drive_db: Some(20.0), // 10x linear gain
            bias: Some(0.0),
            k2: Some(0.0),
            k3: Some(0.0),
            k4: Some(0.0),
        };
        model.update(&cfg, 48000);
        let mut prev = model.process_sample(1.0);
        for _ in 0..100 {
            let next = model.process_sample(1.0);
            assert!(
                (next - prev).abs() < 0.5,
                "drive ramp produced a step: prev={prev} next={next}"
            );
            prev = next;
        }
    }
}
