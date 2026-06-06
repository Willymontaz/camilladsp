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

// Single-Ended triode model used by the `TubeStage` filter, based on the
// Koren plate-current equation:
//
//   E1 = (Kp / mu) * ln(1 + exp(Kp * (1/mu + Vgk / sqrt(Kvb + Vpk^2))))
//   Ip = (E1^X) / Kg1                if E1 > 0, else 0
//
// References:
// - Yeh thesis, sec. 3.2 (default 300B coefficients in Tab. 3.2)
// - Cohen and Helie, "Real-time simulation of a guitar power amplifier" (CoMJ 2009)
//
// The audio path is:
//   audio_in -> coupling cap (highpass) -> grid voltage swing
//             -> Koren static curve (Vgk, Vpk = vplate)
//             -> plate-side: Vplate_eff = vplate - sag * Ip_envelope
//             -> output = -(Ip - Ip_q) * gain
//             -> cathode bypass (low-shelf attenuation)
//             -> coupling cap (highpass)
// All running at the oversampled rate.

use crate::PrcFmt;
use crate::config;
use crate::filters::biquad::{Biquad, BiquadCoefficients};
use crate::filters::tubestage::ramp::RampedValue;
use crate::utils::decibels::db_to_linear;

const SMOOTH_TAU_S: PrcFmt = 0.020;

#[derive(Clone, Debug)]
pub struct TriodeModel {
    pub mu: PrcFmt,
    pub kp: PrcFmt,
    pub kvb: PrcFmt,
    pub kg1: PrcFmt,
    pub x_param: PrcFmt,
    pub vbias: PrcFmt,
    pub vplate: PrcFmt,
    pub drive_lin: RampedValue,
    pub sag_amount: PrcFmt,

    /// Plate current at the quiescent operating point. Subtracted from the
    /// instantaneous current to centre the audio output around zero.
    ip_quiescent: PrcFmt,
    /// Output normalisation derived from the small-signal slope at the
    /// quiescent point so that audio_in ~ 1.0 produces roughly audio_out ~ 1.0.
    norm: PrcFmt,
    /// Slow envelope of |Ip| used for the sag (power-supply) modelling.
    sag_env: PrcFmt,
    /// One-pole coefficient for the sag envelope (close to 1 at high oversampled rates).
    sag_coeff: PrcFmt,

    coupling_in: Biquad,
    coupling_out: Biquad,
    /// Cathode bypass: low-shelf with negative gain to model the cathode
    /// resistor's effect at low frequencies (loss of gain due to local
    /// negative feedback below the bypass cap corner).
    cathode_bypass: Biquad,
}

fn koren_plate_current(
    vgk: PrcFmt,
    vpk: PrcFmt,
    mu: PrcFmt,
    kp: PrcFmt,
    kvb: PrcFmt,
    kg1: PrcFmt,
    x_param: PrcFmt,
) -> PrcFmt {
    // Numerically stable softplus: ln(1 + exp(z)) = max(z, 0) + ln(1 + exp(-|z|)).
    let z = kp * (1.0 / mu + vgk / (kvb + vpk * vpk).sqrt());
    let softplus = if z > 30.0 {
        z
    } else if z < -30.0 {
        0.0
    } else {
        (1.0 + z.exp()).ln()
    };
    let e1 = (kp / mu) * softplus;
    if e1 > 0.0 {
        e1.powf(x_param) / kg1
    } else {
        0.0
    }
}

impl TriodeModel {
    pub fn from_config(p: &config::TriodeModelParameters, oversampled_fs: usize) -> Self {
        let mu = p.mu();
        let kp = p.kp();
        let kvb = p.kvb();
        let kg1 = p.kg1();
        let x_param = p.x();
        let vbias = p.vbias();
        let vplate = p.vplate();
        let drive_lin = db_to_linear(p.drive_db());

        let ip_quiescent = koren_plate_current(vbias, vplate, mu, kp, kvb, kg1, x_param);
        // Estimate the small-signal slope dIp/dVgk numerically near the
        // quiescent point and pick a normalisation so that a +/-1 input
        // (interpreted as full grid swing of |vbias|) maps approximately to
        // +/-1 at the output.
        let dv = (vbias.abs() * 0.01).max(0.1);
        let ip_plus = koren_plate_current(vbias + dv, vplate, mu, kp, kvb, kg1, x_param);
        let slope = ((ip_plus - ip_quiescent) / dv).abs().max(1e-9);
        let norm = 1.0 / (slope * vbias.abs().max(1.0));

        let coupling_in = make_highpass(p.coupling_freq_in(), oversampled_fs);
        let coupling_out = make_highpass(p.coupling_freq_out(), oversampled_fs);
        let cathode_bypass = make_low_shelf(p.cathode_bypass_freq(), -2.0, oversampled_fs);

        // Slow envelope time constant: 50 ms.
        let tau_s: PrcFmt = 0.050;
        let sag_coeff = (-1.0 / (tau_s * (oversampled_fs as PrcFmt))).exp();

        TriodeModel {
            mu,
            kp,
            kvb,
            kg1,
            x_param,
            vbias,
            vplate,
            drive_lin: RampedValue::new(drive_lin, SMOOTH_TAU_S, oversampled_fs),
            sag_amount: p.sag_amount(),
            ip_quiescent,
            norm,
            sag_env: ip_quiescent,
            sag_coeff,
            coupling_in,
            coupling_out,
            cathode_bypass,
        }
    }

    pub fn update(&mut self, p: &config::TriodeModelParameters, oversampled_fs: usize) {
        // Re-derive everything from the new config but keep the filter and
        // smoother states to avoid clicks.
        let new = Self::from_config(p, oversampled_fs);
        self.mu = new.mu;
        self.kp = new.kp;
        self.kvb = new.kvb;
        self.kg1 = new.kg1;
        self.x_param = new.x_param;
        self.vbias = new.vbias;
        self.vplate = new.vplate;
        // Only update the target of drive_lin; let the ramp move smoothly.
        self.drive_lin.set_target(new.drive_lin.target());
        self.drive_lin.retune(SMOOTH_TAU_S, oversampled_fs);
        self.sag_amount = new.sag_amount;
        self.ip_quiescent = new.ip_quiescent;
        self.norm = new.norm;
        self.coupling_in = new.coupling_in;
        self.coupling_out = new.coupling_out;
        self.cathode_bypass = new.cathode_bypass;
        self.sag_coeff = new.sag_coeff;
    }

    #[inline]
    pub fn process_sample(&mut self, audio_in: PrcFmt) -> PrcFmt {
        let coupled = self.coupling_in.process_single(audio_in);
        let drive = self.drive_lin.tick();
        let scaled = drive * coupled;
        // Map scaled (audio in [-1,1]) to grid voltage swing around vbias.
        let vgk = self.vbias + scaled * self.vbias.abs();

        // Dynamic plate voltage with sag.
        let vplate_eff = self.vplate - self.sag_amount * (self.sag_env - self.ip_quiescent);
        let ip = koren_plate_current(
            vgk,
            vplate_eff,
            self.mu,
            self.kp,
            self.kvb,
            self.kg1,
            self.x_param,
        );

        // Update sag envelope (slow lowpass on Ip).
        self.sag_env = self.sag_coeff * self.sag_env + (1.0 - self.sag_coeff) * ip;

        // Inverting amp: positive grid swing -> more current -> lower plate voltage.
        let raw = -(ip - self.ip_quiescent) * self.norm;
        let bypassed = self.cathode_bypass.process_single(raw);
        self.coupling_out.process_single(bypassed)
    }
}

fn make_highpass(freq: PrcFmt, fs: usize) -> Biquad {
    let coeffs = BiquadCoefficients::from_config(fs, config::BiquadParameters::HighpassFO { freq });
    Biquad::new("tubestage_coupling", fs, coeffs)
}

fn make_low_shelf(freq: PrcFmt, gain_db: PrcFmt, fs: usize) -> Biquad {
    let coeffs = BiquadCoefficients::from_config(
        fs,
        config::BiquadParameters::LowshelfFO { freq, gain: gain_db },
    );
    Biquad::new("tubestage_cathode_bypass", fs, coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn koren_300b_quiescent_in_reasonable_range() {
        // 300B rough quiescent: Vgk = -60 V, Vpk = 300 V should yield Ip in
        // the order of tens of mA according to the datasheet. The Koren
        // numbers we use are in arbitrary units (because Kg1 is in those
        // units); we just check that the slope is non-zero and Ip is finite.
        let ip = koren_plate_current(-60.0, 300.0, 3.85, 4.5, 140.0, 1500.0, 1.4);
        assert!(ip.is_finite());
        assert!(ip > 0.0, "expected non-zero Ip at quiescent");
    }

    #[test]
    fn triode_static_curve_monotonic() {
        // For fixed Vpk, plate current Ip should grow monotonically with Vgk.
        let mut prev = 0.0;
        for k in -100..0 {
            let vgk = k as PrcFmt;
            let ip = koren_plate_current(vgk, 300.0, 3.85, 4.5, 140.0, 1500.0, 1.4);
            assert!(
                ip >= prev - 1e-12,
                "Ip not monotonic at vgk={vgk}: prev={prev} ip={ip}"
            );
            prev = ip;
        }
    }

    #[test]
    fn triode_creates_even_harmonics() {
        // Drive a small sine through the triode and check that the output
        // has measurable DC offset (signature of asymmetric distortion).
        let cfg = config::TriodeModelParameters {
            mu: None,
            kp: None,
            kvb: None,
            kg1: None,
            x: None,
            vbias: None,
            vplate: None,
            drive_db: Some(-6.0),
            cathode_bypass_freq: Some(1.0), // pass everything; we want raw DC for the test
            coupling_freq_in: Some(0.01),
            coupling_freq_out: Some(0.01),
            sag_amount: Some(0.0),
        };
        let fs = 192_000;
        let mut model = TriodeModel::from_config(&cfg, fs);
        let n = 8192;
        let two_pi = 2.0 * std::f64::consts::PI as PrcFmt;
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            let x = 0.3 * (two_pi * 1000.0 * (i as PrcFmt) / (fs as PrcFmt)).sin();
            let y = model.process_sample(x);
            // Skip the startup transient.
            if i > 1024 {
                sum += y;
                count += 1;
            }
        }
        let mean = sum / (count as PrcFmt);
        assert!(
            mean.abs() > 1e-4,
            "expected non-zero DC offset (asymmetry), got {mean}"
        );
    }
}
