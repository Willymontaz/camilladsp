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

// `TubeStage` filter: simulates an audiophile single-ended-triode amplifier.
// Pipeline (executed at the oversampled rate, except for the initial up-
// and final down-sampling):
//
//   in -> upsample x M -> [DC blocker] -> waveshaper (Polynomial or Triode)
//      -> [transformer HF shelf] -> [DC blocker] -> makeup gain -> downsample / M -> out

pub mod oversample;
pub mod polynomial;
pub mod ramp;
pub mod triode;

use crate::PrcFmt;
use crate::Res;
use crate::config;
use crate::filters::Filter;
use crate::filters::biquad::{Biquad, BiquadCoefficients};
use crate::utils::decibels::db_to_linear;

use polynomial::PolynomialModel;
use ramp::RampedValue;
use triode::TriodeModel;

use oversample::{HalfbandFirOversampler, IdentityOversampler, Oversampler, RubatoOversampler};

/// Time constant for the makeup-gain smoother (s).
const MAKEUP_TAU_S: PrcFmt = 0.020;

#[derive(Clone, Copy, Debug)]
struct DcBlocker {
    x1: PrcFmt,
    y1: PrcFmt,
    r: PrcFmt,
}

impl DcBlocker {
    fn new(fs: usize) -> Self {
        // 1-pole high-pass: y[n] = x[n] - x[n-1] + r*y[n-1]
        // Place the corner at ~5 Hz to remove DC and infrasonic content
        // without affecting the audio band.
        let cutoff_hz: PrcFmt = 5.0;
        let r = (-2.0 * (std::f64::consts::PI as PrcFmt) * cutoff_hz / (fs as PrcFmt)).exp();
        Self {
            x1: 0.0,
            y1: 0.0,
            r,
        }
    }

    #[inline]
    fn process_single(&mut self, x: PrcFmt) -> PrcFmt {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

enum Model {
    Polynomial(PolynomialModel),
    // Boxed because the Triode variant is significantly larger than the
    // Polynomial one (multiple Biquads + ramps + envelope state).
    Triode(Box<TriodeModel>),
}

pub struct TubeStage {
    name: String,
    samplerate: usize,
    chunksize: usize,

    oversampler: Box<dyn Oversampler>,
    factor: usize,

    model: Model,

    transformer: Option<Biquad>,
    transformer_enabled: bool,

    dc_blocker_in: Option<DcBlocker>,
    dc_blocker_out: Option<DcBlocker>,

    makeup_gain: RampedValue,

    // Scratch buffer for the oversampled domain (length = factor * chunksize).
    scratch: Vec<PrcFmt>,
}

impl TubeStage {
    pub fn from_config(
        name: &str,
        params: config::TubeStageParameters,
        samplerate: usize,
        chunksize: usize,
    ) -> Self {
        debug!(
            "Creating TubeStage filter '{}' at {} Hz, chunksize {}",
            name, samplerate, chunksize
        );

        let oversampling = params.oversampling();
        let factor = oversampling.factor().max(1);
        let oversampled_fs = samplerate * factor;

        let oversampler: Box<dyn Oversampler> = if factor == 1 {
            Box::new(IdentityOversampler)
        } else {
            match oversampling.backend() {
                config::OversamplingBackend::HalfbandFir => {
                    Box::new(HalfbandFirOversampler::new(factor, chunksize))
                }
                config::OversamplingBackend::Rubato => {
                    Box::new(RubatoOversampler::new(factor, chunksize))
                }
            }
        };

        let model = match &params.model {
            config::TubeModel::Polynomial(p) => {
                Model::Polynomial(PolynomialModel::from_config(p, oversampled_fs))
            }
            config::TubeModel::Triode(p) => {
                Model::Triode(Box::new(TriodeModel::from_config(p, oversampled_fs)))
            }
        };

        let transformer_cfg = params.transformer();
        let transformer_enabled = transformer_cfg.enabled();
        let transformer = if transformer_enabled {
            Some(make_transformer_shelf(
                transformer_cfg.hf_shelf_freq(),
                transformer_cfg.hf_shelf_gain_db(),
                oversampled_fs,
            ))
        } else {
            None
        };

        let dc_blocker_enabled = params.dc_blocker();
        let dc_blocker_in = if dc_blocker_enabled {
            Some(DcBlocker::new(oversampled_fs))
        } else {
            None
        };
        let dc_blocker_out = if dc_blocker_enabled {
            Some(DcBlocker::new(oversampled_fs))
        } else {
            None
        };

        let makeup_gain = RampedValue::new(
            db_to_linear(params.makeup_gain_db()),
            MAKEUP_TAU_S,
            oversampled_fs,
        );

        TubeStage {
            name: name.to_string(),
            samplerate,
            chunksize,
            oversampler,
            factor,
            model,
            transformer,
            transformer_enabled,
            dc_blocker_in,
            dc_blocker_out,
            makeup_gain,
            scratch: vec![0.0; factor * chunksize],
        }
    }
}

fn make_transformer_shelf(freq: PrcFmt, gain_db: PrcFmt, fs: usize) -> Biquad {
    let coeffs = BiquadCoefficients::from_config(
        fs,
        config::BiquadParameters::HighshelfFO {
            freq,
            gain: gain_db,
        },
    );
    Biquad::new("tubestage_transformer", fs, coeffs)
}

impl Filter for TubeStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_waveform(&mut self, waveform: &mut [PrcFmt]) -> Res<()> {
        // Resize scratch if the chunk size changed (e.g. variable chunk size from
        // a non-fixed-block source). For the rubato backend this requires a
        // rebuild; we handle that defensively.
        let n = waveform.len();
        if n == 0 {
            return Ok(());
        }
        let expected_high = self.factor * n;
        if self.scratch.len() != expected_high {
            self.scratch.resize(expected_high, 0.0);
            // If the chunk size changed and we're using rubato, the existing
            // resampler is no longer valid; fall back to the halfband backend.
            // This is a defensive path; in practice CamillaDSP keeps a fixed
            // waveform_length per pipeline build.
            if n != self.chunksize {
                debug!(
                    "TubeStage '{}' chunk size changed from {} to {}, rebuilding oversampler with halfband backend",
                    self.name, self.chunksize, n
                );
                self.chunksize = n;
                if self.factor == 1 {
                    self.oversampler = Box::new(IdentityOversampler);
                } else {
                    self.oversampler = Box::new(HalfbandFirOversampler::new(self.factor, n));
                }
            }
        }

        // Up-sample.
        self.oversampler.upsample(waveform, &mut self.scratch);

        // Process at the oversampled rate.
        for sample in self.scratch.iter_mut() {
            let mut s = *sample;
            if let Some(dc) = self.dc_blocker_in.as_mut() {
                s = dc.process_single(s);
            }
            s = match &mut self.model {
                Model::Polynomial(m) => m.process_sample(s),
                Model::Triode(m) => m.process_sample(s),
            };
            if let Some(b) = self.transformer.as_mut() {
                s = b.process_single(s);
            }
            if let Some(dc) = self.dc_blocker_out.as_mut() {
                s = dc.process_single(s);
            }
            s *= self.makeup_gain.tick();
            *sample = s;
        }

        // Down-sample back into the input buffer.
        self.oversampler.downsample(&self.scratch, waveform);

        Ok(())
    }

    fn update_parameters(&mut self, conf: config::Filter) {
        if let config::Filter::TubeStage {
            parameters: params, ..
        } = conf
        {
            // Rebuild the oversampler only if the factor or backend changed.
            let oversampling = params.oversampling();
            let new_factor = oversampling.factor().max(1);
            if new_factor != self.factor {
                debug!(
                    "TubeStage '{}' oversampling factor changed from {} to {}",
                    self.name, self.factor, new_factor
                );
                self.factor = new_factor;
                let new_high = new_factor * self.chunksize;
                self.scratch = vec![0.0; new_high];
                self.oversampler = if new_factor == 1 {
                    Box::new(IdentityOversampler)
                } else {
                    match oversampling.backend() {
                        config::OversamplingBackend::HalfbandFir => {
                            Box::new(HalfbandFirOversampler::new(new_factor, self.chunksize))
                        }
                        config::OversamplingBackend::Rubato => {
                            Box::new(RubatoOversampler::new(new_factor, self.chunksize))
                        }
                    }
                };
            }
            let oversampled_fs = self.samplerate * self.factor.max(1);

            // Update the active model in place when possible to avoid clicks.
            match (&mut self.model, &params.model) {
                (Model::Polynomial(m), config::TubeModel::Polynomial(p)) => {
                    m.update(p, oversampled_fs)
                }
                (Model::Triode(m), config::TubeModel::Triode(p)) => m.update(p, oversampled_fs),
                (_, config::TubeModel::Polynomial(p)) => {
                    self.model = Model::Polynomial(PolynomialModel::from_config(p, oversampled_fs));
                }
                (_, config::TubeModel::Triode(p)) => {
                    self.model =
                        Model::Triode(Box::new(TriodeModel::from_config(p, oversampled_fs)));
                }
            }

            let transformer_cfg = params.transformer();
            self.transformer_enabled = transformer_cfg.enabled();
            self.transformer = if self.transformer_enabled {
                Some(make_transformer_shelf(
                    transformer_cfg.hf_shelf_freq(),
                    transformer_cfg.hf_shelf_gain_db(),
                    oversampled_fs,
                ))
            } else {
                None
            };

            let dc_enabled = params.dc_blocker();
            self.dc_blocker_in = if dc_enabled {
                Some(self.dc_blocker_in.unwrap_or_else(|| DcBlocker::new(oversampled_fs)))
            } else {
                None
            };
            self.dc_blocker_out = if dc_enabled {
                Some(self.dc_blocker_out.unwrap_or_else(|| DcBlocker::new(oversampled_fs)))
            } else {
                None
            };

            self.makeup_gain
                .set_target(db_to_linear(params.makeup_gain_db()));
            self.makeup_gain.retune(MAKEUP_TAU_S, oversampled_fs);
        } else {
            panic!("Invalid config change for TubeStage filter");
        }
    }
}

/// Validate the tube-stage config. Returns Ok if no obvious problem is found.
pub fn validate_config(samplerate: usize, params: &config::TubeStageParameters) -> Res<()> {
    let factor = params.oversampling().factor().max(1);
    let max_freq = (samplerate as PrcFmt) * (factor as PrcFmt) / 2.0;
    let transformer = params.transformer();
    if transformer.enabled() {
        let f = transformer.hf_shelf_freq();
        if f <= 0.0 || f >= max_freq {
            let msg = format!(
                "TubeStage transformer hf_shelf_freq {f} Hz must be in (0, {max_freq}) Hz at sample rate {samplerate} Hz"
            );
            return Err(config::ConfigError::new(&msg).into());
        }
    }
    if let config::TubeModel::Triode(p) = &params.model {
        if p.mu() <= 0.0 || p.kp() <= 0.0 || p.kg1() <= 0.0 {
            let msg = "TubeStage Triode mu, kp and kg1 must be positive".to_string();
            return Err(config::ConfigError::new(&msg).into());
        }
        if p.kvb() <= 0.0 {
            let msg = "TubeStage Triode kvb must be positive".to_string();
            return Err(config::ConfigError::new(&msg).into());
        }
        if p.x() <= 0.0 {
            let msg = "TubeStage Triode x must be positive".to_string();
            return Err(config::ConfigError::new(&msg).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tubestage_polynomial_runs_clean() {
        let cfg = config::TubeStageParameters {
            model: config::TubeModel::Polynomial(config::PolynomialModelParameters {
                drive_db: Some(0.0),
                bias: Some(0.0),
                k2: Some(0.0),
                k3: Some(0.0),
                k4: Some(0.0),
            }),
            transformer: Some(config::TransformerParameters {
                enabled: Some(false),
                hf_shelf_freq: None,
                hf_shelf_gain_db: None,
            }),
            dc_blocker: Some(false),
            makeup_gain_db: Some(0.0),
            oversampling: Some(config::OversamplingParameters {
                factor: Some(config::OversamplingFactor::F1),
                backend: Some(config::OversamplingBackend::HalfbandFir),
            }),
        };
        let mut filter = TubeStage::from_config("t", cfg, 48000, 256);
        let mut buf = vec![0.5; 256];
        filter.process_waveform(&mut buf).unwrap();
        // With drive=0 dB, all coefficients zero, transformer & DC blocker off,
        // factor=1, the filter is a passthrough.
        for &v in &buf {
            assert!((v - 0.5).abs() < 1e-12, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn tubestage_polynomial_oversampled_passthrough() {
        // With all coefficients zero, the filter is essentially a (delayed)
        // passthrough through the oversampling chain. We verify amplitude
        // preservation rather than exact alignment.
        let cfg = config::TubeStageParameters {
            model: config::TubeModel::Polynomial(config::PolynomialModelParameters {
                drive_db: Some(0.0),
                bias: Some(0.0),
                k2: Some(0.0),
                k3: Some(0.0),
                k4: Some(0.0),
            }),
            transformer: Some(config::TransformerParameters {
                enabled: Some(false),
                hf_shelf_freq: None,
                hf_shelf_gain_db: None,
            }),
            dc_blocker: Some(false),
            makeup_gain_db: Some(0.0),
            oversampling: Some(config::OversamplingParameters {
                factor: Some(config::OversamplingFactor::F2),
                backend: Some(config::OversamplingBackend::HalfbandFir),
            }),
        };
        let mut filter = TubeStage::from_config("t", cfg, 48000, 1024);
        // Prime through a few blocks of zeros so the delay line is settled.
        for _ in 0..3 {
            let mut prime = vec![0.0; 1024];
            filter.process_waveform(&mut prime).unwrap();
        }
        let two_pi = 2.0 * std::f64::consts::PI as PrcFmt;
        let mut buf: Vec<PrcFmt> = (0..1024)
            .map(|i| 0.5 * (two_pi * 1000.0 * (i as PrcFmt) / 48000.0).sin())
            .collect();
        let original = buf.clone();
        filter.process_waveform(&mut buf).unwrap();
        // Compare RMS amplitudes; skip an initial window that may include the
        // tail of the transient from previous blocks.
        let skip = 64usize;
        let n = 1024 - skip;
        let in_rms: PrcFmt =
            (original[skip..].iter().map(|x| x * x).sum::<PrcFmt>() / n as PrcFmt).sqrt();
        let out_rms: PrcFmt =
            (buf[skip..].iter().map(|x| x * x).sum::<PrcFmt>() / n as PrcFmt).sqrt();
        let rel = (out_rms - in_rms).abs() / in_rms;
        assert!(
            rel < 0.05,
            "oversampled passthrough amplitude mismatch: in={in_rms} out={out_rms}"
        );
    }

    #[test]
    fn tubestage_hot_swap_drive_no_clicks() {
        // Build a polynomial-mode TubeStage and change the drive at runtime.
        // The output should not have a sample-level discontinuity.
        let cfg = config::TubeStageParameters {
            model: config::TubeModel::Polynomial(config::PolynomialModelParameters {
                drive_db: Some(0.0),
                bias: Some(0.0),
                k2: Some(0.0),
                k3: Some(0.0),
                k4: Some(0.0),
            }),
            transformer: Some(config::TransformerParameters {
                enabled: Some(false),
                hf_shelf_freq: None,
                hf_shelf_gain_db: None,
            }),
            dc_blocker: Some(false),
            makeup_gain_db: Some(0.0),
            oversampling: Some(config::OversamplingParameters {
                factor: Some(config::OversamplingFactor::F1),
                backend: Some(config::OversamplingBackend::HalfbandFir),
            }),
        };
        let mut filter = TubeStage::from_config("t", cfg.clone(), 48000, 64);
        let mut buf = vec![1.0; 64];
        filter.process_waveform(&mut buf).unwrap();

        // Now request a +20 dB drive and a +6 dB makeup gain change. The next
        // block must not jump to the new gain immediately.
        let new_cfg = config::Filter::TubeStage {
            description: None,
            parameters: config::TubeStageParameters {
                model: config::TubeModel::Polynomial(config::PolynomialModelParameters {
                    drive_db: Some(20.0),
                    bias: Some(0.0),
                    k2: Some(0.0),
                    k3: Some(0.0),
                    k4: Some(0.0),
                }),
                transformer: Some(config::TransformerParameters {
                    enabled: Some(false),
                    hf_shelf_freq: None,
                    hf_shelf_gain_db: None,
                }),
                dc_blocker: Some(false),
                makeup_gain_db: Some(6.0),
                oversampling: Some(config::OversamplingParameters {
                    factor: Some(config::OversamplingFactor::F1),
                    backend: Some(config::OversamplingBackend::HalfbandFir),
                }),
            },
        };
        filter.update_parameters(new_cfg);
        let mut buf2 = vec![1.0; 64];
        filter.process_waveform(&mut buf2).unwrap();
        let last_old = buf[buf.len() - 1];
        let first_new = buf2[0];
        assert!(
            (first_new - last_old).abs() < 0.5,
            "click on hot-swap: last_old={last_old} first_new={first_new}"
        );
    }
}
