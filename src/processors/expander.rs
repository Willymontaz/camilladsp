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

// Upward expander: source-domain dynamics restoration.
//
// The inverse of the downward compressor. Where the compressor *attenuates*
// signal above a threshold, the expander *boosts* it, re-opening the macro-
// dynamics that loudness-war mastering crushes.
//
// The sidechain is built for dynamics restoration rather than transient
// catching, so the detector math is a step up from the classic compressor:
//
//   * Channels are linked by an RMS (power) sum, not a plain sample sum, so the
//     detected level is independent of inter-channel correlation. A plain L+R
//     sum reads correlated program up to +6 dB hot and cancels anti-correlated
//     content, both of which would modulate the expansion depth spuriously.
//   * The dB envelope is followed with a smooth decoupled peak detector
//     (Giannoulis, Massberg & Reiss, "Digital Dynamic Range Compressor
//     Design", JAES 2012) instead of a naive branching (attack-if-rising)
//     detector, decoupling the effective attack time from the release state.
//
// On top of that it changes the gain law, adds an optional peak-safety limiter
// (expansion raises peaks), and gates the expansion depth by the program's
// crest factor so it stays near-dormant on material that is already dynamic.

use std::sync::Arc;

use crate::PrcFmt;
use crate::ProcessingParameters;
use crate::Res;
use crate::audiochunk::AudioChunk;
use crate::config;
use crate::config::ExpanderMode;
use crate::filters::limiter::Limiter;
use crate::processors::Processor;
use crate::utils::decibels::db_to_linear;

/// Time constant (seconds) of the one-pole mean-square average feeding the crest
/// estimate. A few hundred ms captures program-level RMS without chasing
/// individual cycles.
const CREST_RMS_TC: PrcFmt = 0.4;
/// Release time constant (seconds) of the peak follower feeding the crest
/// estimate. Slow enough to hold recent transient peaks across quiet stretches,
/// so the peak-to-RMS ratio reflects macro-dynamics rather than instantaneous
/// waveform shape.
const CREST_PEAK_TC: PrcFmt = 1.0;
/// Floor for the running mean-square / peak so the crest ratio stays finite
/// during silence.
const CREST_FLOOR: PrcFmt = 1.0e-20;

#[derive(Clone, Debug)]
pub struct Expander {
    pub name: String,
    pub channels: usize,
    pub monitor_channels: Vec<usize>,
    pub process_channels: Vec<usize>,
    pub attack: PrcFmt,
    pub release: PrcFmt,
    pub threshold: PrcFmt,
    pub ratio: PrcFmt,
    pub max_gain: PrcFmt,
    /// Half the soft-knee width (dB). The gain law transitions from no expansion
    /// to full ratio over `[-half_knee, +half_knee]` around the threshold; 0 gives
    /// a hard knee.
    pub half_knee: PrcFmt,
    pub mode: ExpanderMode,
    pub makeup_gain: PrcFmt,
    pub limiter: Option<Limiter>,
    pub samplerate: usize,
    pub scratch: Vec<PrcFmt>,
    // Smooth decoupled peak detector state (dB): `level_peak` is the
    // instantaneous-attack / exponential-release peak follower, `level_env` its
    // attack-smoothed output that drives the gain computer.
    pub level_peak: PrcFmt,
    pub level_env: PrcFmt,
    // Adaptive (program-relative) threshold. When `adaptive`, the effective
    // threshold tracks `level_slow` (a slow envelope of the program level) plus
    // `relative_offset`, so only transients that stick out above the current
    // musical level are expanded; the fixed `threshold` becomes an absolute
    // floor. `level_slow` is a slowly-smoothed copy of `level_peak` so it
    // converges with `level_env` on steady material (applied gain -> 0).
    pub adaptive: bool,
    pub relative_offset: PrcFmt,
    pub level_slow: PrcFmt,
    pub slow_coef: PrcFmt,
    // Effective threshold (dB) of the last processed sample, for telemetry.
    last_eff_threshold: PrcFmt,
    // Crest-factor driven ratio: when enabled, program crest scales the
    // effective expansion ratio (compressed -> full ratio, dynamic -> ratio 1).
    pub crest_gate: bool,
    pub crest_floor_db: PrcFmt,
    pub crest_ceiling_db: PrcFmt,
    // Running crest estimator state.
    crest_ms: PrcFmt,
    crest_peak: PrcFmt,
    crest_ms_coeff: PrcFmt,
    crest_peak_coeff: PrcFmt,
    processing_params: Arc<ProcessingParameters>,
}

impl Expander {
    /// Creates an Expander from a config struct
    pub fn from_config(
        name: &str,
        config: config::ExpanderParameters,
        samplerate: usize,
        chunksize: usize,
        processing_params: Arc<ProcessingParameters>,
    ) -> Self {
        let name = name.to_string();
        let channels = config.channels;
        let srate = samplerate as PrcFmt;
        let mut monitor_channels = config.monitor_channels();
        if monitor_channels.is_empty() {
            for n in 0..channels {
                monitor_channels.push(n);
            }
        }
        let mut process_channels = config.process_channels();
        if process_channels.is_empty() {
            for n in 0..channels {
                process_channels.push(n);
            }
        }
        let attack = (-1.0 / srate / config.attack).exp();
        let release = (-1.0 / srate / config.release).exp();
        let crest_ms_coeff = (-1.0 / srate / CREST_RMS_TC).exp();
        let crest_peak_coeff = (-1.0 / srate / CREST_PEAK_TC).exp();
        let slow_coef = (-1.0 / srate / config.adapt_time()).exp();
        let adaptive = config.adaptive_threshold();
        // The crest measurement drives the effective expansion ratio (see
        // `calculate_linear_gain`): full ratio on compressed program, tapering to
        // ratio 1 (no expansion) on already-dynamic material. On by default, so
        // the expander does more on crushed material and backs off on good
        // material. Orthogonal to the adaptive threshold (which sets *where* the
        // effect acts relative to the running level).
        let crest_gate = config.crest_gate();
        let clip_limit = config.clip_limit.map(db_to_linear);

        let scratch = vec![0.0; chunksize];

        debug!(
            "Creating expander '{}', channels: {}, monitor_channels: {:?}, process_channels: {:?}, attack: {}, release: {}, threshold: {}, ratio: {}, max_gain_db: {}, knee_db: {}, mode: {:?}, makeup_gain: {}, adaptive_threshold: {}, relative_offset_db: {}, adapt_time: {}, crest_gate: {}, crest_floor_db: {}, crest_ceiling_db: {}, soft_clip: {}, clip_limit: {:?}",
            name,
            channels,
            monitor_channels,
            process_channels,
            attack,
            release,
            config.threshold,
            config.ratio,
            config.max_gain_db(),
            config.knee_db(),
            config.mode(),
            config.makeup_gain(),
            adaptive,
            config.relative_offset_db(),
            config.adapt_time(),
            crest_gate,
            config.crest_floor_db(),
            config.crest_ceiling_db(),
            config.soft_clip(),
            clip_limit
        );
        let limiter = if let Some(limit) = config.clip_limit {
            let limitconf = config::LimiterParameters {
                clip_limit: limit,
                soft_clip: config.soft_clip,
            };
            Some(Limiter::from_config("Limiter", limitconf))
        } else {
            None
        };

        Expander {
            name,
            channels,
            monitor_channels,
            process_channels,
            attack,
            release,
            threshold: config.threshold,
            ratio: config.ratio,
            max_gain: config.max_gain_db(),
            half_knee: 0.5 * config.knee_db(),
            mode: config.mode(),
            makeup_gain: config.makeup_gain(),
            limiter,
            samplerate,
            scratch,
            level_peak: -100.0,
            level_env: -100.0,
            adaptive,
            relative_offset: config.relative_offset_db(),
            level_slow: -100.0,
            slow_coef,
            last_eff_threshold: config.threshold,
            crest_gate,
            crest_floor_db: config.crest_floor_db(),
            crest_ceiling_db: config.crest_ceiling_db(),
            // Start the estimator with a high crest (silent RMS floor, full-scale
            // peak) so the gate begins closed: on startup the expander stays
            // dormant and only engages once it has measured the program to be
            // compressed. This errs on the side of not touching good material.
            crest_ms: CREST_FLOOR,
            crest_peak: 1.0,
            crest_ms_coeff,
            crest_peak_coeff,
            processing_params,
        }
    }

    /// Link the monitored channels into a single per-sample magnitude in
    /// self.scratch, using an RMS (power) sum: `m[n] = sqrt(mean_ch(x_ch[n]^2))`.
    /// Unlike a plain sample sum this is independent of inter-channel
    /// correlation, so correlated and anti-correlated program are detected at the
    /// same level. The result is non-negative and feeds both the crest estimator
    /// and the loudness envelope.
    fn compute_linked_magnitude(&mut self, input: &AudioChunk) {
        let ch0 = self.monitor_channels[0];
        for (acc, val) in self.scratch.iter_mut().zip(input.waveforms[ch0].iter()) {
            *acc = *val * *val;
        }
        for ch in self.monitor_channels.iter().skip(1) {
            for (acc, val) in self.scratch.iter_mut().zip(input.waveforms[*ch].iter()) {
                *acc += *val * *val;
            }
        }
        let norm = 1.0 / self.monitor_channels.len() as PrcFmt;
        for acc in self.scratch.iter_mut() {
            *acc = (*acc * norm).sqrt();
        }
    }

    /// Update the running crest estimate from the (linear) linked monitor
    /// magnitude currently in self.scratch, and return the crest scale in [0, 1]
    /// that scales the effective expansion ratio (see `calculate_linear_gain`):
    /// 1.0 for compressed material (crest <= floor) so the full ratio applies,
    /// ramping to 0.0 as the crest approaches the ceiling (already dynamic) where
    /// the effective ratio collapses to 1 (no expansion). Must be called before
    /// `estimate_loudness`, which overwrites the scratch buffer.
    fn update_crest_scale(&mut self) -> PrcFmt {
        for val in self.scratch.iter() {
            let x2 = *val * *val;
            self.crest_ms = self.crest_ms_coeff * self.crest_ms + (1.0 - self.crest_ms_coeff) * x2;
            let mag = val.abs();
            if mag > self.crest_peak {
                self.crest_peak = mag;
            } else {
                self.crest_peak *= self.crest_peak_coeff;
            }
        }
        let rms = self.crest_ms.max(CREST_FLOOR).sqrt();
        let peak = self.crest_peak.max(CREST_FLOOR);
        let crest_db = 20.0 * (peak / rms).log10();
        if !self.crest_gate {
            return 1.0;
        }
        if crest_db <= self.crest_floor_db {
            1.0
        } else if crest_db >= self.crest_ceiling_db {
            0.0
        } else {
            (self.crest_ceiling_db - crest_db) / (self.crest_ceiling_db - self.crest_floor_db)
        }
    }

    /// Follow the loudness envelope (dB) of the linked magnitude in self.scratch,
    /// replacing each sample with its dB excess `over` above the (possibly
    /// adaptive) threshold — the quantity the gain law consumes.
    ///
    /// Uses the smooth decoupled peak detector of Giannoulis, Massberg & Reiss
    /// (JAES 2012): an instantaneous-attack, exponential-release peak follower
    /// (`level_peak`) whose output is then attack-smoothed (`level_env`). This
    /// gives a single, well-defined "rising level == attack" ballistic for every
    /// expander mode while decoupling the effective attack time from the release
    /// state, unlike a naive branching detector that switches coefficient on the
    /// sign of each sample-to-sample step.
    ///
    /// A second, slower smoothing of `level_peak` (`level_slow`) tracks the
    /// program level. When `adaptive`, the threshold follows it
    /// (`level_slow + relative_offset`, floored at the fixed `threshold`), so the
    /// excess reflects how far a transient pokes above the *current* musical
    /// level rather than above a fixed line. Because `level_slow` and `level_env`
    /// converge on steady material, the excess (and thus the boost) returns to
    /// ~0 in sustained passages and only bounces up on transients.
    fn estimate_loudness(&mut self) {
        for val in self.scratch.iter_mut() {
            let level = 20.0 * (*val + 1.0e-9).log10();
            // Peak follower: jumps up to the level instantly, decays toward it
            // with the release coefficient.
            let released = self.release * self.level_peak + (1.0 - self.release) * level;
            self.level_peak = level.max(released);
            // One-pole attack smoothing of the peak follower's output.
            self.level_env = self.attack * self.level_env + (1.0 - self.attack) * self.level_peak;
            // Slow program-level envelope (same source, longer time constant).
            self.level_slow = self.slow_coef * self.level_slow + (1.0 - self.slow_coef) * self.level_peak;
            let eff_threshold = if self.adaptive {
                self.threshold.max(self.level_slow + self.relative_offset)
            } else {
                self.threshold
            };
            self.last_eff_threshold = eff_threshold;
            *val = self.level_env - eff_threshold;
        }
    }

    /// Calculate linear gain from the per-sample threshold excess `over` already
    /// in self.scratch (written by `estimate_loudness`), replacing it with the
    /// per-sample linear gain. Returns the signed applied expansion (dB, before
    /// makeup) of largest magnitude across the chunk, for activity telemetry
    /// (0 dB == idle).
    ///
    /// `crest_scale` (in [0, 1]) drives the **effective expansion ratio** rather
    /// than gating the output: `effective_ratio = 1 + (ratio - 1) * crest_scale`.
    /// So a compressed program (crest_scale ~1) gets the full ratio while an
    /// already-dynamic program (crest_scale ~0) collapses to ratio 1, i.e. no
    /// expansion. This makes the expander do *more* on crushed material and back
    /// off on good material, instead of applying a fixed ratio and then gating —
    /// which scaled the wrong way with the program's own dynamics.
    fn calculate_linear_gain(&mut self, crest_scale: PrcFmt) -> PrcFmt {
        let effective_ratio = 1.0 + (self.ratio - 1.0) * crest_scale;
        let mut peak_expansion: PrcFmt = 0.0;
        for val in self.scratch.iter_mut() {
            let over = *val;
            // Raw expansion gain in dB, positive above threshold (upward) and
            // negative below (downward), proportional to how far the envelope
            // exceeds/undershoots the threshold. A soft knee smooths the slope
            // discontinuity at the threshold so program dwelling near it is not
            // gain-modulated abruptly (see `soft_knee_upward`).
            let expansion = match self.mode {
                ExpanderMode::Upward => soft_knee_upward(over, effective_ratio, self.half_knee),
                ExpanderMode::Downward => -soft_knee_upward(-over, effective_ratio, self.half_knee),
                // Linear through the origin: already C1-continuous, no knee needed.
                ExpanderMode::Both => over * (effective_ratio - 1.0),
            };
            // A hot passage (or a deep null in downward mode) must not run away.
            let applied = expansion.clamp(-self.max_gain, self.max_gain);
            if applied.abs() > peak_expansion.abs() {
                peak_expansion = applied;
            }
            *val = db_to_linear(applied + self.makeup_gain);
        }
        peak_expansion
    }

    fn apply_gain(&self, input: &mut [PrcFmt]) {
        for (val, gain) in input.iter_mut().zip(self.scratch.iter()) {
            *val *= gain;
        }
    }

    fn apply_limiter(&self, input: &mut [PrcFmt]) {
        if let Some(limiter) = &self.limiter {
            limiter.apply_clip(input);
        }
    }
}

/// Soft-knee gain law for an upward expander: dB expansion as a function of how
/// far the envelope sits above threshold (`over`). Implements the quadratic knee
/// of Giannoulis, Massberg & Reiss (JAES 2012): expansion is zero at/below
/// `over = -half_knee`, ramps with a continuous slope through the knee, and joins
/// the full `over*(ratio-1)` line at/above `over = +half_knee`. With
/// `half_knee == 0` it collapses to the hard knee `max(over, 0)*(ratio-1)`.
///
/// The quadratic is chosen to match both value and slope at each knee edge
/// (value 0 / slope 0 at `-half_knee`; value `half_knee*(ratio-1)` / slope
/// `ratio-1` at `+half_knee`), so the gain trajectory is C1-continuous and free
/// of the derivative kink a hard threshold would impose. Downward expansion
/// reuses this by mirroring: `-soft_knee_upward(-over, ..)`.
fn soft_knee_upward(over: PrcFmt, ratio: PrcFmt, half_knee: PrcFmt) -> PrcFmt {
    let slope = ratio - 1.0;
    if half_knee <= 0.0 || over >= half_knee {
        over.max(0.0) * slope
    } else if over <= -half_knee {
        0.0
    } else {
        let t = over + half_knee;
        slope * t * t / (4.0 * half_knee)
    }
}

impl Processor for Expander {
    fn name(&self) -> &str {
        &self.name
    }

    /// Apply an Expander to an AudioChunk, modifying it in-place.
    fn process_chunk(&mut self, input: &mut AudioChunk) -> Res<()> {
        self.compute_linked_magnitude(input);
        let crest_scale = self.update_crest_scale();
        self.estimate_loudness();
        let expansion = self.calculate_linear_gain(crest_scale);
        for ch in self.process_channels.iter() {
            self.apply_gain(&mut input.waveforms[*ch]);
            self.apply_limiter(&mut input.waveforms[*ch]);
        }
        self.processing_params.set_expansion_gain(expansion as f32);
        // Surface the currently-active threshold (adaptive when enabled, the
        // fixed threshold otherwise) so a monitor can watch it track the program.
        self.processing_params
            .set_adaptive_threshold(self.last_eff_threshold as f32);
        Ok(())
    }

    fn update_parameters(&mut self, config: config::Processor) {
        if let config::Processor::Expander {
            parameters: config, ..
        } = config
        {
            let channels = config.channels;
            let srate = self.samplerate as PrcFmt;
            let mut monitor_channels = config.monitor_channels();
            if monitor_channels.is_empty() {
                for n in 0..channels {
                    monitor_channels.push(n);
                }
            }
            let mut process_channels = config.process_channels();
            if process_channels.is_empty() {
                for n in 0..channels {
                    process_channels.push(n);
                }
            }
            let attack = (-1.0 / srate / config.attack).exp();
            let release = (-1.0 / srate / config.release).exp();
            let slow_coef = (-1.0 / srate / config.adapt_time()).exp();
            let adaptive = config.adaptive_threshold();
            let crest_gate = config.crest_gate();
            let clip_limit = config.clip_limit.map(db_to_linear);

            let limiter = if let Some(limit) = config.clip_limit {
                let limitconf = config::LimiterParameters {
                    clip_limit: limit,
                    soft_clip: config.soft_clip,
                };
                Some(Limiter::from_config("Limiter", limitconf))
            } else {
                None
            };

            self.channels = channels;
            self.monitor_channels = monitor_channels;
            self.process_channels = process_channels;
            self.attack = attack;
            self.release = release;
            self.threshold = config.threshold;
            self.ratio = config.ratio;
            self.max_gain = config.max_gain_db();
            self.half_knee = 0.5 * config.knee_db();
            self.mode = config.mode();
            self.makeup_gain = config.makeup_gain();
            self.adaptive = adaptive;
            self.relative_offset = config.relative_offset_db();
            self.slow_coef = slow_coef;
            self.crest_gate = crest_gate;
            self.crest_floor_db = config.crest_floor_db();
            self.crest_ceiling_db = config.crest_ceiling_db();
            self.limiter = limiter;

            debug!(
                "Updated expander '{}', monitor_channels: {:?}, process_channels: {:?}, attack: {}, release: {}, threshold: {}, ratio: {}, max_gain_db: {}, knee_db: {}, mode: {:?}, makeup_gain: {}, adaptive_threshold: {}, relative_offset_db: {}, adapt_time: {}, crest_gate: {}, crest_floor_db: {}, crest_ceiling_db: {}, soft_clip: {}, clip_limit: {:?}",
                self.name,
                self.monitor_channels,
                self.process_channels,
                attack,
                release,
                config.threshold,
                config.ratio,
                config.max_gain_db(),
                config.knee_db(),
                config.mode(),
                config.makeup_gain(),
                adaptive,
                config.relative_offset_db(),
                config.adapt_time(),
                crest_gate,
                config.crest_floor_db(),
                config.crest_ceiling_db(),
                config.soft_clip(),
                clip_limit
            );
        } else {
            // This should never happen unless there is a bug somewhere else
            panic!("Invalid config change!");
        }
    }
}

/// Validate the expander config, to give a helpful message instead of a panic.
pub fn validate_expander(config: &config::ExpanderParameters) -> Res<()> {
    let channels = config.channels;
    if config.attack <= 0.0 {
        let msg = "Attack value must be larger than zero.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.release <= 0.0 {
        let msg = "Release value must be larger than zero.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.ratio < 1.0 {
        let msg = "Ratio must be greater than or equal to one.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.max_gain_db() < 0.0 {
        let msg = "max_gain_db must be greater than or equal to zero.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.crest_gate() && config.crest_ceiling_db() <= config.crest_floor_db() {
        let msg = "crest_ceiling_db must be greater than crest_floor_db.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.adaptive_threshold() && config.adapt_time() <= 0.0 {
        let msg = "adapt_time must be larger than zero when adaptive_threshold is enabled.";
        return Err(config::ConfigError::new(msg).into());
    }
    for ch in config.monitor_channels().iter() {
        if *ch >= channels {
            let msg = format!(
                "Invalid monitor channel: {}, max is: {}.",
                *ch,
                channels - 1
            );
            return Err(config::ConfigError::new(&msg).into());
        }
    }
    for ch in config.process_channels().iter() {
        if *ch >= channels {
            let msg = format!(
                "Invalid channel to process: {}, max is: {}.",
                *ch,
                channels - 1
            );
            return Err(config::ConfigError::new(&msg).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExpanderParameters;

    const SR: usize = 48000;
    const CHUNK: usize = 256;
    // Whole-chunk run lengths (the fixed-size scratch requires exact multiples of
    // CHUNK, matching the real pipeline). ~1 s and ~6 s respectively.
    const N_1S: usize = 188 * CHUNK; // 48128 samples
    const N_6S: usize = 1128 * CHUNK; // 288768 samples

    /// Base parameters with the crest gate disabled, to isolate the gain law.
    fn params(threshold: PrcFmt, ratio: PrcFmt, mode: ExpanderMode) -> ExpanderParameters {
        ExpanderParameters {
            channels: 1,
            monitor_channels: None,
            process_channels: None,
            attack: 0.001,
            release: 0.001,
            threshold,
            ratio,
            max_gain_db: Some(6.0),
            mode: Some(mode),
            makeup_gain: None,
            soft_clip: None,
            clip_limit: None,
            crest_gate: Some(false),
            crest_floor_db: None,
            crest_ceiling_db: None,
            knee_db: None,
            adaptive_threshold: None,
            relative_offset_db: None,
            adapt_time: None,
        }
    }

    fn make(params: ExpanderParameters) -> (Expander, Arc<ProcessingParameters>) {
        let pp = Arc::new(ProcessingParameters::default());
        let exp = Expander::from_config("test", params, SR, CHUNK, pp.clone());
        (exp, pp)
    }

    /// Run a single-channel signal through the expander in CHUNK-sized blocks,
    /// returning the concatenated output. The signal length must be a multiple of
    /// CHUNK (the fixed-size scratch mirrors the real pipeline).
    fn run(exp: &mut Expander, signal: &[PrcFmt]) -> Vec<PrcFmt> {
        let mut out = Vec::with_capacity(signal.len());
        for block in signal.chunks(CHUNK) {
            let wave = block.to_vec();
            let frames = wave.len();
            let mut chunk = AudioChunk::new(vec![wave], 0.0, 0.0, frames, frames);
            exp.process_chunk(&mut chunk).unwrap();
            out.extend_from_slice(&chunk.waveforms[0]);
        }
        out
    }

    fn constant(len: usize, value: PrcFmt) -> Vec<PrcFmt> {
        vec![value; len]
    }

    /// Run a signal through the expander and return the largest-magnitude
    /// expansion telemetry value seen across all chunks (the reported value only
    /// holds the last chunk, so transient bumps are otherwise missed).
    fn run_peak_expansion(
        exp: &mut Expander,
        signal: &[PrcFmt],
        pp: &Arc<ProcessingParameters>,
    ) -> f32 {
        let mut peak = 0.0f32;
        for block in signal.chunks(CHUNK) {
            let wave = block.to_vec();
            let frames = wave.len();
            let mut chunk = AudioChunk::new(vec![wave], 0.0, 0.0, frames, frames);
            exp.process_chunk(&mut chunk).unwrap();
            let e = pp.expansion_gain();
            if e.abs() > peak.abs() {
                peak = e;
            }
        }
        peak
    }

    fn sine(len: usize, freq: PrcFmt, amp: PrcFmt) -> Vec<PrcFmt> {
        (0..len)
            .map(|n| {
                amp * (2.0 * std::f64::consts::PI as PrcFmt * freq * n as PrcFmt / SR as PrcFmt)
                    .sin()
            })
            .collect()
    }

    /// The steady-state gain of a constant-magnitude signal, measured near the end
    /// of the run (output / input).
    fn steady_gain(out: &[PrcFmt], input_value: PrcFmt) -> PrcFmt {
        let n = out.len();
        out[n - 64..].iter().map(|v| v / input_value).sum::<PrcFmt>() / 64.0
    }

    #[test]
    fn upward_boosts_loud_by_ratio() {
        // threshold -20 dB, ratio 2 -> gain = (level - threshold) dB.
        let threshold = -20.0;
        let ratio = 2.0;
        // Level ABOVE the threshold: -16 dB -> gain = (-16 - -20)*1 = 4 dB
        // (below the 6 dB ceiling, so unclamped).
        let level_db = -16.0;
        let value = db_to_linear(level_db);
        let (mut exp, _pp) = make(params(threshold, ratio, ExpanderMode::Upward));
        let out = run(&mut exp, &constant(N_1S, value));
        let expected = db_to_linear((level_db - threshold) * (ratio - 1.0));
        let gain = steady_gain(&out, value);
        assert!(
            (gain - expected).abs() < 0.02 * expected,
            "gain {} not close to expected {}",
            gain,
            expected
        );
    }

    #[test]
    fn upward_leaves_quiet_untouched() {
        let threshold = -20.0;
        let level_db = -30.0; // below threshold
        let value = db_to_linear(level_db);
        let (mut exp, _pp) = make(params(threshold, 2.0, ExpanderMode::Upward));
        let out = run(&mut exp, &constant(N_1S, value));
        let gain = steady_gain(&out, value);
        assert!((gain - 1.0).abs() < 1e-3, "quiet signal changed, gain {}", gain);
    }

    #[test]
    fn soft_knee_is_continuous_and_matches_asymptotes() {
        let ratio = 3.0;
        let slope = ratio - 1.0;
        let half = 4.0;
        // Well below the knee: no expansion.
        assert_eq!(soft_knee_upward(-10.0, ratio, half), 0.0);
        // Well above the knee: exactly on the hard-knee line.
        assert!((soft_knee_upward(10.0, ratio, half) - 10.0 * slope).abs() < 1e-12);
        // Continuity of value at both knee edges.
        assert!(soft_knee_upward(-half + 1e-6, ratio, half).abs() < 1e-4);
        assert!(
            (soft_knee_upward(half - 1e-6, ratio, half) - half * slope).abs() < 1e-4,
            "value mismatch at upper knee edge"
        );
        // Continuity of slope at the edges (finite-difference derivative).
        let d = 1e-4;
        let deriv = |x: PrcFmt| {
            (soft_knee_upward(x + d, ratio, half) - soft_knee_upward(x - d, ratio, half)) / (2.0 * d)
        };
        assert!(deriv(-half).abs() < 1e-2, "slope not ~0 at lower knee edge");
        assert!((deriv(half) - slope).abs() < 1e-2, "slope not ~ratio-1 at upper knee edge");
        // At threshold the soft knee gives a small positive lift; the hard knee 0.
        assert!(soft_knee_upward(0.0, ratio, half) > 0.0);
        assert_eq!(soft_knee_upward(0.0, ratio, 0.0), 0.0);
    }

    #[test]
    fn soft_knee_lifts_at_threshold() {
        // A signal sitting exactly at the threshold: the soft knee applies a small
        // positive expansion, whereas a hard knee (knee_db 0) leaves it untouched.
        let threshold = -20.0;
        let value = db_to_linear(threshold);

        let mut soft = params(threshold, 2.0, ExpanderMode::Upward);
        soft.knee_db = Some(6.0);
        let (mut exp, _pp) = make(soft);
        let out = run(&mut exp, &constant(N_1S, value));
        let soft_gain = steady_gain(&out, value);
        assert!(soft_gain > 1.0, "soft knee should lift at-threshold signal, gain {}", soft_gain);

        let mut hard = params(threshold, 2.0, ExpanderMode::Upward);
        hard.knee_db = Some(0.0);
        let (mut exp, _pp) = make(hard);
        let out = run(&mut exp, &constant(N_1S, value));
        let hard_gain = steady_gain(&out, value);
        assert!((hard_gain - 1.0).abs() < 1e-3, "hard knee should not lift, gain {}", hard_gain);
    }

    #[test]
    fn upward_clamps_at_max_gain() {
        let threshold = -20.0;
        let ratio = 4.0;
        // level -8 dB -> raw gain = (12)*(3) = 36 dB, clamped to max_gain 6 dB.
        let level_db = -8.0;
        let value = db_to_linear(level_db);
        let (mut exp, _pp) = make(params(threshold, ratio, ExpanderMode::Upward));
        let out = run(&mut exp, &constant(N_1S, value));
        let expected = db_to_linear(6.0);
        let gain = steady_gain(&out, value);
        assert!(
            (gain - expected).abs() < 0.02 * expected,
            "gain {} not clamped to {}",
            gain,
            expected
        );
    }

    #[test]
    fn downward_attenuates_quiet() {
        let threshold = -20.0;
        let ratio = 2.0;
        // level -30 dB (below threshold) -> gain = (-10)*(1) = -10 dB, clamped to
        // -6 dB by max_gain.
        let level_db = -30.0;
        let value = db_to_linear(level_db);
        let (mut exp, _pp) = make(params(threshold, ratio, ExpanderMode::Downward));
        let out = run(&mut exp, &constant(N_1S, value));
        let expected = db_to_linear(-6.0);
        let gain = steady_gain(&out, value);
        assert!(
            (gain - expected).abs() < 0.02 * expected,
            "gain {} not attenuated to {}",
            gain,
            expected
        );
    }

    #[test]
    fn downward_leaves_loud_untouched() {
        let threshold = -20.0;
        let level_db = -10.0; // above threshold
        let value = db_to_linear(level_db);
        let (mut exp, _pp) = make(params(threshold, 2.0, ExpanderMode::Downward));
        let out = run(&mut exp, &constant(N_1S, value));
        let gain = steady_gain(&out, value);
        assert!((gain - 1.0).abs() < 1e-3, "loud signal changed, gain {}", gain);
    }

    #[test]
    fn attack_coefficient_matches_time_constant() {
        // Directly exercise the loudness envelope: after one attack time constant
        // the envelope should have covered ~63% of the step from its start value
        // to the target level. On a rising step the decoupled peak follower
        // saturates to the target immediately, so the attack smoother alone sets
        // the rise and the classic one-time-constant relation holds.
        let attack_s = 0.005;
        let mut p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.attack = attack_s;
        let (mut exp, _pp) = make(p);
        // Constant magnitude -> target envelope level in dB.
        let value: PrcFmt = 0.1;
        let target_db = 20.0 * (value.abs() + 1.0e-9).log10();
        let start = exp.level_env;
        // Feed the same constant magnitude one sample at a time for exactly one
        // time constant. `estimate_loudness` overwrites the scratch in place, so
        // the input must be re-primed before each call.
        exp.scratch = vec![value; 1];
        let n = (SR as PrcFmt * attack_s).round() as usize;
        for _ in 0..n {
            exp.scratch[0] = value;
            exp.estimate_loudness();
        }
        let expected = start + (1.0 - (-1.0 as PrcFmt).exp()) * (target_db - start);
        assert!(
            (exp.level_env - expected).abs() < 0.5,
            "envelope {} after one time constant not near expected {}",
            exp.level_env,
            expected
        );
    }

    #[test]
    fn limiter_caps_peaks() {
        // Big makeup gain that would push a loud tone well past full scale; the
        // clip_limit must hold the output at the limit.
        let mut p = params(-40.0, 2.0, ExpanderMode::Upward);
        p.makeup_gain = Some(12.0);
        p.clip_limit = Some(0.0); // limit at 0 dBFS (linear 1.0)
        let (mut exp, _pp) = make(p);
        let out = run(&mut exp, &sine(N_1S, 1000.0, 0.5));
        let peak = out.iter().cloned().fold(0.0 as PrcFmt, |m, v| m.max(v.abs()));
        assert!(peak <= 1.0 + 1e-6, "limiter failed, peak {}", peak);
        // And it really did reach the limit (gain was applied).
        assert!(peak > 0.99, "expected output driven to the limit, peak {}", peak);
    }

    #[test]
    fn crest_gate_suppresses_dynamic_material() {
        // A high-crest signal: full-scale impulses over near-silence. The gate must
        // drive the applied expansion toward zero even though the impulses sit well
        // above threshold.
        let threshold = -40.0;
        let mut p = params(threshold, 4.0, ExpanderMode::Upward);
        p.crest_gate = Some(true);
        p.crest_floor_db = Some(9.0);
        p.crest_ceiling_db = Some(15.0);
        let (mut exp, pp) = make(p);
        let len = N_6S;
        let mut signal = vec![0.0; len];
        for i in (0..len).step_by(2000) {
            signal[i] = 0.9;
        }
        run(&mut exp, &signal);
        let reported = pp.expansion_gain();
        assert!(
            reported.abs() < 0.5,
            "expected near-zero expansion on dynamic material, got {} dB",
            reported
        );
    }

    #[test]
    fn crest_gate_allows_compressed_material() {
        // A steady tone is low-crest (~3 dB), so the gate opens fully and the loud,
        // above-threshold signal is expanded and reported.
        let threshold = -20.0;
        let mut p = params(threshold, 2.0, ExpanderMode::Upward);
        p.crest_gate = Some(true);
        p.crest_floor_db = Some(9.0);
        p.crest_ceiling_db = Some(15.0);
        let (mut exp, pp) = make(p);
        // -6 dB tone: level well above threshold, crest of a sine ~3 dB < floor.
        let out = run(&mut exp, &sine(N_6S, 1000.0, db_to_linear(-6.0)));
        let reported = pp.expansion_gain();
        assert!(
            reported > 1.0,
            "expected positive expansion on compressed material, got {} dB",
            reported
        );
        // The tone should actually be louder than it went in.
        let in_peak = db_to_linear(-6.0);
        let out_peak = out.iter().cloned().fold(0.0 as PrcFmt, |m, v| m.max(v.abs()));
        assert!(
            out_peak > in_peak * 1.05,
            "compressed tone not expanded: in {} out {}",
            in_peak,
            out_peak
        );
    }

    #[test]
    fn stereo_linking_is_correlation_independent() {
        // Two channels carrying the same tone level, once correlated (L == R) and
        // once anti-correlated (L == -R). The RMS power-sum link must detect the
        // same level in both cases (a plain L+R sum would read the correlated
        // pair +6 dB hot and cancel the anti-correlated pair to silence).
        let threshold = -20.0;
        let level_db = -6.0;
        let value = db_to_linear(level_db);
        let tone = sine(N_1S, 1000.0, value);

        let run_stereo = |right: &[PrcFmt]| -> f32 {
            let mut p = params(threshold, 2.0, ExpanderMode::Upward);
            p.channels = 2;
            let (mut exp, pp) = make(p);
            for (lblk, rblk) in tone.chunks(CHUNK).zip(right.chunks(CHUNK)) {
                let frames = lblk.len();
                let mut chunk = AudioChunk::new(
                    vec![lblk.to_vec(), rblk.to_vec()],
                    0.0,
                    0.0,
                    frames,
                    frames,
                );
                exp.process_chunk(&mut chunk).unwrap();
            }
            pp.expansion_gain()
        };

        let correlated = run_stereo(&tone);
        let anti: Vec<PrcFmt> = tone.iter().map(|v| -v).collect();
        let anti_correlated = run_stereo(&anti);

        assert!(
            (correlated - anti_correlated).abs() < 0.05,
            "correlation changed detected level: correlated {} dB vs anti {} dB",
            correlated,
            anti_correlated
        );
        assert!(
            correlated > 1.0,
            "expected the above-threshold tone to be expanded, got {} dB",
            correlated
        );
    }

    #[test]
    fn telemetry_idle_on_silence() {
        let (mut exp, pp) = make(params(-20.0, 2.0, ExpanderMode::Upward));
        run(&mut exp, &constant(N_1S, 0.0));
        assert!(pp.expansion_gain().abs() < 1e-6);
    }

    #[test]
    fn validation_accepts_sane_config() {
        assert!(validate_expander(&params(-20.0, 2.0, ExpanderMode::Upward)).is_ok());
    }

    #[test]
    fn validation_rejects_bad_values() {
        let mut p = params(-20.0, 0.5, ExpanderMode::Upward); // ratio < 1
        assert!(validate_expander(&p).is_err());

        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.attack = 0.0;
        assert!(validate_expander(&p).is_err());

        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.release = -1.0;
        assert!(validate_expander(&p).is_err());

        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.max_gain_db = Some(-1.0);
        assert!(validate_expander(&p).is_err());

        // Crest ceiling not above floor (gate enabled).
        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.crest_gate = Some(true);
        p.crest_floor_db = Some(12.0);
        p.crest_ceiling_db = Some(10.0);
        assert!(validate_expander(&p).is_err());

        // Out-of-range channel.
        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.process_channels = Some(vec![5]);
        assert!(validate_expander(&p).is_err());

        // Adaptive with non-positive adapt_time.
        p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.adaptive_threshold = Some(true);
        p.adapt_time = Some(0.0);
        assert!(validate_expander(&p).is_err());
    }

    /// Build an adaptive-mode parameter set (crest gate left off so it does not
    /// interfere with observing the adaptive threshold's behavior).
    fn adaptive_params(threshold: PrcFmt, ratio: PrcFmt) -> ExpanderParameters {
        let mut p = params(threshold, ratio, ExpanderMode::Upward);
        p.adaptive_threshold = Some(true);
        p.relative_offset_db = Some(3.0); // == default half_knee (knee_db 6) -> steady over sits at knee edge
        p.adapt_time = Some(0.4);
        p
    }

    /// Key regression: when a whole passage gets louder, adaptive mode must NOT
    /// keep boosting the steady content — the threshold rises with the program
    /// level so a sustained loud tone settles back to ~unity gain. With adaptive
    /// off, the same loud steady tone is boosted.
    #[test]
    fn adaptive_ignores_steady_program_level() {
        let threshold = -50.0; // absolute floor well below the signal
        let value = db_to_linear(-10.0); // loud, far above the floor
        // Adaptive: after the slow envelope settles, steady loud -> ~unity.
        let (mut exp, _pp) = make(adaptive_params(threshold, 4.0));
        let out = run(&mut exp, &constant(N_6S, value));
        let gain = steady_gain(&out, value);
        assert!(
            (gain - 1.0).abs() < 0.03,
            "adaptive steady loud tone should settle to unity, gain {}",
            gain
        );
        // Fixed threshold: the same loud tone sits far above threshold -> boosted
        // (and clamped at max_gain 6 dB).
        let (mut exp, _pp) = make(params(threshold, 4.0, ExpanderMode::Upward));
        let out = run(&mut exp, &constant(N_6S, value));
        let gain_fixed = steady_gain(&out, value);
        assert!(
            gain_fixed > 1.5,
            "fixed-threshold steady loud tone should be boosted, gain {}",
            gain_fixed
        );
    }

    /// The bar bounces on a transient above the running level and returns to ~0
    /// once the slow envelope catches up to the new sustained level.
    #[test]
    fn adaptive_bounces_on_transient_then_settles() {
        let threshold = -50.0;
        let bed = db_to_linear(-30.0);
        let hot = db_to_linear(-10.0);
        let (mut exp, pp) = make(adaptive_params(threshold, 4.0));
        // Settle on the bed level (>> adapt_time).
        run(&mut exp, &constant(N_6S, bed));
        let bed_gain = pp.expansion_gain();
        assert!(
            bed_gain.abs() < 0.5,
            "steady bed should read ~0, got {} dB",
            bed_gain
        );
        // A single chunk jumping to a much higher level: the fast envelope leads
        // the slow one -> strong positive expansion (the bounce).
        run(&mut exp, &constant(CHUNK, hot));
        let bounce = pp.expansion_gain();
        assert!(
            bounce > 3.0,
            "expected the transient to bounce the expander up, got {} dB",
            bounce
        );
        // Hold the higher level: the slow envelope catches up and the boost
        // returns toward 0 (not pumping).
        run(&mut exp, &constant(N_6S, hot));
        let settled = pp.expansion_gain();
        assert!(
            settled.abs() < 0.5,
            "sustained loud level should settle back to ~0, got {} dB",
            settled
        );
    }

    /// The absolute `threshold` remains a hard floor in adaptive mode: content
    /// below it is never boosted, even though the adaptive threshold would sit
    /// lower.
    #[test]
    fn adaptive_respects_absolute_floor() {
        let threshold = -20.0;
        let value = db_to_linear(-30.0); // below the floor
        let (mut exp, _pp) = make(adaptive_params(threshold, 4.0));
        let out = run(&mut exp, &constant(N_6S, value));
        let gain = steady_gain(&out, value);
        assert!(
            (gain - 1.0).abs() < 1e-3,
            "content below the absolute floor must be untouched, gain {}",
            gain
        );
    }

    /// The adaptive-threshold telemetry tracks the program level: it rises when
    /// the sustained level rises, and equals the fixed threshold when adaptive
    /// mode is off.
    #[test]
    fn adaptive_threshold_telemetry_tracks_level() {
        let threshold = -50.0;
        let (mut exp, pp) = make(adaptive_params(threshold, 4.0));
        run(&mut exp, &constant(N_6S, db_to_linear(-30.0)));
        let thr_low = pp.adaptive_threshold();
        run(&mut exp, &constant(N_6S, db_to_linear(-10.0)));
        let thr_high = pp.adaptive_threshold();
        assert!(
            thr_high > thr_low + 10.0,
            "adaptive threshold did not track the level rise: {} -> {}",
            thr_low,
            thr_high
        );
        // Non-adaptive: telemetry reports the fixed threshold.
        let (mut exp, pp) = make(params(-20.0, 2.0, ExpanderMode::Upward));
        run(&mut exp, &constant(N_1S, db_to_linear(-6.0)));
        assert!(
            (pp.adaptive_threshold() - (-20.0)).abs() < 1e-3,
            "non-adaptive threshold telemetry should equal the fixed threshold, got {}",
            pp.adaptive_threshold()
        );
    }

    /// The crest-driven ratio defaults ON in both modes (it is the inverse-
    /// dynamics driver), and an explicit setting is honored.
    #[test]
    fn crest_gate_defaults_on() {
        let mut p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.adaptive_threshold = Some(true);
        p.crest_gate = None;
        assert!(make(p).0.crest_gate, "crest ratio should default on in adaptive mode");

        let mut p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.adaptive_threshold = None;
        p.crest_gate = None;
        assert!(make(p).0.crest_gate, "crest ratio should default on when not adaptive");

        let mut p = params(-20.0, 2.0, ExpanderMode::Upward);
        p.crest_gate = Some(false);
        assert!(!make(p).0.crest_gate, "explicit crest_gate=false should be honored");
    }

    /// The corrected cross-material behavior (the whole point of the crest-driven
    /// ratio): in adaptive mode a compressed, low-crest program with small
    /// transients is expanded, while an already-dynamic, high-crest program is
    /// left essentially alone. Previously the fixed-ratio law did the opposite.
    #[test]
    fn adaptive_expands_compressed_more_than_dynamic() {
        let threshold = -60.0; // absolute floor well below both signals

        // Compressed: a steady bed with small +3 dB bumps -> low crest.
        let mut pc = adaptive_params(threshold, 4.0);
        pc.crest_gate = Some(true);
        pc.relative_offset_db = Some(1.0);
        let (mut exp, pp) = make(pc);
        let bed = db_to_linear(-12.0);
        let bump = db_to_linear(-9.0);
        let mut compressed = vec![bed; N_6S];
        for i in (0..N_6S).step_by(4000) {
            for s in compressed.iter_mut().skip(i).take(200) {
                *s = bump;
            }
        }
        let compressed_boost = run_peak_expansion(&mut exp, &compressed, &pp);

        // Dynamic: sharp full-scale impulses over near-silence -> high crest.
        let mut pd = adaptive_params(threshold, 4.0);
        pd.crest_gate = Some(true);
        pd.relative_offset_db = Some(1.0);
        let (mut exp, pp) = make(pd);
        let mut dynamic = vec![0.0; N_6S];
        for i in (0..N_6S).step_by(4000) {
            dynamic[i] = 0.9;
        }
        let dynamic_boost = run_peak_expansion(&mut exp, &dynamic, &pp);

        assert!(
            compressed_boost > 3.0,
            "compressed material should be expanded, got {} dB",
            compressed_boost
        );
        assert!(
            dynamic_boost.abs() < 0.5,
            "already-dynamic material should be left alone, got {} dB",
            dynamic_boost
        );
        assert!(
            compressed_boost > dynamic_boost + 1.0,
            "expected more expansion on compressed ({} dB) than dynamic ({} dB)",
            compressed_boost,
            dynamic_boost
        );
    }
}
