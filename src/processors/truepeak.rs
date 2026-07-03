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

//! Master-bus true-peak (intersample-peak) limiter.
//!
//! Placed last in the pipeline, this processor protects the actual DAC feed: it
//! runs the signal through a 4x oversampled true-peak detector plus a lookahead
//! peak limiter and caps the linked (all-channel) signal at a configurable
//! ceiling (default -1.0 dBFS). Unlike the former resampler-wrapped guard, it
//! sits *after* every gain/harmonic/EQ stage (e.g. `TubeStage`), so the
//! intersample peaks those stages generate are guarded before reaching the DAC.
//!
//! The reusable DSP lives in [`TruePeakLimiterCore`] (moved here from the old
//! resampler `IspGuard`, which this processor replaces). The applied gain
//! reduction is reported in dB via
//! [`ProcessingParameters::set_truepeak_attenuation`].

use crate::PrcFmt;
use crate::ProcessingParameters;
use crate::Res;
use crate::audiochunk::AudioChunk;
use crate::config;
use crate::processors::Processor;
use std::f64::consts::PI;
use std::sync::Arc;

/// Number of 4x-oversampled samples used as lookahead. At 96 kHz this is
/// `LOOKAHEAD_SAMPLES / 4 / 96000` seconds (~0.42 ms with 16 samples).
pub const LOOKAHEAD_SAMPLES: usize = 16;

/// Length of the halfband prototype FIR used by the 4x true-peak probe.
const HALFBAND_TAPS: usize = 31;

/// Maximum number of channels the true-peak limiter supports (matches the
/// per-frame scratch arrays and [`TruePeakProbe`]'s state allocation).
pub const MAX_CHANNELS: usize = 32;

// ---------------------------------------------------------------------------
// Reusable DSP core (formerly the guts of the resampler `IspGuard`)
// ---------------------------------------------------------------------------

/// Reusable true-peak limiter DSP: a 4x-oversampled true-peak detector feeding
/// a lookahead peak limiter with one-pole gain release.
///
/// The gain reduction is *linked* across channels — the same gain is applied to
/// every channel so stereo/multichannel imaging is preserved — and the limiter
/// caps the true (intersample) peak at `ceiling`.
pub struct TruePeakLimiterCore {
    /// Linear amplitude ceiling (e.g. -1.0 dBFS == 10^(-1.0/20) ~= 0.891).
    ceiling: PrcFmt,
    /// Release coefficient per sample (one-pole low-pass on gain recovery
    /// toward 1.0). Computed from `release_ms` and the sample rate.
    release_coef: PrcFmt,
    /// Current applied gain. 1.0 means no attenuation; <1.0 means limiting.
    current_gain: PrcFmt,
    /// Per-channel lookahead delay buffers (length LOOKAHEAD_SAMPLES).
    delay_buffers: Vec<Vec<PrcFmt>>,
    /// Future peak ring used to compute the look-ahead target gain.
    future_peaks: Vec<PrcFmt>,
    /// 4x oversampling polyphase probe state (per channel).
    oversampler: TruePeakProbe,
    nbr_channels: usize,
}

impl std::fmt::Debug for TruePeakLimiterCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TruePeakLimiterCore")
            .field("ceiling", &self.ceiling)
            .field("nbr_channels", &self.nbr_channels)
            .finish()
    }
}

impl TruePeakLimiterCore {
    pub fn new(
        nbr_channels: usize,
        ceiling_dbfs: f64,
        release_ms: f64,
        samplerate: usize,
    ) -> Self {
        Self {
            ceiling: Self::ceiling_from_dbfs(ceiling_dbfs),
            release_coef: Self::release_from_ms(release_ms, samplerate),
            current_gain: 1.0,
            delay_buffers: vec![vec![0.0 as PrcFmt; LOOKAHEAD_SAMPLES]; nbr_channels],
            future_peaks: vec![0.0 as PrcFmt; LOOKAHEAD_SAMPLES],
            oversampler: TruePeakProbe::new(nbr_channels),
            nbr_channels,
        }
    }

    fn ceiling_from_dbfs(ceiling_dbfs: f64) -> PrcFmt {
        10f64.powf(ceiling_dbfs / 20.0) as PrcFmt
    }

    // One-pole release: gain_n = gain_{n-1} + (1 - gain_{n-1}) * coef
    // For release_ms time-to-reach-(1-1/e), coef = 1 - exp(-1/(rate * release_ms/1000)).
    fn release_from_ms(release_ms: f64, samplerate: usize) -> PrcFmt {
        let coef = if release_ms > 0.0 {
            1.0 - (-1.0 / (samplerate as f64 * release_ms / 1000.0)).exp()
        } else {
            1.0
        };
        coef as PrcFmt
    }

    /// Update ceiling / release without discarding the running limiter state
    /// (used for live parameter changes).
    pub fn set_params(&mut self, ceiling_dbfs: f64, release_ms: f64, samplerate: usize) {
        self.ceiling = Self::ceiling_from_dbfs(ceiling_dbfs);
        self.release_coef = Self::release_from_ms(release_ms, samplerate);
    }

    /// Fixed lookahead latency in samples added by the limiter.
    pub fn lookahead(&self) -> usize {
        LOOKAHEAD_SAMPLES
    }

    /// Process one frame (one sample per channel) through the lookahead limiter.
    ///
    /// Updates the per-channel delay buffers, computes the true-peak for the
    /// just-pushed frame, schedules the target gain in the future-peak ring, and
    /// writes the (already-delayed) limited samples into `out_per_channel`.
    pub fn process_frame(&mut self, in_per_channel: &[PrcFmt], out_per_channel: &mut [PrcFmt]) {
        // True-peak of this new frame across all channels.
        let mut frame_peak: PrcFmt = 0.0;
        for ch in 0..self.nbr_channels {
            let v = in_per_channel[ch];
            let tp = self.oversampler.peak_of_sample(ch, v).max(v.abs());
            if tp > frame_peak {
                frame_peak = tp;
            }
        }

        // Schedule the target gain for this future sample. We store the peak and
        // convert to gain at the moment we apply it. Shift future_peaks left by
        // 1, append frame_peak at the end.
        self.future_peaks.copy_within(1.., 0);
        let last = self.future_peaks.len() - 1;
        self.future_peaks[last] = frame_peak;

        // Compute target gain across the lookahead window: the smallest gain that
        // keeps every future peak <= ceiling.
        let mut target_gain: PrcFmt = 1.0;
        for &p in &self.future_peaks {
            if p > self.ceiling {
                let g = self.ceiling / p;
                if g < target_gain {
                    target_gain = g;
                }
            }
        }

        // Attack is instantaneous (lookahead handles it); release is one-pole.
        if target_gain < self.current_gain {
            self.current_gain = target_gain;
        } else {
            // Recover toward 1.0 (or toward target if it is between current and 1).
            let recovery_target = target_gain.min(1.0);
            self.current_gain += (recovery_target - self.current_gain) * self.release_coef;
        }

        // Apply current gain to the sample that is exiting the delay buffer.
        for ch in 0..self.nbr_channels {
            let v = in_per_channel[ch];
            let buf = &mut self.delay_buffers[ch];
            let out = buf[0] * self.current_gain;
            // Slide buffer one sample.
            buf.copy_within(1.., 0);
            let last = buf.len() - 1;
            buf[last] = v;
            out_per_channel[ch] = out;
        }
    }

    /// Current applied attenuation in dB (0.0 == no limiting, negative == active).
    pub fn attenuation_db(&self) -> f32 {
        let gain = self.current_gain.max(1e-9 as PrcFmt);
        20.0 * (gain as f32).log10()
    }

    pub fn reset(&mut self) {
        for buf in &mut self.delay_buffers {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
        self.future_peaks.iter_mut().for_each(|v| *v = 0.0);
        self.current_gain = 1.0;
        self.oversampler.reset();
    }
}

// ---------------------------------------------------------------------------
// TruePeak processor
// ---------------------------------------------------------------------------

pub struct TruePeak {
    name: String,
    channels: usize,
    ceiling_dbfs: f64,
    release_ms: f64,
    samplerate: usize,
    core: TruePeakLimiterCore,
    processing_params: Arc<ProcessingParameters>,
}

impl std::fmt::Debug for TruePeak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TruePeak")
            .field("name", &self.name)
            .field("channels", &self.channels)
            .field("ceiling_dbfs", &self.ceiling_dbfs)
            .field("release_ms", &self.release_ms)
            .finish()
    }
}

impl TruePeak {
    /// Creates a TruePeak processor from a config struct.
    pub fn from_config(
        name: &str,
        config: config::TruePeakParameters,
        samplerate: usize,
        processing_params: Arc<ProcessingParameters>,
    ) -> Self {
        let name = name.to_string();
        let channels = config.channels;
        let ceiling_dbfs = config.ceiling_dbfs();
        let release_ms = config.release_ms();
        debug!(
            "Creating TruePeak '{}', channels: {}, ceiling_dbfs: {}, release_ms: {}",
            name, channels, ceiling_dbfs, release_ms
        );
        let core = TruePeakLimiterCore::new(channels, ceiling_dbfs, release_ms, samplerate);
        TruePeak {
            name,
            channels,
            ceiling_dbfs,
            release_ms,
            samplerate,
            core,
            processing_params,
        }
    }

    /// Lookahead latency in samples this processor adds to the signal path.
    pub fn latency(&self) -> usize {
        self.core.lookahead()
    }
}

impl Processor for TruePeak {
    fn name(&self) -> &str {
        &self.name
    }

    /// Apply the true-peak limiter to an AudioChunk, modifying it in-place.
    fn process_chunk(&mut self, input: &mut AudioChunk) -> Res<()> {
        let nch = self.channels;
        // Longest waveform sets the frame count; absent/short channels are read
        // as silence and left untouched, mirroring the old guard's robustness.
        let frames = input.waveforms.iter().map(|w| w.len()).max().unwrap_or(0);
        let mut frame_in = [0.0 as PrcFmt; MAX_CHANNELS];
        let mut frame_out = [0.0 as PrcFmt; MAX_CHANNELS];
        for f in 0..frames {
            for ch in 0..nch {
                frame_in[ch] = input
                    .waveforms
                    .get(ch)
                    .and_then(|w| w.get(f))
                    .copied()
                    .unwrap_or(0.0);
            }
            self.core.process_frame(&frame_in[..nch], &mut frame_out[..nch]);
            for ch in 0..nch {
                if let Some(sample) = input.waveforms.get_mut(ch).and_then(|w| w.get_mut(f)) {
                    *sample = frame_out[ch];
                }
            }
        }
        self.processing_params
            .set_truepeak_attenuation(self.core.attenuation_db());
        Ok(())
    }

    fn update_parameters(&mut self, config: config::Processor) {
        if let config::Processor::TruePeak {
            parameters: config, ..
        } = config
        {
            self.channels = config.channels;
            self.ceiling_dbfs = config.ceiling_dbfs();
            self.release_ms = config.release_ms();
            self.core
                .set_params(self.ceiling_dbfs, self.release_ms, self.samplerate);
            debug!(
                "Updating TruePeak '{}', channels: {}, ceiling_dbfs: {}, release_ms: {}",
                self.name, self.channels, self.ceiling_dbfs, self.release_ms
            );
        } else {
            // This should never happen unless there is a bug somewhere else
            panic!("Invalid config change!");
        }
    }
}

/// Validate the TruePeak processor config, to give a helpful message instead of
/// a panic.
pub fn validate_truepeak(config: &config::TruePeakParameters) -> Res<()> {
    if config.channels > MAX_CHANNELS {
        let msg = format!(
            "TruePeak supports at most {} channels, got {}.",
            MAX_CHANNELS, config.channels
        );
        return Err(config::ConfigError::new(&msg).into());
    }
    if config.ceiling_dbfs() > 0.0 {
        let msg = "ceiling_dbfs must be at or below 0 dBFS.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.release_ms() < 0.0 {
        let msg = "release_ms must not be negative.";
        return Err(config::ConfigError::new(msg).into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// True-peak probe: 4x oversampling halfband cascade
// ---------------------------------------------------------------------------

/// 4x true-peak detector. For each input sample, returns the maximum abs value
/// of the corresponding 4 upsampled samples.
///
/// Implemented as a polyphase halfband filter applied twice (effective 4x). The
/// prototype is a Kaiser-windowed sinc at cutoff 0.25.
struct TruePeakProbe {
    nbr_channels: usize,
    /// Polyphase coefficients for one stage of 2x interpolation
    /// (length HALFBAND_TAPS, decomposed into 2 branches of HALFBAND_TAPS/2 each).
    branch_a: Vec<PrcFmt>,
    branch_b: Vec<PrcFmt>,
    /// Per-channel state for each of the two 2x stages.
    state_stage1: Vec<Vec<PrcFmt>>,
    state_stage2: Vec<Vec<PrcFmt>>,
}

impl TruePeakProbe {
    fn new(nbr_channels: usize) -> Self {
        let proto = halfband_kaiser_prototype(HALFBAND_TAPS);
        // Polyphase split: branch_a = even taps, branch_b = odd taps. With an
        // odd-length prototype, branch_a is one tap longer than branch_b; pad
        // branch_b with a trailing zero so both share the same convolution loop.
        let mut branch_a = Vec::new();
        let mut branch_b = Vec::new();
        for (n, &v) in proto.iter().enumerate() {
            if n % 2 == 0 {
                branch_a.push(v);
            } else {
                branch_b.push(v);
            }
        }
        while branch_b.len() < branch_a.len() {
            branch_b.push(0.0 as PrcFmt);
        }
        // Normalize each branch so its sum equals 1 (per-2x-stage unity gain).
        let sa: PrcFmt = branch_a.iter().sum();
        let sb: PrcFmt = branch_b.iter().sum();
        for v in &mut branch_a {
            *v /= sa;
        }
        for v in &mut branch_b {
            *v /= sb;
        }
        let state_stage1 = vec![vec![0.0 as PrcFmt; branch_a.len()]; nbr_channels];
        let state_stage2 = vec![vec![0.0 as PrcFmt; branch_a.len()]; nbr_channels];
        Self {
            nbr_channels,
            branch_a,
            branch_b,
            state_stage1,
            state_stage2,
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.state_stage1 {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
        for buf in &mut self.state_stage2 {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Push one input sample for channel `ch` through the 4x upsampler.
    /// Returns the maximum absolute value among the 4 upsampled output samples.
    fn peak_of_sample(&mut self, ch: usize, v: PrcFmt) -> PrcFmt {
        debug_assert!(ch < self.nbr_channels);
        // Stage 1: 2x interpolation. Produces two output samples (the input
        // passed through via the even-tap branch, and the interpolated midpoint
        // from the odd-tap branch).
        let branch_len = self.branch_a.len();
        let s1 = &mut self.state_stage1[ch];
        let s1_len = s1.len();
        s1.copy_within(0..s1_len - 1, 1);
        s1[0] = v;
        let mut acc_a: PrcFmt = 0.0;
        let mut acc_b: PrcFmt = 0.0;
        for k in 0..branch_len {
            acc_a += self.branch_a[k] * s1[k];
            acc_b += self.branch_b[k] * s1[k];
        }
        let stage1_out = [acc_a, acc_b];

        // Stage 2: 2x interpolation on each of the two stage-1 samples, yielding
        // four upsampled samples in total.
        let mut peak: PrcFmt = 0.0;
        for &x in stage1_out.iter() {
            let s2 = &mut self.state_stage2[ch];
            let s2_len = s2.len();
            s2.copy_within(0..s2_len - 1, 1);
            s2[0] = x;
            let mut a: PrcFmt = 0.0;
            let mut b: PrcFmt = 0.0;
            for k in 0..branch_len {
                a += self.branch_a[k] * s2[k];
                b += self.branch_b[k] * s2[k];
            }
            if a.abs() > peak {
                peak = a.abs();
            }
            if b.abs() > peak {
                peak = b.abs();
            }
        }
        peak
    }
}

/// Kaiser-windowed sinc halfband prototype of length `len` (odd), cutoff 0.25.
fn halfband_kaiser_prototype(mut len: usize) -> Vec<PrcFmt> {
    if len % 2 == 0 {
        len |= 1;
    }
    let half = (len as i64 - 1) / 2;
    let beta = 8.0_f64; // ~80 dB stopband, plenty for peak detection.
    let i0_beta = bessel_i0(beta);
    let mut h = vec![0.0 as PrcFmt; len];
    for n in 0..len {
        let k = n as i64 - half;
        let arg = 0.5 * k as f64;
        let sinc = if k == 0 {
            1.0
        } else {
            (PI * arg).sin() / (PI * arg)
        };
        let win_arg = k as f64 / half as f64;
        let win = bessel_i0(beta * (1.0 - win_arg * win_arg).max(0.0).sqrt()) / i0_beta;
        h[n] = (0.5 * sinc * win) as PrcFmt;
    }
    h
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x_sq = (x * 0.5).powi(2);
    for k in 1..50 {
        term *= half_x_sq / (k as f64 * k as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audiochunk::AudioChunk;

    fn make_processor(ceiling_dbfs: f64, release_ms: f64) -> (TruePeak, Arc<ProcessingParameters>) {
        let pp = Arc::new(ProcessingParameters::default());
        let params = config::TruePeakParameters {
            channels: 2,
            ceiling_dbfs: Some(ceiling_dbfs),
            release_ms: Some(release_ms),
        };
        (
            TruePeak::from_config("tp", params, 48000, pp.clone()),
            pp,
        )
    }

    fn chunk_from(waveforms: Vec<Vec<PrcFmt>>) -> AudioChunk {
        let frames = waveforms[0].len();
        AudioChunk::new(waveforms, 0.0, 0.0, frames, frames)
    }

    /// A half-Nyquist tone sampled at a phase that peaks *between* samples: the
    /// sample values stay well below full scale, but the true (intersample) peak
    /// exceeds 0 dBFS. The limiter must pull the 4x-oversampled peak down to the
    /// ceiling.
    #[test]
    fn truepeak_caps_intersample_peaks() {
        let (mut tp, _pp) = make_processor(-1.0, 20.0);
        let ceiling = 10f64.powf(-1.0 / 20.0) as PrcFmt;
        // Fs/2 tone at 45 degrees: samples are +/-0.9*sin(45)=~0.636 but the
        // reconstructed waveform peaks at ~0.9 between samples. Scale up so the
        // intersample peak is clearly over 0 dBFS.
        let amp = 0.99 as PrcFmt;
        let make_wave = || -> Vec<PrcFmt> {
            (0..2048)
                .map(|n| {
                    let phase = PI * (n as PrcFmt) + PI / 4.0; // Fs/2, +45 deg
                    amp * phase.sin()
                })
                .collect()
        };
        // Run several chunks so the lookahead limiter reaches steady state.
        let mut last = chunk_from(vec![make_wave(), make_wave()]);
        for _ in 0..8 {
            last = chunk_from(vec![make_wave(), make_wave()]);
            tp.process_chunk(&mut last).unwrap();
        }
        // Verify the true peak of the limited output stays at/below ceiling.
        let mut probe = TruePeakProbe::new(1);
        let mut max_tp: PrcFmt = 0.0;
        let n = last.waveforms[0].len();
        for &v in &last.waveforms[0][n / 4..] {
            let tp_val = probe.peak_of_sample(0, v).max(v.abs());
            if tp_val > max_tp {
                max_tp = tp_val;
            }
        }
        assert!(
            max_tp <= ceiling + 5e-3 as PrcFmt,
            "true peak {} exceeded ceiling {}",
            max_tp,
            ceiling
        );
    }

    /// Quiet material well below the ceiling must pass through unattenuated.
    #[test]
    fn truepeak_quiet_is_transparent() {
        let (mut tp, pp) = make_processor(-1.0, 20.0);
        let wave: Vec<PrcFmt> = (0..1024)
            .map(|n| 0.1 * (2.0 * PI * 0.01 * n as PrcFmt).sin())
            .collect();
        let mut chunk = chunk_from(vec![wave.clone(), wave]);
        for _ in 0..4 {
            tp.process_chunk(&mut chunk).unwrap();
        }
        assert!(
            pp.truepeak_attenuation().abs() < 1e-4,
            "quiet material was attenuated by {} dB",
            pp.truepeak_attenuation()
        );
    }

    /// Limiting must be reported as negative dB on hot material. A fresh
    /// full-scale chunk is fed each iteration; reusing one chunk would feed the
    /// already-limited (and delayed) output back in and let the gain release.
    #[test]
    fn truepeak_reports_attenuation() {
        let (mut tp, pp) = make_processor(-1.0, 20.0);
        for _ in 0..8 {
            let wave = vec![1.0 as PrcFmt; 1024];
            let mut chunk = chunk_from(vec![wave.clone(), wave]);
            tp.process_chunk(&mut chunk).unwrap();
        }
        assert!(
            pp.truepeak_attenuation() < -0.1,
            "expected attenuation on full-scale input, got {} dB",
            pp.truepeak_attenuation()
        );
    }

    /// The same gain must be applied to every channel (linked limiting), so a
    /// hot left channel also pulls down a quiet right channel identically.
    #[test]
    fn truepeak_gain_is_linked_across_channels() {
        let (mut tp, _pp) = make_processor(-1.0, 20.0);
        let hot = vec![1.0 as PrcFmt; 1024];
        let quiet = vec![0.05 as PrcFmt; 1024];
        let mut chunk = chunk_from(vec![hot, quiet]);
        // One chunk: after the lookahead fills, the right channel's steady value
        // should be scaled by the same gain the left channel forced.
        tp.process_chunk(&mut chunk).unwrap();
        let n = chunk.waveforms[1].len();
        // Late samples of the quiet channel should be reduced below 0.05.
        let tail = chunk.waveforms[1][n - 1];
        assert!(
            tail < 0.05 as PrcFmt,
            "linked gain not applied to quiet channel: {tail}"
        );
    }

    #[test]
    fn true_peak_probe_runs() {
        // Smoke test: feed a chirp and verify the probe never panics and returns
        // sane peak values.
        let mut probe = TruePeakProbe::new(1);
        let mut max_peak: PrcFmt = 0.0;
        for n in 0..4096 {
            let x = (2.0 * PI * 0.1 * n as f64).sin() as PrcFmt;
            let p = probe.peak_of_sample(0, x);
            if p > max_peak {
                max_peak = p;
            }
        }
        // Reasonable bound: 0 < peak < 1.5 (some headroom for the probe).
        assert!(max_peak > 0.0 && max_peak < 1.5, "peak={max_peak}");
    }
}
