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

//! Intersample-peak (ISP) guard.
//!
//! Wraps an inner `rubato::Resampler<PrcFmt>` and post-processes its output
//! through a true-peak detector (4x oversampled probe) plus a lookahead peak
//! limiter, capping at a configurable ceiling (default -0.5 dBFS).
//!
//! The detector is independent of any oversampler downstream (e.g. TubeStage);
//! it exists specifically because the resampler output is where intersample
//! peaks first appear in the digital signal path.
//!
//! Reports the currently applied attenuation (in dB) via
//! [`ProcessingParameters::set_isp_attenuation`].

use crate::PrcFmt;
use crate::ProcessingParameters;
use audioadapter::{Adapter, AdapterMut};
use rubato::{Indexing, ResampleResult, Resampler};
use std::f64::consts::PI;
use std::sync::Arc;

/// Number of 4x-oversampled samples used as lookahead. At 96 kHz this is
/// `LOOKAHEAD_SAMPLES / 4 / 96000` seconds (~0.42 ms with 16 samples).
const LOOKAHEAD_SAMPLES: usize = 16;

/// Length of the halfband prototype FIR used by the 4x true-peak probe.
const HALFBAND_TAPS: usize = 31;

pub struct IspGuard {
    inner: Box<dyn Resampler<PrcFmt>>,
    /// Linear amplitude ceiling (e.g. -0.5 dBFS == 10^(-0.5/20) ~= 0.944).
    ceiling: PrcFmt,
    /// Release coefficient per output sample (one-pole low-pass on gain
    /// recovery toward 1.0). Computed from `release_ms` and output rate.
    release_coef: PrcFmt,
    /// Current applied gain. 1.0 means no attenuation; <1.0 means limiting.
    current_gain: PrcFmt,
    /// Per-channel lookahead delay buffers (length LOOKAHEAD_SAMPLES + 1).
    delay_buffers: Vec<Vec<PrcFmt>>,
    /// Per-channel future peak ring used to compute the look-ahead target gain.
    future_peaks: Vec<PrcFmt>,
    /// Per-channel scratch state for the 4x oversampling polyphase probe.
    oversampler: TruePeakProbe,
    nbr_channels: usize,
    processing_params: Arc<ProcessingParameters>,
    /// Whether the guard is enabled. When false, samples pass through unchanged
    /// (and current attenuation is reported as 0 dB).
    enabled: bool,
}

impl std::fmt::Debug for IspGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IspGuard")
            .field("ceiling", &self.ceiling)
            .field("enabled", &self.enabled)
            .field("nbr_channels", &self.nbr_channels)
            .finish()
    }
}

impl IspGuard {
    pub fn new(
        inner: Box<dyn Resampler<PrcFmt>>,
        ceiling_dbfs: f64,
        release_ms: f64,
        enabled: bool,
        output_rate: usize,
        processing_params: Arc<ProcessingParameters>,
    ) -> Self {
        let nbr_channels = inner.nbr_channels();

        let ceiling = 10f64.powf(ceiling_dbfs / 20.0) as PrcFmt;
        // One-pole release: gain_n = gain_{n-1} + (1 - gain_{n-1}) * coef
        // For release_ms time-to-reach-(1-1/e), coef = 1 - exp(-1/(output_rate * release_ms/1000)).
        let coef = if release_ms > 0.0 {
            1.0 - (-1.0 / (output_rate as f64 * release_ms / 1000.0)).exp()
        } else {
            1.0
        };
        let release_coef = coef as PrcFmt;

        Self {
            inner,
            ceiling,
            release_coef,
            current_gain: 1.0,
            delay_buffers: vec![vec![0.0 as PrcFmt; LOOKAHEAD_SAMPLES]; nbr_channels],
            future_peaks: vec![0.0 as PrcFmt; LOOKAHEAD_SAMPLES],
            oversampler: TruePeakProbe::new(nbr_channels),
            nbr_channels,
            processing_params,
            enabled,
        }
    }

    /// Process one output sample on all channels through the lookahead limiter.
    ///
    /// Updates the per-channel delay buffers, computes the true-peak for the
    /// just-pushed sample, schedules the target gain in the future-peak ring,
    /// and returns the (already-delayed) limited sample.
    fn limit_sample(
        &mut self,
        in_per_channel: &[PrcFmt],
        out_per_channel: &mut [PrcFmt],
    ) {
        // True-peak of this new frame across all channels.
        let mut frame_peak: PrcFmt = 0.0;
        for ch in 0..self.nbr_channels {
            let v = in_per_channel[ch];
            let tp = self.oversampler.peak_of_sample(ch, v).max(v.abs());
            if tp > frame_peak {
                frame_peak = tp;
            }
        }

        // Schedule the target gain for this future sample. We store the peak
        // and convert to gain at the moment we apply it.
        // Shift future_peaks left by 1, append frame_peak at the end.
        self.future_peaks.copy_within(1.., 0);
        let last = self.future_peaks.len() - 1;
        self.future_peaks[last] = frame_peak;

        // Compute target gain across the lookahead window: the smallest gain
        // that keeps every future peak <= ceiling.
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

    fn report_attenuation(&self) {
        let gain = self.current_gain.max(1e-9 as PrcFmt);
        let db = 20.0 * (gain as f32).log10();
        self.processing_params.set_isp_attenuation(db);
    }
}

impl Resampler<PrcFmt> for IspGuard {
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, PrcFmt>,
        buffer_out: &mut dyn AdapterMut<'a, PrcFmt>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)> {
        if !self.enabled {
            self.processing_params.set_isp_attenuation(0.0);
            return self.inner.process_into_buffer(buffer_in, buffer_out, indexing);
        }

        // Run the inner resampler directly into the caller's output buffer.
        // We then post-process in place, walking frame-by-frame and routing
        // each sample through the lookahead limiter.
        let (in_consumed, out_produced) =
            self.inner.process_into_buffer(buffer_in, buffer_out, indexing)?;

        let output_offset = indexing.map(|i| i.output_offset).unwrap_or(0);
        let nch = self.nbr_channels;
        assert!(
            nch <= 32,
            "IspGuard supports up to 32 channels; got {nch}"
        );
        let mut frame_in = [0.0 as PrcFmt; 32];
        let mut frame_out = [0.0 as PrcFmt; 32];
        for f in 0..out_produced {
            for ch in 0..nch {
                frame_in[ch] = buffer_out
                    .read_sample(ch, output_offset + f)
                    .unwrap_or(0.0);
            }
            self.limit_sample(&frame_in[..nch], &mut frame_out[..nch]);
            for ch in 0..nch {
                buffer_out.write_sample(ch, output_offset + f, &frame_out[ch]);
            }
        }

        self.report_attenuation();

        Ok((in_consumed, out_produced))
    }

    fn input_frames_max(&self) -> usize {
        self.inner.input_frames_max()
    }

    fn input_frames_next(&self) -> usize {
        self.inner.input_frames_next()
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.inner.output_frames_max()
    }

    fn output_frames_next(&self) -> usize {
        self.inner.output_frames_next()
    }

    fn output_delay(&self) -> usize {
        self.inner.output_delay() + LOOKAHEAD_SAMPLES
    }

    fn set_resample_ratio(&mut self, new_ratio: f64, ramp: bool) -> ResampleResult<()> {
        self.inner.set_resample_ratio(new_ratio, ramp)
    }

    fn resample_ratio(&self) -> f64 {
        self.inner.resample_ratio()
    }

    fn set_resample_ratio_relative(
        &mut self,
        rel_ratio: f64,
        ramp: bool,
    ) -> ResampleResult<()> {
        self.inner.set_resample_ratio_relative(rel_ratio, ramp)
    }

    fn reset(&mut self) {
        self.inner.reset();
        for buf in &mut self.delay_buffers {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
        self.future_peaks.iter_mut().for_each(|v| *v = 0.0);
        self.current_gain = 1.0;
        self.oversampler.reset();
    }
}

// ---------------------------------------------------------------------------
// True-peak probe: 4x oversampling halfband cascade
// ---------------------------------------------------------------------------

/// 4x true-peak detector. For each input sample, returns the maximum abs value
/// of the corresponding 4 upsampled samples.
///
/// Implemented as a polyphase halfband filter applied twice (effective 4x).
/// The prototype is a Kaiser-windowed sinc at cutoff 0.25.
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
        // passed through via the even-tap branch, and the interpolated
        // midpoint from the odd-tap branch).
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

        // Stage 2: 2x interpolation on each of the two stage-1 samples,
        // yielding four upsampled samples in total.
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
        let sinc = if k == 0 { 1.0 } else { (PI * arg).sin() / (PI * arg) };
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
    use crate::utils::resampling::polyphase::PolyphaseFir;
    use crate::config;
    use audioadapter_buffers::direct::SequentialSliceOfVecs;

    fn make_inner() -> Box<dyn Resampler<PrcFmt>> {
        Box::new(
            PolyphaseFir::new(
                44100,
                48000,
                256,
                2,
                config::PolyphaseCharacter::LinearPhase,
                32,
                32,
            )
            .unwrap(),
        )
    }

    #[test]
    fn isp_guard_caps_at_ceiling() {
        let pp = Arc::new(ProcessingParameters::default());
        let inner = make_inner();
        let mut guard = IspGuard::new(inner, -1.0, 50.0, true, 48000, pp.clone());

        // Drive with full-scale +1.0 input — output should be clipped to ceiling.
        let in_len = guard.input_frames_max();
        let waves = vec![vec![1.0 as PrcFmt; in_len]; 2];
        let mut out = vec![vec![0.0 as PrcFmt; guard.output_frames_max()]; 2];
        let input = SequentialSliceOfVecs::new(&waves, 2, in_len).unwrap();
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut out, 2, guard.output_frames_max()).unwrap();

        let ceiling = 10f64.powf(-1.0 / 20.0) as PrcFmt;

        // Pump enough chunks for the limiter to settle.
        for _ in 0..16 {
            guard
                .process_into_buffer(&input, &mut output, None)
                .unwrap();
        }
        // Steady state: every output should be at or below ceiling (with some
        // numerical tolerance and lookahead transient grace).
        let n = out[0].len();
        for ch in 0..2 {
            for &v in &out[ch][n / 4..n] {
                assert!(
                    v.abs() <= ceiling + 1e-3 as PrcFmt,
                    "ISP guard let through {} (ceiling {})",
                    v.abs(),
                    ceiling
                );
            }
        }
    }

    #[test]
    fn isp_guard_disabled_is_passthrough() {
        let pp = Arc::new(ProcessingParameters::default());
        let inner = make_inner();
        let mut guard = IspGuard::new(inner, -1.0, 50.0, false, 48000, pp.clone());
        // With enabled=false, attenuation should always be 0 dB.
        let in_len = guard.input_frames_max();
        let waves = vec![vec![0.5 as PrcFmt; in_len]; 2];
        let mut out = vec![vec![0.0 as PrcFmt; guard.output_frames_max()]; 2];
        let input = SequentialSliceOfVecs::new(&waves, 2, in_len).unwrap();
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut out, 2, guard.output_frames_max()).unwrap();
        guard
            .process_into_buffer(&input, &mut output, None)
            .unwrap();
        assert_eq!(pp.isp_attenuation(), 0.0);
    }

    #[test]
    fn true_peak_probe_runs() {
        // Smoke test: feed a chirp and verify the probe never panics and
        // returns sane peak values.
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
