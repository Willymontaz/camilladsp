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

//! Linear-phase polyphase FIR upsampler.
//!
//! Implements `rubato::Resampler<PrcFmt>` for a fixed, exactly rational
//! upsampling ratio. Output chunk size is fixed (matching
//! `rubato::FixedAsync::Output` semantics); the input frame count varies per
//! call.
//!
//! Design in one paragraph: for input rate `fi` and output rate `fo` with
//! `g = gcd(fi, fo)`, the branch count is `L = fo / g` and every output sample
//! advances the read position by exactly `step = fi / g` units of `1/L` input
//! samples. A long Kaiser-windowed sinc prototype of `taps * L + 1` coefficients
//! is designed at the upsampled rate `fi * L` and decomposed into `L` polyphase
//! branches `h_b[k] = h[k*L + b]`, so each output is a single `taps`-long inner
//! product. There is no branch interpolation and no phase drift: the phase is an
//! integer pair `(sample, branch)`.
//!
//! Two properties are exploited:
//!
//! * The prototype is symmetric (`h[m] = h[N-1-m]`, `N-1 = taps*L`), which makes
//!   branch `b` the time-reverse of branch `L-b`. Only `L/2 + 1` branches are
//!   stored and the mirrored ones are evaluated with a forward-reading kernel.
//!   This halves the coefficient table, which is what keeps it inside L2 at
//!   large tap counts.
//! * The stopband edge - not the -6 dB point - is placed at the source Nyquist,
//!   so no part of the transition band images into the output.

use crate::PrcFmt;
use audioadapter::{Adapter, AdapterMut};
use rubato::{Indexing, ResampleError, ResampleResult, Resampler};
use std::f64::consts::PI;
use std::fmt;

#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[path = "polyphase_neon.rs"]
mod neon;

/// Upper bound on the derived branch count. `L = output_rate / gcd(rates)`, so
/// this only rejects pathological rate pairs (e.g. 44100 -> 44101 would need
/// 44101 branches and tens of gigabytes of coefficients).
pub const MAX_OVERSAMPLING: usize = 2048;

/// Stopband attenuation target of the Kaiser prototype, in dB.
const STOPBAND_ATTENUATION_DB: f64 = 140.0;

/// The Kaiser length estimate slightly under-predicts the realised transition
/// width. Widen the assumed width by this factor before shifting the cutoff
/// down, so the stopband is genuinely reached at or before the source Nyquist.
const TRANSITION_SAFETY: f64 = 1.1;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Kaiser transition width for a prototype of `proto_len` coefficients,
/// normalized to the upsampled rate.
fn transition_width(proto_len: usize) -> f64 {
    TRANSITION_SAFETY * (STOPBAND_ATTENUATION_DB - 7.95) / (14.36 * proto_len as f64)
}

/// Branch count and per-output phase step for an exact rational upsampling
/// ratio: returns `(oversampling, step)` where `oversampling = output_rate / g`
/// and `step = input_rate / g` with `g = gcd(input_rate, output_rate)`.
///
/// Shared by the engine and by config validation so the two cannot disagree.
pub fn derived_oversampling(
    input_rate: usize,
    output_rate: usize,
) -> Result<(usize, usize), String> {
    if input_rate == 0 || output_rate == 0 {
        return Err("sample rates must be > 0".to_string());
    }
    if output_rate <= input_rate {
        return Err(format!(
            "the Polyphase resampler only upsamples, but the capture rate {input_rate} Hz is not below the playback rate {output_rate} Hz"
        ));
    }
    let g = gcd(input_rate, output_rate);
    let oversampling = output_rate / g;
    let step = input_rate / g;
    if oversampling > MAX_OVERSAMPLING {
        return Err(format!(
            "resampling {input_rate} Hz -> {output_rate} Hz needs {oversampling} polyphase branches, more than the limit of {MAX_OVERSAMPLING}"
        ));
    }
    Ok((oversampling, step))
}

/// A long linear-phase FIR upsampler built from a Kaiser-windowed sinc
/// prototype, decomposed into `oversampling` polyphase branches. The ratio is
/// fixed at construction.
pub struct PolyphaseFir {
    nbr_channels: usize,
    /// Number of taps per polyphase branch.
    taps: usize,
    /// Polyphase factor (number of branches), derived from the rate ratio.
    oversampling: usize,
    /// Phase advance per output sample, in units of `1 / oversampling` input
    /// samples. Always `< oversampling` because this engine only upsamples.
    step: usize,
    /// `output_rate / input_rate`.
    resample_ratio: f64,
    /// Per-channel input history. Length is `max_buffer_len`.
    buffers: Vec<Vec<PrcFmt>>,
    /// How many valid input samples sit at the start of each buffer.
    buffer_fill: usize,
    /// Input-sample index within `buffers` of the next output sample.
    next_sample: usize,
    /// Polyphase branch of the next output sample, in `0..oversampling`.
    next_branch: usize,
    /// The lower half of the branch table, `oversampling / 2 + 1` branches of
    /// `taps` coefficients each, flattened for locality. Branch `b` lives at
    /// `branches[b * taps .. (b + 1) * taps]`; branches above `oversampling / 2`
    /// are the time-reverse of `oversampling - b` and are not stored.
    branches: Vec<PrcFmt>,
    /// Output frames produced per call (fixed).
    chunk_size: usize,
    /// Allocated length of each channel buffer.
    max_buffer_len: usize,
    /// Channel-active mask. Inactive channels are not processed.
    channel_mask: Vec<bool>,
    /// Indices of the active channels, rebuilt from `channel_mask` on every
    /// call so the output loop can walk them in pairs without allocating.
    active_channels: Vec<usize>,
    /// Steady-state group delay in output frames, used for `output_delay()`.
    output_delay_samples: usize,
}

impl fmt::Debug for PolyphaseFir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolyphaseFir")
            .field("nbr_channels", &self.nbr_channels)
            .field("taps", &self.taps)
            .field("oversampling", &self.oversampling)
            .field("step", &self.step)
            .field("resample_ratio", &self.resample_ratio)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl PolyphaseFir {
    /// Construct a new polyphase FIR upsampler.
    ///
    /// * `input_rate` / `output_rate` set the ratio, fixed for the lifetime of
    ///   the engine. `output_rate` must be strictly greater than `input_rate`.
    /// * `chunk_size` is the number of *output* frames per call.
    /// * `taps` is the per-branch tap count. The conceptual prototype is
    ///   `taps * L` long, so 8192 taps at `L = 320` (44.1 kHz -> 96 kHz) is a
    ///   2.6 M-tap linear-phase lowpass.
    pub fn new(
        input_rate: usize,
        output_rate: usize,
        chunk_size: usize,
        nbr_channels: usize,
        taps: usize,
    ) -> Result<Self, String> {
        if taps == 0 || chunk_size == 0 || nbr_channels == 0 {
            return Err(
                "PolyphaseFir: taps, chunk_size and nbr_channels must all be > 0".to_string(),
            );
        }
        let (oversampling, step) = derived_oversampling(input_rate, output_rate)?;
        let resample_ratio = output_rate as f64 / input_rate as f64;

        let prototype = design_prototype(taps, oversampling);

        // Decompose into polyphase branches, keeping only the lower half plus
        // the middle one. Branch b has taps h_b[k] = h[k*L + b].
        let stored_branches = oversampling / 2 + 1;
        let mut branches = vec![0.0 as PrcFmt; stored_branches * taps];
        for b in 0..stored_branches {
            for k in 0..taps {
                branches[b * taps + k] = prototype[k * oversampling + b];
            }
        }

        // Worst-case new input frames per call, plus headroom for the history
        // the convolution reads back over.
        let max_in_per_call = (chunk_size * step).div_ceil(oversampling) + 2;
        let max_buffer_len = taps + max_in_per_call + 4;
        let buffers = vec![vec![0.0 as PrcFmt; max_buffer_len]; nbr_channels];

        // The prototype's group delay is (taps*L)/2 samples at the upsampled
        // rate, i.e. taps/2 input samples, i.e. taps/2 * L/step output frames.
        let output_delay_samples = taps / 2 * oversampling / step;

        Ok(Self {
            nbr_channels,
            taps,
            oversampling,
            step,
            resample_ratio,
            buffers,
            buffer_fill: 0,
            next_sample: 0,
            next_branch: 0,
            branches,
            chunk_size,
            max_buffer_len,
            channel_mask: vec![true; nbr_channels],
            active_channels: Vec::with_capacity(nbr_channels),
            output_delay_samples,
        })
    }

    /// Number of input frames required for the next call to produce
    /// `chunk_size` output frames. The convolution also reads backwards over up
    /// to `taps` past samples, but those are already in the buffer (or are
    /// zeros during priming), so only the forward reach counts.
    fn input_needed(&self) -> usize {
        let last_up = self.next_branch + (self.chunk_size - 1) * self.step;
        let last_n = self.next_sample + last_up / self.oversampling;
        (last_n + 1).saturating_sub(self.buffer_fill)
    }

    /// Evaluate one output sample: the inner product of polyphase branch
    /// `branch` with the input history ending at `n_p`.
    ///
    /// For `branch <= L/2` this reads the stored branch backwards over the
    /// input. Above that, prototype symmetry (`h[m] = h[N-1-m]` with
    /// `N-1 = taps*L`) gives `h_b[k] = h_{L-b}[taps-1-k]`, so the same result is
    /// obtained by reading the mirrored stored branch and the input both
    /// forwards - no shuffle, and half the table.
    #[inline]
    fn eval(&self, buf: &[PrcFmt], branch: usize, n_p: usize) -> PrcFmt {
        let taps = self.taps;
        let (k_start, k_end) = valid_tap_range(n_p, buf.len(), taps);
        if k_start >= k_end {
            return 0.0 as PrcFmt;
        }
        if branch * 2 <= self.oversampling {
            let coeffs = &self.branches[branch * taps..(branch + 1) * taps];
            convolve_backward(coeffs, buf, n_p, k_start, k_end)
        } else {
            let mirrored = self.oversampling - branch;
            let base = mirrored * taps;
            // j = taps - 1 - k, so k in [k_start, k_end) maps to
            // j in [taps - k_end, taps - k_start), reading the input forwards
            // from index n_p + 1 - k_end.
            let coeffs = &self.branches[base + taps - k_end..base + taps - k_start];
            let samples = &buf[n_p + 1 - k_end..n_p + 1 - k_start];
            convolve_forward(coeffs, samples)
        }
    }

    /// [`Self::eval`] for two channels at once, sharing the coefficient loads.
    ///
    /// Every channel of an output frame uses the same branch at the same input
    /// index, and the coefficient table is the one operand that does not fit
    /// in L1, so evaluating a pair in a single pass halves the L2-streamed
    /// traffic (per tap pair: 1 coefficient load + 2 sample loads + 2 FMAs,
    /// against 2 + 2 + 2 for two separate passes). The per-channel summation
    /// order is unchanged, so the results are bit-identical to two `eval`s.
    ///
    /// Both buffers have the same length, so one tap range serves both.
    #[inline]
    fn eval2(
        &self,
        buf0: &[PrcFmt],
        buf1: &[PrcFmt],
        branch: usize,
        n_p: usize,
    ) -> (PrcFmt, PrcFmt) {
        debug_assert_eq!(buf0.len(), buf1.len());
        let taps = self.taps;
        let (k_start, k_end) = valid_tap_range(n_p, buf0.len(), taps);
        if k_start >= k_end {
            return (0.0 as PrcFmt, 0.0 as PrcFmt);
        }
        if branch * 2 <= self.oversampling {
            let coeffs = &self.branches[branch * taps..(branch + 1) * taps];
            convolve_backward2(coeffs, buf0, buf1, n_p, k_start, k_end)
        } else {
            let mirrored = self.oversampling - branch;
            let base = mirrored * taps;
            let coeffs = &self.branches[base + taps - k_end..base + taps - k_start];
            let samples0 = &buf0[n_p + 1 - k_end..n_p + 1 - k_start];
            let samples1 = &buf1[n_p + 1 - k_end..n_p + 1 - k_start];
            convolve_forward2(coeffs, samples0, samples1)
        }
    }
}

/// Clip the tap range `[0, taps)` to the taps whose input sample
/// `n_p - k` lies inside `0..buf_len`.
#[inline]
fn valid_tap_range(n_p: usize, buf_len: usize, taps: usize) -> (usize, usize) {
    let k_end = (n_p + 1).min(taps);
    let k_start = (n_p + 1).saturating_sub(buf_len);
    (k_start.min(k_end), k_end)
}

/// `sum_k coeffs[k] * buf[n_p - k]` over `k in [k_start, k_end)`.
#[inline]
fn convolve_backward(
    coeffs: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    {
        // SAFETY: NEON is mandatory on AArch64 and the tap range was derived
        // from `buf`/`coeffs` bounds by `valid_tap_range`.
        unsafe { neon::convolve_backward_neon(coeffs, buf, n_p, k_start, k_end) }
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "32bit"))))]
    {
        convolve_backward_scalar(coeffs, buf, n_p, k_start, k_end)
    }
}

/// Portable reference for [`convolve_backward`]; also the NEON test oracle, so
/// it stays compiled even where the NEON kernel is the one that runs.
// Casts between PrcFmt and f64 are no-ops in the default build but are
// required under `feature = "32bit"`, where PrcFmt is f32.
#[allow(clippy::unnecessary_cast)]
#[allow(dead_code)]
#[inline]
fn convolve_backward_scalar(
    coeffs: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= coeffs.len());
    debug_assert!(k_end <= n_p + 1);
    debug_assert!(k_start + buf.len() >= n_p + 1);

    let mut k = k_start;
    let mut acc0 = 0.0_f64;
    let mut acc1 = 0.0_f64;
    let mut acc2 = 0.0_f64;
    let mut acc3 = 0.0_f64;

    unsafe {
        while k + 3 < k_end {
            let sample_idx = n_p - k;
            acc0 += *coeffs.get_unchecked(k) as f64 * *buf.get_unchecked(sample_idx) as f64;
            acc1 += *coeffs.get_unchecked(k + 1) as f64 * *buf.get_unchecked(sample_idx - 1) as f64;
            acc2 += *coeffs.get_unchecked(k + 2) as f64 * *buf.get_unchecked(sample_idx - 2) as f64;
            acc3 += *coeffs.get_unchecked(k + 3) as f64 * *buf.get_unchecked(sample_idx - 3) as f64;
            k += 4;
        }

        let mut acc = acc0 + acc1 + acc2 + acc3;
        while k < k_end {
            acc += *coeffs.get_unchecked(k) as f64 * *buf.get_unchecked(n_p - k) as f64;
            k += 1;
        }
        acc as PrcFmt
    }
}

/// Plain dot product of two equal-length slices, used for the mirrored branches.
#[inline]
fn convolve_forward(coeffs: &[PrcFmt], samples: &[PrcFmt]) -> PrcFmt {
    debug_assert_eq!(coeffs.len(), samples.len());

    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    {
        // SAFETY: NEON is mandatory on AArch64, and the two slices have equal
        // length (asserted above, and guaranteed by `eval`).
        unsafe { neon::convolve_forward_neon(coeffs, samples) }
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "32bit"))))]
    {
        convolve_forward_scalar(coeffs, samples)
    }
}

/// Portable reference for [`convolve_forward`]; see
/// [`convolve_backward_scalar`].
// Casts between PrcFmt and f64 are no-ops in the default build but are
// required under `feature = "32bit"`, where PrcFmt is f32.
#[allow(clippy::unnecessary_cast)]
#[allow(dead_code)]
#[inline]
fn convolve_forward_scalar(coeffs: &[PrcFmt], samples: &[PrcFmt]) -> PrcFmt {
    debug_assert_eq!(coeffs.len(), samples.len());

    let len = coeffs.len();
    let mut i = 0;
    let mut acc0 = 0.0_f64;
    let mut acc1 = 0.0_f64;
    let mut acc2 = 0.0_f64;
    let mut acc3 = 0.0_f64;

    unsafe {
        while i + 3 < len {
            acc0 += *coeffs.get_unchecked(i) as f64 * *samples.get_unchecked(i) as f64;
            acc1 += *coeffs.get_unchecked(i + 1) as f64 * *samples.get_unchecked(i + 1) as f64;
            acc2 += *coeffs.get_unchecked(i + 2) as f64 * *samples.get_unchecked(i + 2) as f64;
            acc3 += *coeffs.get_unchecked(i + 3) as f64 * *samples.get_unchecked(i + 3) as f64;
            i += 4;
        }

        let mut acc = acc0 + acc1 + acc2 + acc3;
        while i < len {
            acc += *coeffs.get_unchecked(i) as f64 * *samples.get_unchecked(i) as f64;
            i += 1;
        }
        acc as PrcFmt
    }
}

/// [`convolve_backward`] for two channels against the same coefficients.
#[inline]
fn convolve_backward2(
    coeffs: &[PrcFmt],
    buf0: &[PrcFmt],
    buf1: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> (PrcFmt, PrcFmt) {
    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    {
        // SAFETY: NEON is mandatory on AArch64 and the tap range was derived
        // from the (equal) buffer lengths and `coeffs` by `valid_tap_range`.
        unsafe { neon::convolve_backward2_neon(coeffs, buf0, buf1, n_p, k_start, k_end) }
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "32bit"))))]
    {
        convolve_backward2_scalar(coeffs, buf0, buf1, n_p, k_start, k_end)
    }
}

/// Portable reference for [`convolve_backward2`]; see
/// [`convolve_backward_scalar`]. Accumulation order per channel matches the
/// single-channel kernel exactly.
// Casts between PrcFmt and f64 are no-ops in the default build but are
// required under `feature = "32bit"`, where PrcFmt is f32.
#[allow(clippy::unnecessary_cast)]
#[allow(dead_code)]
#[inline]
fn convolve_backward2_scalar(
    coeffs: &[PrcFmt],
    buf0: &[PrcFmt],
    buf1: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> (PrcFmt, PrcFmt) {
    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= coeffs.len());
    debug_assert!(k_end <= n_p + 1);
    debug_assert!(k_start + buf0.len() >= n_p + 1);
    debug_assert!(k_start + buf1.len() >= n_p + 1);

    let mut k = k_start;
    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut b0 = 0.0_f64;
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    let mut b3 = 0.0_f64;

    unsafe {
        while k + 3 < k_end {
            let s = n_p - k;
            let c0 = *coeffs.get_unchecked(k) as f64;
            let c1 = *coeffs.get_unchecked(k + 1) as f64;
            let c2 = *coeffs.get_unchecked(k + 2) as f64;
            let c3 = *coeffs.get_unchecked(k + 3) as f64;
            a0 += c0 * *buf0.get_unchecked(s) as f64;
            a1 += c1 * *buf0.get_unchecked(s - 1) as f64;
            a2 += c2 * *buf0.get_unchecked(s - 2) as f64;
            a3 += c3 * *buf0.get_unchecked(s - 3) as f64;
            b0 += c0 * *buf1.get_unchecked(s) as f64;
            b1 += c1 * *buf1.get_unchecked(s - 1) as f64;
            b2 += c2 * *buf1.get_unchecked(s - 2) as f64;
            b3 += c3 * *buf1.get_unchecked(s - 3) as f64;
            k += 4;
        }

        let mut acc_a = a0 + a1 + a2 + a3;
        let mut acc_b = b0 + b1 + b2 + b3;
        while k < k_end {
            let c = *coeffs.get_unchecked(k) as f64;
            acc_a += c * *buf0.get_unchecked(n_p - k) as f64;
            acc_b += c * *buf1.get_unchecked(n_p - k) as f64;
            k += 1;
        }
        (acc_a as PrcFmt, acc_b as PrcFmt)
    }
}

/// [`convolve_forward`] for two channels against the same coefficients.
#[inline]
fn convolve_forward2(
    coeffs: &[PrcFmt],
    samples0: &[PrcFmt],
    samples1: &[PrcFmt],
) -> (PrcFmt, PrcFmt) {
    debug_assert_eq!(coeffs.len(), samples0.len());
    debug_assert_eq!(coeffs.len(), samples1.len());

    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    {
        // SAFETY: NEON is mandatory on AArch64, and the three slices have equal
        // length (asserted above, and guaranteed by `eval2`).
        unsafe { neon::convolve_forward2_neon(coeffs, samples0, samples1) }
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "32bit"))))]
    {
        convolve_forward2_scalar(coeffs, samples0, samples1)
    }
}

/// Portable reference for [`convolve_forward2`]; see
/// [`convolve_backward_scalar`].
// Casts between PrcFmt and f64 are no-ops in the default build but are
// required under `feature = "32bit"`, where PrcFmt is f32.
#[allow(clippy::unnecessary_cast)]
#[allow(dead_code)]
#[inline]
fn convolve_forward2_scalar(
    coeffs: &[PrcFmt],
    samples0: &[PrcFmt],
    samples1: &[PrcFmt],
) -> (PrcFmt, PrcFmt) {
    debug_assert_eq!(coeffs.len(), samples0.len());
    debug_assert_eq!(coeffs.len(), samples1.len());

    let len = coeffs.len();
    let mut i = 0;
    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut b0 = 0.0_f64;
    let mut b1 = 0.0_f64;
    let mut b2 = 0.0_f64;
    let mut b3 = 0.0_f64;

    unsafe {
        while i + 3 < len {
            let c0 = *coeffs.get_unchecked(i) as f64;
            let c1 = *coeffs.get_unchecked(i + 1) as f64;
            let c2 = *coeffs.get_unchecked(i + 2) as f64;
            let c3 = *coeffs.get_unchecked(i + 3) as f64;
            a0 += c0 * *samples0.get_unchecked(i) as f64;
            a1 += c1 * *samples0.get_unchecked(i + 1) as f64;
            a2 += c2 * *samples0.get_unchecked(i + 2) as f64;
            a3 += c3 * *samples0.get_unchecked(i + 3) as f64;
            b0 += c0 * *samples1.get_unchecked(i) as f64;
            b1 += c1 * *samples1.get_unchecked(i + 1) as f64;
            b2 += c2 * *samples1.get_unchecked(i + 2) as f64;
            b3 += c3 * *samples1.get_unchecked(i + 3) as f64;
            i += 4;
        }

        let mut acc_a = a0 + a1 + a2 + a3;
        let mut acc_b = b0 + b1 + b2 + b3;
        while i < len {
            let c = *coeffs.get_unchecked(i) as f64;
            acc_a += c * *samples0.get_unchecked(i) as f64;
            acc_b += c * *samples1.get_unchecked(i) as f64;
            i += 1;
        }
        (acc_a as PrcFmt, acc_b as PrcFmt)
    }
}

impl Resampler<PrcFmt> for PolyphaseFir {
    fn process_into_buffer<'a>(
        &mut self,
        buffer_in: &dyn Adapter<'a, PrcFmt>,
        buffer_out: &mut dyn AdapterMut<'a, PrcFmt>,
        indexing: Option<&Indexing>,
    ) -> ResampleResult<(usize, usize)> {
        // Apply the active-channel mask from the indexing struct.
        if let Some(idx) = indexing {
            if let Some(m) = &idx.active_channels_mask {
                self.channel_mask.copy_from_slice(m);
            } else {
                self.channel_mask.iter_mut().for_each(|v| *v = true);
            }
        } else {
            self.channel_mask.iter_mut().for_each(|v| *v = true);
        }
        self.active_channels.clear();
        self.active_channels.extend(
            self.channel_mask
                .iter()
                .enumerate()
                .filter(|(_, active)| **active)
                .map(|(ch, _)| ch),
        );
        let (input_offset, output_offset) = indexing
            .map(|i| (i.input_offset, i.output_offset))
            .unwrap_or((0, 0));
        let partial_len = indexing.and_then(|i| i.partial_len);

        let needed = self.input_needed();
        let frames_to_read = if let Some(plen) = partial_len {
            plen.min(needed)
        } else {
            needed
        };

        // Validate inputs.
        if buffer_in.channels() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfInputChannels {
                expected: self.nbr_channels,
                actual: buffer_in.channels(),
            });
        }
        if buffer_out.channels() != self.nbr_channels {
            return Err(ResampleError::WrongNumberOfOutputChannels {
                expected: self.nbr_channels,
                actual: buffer_out.channels(),
            });
        }
        if buffer_in.frames() < input_offset + frames_to_read {
            return Err(ResampleError::InsufficientInputBufferSize {
                expected: input_offset + frames_to_read,
                actual: buffer_in.frames(),
            });
        }
        if buffer_out.frames() < output_offset + self.chunk_size {
            return Err(ResampleError::InsufficientOutputBufferSize {
                expected: output_offset + self.chunk_size,
                actual: buffer_out.frames(),
            });
        }

        // Append new input samples after the existing fill. The bound was
        // computed at construction from the exact integer ratio.
        debug_assert!(self.buffer_fill + needed <= self.max_buffer_len);
        for (ch, active) in self.channel_mask.iter().enumerate() {
            let buf = &mut self.buffers[ch];
            if *active {
                let dst = &mut buf[self.buffer_fill..self.buffer_fill + frames_to_read];
                buffer_in.copy_from_channel_to_slice(ch, input_offset, dst);
                // Pad with zeros if partial.
                for v in &mut buf[self.buffer_fill + frames_to_read..self.buffer_fill + needed] {
                    *v = 0.0;
                }
            } else {
                // An inactive channel contributes silence. Zero its history so
                // that stale samples cannot reappear when it becomes active.
                for v in &mut buf[self.buffer_fill..self.buffer_fill + needed] {
                    *v = 0.0;
                }
            }
        }
        self.buffer_fill += needed;

        // Produce output samples. The phase is exact integer arithmetic:
        // `n` is the input sample index, `b` the polyphase branch.
        let oversampling = self.oversampling;
        let mut n = self.next_sample;
        let mut b = self.next_branch;
        // Active channels are walked in pairs so that each coefficient load
        // serves two channels (see `eval2`); an odd channel takes the single
        // path.
        for out_idx in 0..self.chunk_size {
            let mut pairs = self.active_channels.chunks_exact(2);
            for pair in &mut pairs {
                let (ch0, ch1) = (pair[0], pair[1]);
                let (y0, y1) = self.eval2(&self.buffers[ch0], &self.buffers[ch1], b, n);
                buffer_out.write_sample(ch0, output_offset + out_idx, &y0);
                buffer_out.write_sample(ch1, output_offset + out_idx, &y1);
            }
            if let [ch] = pairs.remainder() {
                let y = self.eval(&self.buffers[*ch], b, n);
                buffer_out.write_sample(*ch, output_offset + out_idx, &y);
            }
            b += self.step;
            if b >= oversampling {
                n += b / oversampling;
                b %= oversampling;
            }
        }
        self.next_sample = n;
        self.next_branch = b;

        // Slide the buffer so that exactly `taps` samples of history remain
        // behind the next output position, and drop the rest.
        let consumed = self
            .next_sample
            .saturating_sub(self.taps)
            .min(self.buffer_fill);
        if consumed > 0 {
            for buf in &mut self.buffers {
                buf.copy_within(consumed..self.buffer_fill, 0);
                for v in &mut buf[self.buffer_fill - consumed..self.buffer_fill] {
                    *v = 0.0;
                }
            }
            self.buffer_fill -= consumed;
            self.next_sample -= consumed;
        }

        Ok((needed, self.chunk_size))
    }

    fn input_frames_max(&self) -> usize {
        self.max_buffer_len
    }

    fn input_frames_next(&self) -> usize {
        self.input_needed()
    }

    fn nbr_channels(&self) -> usize {
        self.nbr_channels
    }

    fn output_frames_max(&self) -> usize {
        self.chunk_size
    }

    fn output_frames_next(&self) -> usize {
        self.chunk_size
    }

    fn output_delay(&self) -> usize {
        self.output_delay_samples
    }

    fn set_resample_ratio(&mut self, _new_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        // Fixed at construction. Use `Polyphase` only with a known, stable rate;
        // for `rate_adjust` workflows pick `AsyncSinc` instead.
        Err(ResampleError::SyncNotAdjustable)
    }

    fn resample_ratio(&self) -> f64 {
        self.resample_ratio
    }

    fn set_resample_ratio_relative(&mut self, _rel_ratio: f64, _ramp: bool) -> ResampleResult<()> {
        Err(ResampleError::SyncNotAdjustable)
    }

    fn reset(&mut self) {
        for buf in &mut self.buffers {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
        self.buffer_fill = 0;
        self.next_sample = 0;
        self.next_branch = 0;
        self.channel_mask.iter_mut().for_each(|v| *v = true);
    }
}

// ---------------------------------------------------------------------------
// Prototype filter design
// ---------------------------------------------------------------------------

/// Design the linear-phase prototype lowpass for `taps` per branch and
/// `oversampling` branches.
///
/// The prototype runs at the upsampled rate, so the source Nyquist sits at the
/// normalized frequency `0.5 / oversampling`. The cutoff (the -6 dB point) is
/// placed half a transition width *below* that, which puts the stopband edge on
/// the source Nyquist instead of the middle of the transition: nothing from the
/// transition band images into the output.
fn design_prototype(taps: usize, oversampling: usize) -> Vec<PrcFmt> {
    let proto_len = taps * oversampling;
    let cutoff = 0.5 / oversampling as f64 - transition_width(proto_len) / 2.0;
    design_kaiser_lp(proto_len, cutoff, STOPBAND_ATTENUATION_DB, oversampling)
}

/// Kaiser-window-based lowpass FIR designer.
///
/// * `len` is rounded up to odd, for true linear phase.
/// * `cutoff` is normalized to the sample rate (0.5 = Nyquist).
/// * `attenuation_db` drives the Kaiser beta.
/// * The filter is scaled so that DC gain equals `oversampling`, which gives
///   unity gain after the polyphase decimation by L.
// Casts between PrcFmt and f64 are no-ops in the default build but are
// required under `feature = "32bit"`, where PrcFmt is f32.
#[allow(clippy::unnecessary_cast)]
fn design_kaiser_lp(
    mut len: usize,
    cutoff: f64,
    attenuation_db: f64,
    oversampling: usize,
) -> Vec<PrcFmt> {
    if len.is_multiple_of(2) {
        len |= 1;
    }
    let beta = if attenuation_db > 50.0 {
        0.1102 * (attenuation_db - 8.7)
    } else if attenuation_db >= 21.0 {
        0.5842 * (attenuation_db - 21.0).powf(0.4) + 0.07886 * (attenuation_db - 21.0)
    } else {
        0.0
    };
    let i0_beta = bessel_i0(beta);
    let half = (len as i64 - 1) / 2;
    let mut h = vec![0.0 as PrcFmt; len];
    for (n, slot) in h.iter_mut().enumerate() {
        let k = n as i64 - half;
        let arg = 2.0 * cutoff * k as f64;
        let sinc = if k == 0 {
            1.0
        } else {
            (PI * arg).sin() / (PI * arg)
        };
        let win_arg = k as f64 / half as f64;
        let win = bessel_i0(beta * (1.0 - win_arg * win_arg).max(0.0).sqrt()) / i0_beta;
        *slot = ((2.0 * cutoff) * sinc * win) as PrcFmt;
    }
    // Normalize so DC gain == oversampling (each polyphase branch gets gain 1).
    let dc: f64 = h.iter().map(|v| *v as f64).sum();
    let scale = oversampling as f64 / dc;
    for v in &mut h {
        *v = (*v as f64 * scale) as PrcFmt;
    }
    h
}

/// Modified Bessel function of the first kind, order zero.
/// Power-series approximation; converges quickly for our beta range (<= ~16).
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
// As in the engine itself, PrcFmt <-> f64 casts are no-ops here but are needed
// under `feature = "32bit"`.
#[allow(clippy::unnecessary_cast)]
mod tests {
    use super::*;
    use audioadapter_buffers::direct::SequentialSliceOfVecs;

    fn freq_response_db(h: &[PrcFmt], freq_normalized: f64) -> f64 {
        // Evaluate |H(e^{jw})| at the given normalized frequency (0..0.5).
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let w = 2.0 * PI * freq_normalized;
        for (n, v) in h.iter().enumerate() {
            re += *v as f64 * (n as f64 * w).cos();
            im -= *v as f64 * (n as f64 * w).sin();
        }
        20.0 * (re * re + im * im).sqrt().log10()
    }

    #[test]
    fn derived_oversampling_for_target_rates() {
        assert_eq!(derived_oversampling(44_100, 96_000).unwrap(), (320, 147));
        assert_eq!(derived_oversampling(48_000, 96_000).unwrap(), (2, 1));
        assert_eq!(derived_oversampling(88_200, 96_000).unwrap(), (160, 147));
    }

    #[test]
    fn rejects_downsampling_and_huge_ratio() {
        // Downsampling and pass-through are out of scope for this engine.
        assert!(PolyphaseFir::new(96_000, 44_100, 128, 2, 32).is_err());
        assert!(PolyphaseFir::new(96_000, 96_000, 128, 2, 32).is_err());
        // Coprime rates would need one branch per output-rate hertz.
        assert!(PolyphaseFir::new(44_100, 44_101, 128, 2, 32).is_err());
    }

    #[test]
    fn kaiser_lp_is_symmetric() {
        let h = design_kaiser_lp(257, 0.2, 100.0, 1);
        for n in 0..128 {
            let diff = (h[n] - h[256 - n]).abs();
            assert!(diff < 1e-9 as PrcFmt, "asymmetry at {n}: {diff}");
        }
    }

    #[test]
    fn kaiser_lp_stopband_meets_120db() {
        // Long Kaiser, cutoff at 0.2, stopband should start near 0.25.
        let h = design_kaiser_lp(2049, 0.2, 130.0, 1);
        // At 0.35 we should be well into the stopband.
        let resp = freq_response_db(&h, 0.35);
        assert!(resp <= -120.0, "stopband only reached {resp} dB");
    }

    #[test]
    fn stopband_edge_is_at_nyquist() {
        // The defining property of the cutoff placement: the stopband is
        // already reached at the source Nyquist (0.5 / L of the upsampled
        // rate), rather than the -6 dB point sitting there.
        let taps = 256;
        let oversampling = 320;
        let proto = design_prototype(taps, oversampling);
        let nyquist = 0.5 / oversampling as f64;
        let width = transition_width(taps * oversampling);

        let stop = freq_response_db(&proto, nyquist) - 20.0 * (oversampling as f64).log10();
        assert!(
            stop <= -135.0,
            "response at the source Nyquist is only {stop} dB below passband"
        );

        let pass =
            freq_response_db(&proto, nyquist - 1.5 * width) - 20.0 * (oversampling as f64).log10();
        assert!(
            pass >= -0.01,
            "passband droops to {pass} dB just inside the transition"
        );
    }

    #[test]
    fn half_table_matches_full_table() {
        // Every branch above L/2 is evaluated from the mirrored stored branch
        // with the forward kernel; the result must match a straight full-table
        // convolution to f64 rounding.
        let taps = 32;
        let mut r = PolyphaseFir::new(44_100, 96_000, 128, 1, taps).unwrap();
        let oversampling = r.oversampling;
        let prototype = design_prototype(taps, oversampling);
        for (idx, sample) in r.buffers[0].iter_mut().enumerate() {
            *sample = (((idx * 17 % 31) as f64 - 15.0) / 31.0) as PrcFmt;
        }

        let buf = r.buffers[0].clone();
        let n_p = buf.len() - 1;
        for b in 0..oversampling {
            let mut reference = 0.0_f64;
            let mut magnitude = 0.0_f64;
            for k in 0..taps {
                let term = prototype[k * oversampling + b] as f64 * buf[n_p - k] as f64;
                reference += term;
                magnitude += term.abs();
            }
            let got = r.eval(&buf, b, n_p) as f64;
            let tol = if cfg!(feature = "32bit") { 1e-5 } else { 1e-12 } * magnitude.max(1e-3);
            assert!(
                (got - reference).abs() <= tol,
                "branch {b}: got {got}, reference {reference}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    #[test]
    fn neon_kernels_match_scalar() {
        for &taps in &[1_usize, 2, 3, 4, 7, 8, 31, 64, 65] {
            let coeffs: Vec<PrcFmt> = (0..taps)
                .map(|idx| (((idx * 17 % 31) as f64 - 15.0) / 97.0) as PrcFmt)
                .collect();
            let buf: Vec<PrcFmt> = (0..96)
                .map(|idx| (((idx * 13 % 29) as f64 - 14.0) / 53.0) as PrcFmt)
                .collect();
            let tol = 1e-14 * taps as f64;

            for &n_p in &[0_usize, 1, 5, 31, 63, 95] {
                let (k_start, k_end) = valid_tap_range(n_p, buf.len(), taps);
                if k_start >= k_end {
                    continue;
                }
                let scalar = convolve_backward_scalar(&coeffs, &buf, n_p, k_start, k_end);
                let simd =
                    unsafe { neon::convolve_backward_neon(&coeffs, &buf, n_p, k_start, k_end) };
                assert!(
                    (scalar as f64 - simd as f64).abs() <= tol,
                    "backward mismatch taps={taps}, n_p={n_p}: scalar={scalar}, neon={simd}"
                );
            }

            for &offset in &[0_usize, 1, 7, 20] {
                let samples = &buf[offset..offset + taps];
                let scalar = convolve_forward_scalar(&coeffs, samples);
                let simd = unsafe { neon::convolve_forward_neon(&coeffs, samples) };
                assert!(
                    (scalar as f64 - simd as f64).abs() <= tol,
                    "forward mismatch taps={taps}, offset={offset}: scalar={scalar}, neon={simd}"
                );
            }

            // Pair kernels: the scalar twin is the oracle, and each channel must
            // also agree with the single-channel scalar kernel.
            let buf2: Vec<PrcFmt> = (0..96)
                .map(|idx| (((idx * 7 % 23) as f64 - 11.0) / 41.0) as PrcFmt)
                .collect();
            for &n_p in &[0_usize, 1, 5, 31, 63, 95] {
                let (k_start, k_end) = valid_tap_range(n_p, buf.len(), taps);
                if k_start >= k_end {
                    continue;
                }
                let scalar = convolve_backward2_scalar(&coeffs, &buf, &buf2, n_p, k_start, k_end);
                let simd = unsafe {
                    neon::convolve_backward2_neon(&coeffs, &buf, &buf2, n_p, k_start, k_end)
                };
                let single = (
                    convolve_backward_scalar(&coeffs, &buf, n_p, k_start, k_end),
                    convolve_backward_scalar(&coeffs, &buf2, n_p, k_start, k_end),
                );
                assert!(
                    (scalar.0 as f64 - simd.0 as f64).abs() <= tol
                        && (scalar.1 as f64 - simd.1 as f64).abs() <= tol,
                    "backward2 mismatch taps={taps}, n_p={n_p}: scalar={scalar:?}, neon={simd:?}"
                );
                assert!(
                    (scalar.0 as f64 - single.0 as f64).abs() <= tol
                        && (scalar.1 as f64 - single.1 as f64).abs() <= tol,
                    "backward2 differs from single taps={taps}, n_p={n_p}: pair={scalar:?}, single={single:?}"
                );
            }
            for &offset in &[0_usize, 1, 7, 20] {
                let s0 = &buf[offset..offset + taps];
                let s1 = &buf2[offset..offset + taps];
                let scalar = convolve_forward2_scalar(&coeffs, s0, s1);
                let simd = unsafe { neon::convolve_forward2_neon(&coeffs, s0, s1) };
                let single = (
                    convolve_forward_scalar(&coeffs, s0),
                    convolve_forward_scalar(&coeffs, s1),
                );
                assert!(
                    (scalar.0 as f64 - simd.0 as f64).abs() <= tol
                        && (scalar.1 as f64 - simd.1 as f64).abs() <= tol,
                    "forward2 mismatch taps={taps}, offset={offset}: scalar={scalar:?}, neon={simd:?}"
                );
                assert!(
                    (scalar.0 as f64 - single.0 as f64).abs() <= tol
                        && (scalar.1 as f64 - single.1 as f64).abs() <= tol,
                    "forward2 differs from single taps={taps}, offset={offset}: pair={scalar:?}, single={single:?}"
                );
            }
        }
    }

    #[test]
    fn pair_eval_matches_single_eval() {
        // The coefficient-sharing path must be bit-identical to two single
        // evaluations, on every branch (both kernels) and in the clipped
        // priming region.
        let taps = 32;
        let mut r = PolyphaseFir::new(44_100, 96_000, 128, 2, taps).unwrap();
        for (idx, sample) in r.buffers[0].iter_mut().enumerate() {
            *sample = (((idx * 17 % 31) as f64 - 15.0) / 31.0) as PrcFmt;
        }
        for (idx, sample) in r.buffers[1].iter_mut().enumerate() {
            *sample = (((idx * 7 % 23) as f64 - 11.0) / 23.0) as PrcFmt;
        }
        let buf0 = r.buffers[0].clone();
        let buf1 = r.buffers[1].clone();
        for &n_p in &[buf0.len() - 1, taps + 3, 10, 0] {
            for b in 0..r.oversampling {
                let pair = r.eval2(&buf0, &buf1, b, n_p);
                let single = (r.eval(&buf0, b, n_p), r.eval(&buf1, b, n_p));
                assert!(
                    pair == single,
                    "n_p={n_p} branch {b}: pair {pair:?} != single {single:?}"
                );
            }
        }
    }

    #[test]
    fn odd_and_masked_channel_sets_are_all_written() {
        // Three channels: one pair plus a remainder channel. Then a mask that
        // leaves channels 0 and 2 active, so the pair is formed from
        // non-adjacent channels. Each channel is fed its own DC level and
        // must converge to it.
        let chunk = 512;
        let mut r = PolyphaseFir::new(44_100, 96_000, chunk, 3, 64).unwrap();
        let in_len = r.input_frames_max();
        let levels = [0.25 as PrcFmt, -0.5 as PrcFmt, 0.75 as PrcFmt];
        let waves: Vec<Vec<PrcFmt>> = levels.iter().map(|l| vec![*l; in_len]).collect();
        let mut out = vec![vec![0.0 as PrcFmt; chunk]; 3];

        for mask in [vec![true, true, true], vec![true, false, true]] {
            r.reset();
            let indexing = Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: None,
                active_channels_mask: Some(mask.clone()),
            };
            for _ in 0..4 {
                out.iter_mut()
                    .for_each(|w| w.iter_mut().for_each(|v| *v = 0.0));
                let input = SequentialSliceOfVecs::new(&waves, 3, in_len).unwrap();
                let mut output = SequentialSliceOfVecs::new_mut(&mut out, 3, chunk).unwrap();
                r.process_into_buffer(&input, &mut output, Some(&indexing))
                    .unwrap();
            }
            for (ch, active) in mask.iter().enumerate() {
                let mid = out[ch][chunk / 2];
                let expected = if *active { levels[ch] } else { 0.0 as PrcFmt };
                assert!(
                    (mid - expected).abs() < 1e-3 as PrcFmt,
                    "mask {mask:?} channel {ch}: got {mid}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn integer_phase_advances_exactly() {
        // The phase is integer arithmetic, so it cannot drift no matter how
        // many chunks are processed.
        let chunk = 64;
        let taps = 8;
        let mut r = PolyphaseFir::new(44_100, 96_000, chunk, 1, taps).unwrap();
        let (oversampling, step) = (r.oversampling, r.step);
        let in_len = r.input_frames_max();
        let waves = vec![vec![0.0 as PrcFmt; in_len]; 1];
        let mut out = vec![vec![0.0 as PrcFmt; chunk]; 1];

        let mut consumed_total = 0_usize;
        let mut previous_fill = 0_usize;
        for k in 1..=1000_usize {
            let input = SequentialSliceOfVecs::new(&waves, 1, in_len).unwrap();
            let mut output = SequentialSliceOfVecs::new_mut(&mut out, 1, chunk).unwrap();
            let (read, _) = r.process_into_buffer(&input, &mut output, None).unwrap();
            // consumed = (fill before slide) - (fill after slide)
            consumed_total += previous_fill + read - r.buffer_fill;
            previous_fill = r.buffer_fill;

            let total_up = k * chunk * step;
            assert_eq!(r.next_sample + consumed_total, total_up / oversampling);
            assert_eq!(r.next_branch, total_up % oversampling);
        }
    }

    #[test]
    fn inactive_channel_history_is_zeroed() {
        let mut r = PolyphaseFir::new(44_100, 96_000, 64, 2, 16).unwrap();
        let in_len = r.input_frames_max();
        let waves = vec![vec![0.5 as PrcFmt; in_len]; 2];
        let mut out = vec![vec![0.0 as PrcFmt; 64]; 2];
        let indexing = Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: None,
            active_channels_mask: Some(vec![true, false]),
        };
        let input = SequentialSliceOfVecs::new(&waves, 2, in_len).unwrap();
        let mut output = SequentialSliceOfVecs::new_mut(&mut out, 2, 64).unwrap();
        r.process_into_buffer(&input, &mut output, Some(&indexing))
            .unwrap();

        assert!(
            r.buffers[1].iter().all(|v| *v == 0.0),
            "inactive channel kept a non-zero history"
        );
        assert!(
            r.buffers[0].iter().any(|v| *v != 0.0),
            "active channel lost its history"
        );
    }

    #[test]
    fn polyphase_dc_passes_through_with_unity_gain() {
        // Constant input -> constant output of (approximately) the same value.
        let mut r = PolyphaseFir::new(44_100, 96_000, 512, 1, 64).unwrap();
        let in_len = r.input_frames_max();
        let waves = vec![vec![1.0 as PrcFmt; in_len]; 1];
        let mut out = vec![vec![0.0 as PrcFmt; r.output_frames_max()]; 1];
        let input = SequentialSliceOfVecs::new(&waves, 1, in_len).unwrap();
        let mut output =
            SequentialSliceOfVecs::new_mut(&mut out, 1, r.output_frames_max()).unwrap();
        // Pump several chunks to flush the priming delay.
        for _ in 0..4 {
            r.process_into_buffer(&input, &mut output, None).unwrap();
        }
        // Inspect the middle of the last output chunk (away from edges).
        let mid = out[0][r.output_frames_max() / 2];
        assert!(
            (mid - 1.0).abs() < 1e-3,
            "DC gain off: got {mid} (expected ~1.0)"
        );
    }

    #[test]
    fn polyphase_rejects_rate_adjust() {
        let mut r = PolyphaseFir::new(44_100, 48_000, 128, 2, 32).unwrap();
        assert!(matches!(
            r.set_resample_ratio(1.5, false),
            Err(ResampleError::SyncNotAdjustable)
        ));
        assert!(matches!(
            r.set_resample_ratio_relative(1.1, false),
            Err(ResampleError::SyncNotAdjustable)
        ));
    }

    #[test]
    fn polyphase_follow_capture_samplerate_reconstruction() {
        // Rate changes are handled by reconstructing the engine. Verify that
        // all three supported source rates build and produce full chunks.
        for (in_rate, out_rate) in [(44_100, 96_000), (48_000, 96_000), (88_200, 96_000)] {
            let mut r = PolyphaseFir::new(in_rate, out_rate, 256, 2, 64).unwrap();
            let in_len = r.input_frames_max();
            let waves = vec![vec![0.0 as PrcFmt; in_len]; 2];
            let mut out = vec![vec![0.0 as PrcFmt; r.output_frames_max()]; 2];
            let input = SequentialSliceOfVecs::new(&waves, 2, in_len).unwrap();
            let mut output =
                SequentialSliceOfVecs::new_mut(&mut out, 2, r.output_frames_max()).unwrap();
            let (_in_consumed, out_produced) =
                r.process_into_buffer(&input, &mut output, None).unwrap();
            assert_eq!(out_produced, 256);
        }
    }

    /// Drive the resampler with a unit-amplitude sine at `freq_hz` until the
    /// priming delay is flushed, then collect `collect` output chunks.
    ///
    /// Must feed *exactly* `input_frames_next` per call: feeding more would
    /// leave unused samples in the source array and cause a phase jump in the
    /// engine's view of the input sine on the next call.
    fn collect_steady_output(
        in_rate: usize,
        out_rate: usize,
        freq_hz: f64,
        taps: usize,
        collect: usize,
    ) -> Vec<f64> {
        let chunk_out = 1024;
        let mut r = PolyphaseFir::new(in_rate, out_rate, chunk_out, 1, taps).unwrap();
        let max_in = r.input_frames_max();
        let warmup_chunks = ((r.output_delay() / chunk_out) + 4).max(8);
        let mut collected = Vec::with_capacity(collect * chunk_out);
        let mut out = vec![vec![0.0 as PrcFmt; chunk_out]; 1];
        let mut waves = vec![vec![0.0 as PrcFmt; max_in]; 1];
        let mut sample_counter: usize = 0;
        for chunk_idx in 0..warmup_chunks + collect {
            let needed = r.input_frames_next();
            waves[0].iter_mut().for_each(|v| *v = 0.0);
            for slot in waves[0].iter_mut().take(needed) {
                let t = sample_counter as f64 / in_rate as f64;
                *slot = (2.0 * PI * freq_hz * t).sin() as PrcFmt;
                sample_counter += 1;
            }
            let input = SequentialSliceOfVecs::new(&waves, 1, max_in).unwrap();
            let mut output = SequentialSliceOfVecs::new_mut(&mut out, 1, chunk_out).unwrap();
            r.process_into_buffer(&input, &mut output, None).unwrap();
            if chunk_idx >= warmup_chunks {
                collected.extend(out[0].iter().map(|v| *v as f64));
            }
        }
        collected
    }

    fn measure_peak_amplitude(in_rate: usize, out_rate: usize, freq_hz: f64, taps: usize) -> f64 {
        collect_steady_output(in_rate, out_rate, freq_hz, taps, 4)
            .into_iter()
            .fold(0.0_f64, |peak, v| peak.max(v.abs()))
    }

    /// Amplitude of the tone at `probe_hz` in `signal`, in dB relative to a
    /// unit-amplitude sine. A Kaiser window (beta 20, sidelobes below -190 dB)
    /// keeps a full-scale fundamental from leaking into a far-away probe.
    fn tone_amplitude_db(signal: &[f64], rate: usize, probe_hz: f64) -> f64 {
        let n = signal.len();
        let beta = 20.0_f64;
        let i0_beta = bessel_i0(beta);
        let half = (n - 1) as f64 / 2.0;
        let w = 2.0 * PI * probe_hz / rate as f64;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut win_sum = 0.0_f64;
        for (idx, v) in signal.iter().enumerate() {
            let x = (idx as f64 - half) / half;
            let win = bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / i0_beta;
            win_sum += win;
            let phase = w * idx as f64;
            re += v * win * phase.cos();
            im -= v * win * phase.sin();
        }
        let amplitude = 2.0 * (re * re + im * im).sqrt() / win_sum;
        20.0 * amplitude.max(1e-300).log10()
    }

    #[test]
    fn engine_passband_is_flat() {
        // 44.1k -> 96k. Amplitudes at 1k / 10k / 18k Hz should all be within
        // +-0.5 dB of unity (the input sine has unit amplitude).
        for &freq in &[1000.0_f64, 10_000.0, 18_000.0] {
            let peak = measure_peak_amplitude(44_100, 96_000, freq, 256);
            let db = 20.0 * peak.log10();
            assert!(
                (-0.5..=0.5).contains(&db),
                "passband ripple at {freq} Hz: {db} dB (peak {peak})"
            );
        }
    }

    #[test]
    fn engine_suppresses_images_above_source_nyquist() {
        // Upsampling 44.1k -> 96k replicates the input spectrum around
        // multiples of 44.1 kHz. A 15 kHz tone therefore has an image at
        // 44100 - 15000 = 29100 Hz which the interpolation filter must remove.
        let out_rate = 96_000;
        let signal = collect_steady_output(44_100, out_rate, 15_000.0, 256, 16);
        let fundamental = tone_amplitude_db(&signal, out_rate, 15_000.0);
        assert!(
            (-0.1..=0.1).contains(&fundamental),
            "fundamental should be unity, measured {fundamental} dB"
        );
        let image = tone_amplitude_db(&signal, out_rate, 29_100.0);
        assert!(
            image <= -120.0,
            "image at 29.1 kHz only suppressed to {image} dB"
        );
    }
}

