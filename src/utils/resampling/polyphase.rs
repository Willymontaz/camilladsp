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

//! Audiophile polyphase FIR resampler engine.
//!
//! Implements `rubato::Resampler<PrcFmt>` with a fixed-at-construction ratio
//! and selectable filter character (LinearPhase, MinimumPhase, Apodizing,
//! SlowRollOff). Output chunk size is fixed (matches `rubato::FixedAsync::Output`
//! semantics); the input frame count varies per call based on the ratio.
//!
//! The engine is intentionally a thin, well-documented implementation:
//! per-output-sample inner-product against a polyphase branch, with cubic
//! interpolation across branches for arbitrary (non-integer-ratio) resampling.
//! Quality comes from a long Kaiser prototype designed at init.

use crate::config;
use crate::PrcFmt;
use audioadapter::{Adapter, AdapterMut};
use num_complex::Complex;
use rubato::{Indexing, ResampleError, ResampleResult, Resampler};
use std::f64::consts::PI;
use std::fmt;

#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[path = "polyphase_neon.rs"]
mod neon;

#[cfg(not(feature = "32bit"))]
const DIRECT_PHASE_EPSILON: f64 = 1e-10;
#[cfg(feature = "32bit")]
const DIRECT_PHASE_EPSILON: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectBranch {
    branch_idx: usize,
    sample_shift: isize,
}

/// A long FIR resampler built from a windowed-sinc prototype, decomposed into
/// `oversampling` polyphase branches. The output ratio is fixed at construction.
pub struct PolyphaseFir {
    nbr_channels: usize,
    /// Number of taps per polyphase branch.
    taps: usize,
    /// Polyphase factor (number of branches).
    oversampling: usize,
    /// `output_rate / input_rate`.
    resample_ratio: f64,
    /// Integer advance in upsampled/polyphase units per output sample when
    /// `input_rate * oversampling / output_rate` is integral.
    direct_phase_step: Option<isize>,
    /// Per-channel ring buffer holding the latest input samples. Length is
    /// `taps + max_in_per_call`. We slide window by `consumed` samples per call.
    buffers: Vec<Vec<PrcFmt>>,
    /// How many valid input samples are currently sitting at the start of each buffer.
    buffer_fill: usize,
    /// Fractional position (in input-sample units) within `buffers` of the next
    /// output sample. Updated each call by `-consumed`.
    next_input_pos: f64,
    /// `oversampling` branches, each `taps` long. Stored as one flat Vec for
    /// cache locality; branch `b` is `branches[b * taps .. (b + 1) * taps]`.
    branches: Vec<PrcFmt>,
    /// Maximum output frames per call (fixed).
    chunk_size: usize,
    /// Maximum input frames the buffer can hold (= taps + max_in_per_call_with_headroom).
    max_buffer_len: usize,
    /// Channel-active mask. Inactive channels are not processed.
    channel_mask: Vec<bool>,
    /// Steady-state group delay in output frames, used for `output_delay()`.
    output_delay_samples: usize,
}

impl fmt::Debug for PolyphaseFir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolyphaseFir")
            .field("nbr_channels", &self.nbr_channels)
            .field("taps", &self.taps)
            .field("oversampling", &self.oversampling)
            .field("resample_ratio", &self.resample_ratio)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl PolyphaseFir {
    /// Construct a new polyphase FIR resampler.
    ///
    /// * `input_rate` / `output_rate` set the ratio (fixed for the lifetime).
    /// * `chunk_size` is the number of *output* frames per call.
    /// * `character` selects the FIR character (see [`config::PolyphaseCharacter`]).
    /// * `taps` is the per-branch tap count. With `oversampling = 256` and `taps = 256`,
    ///   the conceptual prototype has ~65k taps - more than enough for >= 120 dB stopband.
    pub fn new(
        input_rate: usize,
        output_rate: usize,
        chunk_size: usize,
        nbr_channels: usize,
        character: config::PolyphaseCharacter,
        taps: usize,
        oversampling: usize,
    ) -> Result<Self, String> {
        if taps == 0 || oversampling == 0 || chunk_size == 0 || nbr_channels == 0 {
            return Err(
                "PolyphaseFir: taps, oversampling, chunk_size, nbr_channels must all be > 0"
                    .to_string(),
            );
        }
        if input_rate == 0 || output_rate == 0 {
            return Err("PolyphaseFir: sample rates must be > 0".to_string());
        }
        let resample_ratio = output_rate as f64 / input_rate as f64;
        let direct_phase_step = direct_phase_step(input_rate, output_rate, oversampling);

        // Design the prototype lowpass at the upsampled rate. Cutoff is
        // 0.5 * min(1, ratio) / oversampling (in normalized-to-upsampled-rate terms).
        // The factor min(1, ratio) protects the downsampling case from aliasing.
        let ratio_for_cutoff = resample_ratio.min(1.0);
        let cutoff_normalized = 0.5 * ratio_for_cutoff / oversampling as f64;
        let proto_len = taps * oversampling;

        let prototype = design_prototype(character, proto_len, cutoff_normalized, oversampling);

        // Decompose into polyphase branches. Branch b has taps h_b[k] = h[k*L + b].
        let mut branches = vec![0.0 as PrcFmt; oversampling * taps];
        for b in 0..oversampling {
            for k in 0..taps {
                branches[b * taps + k] = prototype[k * oversampling + b];
            }
        }

        // Headroom for the input-buffer ring: enough samples to produce
        // `chunk_size` output frames at the worst-case ratio, plus the FIR length.
        let max_in_per_call = (chunk_size as f64 / resample_ratio).ceil() as usize + 4;
        let max_buffer_len = taps + max_in_per_call + 4;
        let buffers = vec![vec![0.0 as PrcFmt; max_buffer_len]; nbr_channels];

        // The output is delayed by the steady-state group delay of the prototype
        // (taps/2 input samples for the linear-phase character). Reported to
        // callers via `output_delay()` per the rubato Resampler trait convention.
        let output_delay_samples = ((taps as f64 / 2.0) * resample_ratio) as usize;

        Ok(Self {
            nbr_channels,
            taps,
            oversampling,
            resample_ratio,
            direct_phase_step,
            buffers,
            buffer_fill: 0,
            next_input_pos: 0.0,
            branches,
            chunk_size,
            max_buffer_len,
            channel_mask: vec![true; nbr_channels],
            output_delay_samples,
        })
    }

    /// Number of input frames required for the next call to produce `chunk_size`
    /// output frames. The convolution at the last output reads back up to
    /// `taps` samples, but those past samples are already in the ring buffer
    /// (or zero-padded); only the forward-reaching part (`n_p_max + 1`) needs
    /// to be provided.
    fn input_needed(&self) -> usize {
        let last_pos = self.next_input_pos + (self.chunk_size - 1) as f64 / self.resample_ratio;
        // Worst case: cubic branch interpolation wraps to sample_shift = +1.
        let max_n_p = last_pos.floor() as isize + 1;
        let needed_count = (max_n_p + 1).max(0) as usize;
        needed_count.saturating_sub(self.buffer_fill)
    }

    /// Evaluate the polyphase FIR at the given fractional input position for one channel.
    /// Uses direct branch evaluation when the phase lands on a polyphase branch,
    /// otherwise falls back to cubic Lagrange interpolation across 4 adjacent branches.
    #[inline]
    fn eval(&self, channel: usize, frac_input_pos: f64) -> PrcFmt {
        let buf = &self.buffers[channel];
        let int_pos = frac_input_pos.floor();
        let frac = frac_input_pos - int_pos; // in [0, 1)
        let center = int_pos as isize;
        let bf = frac * self.oversampling as f64;

        if let Some(direct) = direct_branch(bf, self.oversampling) {
            return self.eval_direct(buf, direct.branch_idx, center + direct.sample_shift);
        }

        self.eval_cubic(buf, center, bf)
    }

    #[inline]
    fn eval_direct(&self, buf: &[PrcFmt], branch_idx: usize, n_p: isize) -> PrcFmt {
        let branch = &self.branches[branch_idx * self.taps..(branch_idx + 1) * self.taps];
        let (k_start, k_end) = valid_tap_range(n_p, buf.len(), self.taps);
        if k_start >= k_end {
            return 0.0 as PrcFmt;
        }
        convolve_direct(branch, buf, n_p, k_start, k_end)
    }

    #[inline]
    fn eval_branch_scalar(&self, buf: &[PrcFmt], branch_idx: usize, n_p: isize) -> PrcFmt {
        let branch = &self.branches[branch_idx * self.taps..(branch_idx + 1) * self.taps];
        let (k_start, k_end) = valid_tap_range(n_p, buf.len(), self.taps);
        if k_start >= k_end {
            return 0.0 as PrcFmt;
        }
        convolve_direct_scalar(branch, buf, n_p, k_start, k_end)
    }

    #[inline]
    fn eval_cubic(&self, buf: &[PrcFmt], center: isize, bf: f64) -> PrcFmt {
        // Branch is `frac * oversampling` within [0, oversampling). For cubic
        // interpolation we evaluate 4 adjacent branches and Lagrange-interpolate.
        let b_int = bf.floor() as isize;
        let b_frac = bf - b_int as f64; // in [0, 1)

        let mut taps_out = [0.0 as PrcFmt; 4];
        for j in 0..4_isize {
            // Four branches: b-1, b, b+1, b+2. When the raw branch index
            // crosses the modular boundary, the equivalent (branch, n_p)
            // shifts the input sample index by +/-1.
            let b_raw = b_int + j - 1;
            let (branch_idx, sample_shift) = if b_raw < 0 {
                ((b_raw + self.oversampling as isize) as usize, -1_isize)
            } else if b_raw >= self.oversampling as isize {
                ((b_raw - self.oversampling as isize) as usize, 1_isize)
            } else {
                (b_raw as usize, 0_isize)
            };
            // Polyphase convolution at integer upsampled position p_up:
            //   y_up[p_up] = sum_k h_b[k] * x[n_p - k]
            // where b = p_up mod L and n_p = p_up div L.
            let n_p = center + sample_shift;
            taps_out[j as usize] = self.eval_branch_scalar(buf, branch_idx, n_p);
        }

        // Cubic Lagrange at offset b_frac over indices {-1, 0, 1, 2}. The four
        // denominators reduce to {-6, 2, -2, 6}.
        let x = b_frac;
        let xm1 = x + 1.0;
        let x0 = x;
        let x1 = x - 1.0;
        let x2 = x - 2.0;
        let l_m1 = (x0 * x1 * x2) / -6.0;
        let l_0 = (xm1 * x1 * x2) / 2.0;
        let l_1 = (xm1 * x0 * x2) / -2.0;
        let l_2 = (xm1 * x0 * x1) / 6.0;
        let result = (taps_out[0] as f64) * l_m1
            + (taps_out[1] as f64) * l_0
            + (taps_out[2] as f64) * l_1
            + (taps_out[3] as f64) * l_2;
        result as PrcFmt
    }
}

#[inline]
fn direct_branch(bf: f64, oversampling: usize) -> Option<DirectBranch> {
    let nearest = bf.round();
    if (bf - nearest).abs() > DIRECT_PHASE_EPSILON {
        return None;
    }

    let oversampling = oversampling as isize;
    let branch = nearest as isize;
    if branch == oversampling {
        Some(DirectBranch {
            branch_idx: 0,
            sample_shift: 1,
        })
    } else if (0..oversampling).contains(&branch) {
        Some(DirectBranch {
            branch_idx: branch as usize,
            sample_shift: 0,
        })
    } else {
        None
    }
}

#[inline]
fn direct_phase_step(input_rate: usize, output_rate: usize, oversampling: usize) -> Option<isize> {
    let numerator = input_rate as u128 * oversampling as u128;
    let output_rate = output_rate as u128;
    if numerator % output_rate != 0 {
        return None;
    }

    let step = numerator / output_rate;
    if step <= isize::MAX as u128 {
        Some(step as isize)
    } else {
        None
    }
}

#[inline]
fn snap_to_branch_grid(pos: f64, oversampling: usize) -> f64 {
    (pos * oversampling as f64).round() / oversampling as f64
}

#[inline]
fn valid_tap_range(n_p: isize, buf_len: usize, taps: usize) -> (usize, usize) {
    if n_p < 0 {
        return (0, 0);
    }

    let lower_bound = n_p - buf_len as isize + 1;
    let k_start = lower_bound.max(0) as usize;
    let k_end = (n_p as usize + 1).min(taps);
    (k_start.min(k_end), k_end)
}

#[inline]
fn convolve_direct(
    branch: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: isize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    {
        // SAFETY: NEON is mandatory on AArch64 and the tap range was derived
        // from `buf`/`branch` bounds by `valid_tap_range`.
        unsafe { neon::convolve_direct_neon(branch, buf, n_p, k_start, k_end) }
    }

    #[cfg(not(all(target_arch = "aarch64", not(feature = "32bit"))))]
    {
        convolve_direct_scalar(branch, buf, n_p, k_start, k_end)
    }
}

#[inline]
fn convolve_direct_scalar(
    branch: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: isize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= branch.len());
    debug_assert!(n_p >= 0);

    let n_p = n_p as usize;
    let mut k = k_start;
    let mut acc0 = 0.0_f64;
    let mut acc1 = 0.0_f64;
    let mut acc2 = 0.0_f64;
    let mut acc3 = 0.0_f64;

    unsafe {
        while k + 3 < k_end {
            let sample_idx = n_p - k;
            acc0 += *branch.get_unchecked(k) as f64 * *buf.get_unchecked(sample_idx) as f64;
            acc1 += *branch.get_unchecked(k + 1) as f64 * *buf.get_unchecked(sample_idx - 1) as f64;
            acc2 += *branch.get_unchecked(k + 2) as f64 * *buf.get_unchecked(sample_idx - 2) as f64;
            acc3 += *branch.get_unchecked(k + 3) as f64 * *buf.get_unchecked(sample_idx - 3) as f64;
            k += 4;
        }

        let mut acc = acc0 + acc1 + acc2 + acc3;
        while k < k_end {
            let sample_idx = n_p - k;
            acc += *branch.get_unchecked(k) as f64 * *buf.get_unchecked(sample_idx) as f64;
            k += 1;
        }
        acc as PrcFmt
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

        // Append new input samples into the per-channel buffer (after existing fill).
        let new_fill = self.buffer_fill + needed;
        if new_fill > self.max_buffer_len {
            // Reallocate to fit (rare; only if something asks for more than the
            // upper bound we computed at construction).
            for buf in &mut self.buffers {
                buf.resize(new_fill + self.taps, 0.0);
            }
            self.max_buffer_len = new_fill + self.taps;
        }
        for (ch, active) in self.channel_mask.iter().enumerate() {
            if *active {
                let dst =
                    &mut self.buffers[ch][self.buffer_fill..self.buffer_fill + frames_to_read];
                buffer_in.copy_from_channel_to_slice(ch, input_offset, dst);
                // Pad with zeros if partial.
                if frames_to_read < needed {
                    for v in &mut self.buffers[ch]
                        [self.buffer_fill + frames_to_read..self.buffer_fill + needed]
                    {
                        *v = 0.0;
                    }
                }
            }
        }
        self.buffer_fill += needed;

        // Produce output samples.
        let step = 1.0 / self.resample_ratio;
        if let Some(direct_phase_step) = self.direct_phase_step {
            let start_up = (self.next_input_pos * self.oversampling as f64).round() as isize;
            let oversampling = self.oversampling as isize;
            for out_idx in 0..self.chunk_size {
                let up_pos = start_up + out_idx as isize * direct_phase_step;
                let center = up_pos.div_euclid(oversampling);
                let branch_idx = up_pos.rem_euclid(oversampling) as usize;
                for (ch, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        let y = self.eval_direct(&self.buffers[ch], branch_idx, center);
                        buffer_out.write_sample(ch, output_offset + out_idx, &y);
                    }
                }
            }
        } else {
            for out_idx in 0..self.chunk_size {
                let pos = self.next_input_pos + out_idx as f64 * step;
                for (ch, active) in self.channel_mask.iter().enumerate() {
                    if *active {
                        let y = self.eval(ch, pos);
                        buffer_out.write_sample(ch, output_offset + out_idx, &y);
                    }
                }
            }
        }

        // Advance to the position of the first output of the next call.
        self.next_input_pos += self.chunk_size as f64 * step;

        // Slide the ring buffer so we keep ~`taps` samples of history behind
        // `next_input_pos` (enough for the backward-looking convolution) and
        // drop the older samples. This keeps `next_input_pos` bounded.
        let target_history = self.taps as f64;
        let consumed = (self.next_input_pos.floor() - target_history).max(0.0) as usize;
        let consumed = consumed.min(self.buffer_fill);
        if consumed > 0 {
            for buf in &mut self.buffers {
                buf.copy_within(consumed..self.buffer_fill, 0);
                for v in &mut buf[self.buffer_fill - consumed..self.buffer_fill] {
                    *v = 0.0;
                }
            }
            self.buffer_fill -= consumed;
            self.next_input_pos -= consumed as f64;
        }
        if self.direct_phase_step.is_some() {
            self.next_input_pos = snap_to_branch_grid(self.next_input_pos, self.oversampling);
        }

        Ok((needed, self.chunk_size))
    }

    fn input_frames_max(&self) -> usize {
        // Upper bound: the worst case is start-of-stream when we need taps/2
        // primer samples plus a full chunk worth at the slowest ratio.
        (self.chunk_size as f64 / self.resample_ratio).ceil() as usize + self.taps + 4
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
        self.next_input_pos = 0.0;
        self.channel_mask.iter_mut().for_each(|v| *v = true);
    }
}

// ---------------------------------------------------------------------------
// Prototype filter design
// ---------------------------------------------------------------------------

/// Design a prototype lowpass for the given character.
///
/// `proto_len` is the total prototype length (taps * oversampling).
/// `cutoff` is normalized to the upsampled rate (so 0.5 means Nyquist of the
/// upsampled stream). `oversampling` is passed so the per-output gain can be
/// scaled correctly (each branch is a one-of-L decimation of the prototype).
fn design_prototype(
    character: config::PolyphaseCharacter,
    proto_len: usize,
    cutoff: f64,
    oversampling: usize,
) -> Vec<PrcFmt> {
    match character {
        config::PolyphaseCharacter::LinearPhase => {
            design_kaiser_lp(proto_len, cutoff, 140.0, oversampling)
        }
        config::PolyphaseCharacter::Apodizing => {
            // Apodizing trades a small amount of stopband attenuation for a
            // shorter / gentler impulse response. Cutoff is moved in to 0.9 of
            // the design cutoff so the transition starts at ~0.45 fs/2 and
            // reaches >= 100 dB by ~0.55 fs/2.
            //
            // Length is half the linear-phase prototype: long enough to meet
            // the 100 dB stopband at any reasonable oversampling, short enough
            // to reduce pre-ring vs the full linear-phase design.
            let mut apod_len = (proto_len / 2).max(64);
            if apod_len % 2 == 0 {
                apod_len |= 1;
            }
            let apod = design_kaiser_lp(apod_len, cutoff * 0.9, 100.0, oversampling);
            let mut padded = vec![0.0 as PrcFmt; proto_len];
            let offset = (proto_len - apod_len) / 2;
            padded[offset..offset + apod_len].copy_from_slice(&apod);
            padded
        }
        config::PolyphaseCharacter::SlowRollOff => {
            // Short Hann prototype: ~-60 dB stopband for a compact impulse
            // response. Genre choice (minimal ringing), not maximal stopband.
            let mut slow_len = (64 * oversampling).min(proto_len).max(33);
            if slow_len % 2 == 0 {
                slow_len |= 1;
            }
            let slow = design_hann_lp(slow_len, cutoff, oversampling);
            let mut padded = vec![0.0 as PrcFmt; proto_len];
            let offset = (proto_len - slow_len) / 2;
            padded[offset..offset + slow_len].copy_from_slice(&slow);
            padded
        }
        config::PolyphaseCharacter::MinimumPhase => {
            // Start from the same long Kaiser linear-phase design, then perform
            // cepstral spectral factorization to obtain the minimum-phase
            // equivalent (same magnitude response, all energy at the front).
            let lp = design_kaiser_lp(proto_len, cutoff, 140.0, oversampling);
            min_phase_from_linear(&lp)
        }
    }
}

/// Kaiser-window-based lowpass FIR designer.
///
/// * `len` should be odd (we enforce odd for true linear phase).
/// * `cutoff` is normalized to the sample rate (0.5 = Nyquist).
/// * `attenuation_db` drives the Kaiser beta.
/// * The filter is scaled so that DC gain equals `oversampling` (which gives
///   unity gain after the polyphase decimation by L).
fn design_kaiser_lp(
    mut len: usize,
    cutoff: f64,
    attenuation_db: f64,
    oversampling: usize,
) -> Vec<PrcFmt> {
    if len % 2 == 0 {
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
    for n in 0..len {
        let k = n as i64 - half;
        let arg = 2.0 * cutoff * k as f64;
        let sinc = if k == 0 {
            1.0
        } else {
            (PI * arg).sin() / (PI * arg)
        };
        let win_arg = k as f64 / half as f64;
        let win = bessel_i0(beta * (1.0 - win_arg * win_arg).max(0.0).sqrt()) / i0_beta;
        h[n] = ((2.0 * cutoff) * sinc * win) as PrcFmt;
    }
    // Normalize so DC gain == oversampling (each polyphase branch gets gain 1).
    let dc: f64 = h.iter().map(|v| *v as f64).sum();
    let target = oversampling as f64;
    let scale = target / dc;
    for v in &mut h {
        *v = (*v as f64 * scale) as PrcFmt;
    }
    h
}

/// Hann-window-based lowpass FIR designer for the SlowRollOff character.
fn design_hann_lp(mut len: usize, cutoff: f64, oversampling: usize) -> Vec<PrcFmt> {
    if len % 2 == 0 {
        len |= 1;
    }
    let half = (len as i64 - 1) / 2;
    let mut h = vec![0.0 as PrcFmt; len];
    for n in 0..len {
        let k = n as i64 - half;
        let arg = 2.0 * cutoff * k as f64;
        let sinc = if k == 0 {
            1.0
        } else {
            (PI * arg).sin() / (PI * arg)
        };
        let win = 0.5 * (1.0 + (PI * k as f64 / half as f64).cos());
        h[n] = ((2.0 * cutoff) * sinc * win) as PrcFmt;
    }
    let dc: f64 = h.iter().map(|v| *v as f64).sum();
    let target = oversampling as f64;
    let scale = target / dc;
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

/// Compute the minimum-phase equivalent of `lin` (same magnitude response,
/// all energy at the start of the impulse response).
///
/// Real-cepstrum spectral factorization:
///   1. Zero-pad to >= 4x for accurate cepstral estimation.
///   2. log_mag = ln(max(|H|, floor)).
///   3. cepstrum c = ifft(log_mag), real part.
///   4. Window: c_min[0] = c[0]; c_min[1..N/2] = 2*c[1..N/2]; rest = 0.
///   5. min_phase_log_H = fft(c_min); H_min = exp(min_phase_log_H).
///   6. h_min = real(ifft(H_min)), truncated to original length.
fn min_phase_from_linear(lin: &[PrcFmt]) -> Vec<PrcFmt> {
    let orig_len = lin.len();
    let mut n = (orig_len * 4).next_power_of_two();
    if n < 16 {
        n = 16;
    }

    let mut padded: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n];
    for (i, v) in lin.iter().enumerate() {
        padded[i] = Complex::new(*v as f64, 0.0);
    }

    // Forward FFT of the zero-padded linear-phase response.
    let spec = complex_fft(&padded);

    // Log-magnitude with a floor at -200 dB to avoid -Inf at deep stopband nulls.
    let mag_floor = 10f64.powf(-200.0 / 20.0);
    let log_mag: Vec<Complex<f64>> = spec
        .iter()
        .map(|c| Complex::new(c.norm().max(mag_floor).ln(), 0.0))
        .collect();

    // Inverse FFT to get the real cepstrum.
    let cep = complex_ifft(&log_mag);

    // Window the cepstrum so it corresponds to a minimum-phase sequence:
    //   c_min[0]       = c[0]
    //   c_min[1..N/2]  = 2 * c[1..N/2]
    //   c_min[N/2]     = c[N/2]
    //   c_min[N/2+1..] = 0
    let mut cmin = vec![Complex::<f64>::new(0.0, 0.0); n];
    cmin[0] = cep[0];
    for k in 1..n / 2 {
        cmin[k] = Complex::new(2.0 * cep[k].re, 0.0);
    }
    cmin[n / 2] = cep[n / 2];

    // Forward FFT to get min-phase log spectrum, then exp() to get H_min(z) on
    // the unit circle.
    let min_log_spec = complex_fft(&cmin);
    let h_min_spec: Vec<Complex<f64>> = min_log_spec.iter().map(|c| c.exp()).collect();

    // Inverse FFT to get the min-phase impulse response. It is causal and
    // concentrated at the start, so truncating to the original length is clean.
    let h_min_time = complex_ifft(&h_min_spec);

    let mut out = vec![0.0 as PrcFmt; orig_len];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = h_min_time[i].re as PrcFmt;
    }
    out
}

/// One-shot complex FFT used only by the min-phase factorization at init.
/// Performance is not critical; correctness is. The min-phase path needs
/// complex-to-complex transforms (`realfft` only provides real-to-complex).
fn complex_fft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out = input.to_vec();
    fft_radix2(&mut out, false);
    out
}

fn complex_ifft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    let mut out = input.to_vec();
    fft_radix2(&mut out, true);
    let inv_n = 1.0 / n as f64;
    for v in &mut out {
        *v *= inv_n;
    }
    out
}

/// In-place iterative radix-2 FFT. `inverse=true` uses positive exponent.
fn fft_radix2(a: &mut [Complex<f64>], inverse: bool) {
    let n = a.len();
    assert!(n.is_power_of_two(), "fft_radix2 needs power-of-two length");
    // Bit-reversal permutation.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    // Cooley-Tukey butterflies.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = if inverse { 2.0 * PI } else { -2.0 * PI } / len as f64;
        let wlen = Complex::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = a[i + k];
                let v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
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

    fn sample_tol() -> f64 {
        #[cfg(not(feature = "32bit"))]
        {
            1e-10
        }
        #[cfg(feature = "32bit")]
        {
            1e-5
        }
    }

    fn branch_position(input_rate: usize, output_rate: usize, out_idx: usize) -> f64 {
        let step = input_rate as f64 / output_rate as f64;
        let pos = out_idx as f64 * step;
        let frac = pos - pos.floor();
        frac * 320.0
    }

    #[test]
    fn oversampling_320_classifies_common_ratios_as_direct() {
        for &(input_rate, output_rate, phase_step) in &[
            (44_100, 96_000, 147),
            (48_000, 96_000, 160),
            (88_200, 96_000, 294),
        ] {
            assert_eq!(
                direct_phase_step(input_rate, output_rate, 320),
                Some(phase_step)
            );
            for out_idx in 0..2048 {
                let bf = branch_position(input_rate, output_rate, out_idx);
                assert!(
                    direct_branch(bf, 320).is_some(),
                    "{input_rate}->{output_rate} output {out_idx} was not direct: bf={bf}"
                );
            }
        }
        assert_eq!(direct_phase_step(96_000, 44_100, 320), None);
    }

    #[test]
    fn direct_branch_handles_zero_wrap_and_fractional_phase() {
        let eps = DIRECT_PHASE_EPSILON * 0.5;
        assert_eq!(
            direct_branch(eps, 320),
            Some(DirectBranch {
                branch_idx: 0,
                sample_shift: 0
            })
        );
        assert_eq!(
            direct_branch(320.0 - eps, 320),
            Some(DirectBranch {
                branch_idx: 0,
                sample_shift: 1
            })
        );
        assert!(direct_branch(12.25, 320).is_none());
    }

    #[test]
    fn direct_eval_matches_cubic_on_integer_phase() {
        let mut r = PolyphaseFir::new(
            44_100,
            96_000,
            128,
            1,
            config::PolyphaseCharacter::LinearPhase,
            32,
            320,
        )
        .unwrap();
        for (idx, sample) in r.buffers[0].iter_mut().enumerate() {
            let value = ((idx * 17 % 31) as f64 - 15.0) / 31.0;
            *sample = value as PrcFmt;
        }

        let pos: f64 = 80.0 + 147.0 / 320.0;
        let center = pos.floor() as isize;
        let bf = (pos - pos.floor()) * r.oversampling as f64;
        assert!(direct_branch(bf, r.oversampling).is_some());

        let fast = r.eval(0, pos);
        let cubic = r.eval_cubic(&r.buffers[0], center, bf);
        let diff = (fast as f64 - cubic as f64).abs();
        assert!(
            diff <= sample_tol(),
            "direct/cubic mismatch at integer phase: fast={fast}, cubic={cubic}, diff={diff}"
        );
    }

    #[test]
    fn fractional_phase_uses_cubic_path() {
        let mut r = PolyphaseFir::new(
            44_100,
            96_000,
            128,
            1,
            config::PolyphaseCharacter::LinearPhase,
            32,
            320,
        )
        .unwrap();
        for (idx, sample) in r.buffers[0].iter_mut().enumerate() {
            let value = ((idx * 13 % 29) as f64 - 14.0) / 29.0;
            *sample = value as PrcFmt;
        }

        let pos: f64 = 80.12345;
        let center = pos.floor() as isize;
        let bf = (pos - pos.floor()) * r.oversampling as f64;
        assert!(direct_branch(bf, r.oversampling).is_none());

        let eval = r.eval(0, pos);
        let cubic = r.eval_cubic(&r.buffers[0], center, bf);
        let diff = (eval as f64 - cubic as f64).abs();
        assert!(
            diff <= sample_tol(),
            "fractional phase should use cubic path: eval={eval}, cubic={cubic}, diff={diff}"
        );
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
    #[test]
    fn direct_neon_matches_scalar() {
        for &taps in &[1_usize, 2, 3, 4, 7, 8, 31, 64, 65] {
            let branch: Vec<PrcFmt> = (0..taps)
                .map(|idx| (((idx * 17 % 31) as f64 - 15.0) / 97.0) as PrcFmt)
                .collect();
            let buf: Vec<PrcFmt> = (0..96)
                .map(|idx| (((idx * 13 % 29) as f64 - 14.0) / 53.0) as PrcFmt)
                .collect();

            for &n_p in &[0_isize, 1, 5, 31, 63, 95, 101] {
                let (k_start, k_end) = valid_tap_range(n_p, buf.len(), taps);
                if k_start >= k_end {
                    continue;
                }
                let scalar = convolve_direct_scalar(&branch, &buf, n_p, k_start, k_end);
                let neon =
                    unsafe { neon::convolve_direct_neon(&branch, &buf, n_p, k_start, k_end) };
                let diff = (scalar as f64 - neon as f64).abs();
                let tol = 1e-14 * taps as f64;
                assert!(
                    diff <= tol,
                    "NEON/scalar mismatch taps={taps}, n_p={n_p}, range={k_start}..{k_end}: scalar={scalar}, neon={neon}, diff={diff}, tol={tol}"
                );
            }
        }
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
    fn min_phase_is_front_loaded() {
        // Defining property of a minimum-phase filter: its impulse-response
        // peak is near the start, not the center. (The "no pre-ring" audible
        // property follows from this: there is no symmetric ringing extending
        // backward in time from a transient.)
        let lp = design_kaiser_lp(513, 0.2, 120.0, 1);
        let mp = min_phase_from_linear(&lp);
        let mp_peak_idx = mp
            .iter()
            .enumerate()
            .max_by(|a, b| (a.1.abs()).partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        // Compared to linear phase, whose peak is at lp.len() / 2 = 256, the
        // min-phase peak must be in the first 10% of the response.
        assert!(
            mp_peak_idx < lp.len() / 10,
            "min-phase peak at {mp_peak_idx} (expected < {})",
            lp.len() / 10
        );
    }

    #[test]
    fn min_phase_energy_is_in_front_half() {
        // Energy-distribution form of the "no pre-ring" property: virtually
        // all the energy of the min-phase response sits in the first half of
        // the filter, leaving the second half at the noise floor. Linear-phase
        // would split this energy ~50/50 by symmetry.
        let lp = design_kaiser_lp(513, 0.2, 120.0, 1);
        let mp = min_phase_from_linear(&lp);
        let center = lp.len() / 2;
        let first: f64 = mp[..center].iter().map(|v| (*v as f64).powi(2)).sum();
        let second: f64 = mp[center..].iter().map(|v| (*v as f64).powi(2)).sum();
        // First half holds >= 1000x more energy than the second half.
        assert!(
            first > 1000.0 * second,
            "min-phase not front-loaded: first-half energy {first}, second-half {second}"
        );
    }

    #[test]
    fn min_phase_preserves_magnitude_response() {
        // |H_min(w)| should equal |H_lin(w)| at every probed frequency.
        let lp = design_kaiser_lp(513, 0.25, 100.0, 1);
        let mp = min_phase_from_linear(&lp);
        for &f in &[0.05, 0.1, 0.2, 0.3, 0.4] {
            let lin_db = freq_response_db(&lp, f);
            let min_db = freq_response_db(&mp, f);
            assert!(
                (lin_db - min_db).abs() < 1.0,
                "magnitude mismatch at f={f}: lin={lin_db} dB, min={min_db} dB"
            );
        }
    }

    #[test]
    fn polyphase_dc_passes_through_with_unity_gain() {
        // Constant input -> constant output of (approximately) the same value.
        let mut r = PolyphaseFir::new(
            44100,
            96000,
            512,
            1,
            config::PolyphaseCharacter::LinearPhase,
            64,
            64,
        )
        .unwrap();
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
        let mut r = PolyphaseFir::new(
            44100,
            48000,
            128,
            2,
            config::PolyphaseCharacter::LinearPhase,
            32,
            32,
        )
        .unwrap();
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
        // The plan supports rate changes by reconstructing the engine. Verify
        // construction at two different rates both work and produce valid output.
        for (in_rate, out_rate) in [(44100, 96000), (48000, 96000), (96000, 44100)] {
            let mut r = PolyphaseFir::new(
                in_rate,
                out_rate,
                256,
                2,
                config::PolyphaseCharacter::LinearPhase,
                64,
                64,
            )
            .unwrap();
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

    #[test]
    fn apodizing_meets_transition_mask() {
        // Apodizing prototype designed at cutoff 0.45 of nominal Nyquist.
        // Probe the response: should be > -6 dB up to 0.45 and >= 100 dB
        // stopband beyond 0.55.
        let proto = design_prototype(
            config::PolyphaseCharacter::Apodizing,
            1024 * 8,
            0.45 / 2.0, // cutoff at "0.45 of half-Nyquist" relative to upsampled rate
            1,
        );
        // Probe at 0.4 * 0.5 (well in passband) and 0.55 * 0.5 (stopband).
        let pass = freq_response_db(&proto, 0.4 * 0.5);
        let stop = freq_response_db(&proto, 0.55 * 0.5);
        assert!(pass > -6.0, "passband at 0.4 fs/2 only {pass} dB");
        assert!(stop < -100.0, "stopband at 0.55 fs/2 only {stop} dB");
    }

    #[test]
    fn slow_rolloff_is_short() {
        // Most of the prototype should be zero (short kernel padded out).
        let proto = design_prototype(config::PolyphaseCharacter::SlowRollOff, 4096, 0.4 / 2.0, 1);
        let nonzero = proto.iter().filter(|v| v.abs() > 1e-9 as PrcFmt).count();
        // Should be roughly 64 taps wide, not the full 4096.
        assert!(
            nonzero < 200,
            "SlowRollOff prototype has {nonzero} non-zero taps (expected short)"
        );
    }

    /// Drive the resampler with a sine at frequency `freq_hz` until the priming
    /// delay is flushed and steady state is reached, then return the peak
    /// absolute amplitude of the output in the last collected chunks.
    ///
    /// Must feed *exactly* `input_frames_next` per call: feeding more would
    /// leave unused samples in the source array and cause a phase jump in the
    /// engine's view of the input sine on the next call.
    fn measure_peak_amplitude(
        in_rate: usize,
        out_rate: usize,
        freq_hz: f64,
        character: config::PolyphaseCharacter,
        taps: usize,
        oversampling: usize,
    ) -> f64 {
        let chunk_out = 1024;
        let mut r = PolyphaseFir::new(
            in_rate,
            out_rate,
            chunk_out,
            1,
            character,
            taps,
            oversampling,
        )
        .unwrap();
        let max_in = r.input_frames_max();
        let warmup_chunks = ((r.output_delay() / chunk_out) + 4).max(8);
        let mut peak = 0.0_f64;
        let mut out = vec![vec![0.0 as PrcFmt; chunk_out]; 1];
        let mut sample_counter: usize = 0;
        for chunk_idx in 0..warmup_chunks + 4 {
            let needed = r.input_frames_next();
            let mut waves = vec![vec![0.0 as PrcFmt; max_in]; 1];
            for slot in waves[0].iter_mut().take(needed) {
                let t = sample_counter as f64 / in_rate as f64;
                *slot = (2.0 * PI * freq_hz * t).sin() as PrcFmt;
                sample_counter += 1;
            }
            let input = SequentialSliceOfVecs::new(&waves, 1, max_in).unwrap();
            let mut output = SequentialSliceOfVecs::new_mut(&mut out, 1, chunk_out).unwrap();
            r.process_into_buffer(&input, &mut output, None).unwrap();
            if chunk_idx >= warmup_chunks {
                for &v in out[0].iter() {
                    let a = (v as f64).abs();
                    if a > peak {
                        peak = a;
                    }
                }
            }
        }
        peak
    }

    #[test]
    fn engine_passband_is_flat_for_linear_phase_downsample() {
        // 96k -> 44.1k with the recommended LinearPhase character. Probe the
        // passband: amplitudes at 1k / 10k / 18k Hz should all be within
        // +-0.5 dB of unity (the input sine has unit amplitude).
        for &freq in &[1000.0_f64, 10_000.0, 18_000.0] {
            let peak = measure_peak_amplitude(
                96000,
                44100,
                freq,
                config::PolyphaseCharacter::LinearPhase,
                64,
                64,
            );
            let db = 20.0 * peak.log10();
            assert!(
                (-0.5..=0.5).contains(&db),
                "passband ripple at {freq} Hz: {db} dB (peak {peak})"
            );
        }
    }

    #[test]
    fn engine_stopband_rejects_aliasing_at_realistic_params() {
        // 96k -> 44.1k. Output Nyquist is 22050 Hz; a 30 kHz input would alias
        // to 44100 - 30000 = 14100 Hz unless the antialias filter suppresses
        // it. With the LinearPhase character and 64x64 taps the suppression
        // should be > 90 dB - well below audibility for typical playback.
        let peak = measure_peak_amplitude(
            96000,
            44100,
            30_000.0,
            config::PolyphaseCharacter::LinearPhase,
            64,
            64,
        );
        let db = 20.0 * peak.max(1e-30).log10();
        assert!(
            db <= -90.0,
            "alias rejection at 30 kHz only {db} dB (peak {peak})"
        );
    }
}
