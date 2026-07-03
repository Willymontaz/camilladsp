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

// Declipper: source-domain repair for clipped (flat-topped) masters.
//
// Two stages: detect flat-topped clip runs, then reconstruct the missing
// waveform peak from the surrounding unclipped context so the restored peak
// overshoots the clip level. Reconstruction needs unclipped samples on *both*
// sides of a clip, so the processor runs a per-channel delay line: output is
// delayed by `lookahead` samples, and clips that straddle chunk boundaries are
// repaired once their trailing context has arrived.

use std::sync::Arc;

use crate::PrcFmt;
use crate::ProcessingParameters;
use crate::Res;
use crate::audiochunk::AudioChunk;
use crate::config;
use crate::config::DeclipMethod;
use crate::processors::Processor;
use crate::utils::decibels::db_to_linear;

/// Maximum difference between adjacent samples for them to count as part of the
/// same flat clip plateau. Hard digital clipping produces bit-identical samples
/// (difference 0), so a small epsilon detects real clips while staying far below
/// the curvature of a reconstructed (smooth) peak, which prevents an already
/// repaired region from being re-detected as a clip on the next chunk.
const FLAT_EPS: PrcFmt = 1e-5;

pub struct Declipper {
    pub name: String,
    pub channels: usize,
    pub process_channels: Vec<usize>,
    is_process: Vec<bool>,
    clip_threshold: PrcFmt,
    min_clip_len: usize,
    max_clip_len: usize,
    method: DeclipMethod,
    ar_order: usize,
    makeup: PrcFmt,
    /// Left/right context length kept for reconstruction (also the delay-line
    /// lookback). Fixed at construction so the delay stays continuous across
    /// parameter updates.
    ctx: usize,
    /// Output delay in samples (lookahead depth): `max_clip_len + ctx`.
    delay: usize,
    /// Largest clip gap the autoregressive scratch buffers can hold; longer gaps
    /// fall back to cubic reconstruction.
    ar_gap_cap: usize,
    #[allow(dead_code)]
    samplerate: usize,
    /// Per-channel delay/analysis buffers. Each holds `ctx + delay` samples at
    /// rest: the first `ctx` are already-emitted left context, the last `delay`
    /// are received-but-not-yet-emitted samples.
    buffers: Vec<Vec<PrcFmt>>,
    // Reusable scratch for the autoregressive method (no allocation in process).
    ar_r: Vec<PrcFmt>,
    ar_a: Vec<PrcFmt>,
    ar_tmp: Vec<PrcFmt>,
    /// Deterministic autocorrelation of the LPC coefficients, R(d) = sum_k a[k]*a[k+d].
    ar_rc: Vec<PrcFmt>,
    /// Lower-triangular band storage for the banded Cholesky solve of the
    /// Janssen normal equations. Sized `ar_gap_cap * (ctx + 1)`.
    ar_band: Vec<PrcFmt>,
    /// Right-hand side and solution scratch for the gap linear system.
    ar_rhs: Vec<PrcFmt>,
    ar_x: Vec<PrcFmt>,
    processing_params: Arc<ProcessingParameters>,
    long_clip_warned: bool,
}

fn build_is_process(channels: usize, process_channels: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; channels];
    if process_channels.is_empty() {
        // Default: process every channel.
        mask.iter_mut().for_each(|m| *m = true);
    } else {
        for ch in process_channels {
            if *ch < channels {
                mask[*ch] = true;
            }
        }
    }
    mask
}

impl Declipper {
    /// Creates a Declipper from a config struct
    pub fn from_config(
        name: &str,
        config: config::DeclipperParameters,
        samplerate: usize,
        chunksize: usize,
        processing_params: Arc<ProcessingParameters>,
    ) -> Self {
        let name = name.to_string();
        let channels = config.channels;
        let process_channels = config.process_channels();
        let is_process = build_is_process(channels, &process_channels);
        let clip_threshold = config.clip_threshold();
        let min_clip_len = config.min_clip_len().max(1);
        let max_clip_len = config.max_clip_len().max(min_clip_len);
        let method = config.method();
        let ar_order = config.ar_order().max(1);
        let makeup = db_to_linear(config.makeup_gain_db());

        // Context must comfortably exceed the AR order so the autocorrelation and
        // prediction recursion are well conditioned.
        let ctx = (4 * ar_order).max(128);
        let delay = max_clip_len + ctx;
        let ar_gap_cap = max_clip_len;

        let buf_len = ctx + delay;
        let capacity = buf_len + chunksize;
        let mut buffers = Vec::with_capacity(channels);
        for _ in 0..channels {
            let mut b = Vec::with_capacity(capacity);
            b.resize(buf_len, 0.0);
            buffers.push(b);
        }

        debug!(
            "Creating declipper '{}', channels: {}, process_channels: {:?}, clip_threshold: {}, min_clip_len: {}, max_clip_len: {}, method: {:?}, ar_order: {}, makeup_gain_db: {}, delay: {} samples",
            name,
            channels,
            process_channels,
            clip_threshold,
            min_clip_len,
            max_clip_len,
            method,
            ar_order,
            config.makeup_gain_db(),
            delay,
        );

        Declipper {
            name,
            channels,
            process_channels,
            is_process,
            clip_threshold,
            min_clip_len,
            max_clip_len,
            method,
            ar_order,
            makeup,
            ctx,
            delay,
            ar_gap_cap,
            samplerate,
            buffers,
            ar_r: vec![0.0; ctx + 1],
            ar_a: vec![0.0; ctx + 1],
            ar_tmp: vec![0.0; ctx + 1],
            ar_rc: vec![0.0; ctx + 1],
            // Half-bandwidth of the normal-equation matrix equals the model order,
            // which never exceeds `ctx` (it is capped by `r.len() - 1`). Sizing the
            // band on `ctx` keeps the scratch valid even if `ar_order` is raised by
            // a later parameter update.
            ar_band: vec![0.0; ar_gap_cap * (ctx + 1)],
            ar_rhs: vec![0.0; ar_gap_cap],
            ar_x: vec![0.0; ar_gap_cap],
            processing_params,
            long_clip_warned: false,
        }
    }

    /// The output latency introduced by the declipper, in samples.
    pub fn latency(&self) -> usize {
        self.delay
    }
}

/// Compute the autocorrelation (lags 0..=order) of `seg` into `r`, with a small
/// white-noise floor added for numerical stability.
fn autocorrelation(seg: &[PrcFmt], order: usize, r: &mut [PrcFmt]) {
    for (k, rk) in r.iter_mut().enumerate().take(order + 1) {
        let mut sum = 0.0;
        for n in 0..seg.len() - k {
            sum += seg[n] * seg[n + k];
        }
        *rk = sum;
    }
    // Regularize: ensures r[0] > 0 and improves conditioning.
    r[0] = r[0] * (1.0 + 1e-6) + 1e-12;
}

/// Levinson-Durbin recursion. On success `a[0..=order]` holds LPC coefficients
/// with the convention `x_hat[n] = -sum_{k=1..=order} a[k] * x[n-k]` (a[0] = 1).
/// Returns false if the signal is (near) silent and no model can be fit.
fn levinson(r: &[PrcFmt], order: usize, a: &mut [PrcFmt], tmp: &mut [PrcFmt]) -> bool {
    let mut err = r[0];
    if err <= 0.0 || !err.is_finite() {
        return false;
    }
    a[0] = 1.0;
    for ai in a.iter_mut().take(order + 1).skip(1) {
        *ai = 0.0;
    }
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / err;
        if !k.is_finite() {
            return false;
        }
        tmp[1..i].copy_from_slice(&a[1..i]);
        for j in 1..i {
            a[j] = tmp[j] + k * tmp[i - j];
        }
        a[i] = k;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            // Numerically singular; the coefficients computed so far are a valid
            // (lower-order) model, leave the rest at zero.
            break;
        }
    }
    true
}

/// Cubic Hermite reconstruction across the gap `[s, e)`, using the two good
/// samples on each side. The boundary tangents make the interpolant overshoot
/// the clip level, restoring the peak. Requires `s >= 2` and `e + 1 < buf.len()`.
fn reconstruct_cubic(buf: &mut [PrcFmt], s: usize, e: usize) {
    let g = (e - s) as PrcFmt;
    let p0 = buf[s - 1];
    let p1 = buf[e];
    // Tangents scaled to the [0, 1] Hermite interval, which spans (g + 1) samples.
    let m0 = (buf[s - 1] - buf[s - 2]) * (g + 1.0);
    let m1 = (buf[e + 1] - buf[e]) * (g + 1.0);
    for (offset, j) in (s..e).enumerate() {
        let t = (offset as PrcFmt + 1.0) / (g + 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        buf[j] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }
}

/// Deterministic autocorrelation of the LPC (whitening-filter) coefficients:
/// `rc[d] = sum_k a[k] * a[k + d]` for `d = 0..=order`, with `a[0] = 1`. These are
/// the Toeplitz coefficients of the Janssen normal-equation matrix.
fn coeff_autocorr(a: &[PrcFmt], order: usize, rc: &mut [PrcFmt]) {
    for d in 0..=order {
        let mut s = 0.0;
        for k in 0..=order - d {
            s += a[k] * a[k + d];
        }
        rc[d] = s;
    }
}

/// Solve `A x = rhs` where `A` is the symmetric positive-definite banded Toeplitz
/// matrix with `A[i][j] = rc[|i - j|]` for `|i - j| <= bw` (zero outside the band),
/// of size `g`. Uses an in-place banded Cholesky (`A = L Lᵀ`). `band` must have at
/// least `g * (bw + 1)` entries; the factor `L` is written into it as
/// `band[i*(bw+1) + d] = L[i][i-d]`. Returns false if `A` is not positive definite.
fn solve_banded_toeplitz(
    rc: &[PrcFmt],
    bw: usize,
    g: usize,
    rhs: &[PrcFmt],
    band: &mut [PrcFmt],
    x: &mut [PrcFmt],
) -> bool {
    let stride = bw + 1;
    // Materialise the lower band of A from its Toeplitz coefficients.
    for i in 0..g {
        let dmax = bw.min(i);
        for d in 0..=dmax {
            band[i * stride + d] = rc[d];
        }
    }
    // Banded Cholesky. For each row i, fill columns j = i-d with d descending so
    // that within-row entries L[i][k] (k > j, i.e. larger band index) are ready,
    // and cross-row entries L[j][*] come from already-factored rows j < i.
    for i in 0..g {
        let dmax = bw.min(i);
        let k_start = i.saturating_sub(bw);
        for d in (0..=dmax).rev() {
            let j = i - d;
            let mut sum = band[i * stride + d];
            for k in k_start..j {
                let dik = i - k;
                let djk = j - k;
                if dik <= bw && djk <= bw {
                    sum -= band[i * stride + dik] * band[j * stride + djk];
                }
            }
            if d == 0 {
                if sum <= 0.0 || !sum.is_finite() {
                    return false;
                }
                band[i * stride] = sum.sqrt();
            } else {
                band[i * stride + d] = sum / band[j * stride];
            }
        }
    }
    // Forward solve L y = rhs (in place into x).
    for i in 0..g {
        let dmax = bw.min(i);
        let mut sum = rhs[i];
        for d in 1..=dmax {
            sum -= band[i * stride + d] * x[i - d];
        }
        x[i] = sum / band[i * stride];
    }
    // Back solve Lᵀ x = y.
    for i in (0..g).rev() {
        let mut sum = x[i];
        let lmax = (i + bw).min(g - 1);
        for l in (i + 1)..=lmax {
            sum -= band[l * stride + (l - i)] * x[l];
        }
        x[i] = sum / band[i * stride];
    }
    true
}

/// Fill the Janssen normal-equation right-hand side for the gap `[gs, ge)` under the
/// current coefficient autocorrelation `rc`, solve for those samples, and write the
/// result into `buf`. Every sample outside `[gs, ge)` — left/right model context *and*
/// any samples pinned at the clip boundary by the constrained refinement — is treated
/// as a known neighbour entering through the RHS; the missing–missing coupling inside
/// the gap is carried by the banded Toeplitz matrix. Returns false on an
/// ill-conditioned factorisation or a non-finite solution.
#[allow(clippy::too_many_arguments)]
fn solve_gap(
    buf: &mut [PrcFmt],
    gs: usize,
    ge: usize,
    order: usize,
    rc: &[PrcFmt],
    band: &mut [PrcFmt],
    rhs: &mut [PrcFmt],
    x: &mut [PrcFmt],
) -> bool {
    let g = ge - gs;
    // RHS: contributions of the known samples straddling the gap edges. For missing
    // index gs+i, a neighbour gs+i+o (|o| <= order) lying outside [gs, ge) is known and
    // contributes -rc[|o|] * buf[idx].
    for (i, rhs_i) in rhs.iter_mut().take(g).enumerate() {
        let mut acc = 0.0;
        let base = gs + i;
        for o in 1..=order {
            // Right neighbour is known once it reaches the samples at/after ge.
            if base + o >= ge {
                acc += rc[o] * buf[base + o];
            }
            // Left neighbour is known once it steps before gs; the in-gap neighbours
            // (o <= i) are handled by the matrix, not the RHS.
            if o > i {
                acc += rc[o] * buf[base - o];
            }
        }
        *rhs_i = -acc;
    }
    if !solve_banded_toeplitz(rc, order, g, rhs, band, x) {
        return false;
    }
    for (idx, j) in (gs..ge).enumerate() {
        if !x[idx].is_finite() {
            return false;
        }
        buf[j] = x[idx];
    }
    true
}

/// Autoregressive gap reconstruction via the **Janssen iterative least-squares AR
/// interpolation** (Janssen, Veldhuis & Vries, 1986 — the standard method of
/// Godsill & Rayner's *Digital Audio Restoration*), extended with the
/// **clipping-consistency constraint** that turns AR *interpolation* into AR
/// *declipping*. Rather than extrapolating each side independently and crossfading —
/// which lets a stable AR model decay toward the mean and under-restore the peak
/// mid-gap — this jointly estimates the missing samples `[s, e)` as the values that
/// minimise the total AR residual energy under a single model spanning both sides. The
/// estimate and the model are refined by alternating: (re)fit the AR coefficients on
/// the currently-filled signal, then re-solve the normal equations for the gap.
///
/// A clipped sample carries more information than a merely *missing* one: its true
/// magnitude was provably `>= clip level` (that is why it flat-topped). Unconstrained
/// LSAR ignores this and connects smoothly to the sub-threshold shoulders, so it dips
/// below the clip level near the gap edges — a reconstruction inconsistent with the
/// observation. After the interpolation converges we therefore run an active-set pass
/// that pins the violating edge samples to the clip boundary and re-solves the
/// interior with them held as known context, restoring both AR-optimality inside and
/// physical consistency (`|x| >= clip level`) throughout. Returns false (caller falls
/// back to cubic) when there is not enough context, the gap exceeds scratch capacity,
/// or the model/solve is ill-conditioned.
#[allow(clippy::too_many_arguments)]
fn reconstruct_ar(
    buf: &mut [PrcFmt],
    s: usize,
    e: usize,
    max_ctx: usize,
    threshold: PrcFmt,
    req_order: usize,
    r: &mut [PrcFmt],
    a: &mut [PrcFmt],
    tmp: &mut [PrcFmt],
    rc: &mut [PrcFmt],
    band: &mut [PrcFmt],
    rhs: &mut [PrcFmt],
    x: &mut [PrcFmt],
) -> bool {
    /// Alternations of (fit model, solve gap). Janssen converges in a handful of
    /// iterations; 5 is comfortably past the knee for audio-length gaps.
    const ITERATIONS: usize = 5;

    let n = buf.len();
    let g = e - s;
    if g == 0 || g > rhs.len() || g > x.len() {
        return false;
    }
    // Left context is the already-reconstructed / clean region before the gap.
    let left_len = s.min(max_ctx);
    // Right context: clean downstream samples used to fit the model. It must stop at
    // the *next clip plateau* (an unrepaired flat top would bias the model), but must
    // include the unclipped shoulder that immediately follows this gap — those samples
    // often sit above `threshold` in magnitude yet are perfectly good signal, so a
    // plain magnitude test would wrongly truncate the context to nothing.
    let mut right_len = 0;
    while right_len < max_ctx && e + right_len < n {
        let p = e + right_len;
        let flat_top = buf[p].abs() >= threshold && (buf[p] - buf[p - 1]).abs() <= FLAT_EPS;
        if flat_top {
            break;
        }
        right_len += 1;
    }
    let order = req_order
        .min(left_len.saturating_sub(1))
        .min(right_len.saturating_sub(1))
        .min(r.len() - 1);
    // Need enough context on each side to both fit the model and supply the `order`
    // known neighbours the normal equations reference across each gap edge.
    if order < 1 || left_len < order || right_len < order {
        return false;
    }
    if g * (order + 1) > band.len() {
        return false;
    }

    // The physical clip level is the flat-top plateau value, captured before the cubic
    // seed overwrites it. The true signal magnitude at every clipped sample was at
    // least this large, with this sign — the constraint enforced after interpolation.
    let clip_val = buf[s];
    let positive = clip_val > 0.0;
    let clip_mag = clip_val.abs();

    // Seed the gap with the cubic estimate so the first coefficient fit sees a
    // plausible waveform rather than a discontinuity.
    reconstruct_cubic(buf, s, e);

    let win_start = s - left_len;
    let win_end = e + right_len;

    for _ in 0..ITERATIONS {
        // 1. Fit an AR model to the whole currently-filled window (both contexts
        //    plus the current gap estimate) — a single, consistent model.
        autocorrelation(&buf[win_start..win_end], order, r);
        if !levinson(r, order, a, tmp) {
            return false;
        }
        // 2. Toeplitz coefficients of the normal-equation matrix.
        coeff_autocorr(a, order, rc);
        // 3. Solve for the gap that minimises the residual energy under the fitted
        //    model, writing the estimate back into the buffer.
        if !solve_gap(buf, s, e, order, rc, band, rhs, x) {
            return false;
        }
    }

    // Clipping-consistency (active-set) refinement. The unconstrained fit connects to
    // the sub-threshold shoulders, so it typically dips below the clip level at the
    // gap edges — impossible for samples that were clipped. Pin each violating sample
    // to the clip boundary and re-solve the still-free interior with the pinned
    // samples as known context; repeat as newly exposed edges fall below the boundary.
    // The active set only grows and is bounded by the gap length, so this terminates;
    // `rc`/`a` from the final interpolation iteration define the (fixed) quadratic.
    // Clamping never lowers a magnitude, so the restored peak can only rise — the
    // overshoot the reconstruction is there to recover is preserved.
    let boundary = if positive { clip_mag } else { -clip_mag };
    let violates = |v: PrcFmt| if positive { v < clip_mag } else { v > -clip_mag };
    let mut gs = s;
    let mut ge = e;
    loop {
        // Grow the pinned prefix/suffix over the samples that fell below the boundary.
        let mut moved = false;
        while gs < ge && violates(buf[gs]) {
            buf[gs] = boundary;
            gs += 1;
            moved = true;
        }
        while ge > gs && violates(buf[ge - 1]) {
            buf[ge - 1] = boundary;
            ge -= 1;
            moved = true;
        }
        if gs >= ge {
            // Every remaining sample is pinned to the boundary; nothing left to solve.
            break;
        }
        // A violation strictly inside the free interior means the active set is not a
        // pair of contiguous edge runs (rare, for non-monotone gaps). Project those
        // samples directly rather than attempting a non-banded reduced solve.
        if buf[gs..ge].iter().any(|&v| violates(v)) {
            for v in buf[gs..ge].iter_mut() {
                if violates(*v) {
                    *v = boundary;
                }
            }
            break;
        }
        if !moved {
            // Interior is feasible and no edge moved this round: constrained optimum.
            break;
        }
        // Re-solve the free interior with the pinned samples acting as known context.
        if !solve_gap(buf, gs, ge, order, rc, band, rhs, x) {
            // Fall back to a plain projection of any residual violations.
            for v in buf[gs..ge].iter_mut() {
                if violates(*v) {
                    *v = boundary;
                }
            }
            break;
        }
    }
    true
}

impl Processor for Declipper {
    fn name(&self) -> &str {
        &self.name
    }

    /// Apply the declipper to an AudioChunk, modifying it in-place. Every channel
    /// is delayed by `self.delay` samples to keep channels aligned; the selected
    /// process channels additionally have their clip runs repaired.
    fn process_chunk(&mut self, input: &mut AudioChunk) -> Res<()> {
        let lb = self.ctx;
        let threshold = self.clip_threshold;
        let min_len = self.min_clip_len;
        let max_len = self.max_clip_len;
        let method = self.method;
        let req_order = self.ar_order;
        let ar_gap_cap = self.ar_gap_cap;
        let ctx = self.ctx;
        let makeup = self.makeup;
        let n_ch = self.channels.min(input.waveforms.len());

        let mut declipped = 0usize;
        let mut long_clip_seen: Option<usize> = None;

        for ch in 0..n_ch {
            let wave = &mut input.waveforms[ch];
            let frames = wave.len();
            let buf = &mut self.buffers[ch];
            // Append the incoming samples to the delay/analysis buffer.
            buf.extend_from_slice(wave);
            let n = buf.len();

            if self.is_process[ch] {
                // Scan for clip runs whose plateau *starts* within the emit
                // region [lb, lb + frames). Each clip is therefore repaired
                // exactly once, in the chunk where its start reaches the emit
                // region, at which point its trailing context is guaranteed
                // present in the buffer.
                let scan_end = (lb + frames).min(n);
                let mut i = lb;
                while i < scan_end {
                    if buf[i].abs() < threshold {
                        i += 1;
                        continue;
                    }
                    let positive = buf[i] > 0.0;
                    // Skip continuations of a plateau that began earlier (already
                    // handled or a passed-through long clip).
                    if i > 0
                        && buf[i - 1].abs() >= threshold
                        && (buf[i - 1] > 0.0) == positive
                        && (buf[i] - buf[i - 1]).abs() <= FLAT_EPS
                    {
                        i += 1;
                        continue;
                    }
                    // Extend the flat plateau to the right.
                    let s = i;
                    let mut e = s + 1;
                    while e < n
                        && buf[e].abs() >= threshold
                        && (buf[e] > 0.0) == positive
                        && (buf[e] - buf[e - 1]).abs() <= FLAT_EPS
                    {
                        e += 1;
                    }
                    let len = e - s;
                    // e == n: plateau not terminated within the buffer yet. This
                    // cannot happen for an emit-region start given the lookahead,
                    // but guard defensively and pass through.
                    if e >= n || len < min_len {
                        i = e.max(s + 1);
                        continue;
                    }
                    if len > max_len {
                        long_clip_seen = Some(len);
                        i = e;
                        continue;
                    }
                    // Need at least two good samples of right context for cubic.
                    if n - e < 2 {
                        i = e;
                        continue;
                    }
                    let repaired = if method == DeclipMethod::Ar && len <= ar_gap_cap {
                        reconstruct_ar(
                            buf,
                            s,
                            e,
                            ctx,
                            threshold,
                            req_order,
                            &mut self.ar_r,
                            &mut self.ar_a,
                            &mut self.ar_tmp,
                            &mut self.ar_rc,
                            &mut self.ar_band,
                            &mut self.ar_rhs,
                            &mut self.ar_x,
                        )
                    } else {
                        false
                    };
                    if !repaired {
                        reconstruct_cubic(buf, s, e);
                    }
                    declipped += len;
                    i = e;
                }
            }

            // Emit the oldest `frames` pending samples (delayed by `self.delay`),
            // applying makeup gain, then drop them from the buffer.
            for (dst, src) in wave.iter_mut().zip(buf[lb..lb + frames].iter()) {
                *dst = *src * makeup;
            }
            buf.drain(0..frames);
        }

        if declipped > 0 {
            self.processing_params.add_declipped_samples(declipped);
        }
        if let Some(len) = long_clip_seen
            && !self.long_clip_warned
        {
            warn!(
                "Declipper '{}': clip run of {} samples exceeds max_clip_len ({}), passing through unrepaired (further such events suppressed)",
                self.name, len, self.max_clip_len
            );
            self.long_clip_warned = true;
        }
        Ok(())
    }

    fn update_parameters(&mut self, config: config::Processor) {
        if let config::Processor::Declipper {
            parameters: config, ..
        } = config
        {
            let channels = config.channels;
            let process_channels = config.process_channels();
            self.is_process = build_is_process(channels, &process_channels);
            // If the channel count changed, resize the delay buffers. New
            // buffers start primed with silence; this only happens on a config
            // change that also passed validation against the pipeline width.
            if channels != self.channels {
                let buf_len = self.ctx + self.delay;
                self.buffers.resize_with(channels, || vec![0.0; buf_len]);
                self.channels = channels;
            }
            self.process_channels = process_channels;
            self.clip_threshold = config.clip_threshold();
            self.min_clip_len = config.min_clip_len().max(1);
            self.max_clip_len = config.max_clip_len().max(self.min_clip_len);
            self.method = config.method();
            self.ar_order = config.ar_order().max(1);
            self.makeup = db_to_linear(config.makeup_gain_db());
            self.long_clip_warned = false;

            debug!(
                "Updated declipper '{}', process_channels: {:?}, clip_threshold: {}, min_clip_len: {}, max_clip_len: {}, method: {:?}, ar_order: {}, makeup_gain_db: {}",
                self.name,
                self.process_channels,
                self.clip_threshold,
                self.min_clip_len,
                self.max_clip_len,
                self.method,
                self.ar_order,
                config.makeup_gain_db(),
            );
        } else {
            // This should never happen unless there is a bug somewhere else
            panic!("Invalid config change!");
        }
    }
}

/// Validate the declipper config, to give a helpful message instead of a panic.
pub fn validate_declipper(config: &config::DeclipperParameters) -> Res<()> {
    let channels = config.channels;
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
    let threshold = config.clip_threshold();
    if threshold <= 0.0 || threshold > 1.0 {
        let msg = "clip_threshold must be in the range (0, 1].";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.min_clip_len() < 1 {
        let msg = "min_clip_len must be at least 1.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.max_clip_len() < config.min_clip_len() {
        let msg = "max_clip_len must be greater than or equal to min_clip_len.";
        return Err(config::ConfigError::new(msg).into());
    }
    if config.ar_order() < 1 {
        let msg = "ar_order must be at least 1.";
        return Err(config::ConfigError::new(msg).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DeclipperParameters;

    const SR: usize = 48000;
    const CHUNK: usize = 256;

    fn params(
        method: DeclipMethod,
        clip_threshold: PrcFmt,
        min_clip_len: usize,
        max_clip_len: usize,
    ) -> DeclipperParameters {
        DeclipperParameters {
            channels: 1,
            process_channels: None,
            clip_threshold: Some(clip_threshold),
            min_clip_len: Some(min_clip_len),
            max_clip_len: Some(max_clip_len),
            method: Some(method),
            ar_order: Some(24),
            makeup_gain_db: None,
        }
    }

    fn make(params: DeclipperParameters) -> (Declipper, Arc<ProcessingParameters>) {
        let pp = Arc::new(ProcessingParameters::default());
        let dec = Declipper::from_config("test", params, SR, CHUNK, pp.clone());
        (dec, pp)
    }

    /// Run a single-channel signal through the declipper in chunks, returning the
    /// concatenated output (same length as the input).
    fn run(dec: &mut Declipper, signal: &[PrcFmt]) -> Vec<PrcFmt> {
        let mut out = Vec::with_capacity(signal.len());
        for block in signal.chunks(CHUNK) {
            let wave = block.to_vec();
            let frames = wave.len();
            let mut chunk = AudioChunk::new(vec![wave], 0.0, 0.0, frames, frames);
            dec.process_chunk(&mut chunk).unwrap();
            out.extend_from_slice(&chunk.waveforms[0]);
        }
        out
    }

    fn sine(len: usize, freq: PrcFmt, amp: PrcFmt) -> Vec<PrcFmt> {
        (0..len)
            .map(|n| {
                amp * (2.0 * std::f64::consts::PI as PrcFmt * freq * n as PrcFmt / SR as PrcFmt)
                    .sin()
            })
            .collect()
    }

    fn hard_clip(signal: &[PrcFmt], level: PrcFmt) -> Vec<PrcFmt> {
        signal.iter().map(|&x| x.clamp(-level, level)).collect()
    }

    fn error_energy(a: &[PrcFmt], b: &[PrcFmt], range: std::ops::Range<usize>) -> PrcFmt {
        range.clone().map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
    }

    #[test]
    fn delay_line_is_identity_without_clips() {
        // A signal that never reaches the threshold must be passed through
        // unchanged, delayed by exactly `delay` samples.
        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.985, 2, 64));
        let delay = dec.delay;
        let signal = sine(4096, 1000.0, 0.5);
        let out = run(&mut dec, &signal);
        for k in 0..(signal.len() - delay) {
            assert!(
                (out[delay + k] - signal[k]).abs() < 1e-9,
                "mismatch at {}: {} vs {}",
                k,
                out[delay + k],
                signal[k]
            );
        }
        // Leading samples are the primed silence.
        for v in out.iter().take(delay) {
            assert_eq!(*v, 0.0);
        }
        assert_eq!(pp.declipped_samples(), 0);
    }

    #[test]
    fn cubic_reduces_error_and_overshoots() {
        let orig = sine(8192, 1000.0, 0.95);
        let clip_level = 0.7;
        let clipped = hard_clip(&orig, clip_level);
        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.6, 2, 64));
        let delay = dec.delay;
        let out = run(&mut dec, &clipped);

        // Compare over the fully-emitted region, skipping warmup.
        let range = 512..(orig.len() - delay);
        let declipped_aligned: Vec<PrcFmt> = range.clone().map(|k| out[delay + k]).collect();
        let orig_slice: Vec<PrcFmt> = range.clone().map(|k| orig[k]).collect();
        let clipped_slice: Vec<PrcFmt> = range.clone().map(|k| clipped[k]).collect();
        let n = declipped_aligned.len();

        let err_declipped = error_energy(&declipped_aligned, &orig_slice, 0..n);
        let err_clipped = error_energy(&clipped_slice, &orig_slice, 0..n);
        assert!(
            err_declipped < 0.5 * err_clipped,
            "declipped error {} not substantially below clipped error {}",
            err_declipped,
            err_clipped
        );
        // Reconstructed peak must overshoot the clip level.
        let peak = declipped_aligned
            .iter()
            .cloned()
            .fold(0.0_f64 as PrcFmt, |m, v| m.max(v.abs()));
        assert!(peak > clip_level + 0.05, "peak {} did not overshoot", peak);
        assert!(pp.declipped_samples() > 0);
    }

    #[test]
    fn ar_reduces_error() {
        let orig = sine(8192, 1000.0, 0.95);
        let clip_level = 0.7;
        let clipped = hard_clip(&orig, clip_level);
        let (mut dec, _pp) = make(params(DeclipMethod::Ar, 0.6, 2, 64));
        let delay = dec.delay;
        let out = run(&mut dec, &clipped);
        let range = 512..(orig.len() - delay);
        let n = range.len();
        let declipped_aligned: Vec<PrcFmt> = range.clone().map(|k| out[delay + k]).collect();
        let orig_slice: Vec<PrcFmt> = range.clone().map(|k| orig[k]).collect();
        let clipped_slice: Vec<PrcFmt> = range.map(|k| clipped[k]).collect();
        let err_declipped = error_energy(&declipped_aligned, &orig_slice, 0..n);
        let err_clipped = error_energy(&clipped_slice, &orig_slice, 0..n);
        assert!(
            err_declipped < err_clipped,
            "AR declipped error {} not below clipped error {}",
            err_declipped,
            err_clipped
        );
        // Reconstruction must stay bounded (no runaway extrapolation).
        for v in &declipped_aligned {
            assert!(v.is_finite() && v.abs() < 4.0, "unbounded AR output: {}", v);
        }
    }

    #[test]
    fn repairs_clip_straddling_chunk_boundary() {
        // Build a signal with a single flat-topped plateau straddling the first
        // chunk boundary (index 256).
        let len = 2048;
        let mut signal = sine(len, 500.0, 0.4);
        let plateau_start = CHUNK - 5;
        let plateau_len = 12;
        for s in signal.iter_mut().skip(plateau_start).take(plateau_len) {
            *s = 0.9;
        }
        // Make the samples on each side of the plateau a clean rising/falling edge
        // so the plateau is well defined.
        signal[plateau_start - 1] = 0.7;
        signal[plateau_start + plateau_len] = 0.7;

        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.85, 2, 64));
        let delay = dec.delay;
        let out = run(&mut dec, &signal);

        let repaired: Vec<PrcFmt> = (plateau_start..plateau_start + plateau_len)
            .map(|k| out[delay + k])
            .collect();
        let max = repaired
            .iter()
            .cloned()
            .fold(0.0 as PrcFmt, |m, v| m.max(v));
        let min = repaired
            .iter()
            .cloned()
            .fold(2.0 as PrcFmt, |m, v| m.min(v));
        assert!(max > 0.9, "boundary plateau not lifted: max {}", max);
        assert!(max - min > 1e-3, "boundary plateau still flat");
        assert_eq!(pp.declipped_samples(), plateau_len);
    }

    #[test]
    fn passes_through_clips_longer_than_max() {
        let len = 2048;
        let mut signal = sine(len, 500.0, 0.4);
        let plateau_start = 600;
        let plateau_len = 30;
        for s in signal.iter_mut().skip(plateau_start).take(plateau_len) {
            *s = 0.9;
        }
        signal[plateau_start - 1] = 0.7;
        signal[plateau_start + plateau_len] = 0.7;

        // max_clip_len smaller than the plateau -> pass through untouched.
        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.85, 2, 8));
        let delay = dec.delay;
        let out = run(&mut dec, &signal);
        for k in plateau_start..plateau_start + plateau_len {
            assert!(
                (out[delay + k] - 0.9).abs() < 1e-9,
                "long clip modified at {}: {}",
                k,
                out[delay + k]
            );
        }
        assert_eq!(pp.declipped_samples(), 0);
    }

    #[test]
    fn telemetry_counts_exact_plateau() {
        let len = 2048;
        let mut signal = sine(len, 500.0, 0.4);
        let plateau_start = 700;
        let plateau_len = 10;
        for s in signal.iter_mut().skip(plateau_start).take(plateau_len) {
            *s = -0.95; // negative-going clip
        }
        signal[plateau_start - 1] = -0.7;
        signal[plateau_start + plateau_len] = -0.7;
        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.85, 2, 64));
        run(&mut dec, &signal);
        assert_eq!(pp.declipped_samples(), plateau_len);
    }

    #[test]
    fn ignores_short_runs_below_min_len() {
        // A 1-sample full-scale transient must not be treated as a clip.
        let len = 2048;
        let mut signal = sine(len, 500.0, 0.4);
        signal[700] = 0.99;
        signal[699] = 0.5;
        signal[701] = 0.5;
        let (mut dec, pp) = make(params(DeclipMethod::Cubic, 0.85, 2, 64));
        let delay = dec.delay;
        let out = run(&mut dec, &signal);
        assert_eq!(pp.declipped_samples(), 0);
        assert!((out[delay + 700] - 0.99).abs() < 1e-9);
    }

    /// The banded Cholesky solver must reproduce a dense reference solution for a
    /// symmetric positive-definite banded Toeplitz system.
    #[test]
    fn banded_solver_matches_dense() {
        let order = 4;
        // A diagonally-dominant symmetric band: rc[0] large, decaying off-diagonals.
        let rc = [4.0, -1.5, 0.5, -0.2, 0.05];
        let g = 12;
        // Reference dense solve of A x = rhs via Gaussian elimination.
        let mut mat = vec![vec![0.0 as PrcFmt; g]; g];
        for i in 0..g {
            for j in 0..g {
                let d = (i as isize - j as isize).unsigned_abs();
                if d <= order {
                    mat[i][j] = rc[d];
                }
            }
        }
        let rhs: Vec<PrcFmt> = (0..g).map(|i| ((i * 7 + 3) % 11) as PrcFmt - 5.0).collect();
        // Naive dense Gaussian elimination (partial pivoting) for the reference.
        let mut aug: Vec<Vec<PrcFmt>> = (0..g)
            .map(|i| {
                let mut row = mat[i].clone();
                row.push(rhs[i]);
                row
            })
            .collect();
        for col in 0..g {
            let mut piv = col;
            for row in (col + 1)..g {
                if aug[row][col].abs() > aug[piv][col].abs() {
                    piv = row;
                }
            }
            aug.swap(col, piv);
            let d = aug[col][col];
            for row in (col + 1)..g {
                let f = aug[row][col] / d;
                for k in col..=g {
                    aug[row][k] -= f * aug[col][k];
                }
            }
        }
        let mut dense = vec![0.0 as PrcFmt; g];
        for i in (0..g).rev() {
            let mut s = aug[i][g];
            for j in (i + 1)..g {
                s -= aug[i][j] * dense[j];
            }
            dense[i] = s / aug[i][i];
        }
        // Banded solver.
        let mut band = vec![0.0 as PrcFmt; g * (order + 1)];
        let mut x = vec![0.0 as PrcFmt; g];
        assert!(solve_banded_toeplitz(&rc, order, g, &rhs, &mut band, &mut x));
        for i in 0..g {
            assert!(
                (x[i] - dense[i]).abs() < 1e-4,
                "solver mismatch at {}: {} vs {}",
                i,
                x[i],
                dense[i]
            );
        }
    }

    /// Clipping consistency: every AR-repaired sample must have magnitude at least the
    /// clip level with the correct sign, since the true signal was provably >= the clip
    /// level wherever it flat-topped. This is what distinguishes declipping from plain
    /// interpolation — unconstrained LSAR dips below the clip level near the gap edges.
    #[test]
    fn ar_reconstruction_is_clipping_consistent() {
        // A low-frequency tone clipped hard produces long plateaus with edges where an
        // unconstrained fit would sag below the clip level.
        let orig = sine(8192, 300.0, 0.98);
        let clip_level = 0.55;
        let clipped = hard_clip(&orig, clip_level);
        let (mut dec, pp) = make(params(DeclipMethod::Ar, 0.5, 2, 64));
        let delay = dec.delay;
        let out = run(&mut dec, &clipped);

        // Walk the clipped input; for every plateau, every corresponding repaired
        // output sample must be at least the clip level (same sign), within a tiny
        // numerical margin.
        let margin = 1e-6;
        let mut checked = 0usize;
        let mut k = 0;
        while k < clipped.len() {
            if clipped[k].abs() < clip_level - 1e-9 {
                k += 1;
                continue;
            }
            let positive = clipped[k] > 0.0;
            let start = k;
            while k < clipped.len()
                && clipped[k].abs() >= clip_level - 1e-9
                && (clipped[k] > 0.0) == positive
            {
                k += 1;
            }
            let len = k - start;
            // Only plateaus the declipper actually repairs (length within limits and
            // with enough emitted context around them).
            if len < 2 || len > 64 || start < 512 || k + delay + 8 >= out.len() {
                continue;
            }
            for j in start..k {
                let v = out[delay + j];
                if positive {
                    assert!(
                        v >= clip_level - margin,
                        "positive clip under-restored at {}: {} < {}",
                        j,
                        v,
                        clip_level
                    );
                } else {
                    assert!(
                        v <= -clip_level + margin,
                        "negative clip under-restored at {}: {} > {}",
                        j,
                        v,
                        -clip_level
                    );
                }
            }
            checked += len;
        }
        assert!(checked > 0, "no plateaus were exercised");
        assert!(pp.declipped_samples() > 0);
    }

    /// Janssen LSAR reconstruction of a clipped tone should both restore the peak
    /// (overshoot the clip level) and land much closer to the original than cubic —
    /// the whole point of using a signal model instead of a boundary spline.
    #[test]
    fn ar_restores_peak_and_beats_cubic() {
        let orig = sine(8192, 1200.0, 0.97);
        let clip_level = 0.6;
        let clipped = hard_clip(&orig, clip_level);

        let (mut dec_ar, _) = make(params(DeclipMethod::Ar, 0.5, 2, 64));
        let (mut dec_cu, _) = make(params(DeclipMethod::Cubic, 0.5, 2, 64));
        let delay = dec_ar.delay;
        let out_ar = run(&mut dec_ar, &clipped);
        let out_cu = run(&mut dec_cu, &clipped);

        let range = 512..(orig.len() - delay);
        let n = range.len();
        let ar: Vec<PrcFmt> = range.clone().map(|k| out_ar[delay + k]).collect();
        let cu: Vec<PrcFmt> = range.clone().map(|k| out_cu[delay + k]).collect();
        let og: Vec<PrcFmt> = range.map(|k| orig[k]).collect();

        let err_ar = error_energy(&ar, &og, 0..n);
        let err_cu = error_energy(&cu, &og, 0..n);
        assert!(
            err_ar < err_cu,
            "AR error {} should beat cubic error {}",
            err_ar,
            err_cu
        );
        // The restored peak must overshoot the clip level and stay bounded.
        let peak = ar.iter().cloned().fold(0.0 as PrcFmt, |m, v| m.max(v.abs()));
        assert!(
            peak > clip_level + 0.05,
            "AR peak {} did not overshoot clip level {}",
            peak,
            clip_level
        );
        for v in &ar {
            assert!(v.is_finite() && v.abs() < 4.0, "unbounded AR output: {}", v);
        }
    }
}
