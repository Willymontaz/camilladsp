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

//! NEON polyphase inner products for f64 builds on AArch64.
//!
//! Both kernels are load-bound rather than FMA-bound: four accumulators are
//! enough to hide the FMA latency, and adding more measured no faster.

use crate::PrcFmt;

/// Backward polyphase dot product, used for the stored branches.
///
/// Computes `sum_k coeffs[k] * buf[n_p - k]` over the already-clipped tap
/// interval `[k_start, k_end)`. Coefficients are contiguous forward while input
/// samples run backward, so each NEON load reverses a pair of input lanes
/// before multiplying.
#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_backward_neon(
    coeffs: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    use std::arch::aarch64::*;

    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= coeffs.len());
    debug_assert!(k_end <= n_p + 1);
    debug_assert!(k_start + buf.len() >= n_p + 1);

    if k_start >= k_end {
        return 0.0 as PrcFmt;
    }

    let coeff_ptr = coeffs.as_ptr();
    let buf_ptr = buf.as_ptr();
    let mut k = k_start;

    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);

        while k + 7 < k_end {
            let sample0 = n_p - k;
            let sample1 = n_p - (k + 2);
            let sample2 = n_p - (k + 4);
            let sample3 = n_p - (k + 6);

            let coeff0 = vld1q_f64(coeff_ptr.add(k));
            let coeff1 = vld1q_f64(coeff_ptr.add(k + 2));
            let coeff2 = vld1q_f64(coeff_ptr.add(k + 4));
            let coeff3 = vld1q_f64(coeff_ptr.add(k + 6));

            let samples0_raw = vld1q_f64(buf_ptr.add(sample0 - 1));
            let samples1_raw = vld1q_f64(buf_ptr.add(sample1 - 1));
            let samples2_raw = vld1q_f64(buf_ptr.add(sample2 - 1));
            let samples3_raw = vld1q_f64(buf_ptr.add(sample3 - 1));

            let samples0 = vextq_f64::<1>(samples0_raw, samples0_raw);
            let samples1 = vextq_f64::<1>(samples1_raw, samples1_raw);
            let samples2 = vextq_f64::<1>(samples2_raw, samples2_raw);
            let samples3 = vextq_f64::<1>(samples3_raw, samples3_raw);

            acc0 = vfmaq_f64(acc0, coeff0, samples0);
            acc1 = vfmaq_f64(acc1, coeff1, samples1);
            acc2 = vfmaq_f64(acc2, coeff2, samples2);
            acc3 = vfmaq_f64(acc3, coeff3, samples3);

            k += 8;
        }

        let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        while k + 1 < k_end {
            let sample = n_p - k;
            let coeff_pair = vld1q_f64(coeff_ptr.add(k));
            let samples_raw = vld1q_f64(buf_ptr.add(sample - 1));
            let samples = vextq_f64::<1>(samples_raw, samples_raw);
            acc = vfmaq_f64(acc, coeff_pair, samples);
            k += 2;
        }

        let mut sum = vgetq_lane_f64::<0>(acc) + vgetq_lane_f64::<1>(acc);
        if k < k_end {
            sum += *coeffs.get_unchecked(k) * *buf.get_unchecked(n_p - k);
        }
        sum as PrcFmt
    }
}

/// Forward dot product of two equal-length slices, used for the mirrored
/// branches. Both operands stream forward so there is no lane shuffle.
#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_forward_neon(coeffs: &[PrcFmt], samples: &[PrcFmt]) -> PrcFmt {
    use std::arch::aarch64::*;

    debug_assert_eq!(coeffs.len(), samples.len());

    let len = coeffs.len();
    let coeff_ptr = coeffs.as_ptr();
    let sample_ptr = samples.as_ptr();
    let mut i = 0;

    unsafe {
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);
        let mut acc2 = vdupq_n_f64(0.0);
        let mut acc3 = vdupq_n_f64(0.0);

        while i + 7 < len {
            acc0 = vfmaq_f64(
                acc0,
                vld1q_f64(coeff_ptr.add(i)),
                vld1q_f64(sample_ptr.add(i)),
            );
            acc1 = vfmaq_f64(
                acc1,
                vld1q_f64(coeff_ptr.add(i + 2)),
                vld1q_f64(sample_ptr.add(i + 2)),
            );
            acc2 = vfmaq_f64(
                acc2,
                vld1q_f64(coeff_ptr.add(i + 4)),
                vld1q_f64(sample_ptr.add(i + 4)),
            );
            acc3 = vfmaq_f64(
                acc3,
                vld1q_f64(coeff_ptr.add(i + 6)),
                vld1q_f64(sample_ptr.add(i + 6)),
            );
            i += 8;
        }

        let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        while i + 1 < len {
            acc = vfmaq_f64(
                acc,
                vld1q_f64(coeff_ptr.add(i)),
                vld1q_f64(sample_ptr.add(i)),
            );
            i += 2;
        }

        let mut sum = vgetq_lane_f64::<0>(acc) + vgetq_lane_f64::<1>(acc);
        if i < len {
            sum += *coeffs.get_unchecked(i) * *samples.get_unchecked(i);
        }
        sum as PrcFmt
    }
}

/// Two-channel backward dot product: `convolve_backward_neon` for `buf0` and
/// `buf1` against the same coefficients in a single pass.
///
/// This is the coefficient-sharing path. Per unrolled step it issues 4
/// coefficient loads, 4 + 4 sample loads and 8 FMAs, against 8 + 8 loads and 8
/// FMAs for two separate passes: the L2-streamed coefficient traffic is halved
/// while the L1-resident sample traffic is unchanged. The per-channel
/// accumulation order is identical to the single-channel kernel, so the results
/// are bit-identical to two separate calls.
#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_backward2_neon(
    coeffs: &[PrcFmt],
    buf0: &[PrcFmt],
    buf1: &[PrcFmt],
    n_p: usize,
    k_start: usize,
    k_end: usize,
) -> (PrcFmt, PrcFmt) {
    use std::arch::aarch64::*;

    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= coeffs.len());
    debug_assert!(k_end <= n_p + 1);
    debug_assert!(k_start + buf0.len() >= n_p + 1);
    debug_assert!(k_start + buf1.len() >= n_p + 1);

    if k_start >= k_end {
        return (0.0 as PrcFmt, 0.0 as PrcFmt);
    }

    let coeff_ptr = coeffs.as_ptr();
    let buf0_ptr = buf0.as_ptr();
    let buf1_ptr = buf1.as_ptr();
    let mut k = k_start;

    unsafe {
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let mut b0 = vdupq_n_f64(0.0);
        let mut b1 = vdupq_n_f64(0.0);
        let mut b2 = vdupq_n_f64(0.0);
        let mut b3 = vdupq_n_f64(0.0);

        while k + 7 < k_end {
            let sample0 = n_p - k;
            let sample1 = n_p - (k + 2);
            let sample2 = n_p - (k + 4);
            let sample3 = n_p - (k + 6);

            let coeff0 = vld1q_f64(coeff_ptr.add(k));
            let coeff1 = vld1q_f64(coeff_ptr.add(k + 2));
            let coeff2 = vld1q_f64(coeff_ptr.add(k + 4));
            let coeff3 = vld1q_f64(coeff_ptr.add(k + 6));

            let x0 = vld1q_f64(buf0_ptr.add(sample0 - 1));
            let x1 = vld1q_f64(buf0_ptr.add(sample1 - 1));
            let x2 = vld1q_f64(buf0_ptr.add(sample2 - 1));
            let x3 = vld1q_f64(buf0_ptr.add(sample3 - 1));
            let y0 = vld1q_f64(buf1_ptr.add(sample0 - 1));
            let y1 = vld1q_f64(buf1_ptr.add(sample1 - 1));
            let y2 = vld1q_f64(buf1_ptr.add(sample2 - 1));
            let y3 = vld1q_f64(buf1_ptr.add(sample3 - 1));

            a0 = vfmaq_f64(a0, coeff0, vextq_f64::<1>(x0, x0));
            a1 = vfmaq_f64(a1, coeff1, vextq_f64::<1>(x1, x1));
            a2 = vfmaq_f64(a2, coeff2, vextq_f64::<1>(x2, x2));
            a3 = vfmaq_f64(a3, coeff3, vextq_f64::<1>(x3, x3));
            b0 = vfmaq_f64(b0, coeff0, vextq_f64::<1>(y0, y0));
            b1 = vfmaq_f64(b1, coeff1, vextq_f64::<1>(y1, y1));
            b2 = vfmaq_f64(b2, coeff2, vextq_f64::<1>(y2, y2));
            b3 = vfmaq_f64(b3, coeff3, vextq_f64::<1>(y3, y3));

            k += 8;
        }

        let mut acc_a = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        let mut acc_b = vaddq_f64(vaddq_f64(b0, b1), vaddq_f64(b2, b3));
        while k + 1 < k_end {
            let sample = n_p - k;
            let coeff_pair = vld1q_f64(coeff_ptr.add(k));
            let x = vld1q_f64(buf0_ptr.add(sample - 1));
            let y = vld1q_f64(buf1_ptr.add(sample - 1));
            acc_a = vfmaq_f64(acc_a, coeff_pair, vextq_f64::<1>(x, x));
            acc_b = vfmaq_f64(acc_b, coeff_pair, vextq_f64::<1>(y, y));
            k += 2;
        }

        let mut sum_a = vgetq_lane_f64::<0>(acc_a) + vgetq_lane_f64::<1>(acc_a);
        let mut sum_b = vgetq_lane_f64::<0>(acc_b) + vgetq_lane_f64::<1>(acc_b);
        if k < k_end {
            let c = *coeffs.get_unchecked(k);
            sum_a += c * *buf0.get_unchecked(n_p - k);
            sum_b += c * *buf1.get_unchecked(n_p - k);
        }
        (sum_a as PrcFmt, sum_b as PrcFmt)
    }
}

/// Two-channel forward dot product: `convolve_forward_neon` for `samples0` and
/// `samples1` against the same coefficients in a single pass. See
/// [`convolve_backward2_neon`] for the load accounting.
#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_forward2_neon(
    coeffs: &[PrcFmt],
    samples0: &[PrcFmt],
    samples1: &[PrcFmt],
) -> (PrcFmt, PrcFmt) {
    use std::arch::aarch64::*;

    debug_assert_eq!(coeffs.len(), samples0.len());
    debug_assert_eq!(coeffs.len(), samples1.len());

    let len = coeffs.len();
    let coeff_ptr = coeffs.as_ptr();
    let s0_ptr = samples0.as_ptr();
    let s1_ptr = samples1.as_ptr();
    let mut i = 0;

    unsafe {
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let mut b0 = vdupq_n_f64(0.0);
        let mut b1 = vdupq_n_f64(0.0);
        let mut b2 = vdupq_n_f64(0.0);
        let mut b3 = vdupq_n_f64(0.0);

        while i + 7 < len {
            let coeff0 = vld1q_f64(coeff_ptr.add(i));
            let coeff1 = vld1q_f64(coeff_ptr.add(i + 2));
            let coeff2 = vld1q_f64(coeff_ptr.add(i + 4));
            let coeff3 = vld1q_f64(coeff_ptr.add(i + 6));

            a0 = vfmaq_f64(a0, coeff0, vld1q_f64(s0_ptr.add(i)));
            a1 = vfmaq_f64(a1, coeff1, vld1q_f64(s0_ptr.add(i + 2)));
            a2 = vfmaq_f64(a2, coeff2, vld1q_f64(s0_ptr.add(i + 4)));
            a3 = vfmaq_f64(a3, coeff3, vld1q_f64(s0_ptr.add(i + 6)));
            b0 = vfmaq_f64(b0, coeff0, vld1q_f64(s1_ptr.add(i)));
            b1 = vfmaq_f64(b1, coeff1, vld1q_f64(s1_ptr.add(i + 2)));
            b2 = vfmaq_f64(b2, coeff2, vld1q_f64(s1_ptr.add(i + 4)));
            b3 = vfmaq_f64(b3, coeff3, vld1q_f64(s1_ptr.add(i + 6)));

            i += 8;
        }

        let mut acc_a = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        let mut acc_b = vaddq_f64(vaddq_f64(b0, b1), vaddq_f64(b2, b3));
        while i + 1 < len {
            let coeff_pair = vld1q_f64(coeff_ptr.add(i));
            acc_a = vfmaq_f64(acc_a, coeff_pair, vld1q_f64(s0_ptr.add(i)));
            acc_b = vfmaq_f64(acc_b, coeff_pair, vld1q_f64(s1_ptr.add(i)));
            i += 2;
        }

        let mut sum_a = vgetq_lane_f64::<0>(acc_a) + vgetq_lane_f64::<1>(acc_a);
        let mut sum_b = vgetq_lane_f64::<0>(acc_b) + vgetq_lane_f64::<1>(acc_b);
        if i < len {
            let c = *coeffs.get_unchecked(i);
            sum_a += c * *samples0.get_unchecked(i);
            sum_b += c * *samples1.get_unchecked(i);
        }
        (sum_a as PrcFmt, sum_b as PrcFmt)
    }
}
