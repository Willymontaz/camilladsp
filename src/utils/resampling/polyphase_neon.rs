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

use crate::PrcFmt;

/// Direct polyphase dot product for f64 builds on AArch64.
///
/// Computes `sum_k branch[k] * buf[n_p - k]` over the already-clipped tap
/// interval `[k_start, k_end)`. Branch samples are contiguous forward; input
/// samples are contiguous backward, so each NEON load reverses a pair of input
/// lanes before multiplying.
#[cfg(all(target_arch = "aarch64", not(feature = "32bit")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn convolve_direct_neon(
    branch: &[PrcFmt],
    buf: &[PrcFmt],
    n_p: isize,
    k_start: usize,
    k_end: usize,
) -> PrcFmt {
    use std::arch::aarch64::*;

    debug_assert!(k_start <= k_end);
    debug_assert!(k_end <= branch.len());
    debug_assert!(n_p >= 0);

    if k_start >= k_end {
        return 0.0 as PrcFmt;
    }

    let n_p = n_p as usize;
    let branch_ptr = branch.as_ptr() as *const f64;
    let buf_ptr = buf.as_ptr() as *const f64;
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

            let branch0 = vld1q_f64(branch_ptr.add(k));
            let branch1 = vld1q_f64(branch_ptr.add(k + 2));
            let branch2 = vld1q_f64(branch_ptr.add(k + 4));
            let branch3 = vld1q_f64(branch_ptr.add(k + 6));

            let samples0_raw = vld1q_f64(buf_ptr.add(sample0 - 1));
            let samples1_raw = vld1q_f64(buf_ptr.add(sample1 - 1));
            let samples2_raw = vld1q_f64(buf_ptr.add(sample2 - 1));
            let samples3_raw = vld1q_f64(buf_ptr.add(sample3 - 1));

            let samples0 = vextq_f64::<1>(samples0_raw, samples0_raw);
            let samples1 = vextq_f64::<1>(samples1_raw, samples1_raw);
            let samples2 = vextq_f64::<1>(samples2_raw, samples2_raw);
            let samples3 = vextq_f64::<1>(samples3_raw, samples3_raw);

            acc0 = vfmaq_f64(acc0, branch0, samples0);
            acc1 = vfmaq_f64(acc1, branch1, samples1);
            acc2 = vfmaq_f64(acc2, branch2, samples2);
            acc3 = vfmaq_f64(acc3, branch3, samples3);

            k += 8;
        }

        let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
        while k + 1 < k_end {
            let sample = n_p - k;
            let branch_pair = vld1q_f64(branch_ptr.add(k));
            let samples_raw = vld1q_f64(buf_ptr.add(sample - 1));
            let samples = vextq_f64::<1>(samples_raw, samples_raw);
            acc = vfmaq_f64(acc, branch_pair, samples);
            k += 2;
        }

        let mut sum = vgetq_lane_f64::<0>(acc) + vgetq_lane_f64::<1>(acc);
        if k < k_end {
            let sample = n_p - k;
            sum += *branch.get_unchecked(k) as f64 * *buf.get_unchecked(sample) as f64;
        }
        sum as PrcFmt
    }
}
