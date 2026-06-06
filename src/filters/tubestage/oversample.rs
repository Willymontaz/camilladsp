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

// Anti-aliasing oversampling backends used by the `TubeStage` filter.
//
// Two backends are available:
// - `HalfbandFirOversampler`: cascade of 2x halfband FIR stages (Kaiser-windowed sinc).
//   Low latency, no extra dependency, suitable for real-time audio.
// - `RubatoOversampler`: synchronous FFT resampler from rubato. Fixed chunk size required.

use crate::PrcFmt;
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Indexing, Resampler};

pub trait Oversampler: Send {
    fn factor(&self) -> usize;

    /// Upsample `input` to `output`. `output.len()` must be `factor * input.len()`.
    fn upsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]);

    /// Downsample `input` to `output`. `input.len()` must be `factor * output.len()`.
    fn downsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]);
}

// ------------------------------- Identity (factor = 1) -------------------------------

pub struct IdentityOversampler;

impl Oversampler for IdentityOversampler {
    fn factor(&self) -> usize {
        1
    }

    fn upsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        output.copy_from_slice(input);
    }

    fn downsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        output.copy_from_slice(input);
    }
}

// ------------------------------- Halfband FIR -------------------------------

const HALFBAND_LENGTH: usize = 31; // odd, center at index 15 (odd) -> halfband natural
const KAISER_BETA: f64 = 8.0; // ~80 dB stopband attenuation

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..60 {
        term *= half_x / (k as f64);
        let new = term * term;
        sum += new;
        if new < 1e-30 * sum {
            break;
        }
    }
    sum
}

fn kaiser_window(n: usize, length: usize, beta: f64) -> f64 {
    let center = (length - 1) as f64 / 2.0;
    let arg = (n as f64 - center) / center;
    let inside = (1.0 - arg * arg).max(0.0);
    bessel_i0(beta * inside.sqrt()) / bessel_i0(beta)
}

/// Build a length-N halfband prototype lowpass with cutoff = fs/4, DC gain = 1.
/// For `length` of the form 4k+3 (e.g. 31, 35, ...) the natural sinc places zeros
/// at every odd-index tap except the center, giving a true halfband filter.
fn make_halfband_coeffs(length: usize, beta: f64) -> Vec<PrcFmt> {
    assert!(length % 2 == 1, "halfband filter length must be odd");
    let center = length / 2;
    let mut h = vec![0.0_f64; length];
    for (n, slot) in h.iter_mut().enumerate() {
        let m = n as i64 - center as i64;
        // Cutoff = 0.25 * fs (i.e. fs/4 = half of Nyquist).
        // Filter impulse response: h_lp[n] = 2*fc * sinc(2*fc*(n-center)) = 0.5 * sinc((n-center)/2)
        let sinc_val = if m == 0 {
            0.5
        } else {
            let arg = std::f64::consts::PI * (m as f64) / 2.0;
            arg.sin() / (std::f64::consts::PI * m as f64)
        };
        *slot = sinc_val * kaiser_window(n, length, beta);
    }
    // Normalize to DC gain = 1 so the same coefficients can be reused for both
    // up- and down-sampling (the upsampler scales output by 2 to compensate for
    // zero stuffing).
    let dc_gain: f64 = h.iter().sum();
    h.iter().map(|v| (v / dc_gain) as PrcFmt).collect()
}

/// Single 2x halfband stage. Holds independent up- and down-sampling state.
struct HalfbandStage {
    coeffs: Vec<PrcFmt>,
    up_dl: Vec<PrcFmt>,   // length = HALFBAND_LENGTH, indexed at high rate
    down_dl: Vec<PrcFmt>, // length = HALFBAND_LENGTH, indexed at high rate
}

impl HalfbandStage {
    fn new() -> Self {
        let coeffs = make_halfband_coeffs(HALFBAND_LENGTH, KAISER_BETA);
        Self {
            coeffs,
            up_dl: vec![0.0; HALFBAND_LENGTH],
            down_dl: vec![0.0; HALFBAND_LENGTH],
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.up_dl.iter_mut().for_each(|v| *v = 0.0);
        self.down_dl.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Upsample one input sample x[n] to two output samples y[2n], y[2n+1].
    /// Implementation note: we zero-stuff and convolve at the high rate, skipping
    /// taps known to be zero by construction. The output is scaled by 2 to
    /// compensate for the zero-stuffing energy loss.
    #[inline]
    fn upsample_one(&mut self, x: PrcFmt) -> (PrcFmt, PrcFmt) {
        // Push x then 0 into the high-rate delay line, computing one output per push.
        // First push: insert x.
        for i in (1..HALFBAND_LENGTH).rev() {
            self.up_dl[i] = self.up_dl[i - 1];
        }
        self.up_dl[0] = x;
        let y0 = self.fir_eval(&self.up_dl);
        // Second push: insert 0.
        for i in (1..HALFBAND_LENGTH).rev() {
            self.up_dl[i] = self.up_dl[i - 1];
        }
        self.up_dl[0] = 0.0;
        let y1 = self.fir_eval(&self.up_dl);
        (2.0 * y0, 2.0 * y1)
    }

    /// Downsample two input samples to one output sample.
    /// The two inputs are consumed in order (x_first then x_second), the second
    /// producing the output.
    #[inline]
    fn downsample_pair(&mut self, x_first: PrcFmt, x_second: PrcFmt) -> PrcFmt {
        for i in (1..HALFBAND_LENGTH).rev() {
            self.down_dl[i] = self.down_dl[i - 1];
        }
        self.down_dl[0] = x_first;
        for i in (1..HALFBAND_LENGTH).rev() {
            self.down_dl[i] = self.down_dl[i - 1];
        }
        self.down_dl[0] = x_second;
        self.fir_eval(&self.down_dl)
    }

    #[inline]
    fn fir_eval(&self, dl: &[PrcFmt]) -> PrcFmt {
        let mut sum = 0.0;
        for (k, &sample) in dl.iter().enumerate().take(HALFBAND_LENGTH) {
            // Many of these coefficients are exactly zero by construction
            // (halfband). The compiler can not prove that, so we guard the mul.
            let c = self.coeffs[k];
            if c != 0.0 {
                sum += c * sample;
            }
        }
        sum
    }
}

pub struct HalfbandFirOversampler {
    factor: usize,
    stages: Vec<HalfbandStage>,
    /// Intermediate buffers at every intermediate sample rate (excluding the
    /// final/lowest one). `scratch[i]` holds samples at rate `2^(i+1) * fs_base`.
    scratch: Vec<Vec<PrcFmt>>,
}

impl HalfbandFirOversampler {
    pub fn new(factor: usize, base_chunksize: usize) -> Self {
        assert!(
            factor.is_power_of_two() && factor >= 2,
            "halfband oversampler factor must be a power of two >= 2"
        );
        let num_stages = factor.trailing_zeros() as usize;
        let mut stages = Vec::with_capacity(num_stages);
        for _ in 0..num_stages {
            stages.push(HalfbandStage::new());
        }
        let mut scratch = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            scratch.push(vec![0.0; base_chunksize * (1 << (i + 1))]);
        }
        Self {
            factor,
            stages,
            scratch,
        }
    }
}

impl Oversampler for HalfbandFirOversampler {
    fn factor(&self) -> usize {
        self.factor
    }

    fn upsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        debug_assert_eq!(output.len(), self.factor * input.len());
        let num_stages = self.stages.len();
        let n_in = input.len();
        // First stage: input -> scratch[0]
        {
            let stage = &mut self.stages[0];
            let dst = &mut self.scratch[0];
            // Resize if chunk grew (rare; only on misuse). We re-allocate to keep
            // semantics consistent.
            if dst.len() != 2 * n_in {
                dst.resize(2 * n_in, 0.0);
            }
            for (i, &x) in input.iter().enumerate() {
                let (y0, y1) = stage.upsample_one(x);
                dst[2 * i] = y0;
                dst[2 * i + 1] = y1;
            }
        }
        // Intermediate stages: scratch[s-1] -> scratch[s]
        for s in 1..num_stages {
            // Split scratch borrow.
            let (left, right) = self.scratch.split_at_mut(s);
            let src = &left[s - 1];
            let dst = &mut right[0];
            let n_src = src.len();
            if dst.len() != 2 * n_src {
                dst.resize(2 * n_src, 0.0);
            }
            let stage = &mut self.stages[s];
            for i in 0..n_src {
                let (y0, y1) = stage.upsample_one(src[i]);
                dst[2 * i] = y0;
                dst[2 * i + 1] = y1;
            }
        }
        // Last stage's scratch -> output
        let last = &self.scratch[num_stages - 1];
        debug_assert_eq!(last.len(), output.len());
        output.copy_from_slice(last);
    }

    fn downsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        debug_assert_eq!(input.len(), self.factor * output.len());
        let num_stages = self.stages.len();
        // We process from the highest rate down to base. The first downsample
        // (highest stage) reads from `input`, subsequent ones read from scratch.
        // We use scratch[s] to hold the rate after stage s+1 has been applied.
        // For clarity we walk top-down: the top stage is `stages[num_stages - 1]`.
        //
        // input -> scratch[num_stages - 2] -> ... -> scratch[0] -> output

        if num_stages == 1 {
            let stage = &mut self.stages[0];
            for (i, chunk) in input.chunks_exact(2).enumerate() {
                output[i] = stage.downsample_pair(chunk[0], chunk[1]);
            }
            return;
        }

        // Top stage: input -> scratch[num_stages - 2]
        {
            let s_top = num_stages - 1;
            let dst_idx = num_stages - 2;
            let stage = &mut self.stages[s_top];
            let n_out = input.len() / 2;
            let dst = &mut self.scratch[dst_idx];
            if dst.len() != n_out {
                dst.resize(n_out, 0.0);
            }
            for (i, chunk) in input.chunks_exact(2).enumerate() {
                dst[i] = stage.downsample_pair(chunk[0], chunk[1]);
            }
        }
        // Intermediate stages: scratch[s] -> scratch[s-1]
        for s in (1..num_stages - 1).rev() {
            let (left, right) = self.scratch.split_at_mut(s);
            let src = &right[0]; // scratch[s]
            let dst = &mut left[s - 1]; // scratch[s-1]
            let n_out = src.len() / 2;
            if dst.len() != n_out {
                dst.resize(n_out, 0.0);
            }
            let stage = &mut self.stages[s];
            for i in 0..n_out {
                dst[i] = stage.downsample_pair(src[2 * i], src[2 * i + 1]);
            }
        }
        // Last stage (closest to base rate): scratch[0] -> output
        {
            let stage = &mut self.stages[0];
            let src = &self.scratch[0];
            for (i, chunk) in src.chunks_exact(2).enumerate() {
                output[i] = stage.downsample_pair(chunk[0], chunk[1]);
            }
        }
    }
}

// ------------------------------- Rubato (FFT) -------------------------------

pub struct RubatoOversampler {
    factor: usize,
    chunksize: usize,
    upsampler: Box<dyn Resampler<PrcFmt> + Send>,
    downsampler: Box<dyn Resampler<PrcFmt> + Send>,
    in_buf: Vec<Vec<PrcFmt>>,
    out_buf_high: Vec<Vec<PrcFmt>>,
    out_buf_low: Vec<Vec<PrcFmt>>,
    indexing: Indexing,
}

impl RubatoOversampler {
    pub fn new(factor: usize, chunksize: usize) -> Self {
        let upsampler = Box::new(
            Fft::<PrcFmt>::new(1, factor, factor * chunksize, 2, 1, FixedSync::Output).expect(
                "Failed to create rubato Fft upsampler for TubeStage. \
                 Use a different chunk size or oversampling factor.",
            ),
        );
        let downsampler = Box::new(
            Fft::<PrcFmt>::new(factor, 1, chunksize, 2, 1, FixedSync::Output).expect(
                "Failed to create rubato Fft downsampler for TubeStage. \
                 Use a different chunk size or oversampling factor.",
            ),
        );
        Self {
            factor,
            chunksize,
            upsampler,
            downsampler,
            in_buf: vec![vec![0.0; chunksize]],
            out_buf_high: vec![vec![0.0; factor * chunksize]],
            out_buf_low: vec![vec![0.0; chunksize]],
            indexing: Indexing {
                input_offset: 0,
                output_offset: 0,
                partial_len: None,
                active_channels_mask: None,
            },
        }
    }
}

impl Oversampler for RubatoOversampler {
    fn factor(&self) -> usize {
        self.factor
    }

    fn upsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        debug_assert_eq!(input.len(), self.chunksize);
        debug_assert_eq!(output.len(), self.factor * self.chunksize);
        self.in_buf[0].clear();
        self.in_buf[0].extend_from_slice(input);
        let in_adapter = SequentialSliceOfVecs::new(&self.in_buf, 1, self.chunksize)
            .expect("Failed to build input adapter for rubato upsampler");
        let mut out_adapter =
            SequentialSliceOfVecs::new_mut(&mut self.out_buf_high, 1, self.factor * self.chunksize)
                .expect("Failed to build output adapter for rubato upsampler");
        self.upsampler
            .process_into_buffer(&in_adapter, &mut out_adapter, Some(&self.indexing))
            .expect("rubato upsampler processing failed");
        output.copy_from_slice(&self.out_buf_high[0]);
    }

    fn downsample(&mut self, input: &[PrcFmt], output: &mut [PrcFmt]) {
        debug_assert_eq!(input.len(), self.factor * self.chunksize);
        debug_assert_eq!(output.len(), self.chunksize);
        // We need a Vec<Vec<PrcFmt>> as the input buffer for the audioadapter.
        if self.out_buf_high[0].len() != input.len() {
            self.out_buf_high[0].resize(input.len(), 0.0);
        }
        self.out_buf_high[0].copy_from_slice(input);
        let in_adapter =
            SequentialSliceOfVecs::new(&self.out_buf_high, 1, self.factor * self.chunksize)
                .expect("Failed to build input adapter for rubato downsampler");
        let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut self.out_buf_low, 1, self.chunksize)
            .expect("Failed to build output adapter for rubato downsampler");
        self.downsampler
            .process_into_buffer(&in_adapter, &mut out_adapter, Some(&self.indexing))
            .expect("rubato downsampler processing failed");
        output.copy_from_slice(&self.out_buf_low[0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrcFmt;

    fn rms(a: &[PrcFmt]) -> PrcFmt {
        let n = a.len() as PrcFmt;
        let s: PrcFmt = a.iter().map(|x| x * x).sum();
        (s / n).sqrt()
    }

    fn make_sine(n: usize, freq: PrcFmt, fs: PrcFmt) -> Vec<PrcFmt> {
        let mut out = vec![0.0; n];
        let two_pi = 2.0 * std::f64::consts::PI as PrcFmt;
        for (i, x) in out.iter_mut().enumerate() {
            *x = (two_pi * freq * (i as PrcFmt) / fs).sin();
        }
        out
    }

    #[test]
    fn identity_round_trip() {
        let mut over = IdentityOversampler;
        let input = make_sine(256, 1000.0, 44100.0);
        let mut up = vec![0.0; 256];
        over.upsample(&input, &mut up);
        let mut back = vec![0.0; 256];
        over.downsample(&up, &mut back);
        let n = input.len();
        let s: PrcFmt = input
            .iter()
            .zip(back.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        assert!((s / n as PrcFmt).sqrt() < 1e-12);
    }

    #[test]
    fn halfband_round_trip_amplitude_preserved_2x() {
        // Verify that round-trip preserves the RMS amplitude of a low-frequency
        // sine within the filter passband. We do not check exact alignment as
        // group-delay compensation isn't an integer number of low-rate samples.
        let n = 4096;
        let mut over = HalfbandFirOversampler::new(2, n);
        let input = make_sine(n, 1000.0, 44100.0);
        let mut up = vec![0.0; 2 * n];
        let mut back = vec![0.0; n];

        // Prime through a couple of blocks to flush the delay lines.
        let zeros = vec![0.0; n];
        let mut up_pre = vec![0.0; 2 * n];
        let mut back_pre = vec![0.0; n];
        for _ in 0..2 {
            over.upsample(&zeros, &mut up_pre);
            over.downsample(&up_pre, &mut back_pre);
        }

        over.upsample(&input, &mut up);
        over.downsample(&up, &mut back);

        // Compare RMS of inner window (skip filter transient at start).
        let skip = 64usize;
        let in_rms = rms(&input[..n - skip]);
        let out_rms = rms(&back[skip..]);
        let err = (out_rms - in_rms).abs() / in_rms;
        assert!(err < 0.02, "halfband 2x amplitude mismatch: {err} (in={in_rms}, out={out_rms})");
    }

    #[test]
    fn halfband_round_trip_amplitude_preserved_4x() {
        let n = 4096;
        let mut over = HalfbandFirOversampler::new(4, n);
        let input = make_sine(n, 1000.0, 44100.0);
        let mut up = vec![0.0; 4 * n];
        let mut back = vec![0.0; n];

        let zeros = vec![0.0; n];
        let mut up_pre = vec![0.0; 4 * n];
        let mut back_pre = vec![0.0; n];
        for _ in 0..3 {
            over.upsample(&zeros, &mut up_pre);
            over.downsample(&up_pre, &mut back_pre);
        }

        over.upsample(&input, &mut up);
        over.downsample(&up, &mut back);

        let skip = 64usize;
        let in_rms = rms(&input[..n - skip]);
        let out_rms = rms(&back[skip..]);
        let err = (out_rms - in_rms).abs() / in_rms;
        assert!(err < 0.05, "halfband 4x amplitude mismatch: {err} (in={in_rms}, out={out_rms})");
    }

    #[test]
    fn halfband_dc_passthrough() {
        let n = 256;
        let mut over = HalfbandFirOversampler::new(2, n);
        let input = vec![1.0; n];
        let mut up = vec![0.0; 2 * n];
        over.upsample(&input, &mut up);
        // After flushing the delay line, all up-sampled samples should be ~1.0
        // (DC passes through with gain 1).
        let tail = &up[2 * (HALFBAND_LENGTH + 4)..];
        for &v in tail {
            assert!((v - 1.0).abs() < 1e-3, "DC not passed through: {v}");
        }
    }
}
