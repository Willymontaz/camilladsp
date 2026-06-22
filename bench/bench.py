#!/usr/bin/env python3
"""
Polyphase 320 listening-quality benchmark for CamillaDSP.

This bench is intentionally narrower than the historical M4/GPU experiment:
it only exercises the current `Polyphase` resampler, using oversampling=320 for
the common listening paths 44.1/48/88.2 kHz -> 96 kHz.

The figures focus on things that can plausibly matter while listening:
  - transition-band / anti-aliasing behavior around the source Nyquist
  - impulse pre/post ringing and tail length
  - high-frequency sweep residue / imaging
  - 1 kHz sine floor and spurious tones
  - CPU cost as realtime factor
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PYPLOT = None
_SCIPY_SIGNAL = None


def pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def signal_module():
    global _SCIPY_SIGNAL
    if _SCIPY_SIGNAL is None:
        from scipy import signal as scisig

        _SCIPY_SIGNAL = scisig
    return _SCIPY_SIGNAL

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
BENCH_NAME = "polyphase_320"
SIGNALS = BENCH / "signals" / BENCH_NAME
OUTPUTS = BENCH / "outputs" / BENCH_NAME
CONFIGS = BENCH / "configs" / BENCH_NAME
FIGS = BENCH / "figures" / BENCH_NAME
LOGS = BENCH / "logs" / BENCH_NAME
RESULTS = BENCH / "results"
CSV_PATH = RESULTS / f"{BENCH_NAME}.csv"
BIN = ROOT / "target" / "release" / "camilladsp"

for path in (SIGNALS, OUTPUTS, CONFIGS, FIGS, LOGS, RESULTS):
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Audio + matrix
# -----------------------------------------------------------------------------
DST_RATE = 96_000
SRC_RATES = [44_100, 48_000, 88_200]
TAPS = [256, 512, 2048, 4096, 8192]
OVERSAMPLING = 320
CHUNKSIZE = 1024
EXTRA_SAMPLES = 65_536
RUN_TIMEOUT_S = 300

SWEEP_DURATION_S = 10.0
IMPULSE_DURATION_S = 4.0
NOISE_DURATION_S = 5.0
SINE_DURATION_S = 5.0

SMOKE_SWEEP_DURATION_S = 1.0
SMOKE_IMPULSE_DURATION_S = 1.0
SMOKE_NOISE_DURATION_S = 1.0
SMOKE_SINE_DURATION_S = 1.0

CSV_COLUMNS = [
    "rate",
    "signal",
    "taps",
    "oversampling",
    "duration_s",
    "audio_s",
    "realtime_x",
    "output_frames",
    "ok",
    "error",
    "config",
    "output",
    "log",
]


@dataclasses.dataclass(frozen=True)
class PolyphaseSpec:
    taps: int

    @property
    def key(self) -> str:
        return f"taps_{self.taps}"

    @property
    def label(self) -> str:
        return f"{self.taps} taps"

    @property
    def yaml(self) -> str:
        return f"""\
    type: Polyphase
    character: LinearPhase
    taps: {self.taps}
    oversampling: {OVERSAMPLING}
"""


@dataclasses.dataclass
class RunRecord:
    ok: bool
    rate: int
    signal: str
    spec: PolyphaseSpec
    duration_s: float | None
    output_frames: int
    config_path: Path
    output_path: Path
    log_path: Path
    error: str = ""

    def csv_row(self) -> dict[str, object]:
        audio_s = self.output_frames / DST_RATE if self.output_frames else None
        realtime_x = audio_s / self.duration_s if audio_s and self.duration_s else None
        return {
            "rate": self.rate,
            "signal": self.signal,
            "taps": self.spec.taps,
            "oversampling": OVERSAMPLING,
            "duration_s": format_decimal(self.duration_s, 6),
            "audio_s": format_decimal(audio_s, 6),
            "realtime_x": format_decimal(realtime_x, 3),
            "output_frames": self.output_frames,
            "ok": int(self.ok),
            "error": self.error.replace("\n", " | "),
            "config": rel(self.config_path),
            "output": rel(self.output_path) if self.output_path.exists() else "",
            "log": rel(self.log_path),
        }


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def format_decimal(value: float | None, places: int) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{places}f}"


def write_csv(records: list[RunRecord]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.csv_row())


# -----------------------------------------------------------------------------
# Signal generators
# -----------------------------------------------------------------------------
def write_f32(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(samples, dtype=np.float32).tofile(path)


def gen_sweep(rate: int, duration_s: float, amp: float = 0.5) -> Path:
    scisig = signal_module()
    n = int(duration_s * rate)
    t = np.arange(n) / rate
    f0 = 20.0
    f1 = rate / 2 - 10.0
    x = amp * scisig.chirp(t, f0=f0, f1=f1, t1=duration_s, method="logarithmic")
    fade = max(1, int(0.05 * rate))
    fade = min(fade, n // 2)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]
    path = SIGNALS / f"rate_{rate}" / "sweep_src.raw"
    write_f32(path, x)
    return path


def gen_dirac(rate: int, duration_s: float) -> Path:
    n = int(duration_s * rate)
    x = np.zeros(n, dtype=np.float32)
    x[n // 2] = 1.0
    path = SIGNALS / f"rate_{rate}" / "dirac_src.raw"
    write_f32(path, x)
    return path


def gen_white_noise(rate: int, duration_s: float, amp: float = 0.25) -> Path:
    rng = np.random.default_rng(0xC0FFEE + rate)
    n = int(duration_s * rate)
    x = amp * rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    path = SIGNALS / f"rate_{rate}" / "noise_src.raw"
    write_f32(path, x)
    return path


def gen_sine(rate: int, duration_s: float, freq_hz: float = 1000.0, amp: float = 0.5) -> Path:
    n = int(duration_s * rate)
    t = np.arange(n) / rate
    x = amp * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    path = SIGNALS / f"rate_{rate}" / "sine1k_src.raw"
    write_f32(path, x)
    return path


def generate_sources(rate: int, smoke: bool) -> list[tuple[str, Path]]:
    sweep_s = SMOKE_SWEEP_DURATION_S if smoke else SWEEP_DURATION_S
    impulse_s = SMOKE_IMPULSE_DURATION_S if smoke else IMPULSE_DURATION_S
    noise_s = SMOKE_NOISE_DURATION_S if smoke else NOISE_DURATION_S
    sine_s = SMOKE_SINE_DURATION_S if smoke else SINE_DURATION_S
    return [
        ("sweep", gen_sweep(rate, sweep_s)),
        ("impulse", gen_dirac(rate, impulse_s)),
        ("noise", gen_white_noise(rate, noise_s)),
        ("sine1k", gen_sine(rate, sine_s)),
    ]


# -----------------------------------------------------------------------------
# YAML + runner
# -----------------------------------------------------------------------------
def build_yaml(rate: int, spec: PolyphaseSpec, in_path: Path, out_path: Path, in_bytes: int) -> str:
    return f"""---
devices:
  samplerate: {DST_RATE}
  chunksize: {CHUNKSIZE}
  capture_samplerate: {rate}
  enable_rate_adjust: false
  resampler:
{spec.yaml}  capture:
    type: RawFile
    filename: "{in_path}"
    channels: 1
    format: F32_LE
    skip_bytes: 0
    read_bytes: {in_bytes}
    extra_samples: {EXTRA_SAMPLES}
  playback:
    type: File
    filename: "{out_path}"
    channels: 1
    format: F32_LE
    wav_header: false
filters: {{}}
mixers: {{}}
pipeline: []
"""


def run_one(rate: int, signal_name: str, spec: PolyphaseSpec, in_path: Path, bin_path: Path) -> RunRecord:
    stem = f"rate_{rate}__{signal_name}__{spec.key}"
    out_path = OUTPUTS / f"{stem}.raw"
    cfg_path = CONFIGS / f"{stem}.yml"
    log_path = LOGS / f"{stem}.log"
    for path in (out_path, cfg_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    cfg_path.write_text(build_yaml(rate, spec, in_path, out_path, in_path.stat().st_size))

    check = subprocess.run([str(bin_path), "--check", str(cfg_path)], capture_output=True, text=True)
    if check.returncode != 0:
        log_text = f"=== check stdout ===\n{check.stdout}\n=== check stderr ===\n{check.stderr}\n"
        log_path.write_text(log_text)
        return RunRecord(
            ok=False,
            rate=rate,
            signal=signal_name,
            spec=spec,
            duration_s=None,
            output_frames=0,
            config_path=cfg_path,
            output_path=out_path,
            log_path=log_path,
            error=f"--check failed with {check.returncode}",
        )

    cmd = [str(bin_path), "-l", "warn", str(cfg_path)]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
        duration_s = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        log_text = f"=== timeout stdout ===\n{exc.stdout or ''}\n=== timeout stderr ===\n{exc.stderr or ''}\n"
        log_path.write_text(log_text)
        return RunRecord(
            ok=False,
            rate=rate,
            signal=signal_name,
            spec=spec,
            duration_s=None,
            output_frames=0,
            config_path=cfg_path,
            output_path=out_path,
            log_path=log_path,
            error=f"timeout after {RUN_TIMEOUT_S}s",
        )

    log_text = (
        f"=== command ===\n{' '.join(cmd)}\n"
        f"=== stdout ===\n{proc.stdout}\n"
        f"=== stderr ===\n{proc.stderr}\n"
    )
    log_path.write_text(log_text)
    output_frames = out_path.stat().st_size // 4 if out_path.exists() else 0
    ok = proc.returncode == 0 and output_frames > 0
    error = ""
    if proc.returncode != 0:
        error = f"camilladsp exited {proc.returncode}"
    elif output_frames == 0:
        error = "no output produced"

    return RunRecord(
        ok=ok,
        rate=rate,
        signal=signal_name,
        spec=spec,
        duration_s=duration_s,
        output_frames=output_frames,
        config_path=cfg_path,
        output_path=out_path,
        log_path=log_path,
        error=error,
    )


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).astype(np.float64)


# -----------------------------------------------------------------------------
# Audio analysis helpers
# -----------------------------------------------------------------------------
def trim_active(
    y: np.ndarray,
    fs: int = DST_RATE,
    head_ms: float = 50.0,
    tail_ms: float = 50.0,
    threshold: float = 1e-5,
) -> np.ndarray:
    abs_y = np.abs(y)
    active = np.where(abs_y > threshold)[0]
    if len(active) == 0:
        return y
    head = active[0] + int(head_ms * 1e-3 * fs)
    tail = active[-1] - int(tail_ms * 1e-3 * fs)
    if tail <= head:
        return y[active[0] : active[-1] + 1]
    return y[head : tail + 1]


def common_grid(n: int, cols: int = 3) -> tuple[int, int]:
    return (n + cols - 1) // cols, cols


def output_map(records: list[RunRecord], rate: int, signal_name: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for record in records:
        if record.rate == rate and record.signal == signal_name and record.ok and record.output_path.exists():
            out[record.spec.taps] = read_f32(record.output_path)
    return dict(sorted(out.items()))


def welch_db(y: np.ndarray, fs: int = DST_RATE, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    scisig = signal_module()
    y = trim_active(y, fs=fs)
    nperseg = min(nperseg, max(256, len(y)))
    f, pxx = scisig.welch(
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        scaling="density",
        detrend=False,
    )
    passband = pxx[(f > 100.0) & (f < min(10_000.0, fs / 2))]
    ref = np.median(passband) if len(passband) else np.max(pxx)
    mag = 10 * np.log10(np.maximum(pxx / max(ref, 1e-30), 1e-24))
    return f, mag


def fft_db(y: np.ndarray, fs: int = DST_RATE, nfft: int = 1 << 17) -> tuple[np.ndarray, np.ndarray]:
    scisig = signal_module()
    y = trim_active(y, fs=fs)
    if len(y) < nfft:
        nfft = 1 << int(np.floor(np.log2(max(256, len(y)))))
    y = y[:nfft]
    w = scisig.get_window("blackmanharris", nfft)
    cg = w.sum() / nfft
    spectrum = np.fft.rfft(y * w) / (nfft * cg / 2.0)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-15))
    return freqs, mag


def thd_n(y: np.ndarray, fs: int = DST_RATE, f0: float = 1000.0) -> tuple[float, float]:
    y = trim_active(y, fs=fs)
    n = len(y)
    if n == 0:
        return math.nan, math.nan
    t = np.arange(n) / fs
    matrix = np.vstack((np.cos(2 * np.pi * f0 * t), np.sin(2 * np.pi * f0 * t))).T
    coeffs, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    fundamental = matrix @ coeffs
    residual = y - fundamental
    fund_rms = np.sqrt(np.mean(fundamental**2))
    resid_rms = np.sqrt(np.mean(residual**2))
    thdn_db = 20 * np.log10(max(resid_rms, 1e-20) / max(fund_rms, 1e-20))
    fund_dbfs = 20 * np.log10(max(fund_rms * np.sqrt(2), 1e-20))
    return float(thdn_db), float(fund_dbfs)


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def plot_performance(records: list[RunRecord], fig_dir: Path) -> None:
    plt = pyplot()
    fig_dir.mkdir(parents=True, exist_ok=True)
    rates = SRC_RATES
    taps_values = TAPS
    matrix = np.full((len(rates), len(taps_values)), np.nan)

    for i, rate in enumerate(rates):
        for j, taps in enumerate(taps_values):
            values: list[float] = []
            for record in records:
                if record.ok and record.rate == rate and record.spec.taps == taps and record.duration_s:
                    audio_s = record.output_frames / DST_RATE
                    values.append(audio_s / record.duration_s)
            if values:
                matrix[i, j] = float(np.median(values))

    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(taps_values)), labels=[str(t) for t in taps_values])
    ax.set_yticks(range(len(rates)), labels=[f"{r/1000:g}k -> 96k" for r in rates])
    ax.set_xlabel("Taps")
    ax.set_title("Realtime factor, median over listening-quality signals (higher is cheaper)")
    for i in range(len(rates)):
        for j in range(len(taps_values)):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.1f}x", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Realtime factor")
    fig.tight_layout()
    out = fig_dir / "performance_realtime_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")

    fig, axes = plt.subplots(len(rates), 1, figsize=(10, 2.8 * len(rates)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rate, row in zip(axes, rates, matrix):
        ax.bar([str(t) for t in taps_values], row)
        ax.set_ylabel("Realtime x")
        ax.set_title(f"{rate} -> {DST_RATE}")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xlabel("Taps")
    fig.suptitle("CPU cost by taps: choose the smallest taps that still sounds clean", y=1.0)
    fig.tight_layout()
    out = fig_dir / "performance_realtime_bars.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")


def plot_noise_transition(rate: int, outputs: dict[int, np.ndarray], fig_dir: Path) -> None:
    plt = pyplot()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for taps, y in outputs.items():
        f, mag = welch_db(y)
        ax.plot(f, mag, lw=1.0, label=f"{taps} taps")
    ax.axvline(rate / 2, color="k", lw=0.8, ls="--", alpha=0.6, label="source Nyquist")
    ax.set_xlim(0, DST_RATE / 2)
    ax.set_ylim(-160, 10)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB, passband-normalized]")
    ax.set_title(f"White-noise anti-aliasing after {rate}->{DST_RATE}: full audible band")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    out = fig_dir / "noise_fullband_psd.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for taps, y in outputs.items():
        f, mag = welch_db(y, nperseg=16_384)
        ax.plot(f, mag, lw=1.0, label=f"{taps} taps")
    nyq = rate / 2
    ax.axvline(nyq, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(max(nyq - 8000, 0), min(nyq + 12_000, DST_RATE / 2))
    ax.set_ylim(-170, 10)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB, passband-normalized]")
    ax.set_title(f"Transition zoom around source Nyquist ({nyq:.0f} Hz): taps should separate here")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.tight_layout()
    out = fig_dir / "noise_transition_zoom.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")


def plot_sweep_spectrogram(rate: int, outputs: dict[int, np.ndarray], fig_dir: Path) -> None:
    plt = pyplot()
    scisig = signal_module()
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = common_grid(len(outputs))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.6 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    im = None
    for ax, (taps, y) in zip(axes, outputs.items()):
        y = trim_active(y, head_ms=20.0, tail_ms=20.0)
        f, t, sxx = scisig.spectrogram(
            y,
            fs=DST_RATE,
            nperseg=4096,
            noverlap=3072,
            window="hann",
            scaling="spectrum",
            mode="magnitude",
        )
        sdb = 20 * np.log10(np.maximum(sxx, 1e-12))
        im = ax.pcolormesh(t, f, sdb, vmin=-140, vmax=-10, shading="auto", cmap="magma")
        ax.axhline(rate / 2, color="cyan", lw=0.7, ls="--", alpha=0.8)
        ax.set_title(f"{taps} taps")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_ylim(0, DST_RATE / 2)
    for ax in axes[len(outputs) :]:
        ax.set_visible(False)
    fig.suptitle(f"Log sweep residue after {rate}->{DST_RATE}: aliases/images show as extra traces", y=1.0)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), shrink=0.84, location="right")
        cbar.set_label("Magnitude [dB]")
    fig.tight_layout()
    out = fig_dir / "sweep_spectrogram_by_taps.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")


def impulse_segment(y: np.ndarray, half_window_s: float) -> tuple[np.ndarray, np.ndarray]:
    peak = int(np.argmax(np.abs(y)))
    window = int(half_window_s * DST_RATE)
    lo = max(0, peak - window)
    hi = min(len(y), peak + window + 1)
    t = (np.arange(lo, hi) - peak) / DST_RATE
    seg = y[lo:hi]
    peak_amp = np.max(np.abs(seg))
    if peak_amp > 0:
        seg = seg / peak_amp
    return t, seg


def plot_impulse(rate: int, outputs: dict[int, np.ndarray], fig_dir: Path) -> None:
    plt = pyplot()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for half_window_s, name, unit in [
        (0.002, "impulse_zoom_2ms.png", "ms"),
        (0.100, "impulse_zoom_100ms.png", "ms"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for taps, y in outputs.items():
            t, seg = impulse_segment(y, half_window_s)
            x = t * 1000 if unit == "ms" else t
            ax.plot(x, seg, lw=1.0, label=f"{taps} taps")
        ax.axvline(0, color="k", lw=0.6, alpha=0.4)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("Time relative to peak [ms]")
        ax.set_ylabel("Normalized amplitude")
        ax.set_title(f"Impulse ringing after {rate}->{DST_RATE}, +/- {half_window_s * 1000:.0f} ms")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=9)
        fig.tight_layout()
        out = fig_dir / name
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {rel(out)}")

    fig, ax = plt.subplots(figsize=(12, 5.8))
    for taps, y in outputs.items():
        t, seg = impulse_segment(y, 0.6)
        db = 20 * np.log10(np.maximum(np.abs(seg), 1e-12))
        if len(db) > 16_000:
            bins = int(np.ceil(len(db) / 16_000))
            usable = (len(db) // bins) * bins
            t_plot = t[:usable].reshape(-1, bins).mean(axis=1)
            db_plot = db[:usable].reshape(-1, bins).max(axis=1)
        else:
            t_plot, db_plot = t, db
        ax.plot(t_plot, db_plot, lw=1.0, label=f"{taps} taps")
    ax.axvline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_ylim(-180, 5)
    ax.set_xlabel("Time relative to peak [s]")
    ax.set_ylabel("|impulse| [dB, normalized]")
    ax.set_title(f"Impulse tail after {rate}->{DST_RATE}: long low-level ringing comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout()
    out = fig_dir / "impulse_log_tail_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")


def plot_sine_floor(rate: int, outputs: dict[int, np.ndarray], fig_dir: Path) -> None:
    plt = pyplot()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    thdn_rows: list[tuple[int, float, float]] = []
    for taps, y in outputs.items():
        f, mag = fft_db(y)
        thdn_db, fund_dbfs = thd_n(y)
        thdn_rows.append((taps, thdn_db, fund_dbfs))
        ax.plot(f, mag, lw=0.9, label=f"{taps} taps [THD+N {thdn_db:+.1f} dB]")
    ax.set_xlim(0, DST_RATE / 2)
    ax.set_ylim(-180, 5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dBFS]")
    ax.set_title(f"1 kHz sine floor after {rate}->{DST_RATE}: look for spurs/noise, not pretty curves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    out = fig_dir / "sine1k_floor_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {rel(out)}")

    print(f"    THD+N after {rate}->{DST_RATE}:")
    for taps, thdn_db, fund_dbfs in sorted(thdn_rows):
        print(f"      {taps:5d} taps: THD+N {thdn_db:+8.2f} dB, fundamental {fund_dbfs:+6.2f} dBFS")


def plot_rate_figures(rate: int, records: list[RunRecord]) -> None:
    rate_dir = FIGS / f"rate_{rate}"
    noise = output_map(records, rate, "noise")
    sweep = output_map(records, rate, "sweep")
    impulse = output_map(records, rate, "impulse")
    sine = output_map(records, rate, "sine1k")

    if noise:
        print(f"[figures {rate}] transition/noise")
        plot_noise_transition(rate, noise, rate_dir / "transition")
    if sweep:
        print(f"[figures {rate}] sweep spectrogram")
        plot_sweep_spectrogram(rate, sweep, rate_dir / "sweep")
    if impulse:
        print(f"[figures {rate}] impulse listening view")
        plot_impulse(rate, impulse, rate_dir / "impulse")
    if sine:
        print(f"[figures {rate}] sine/noise floor")
        plot_sine_floor(rate, sine, rate_dir / "sine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Polyphase oversampling=320 listening-quality bench.")
    parser.add_argument("--bin", type=Path, default=BIN, help="Path to target/release/camilladsp")
    parser.add_argument("--smoke", action="store_true", help="Run one short 48k->96k sine case for quick validation.")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bin_path = args.bin
    if not bin_path.exists():
        print(f"ERROR: camilladsp binary not found at {bin_path}", file=sys.stderr)
        print('Build first: RUSTFLAGS="-C target-cpu=native" cargo build --release', file=sys.stderr)
        sys.exit(2)

    rates = [48_000] if args.smoke else SRC_RATES
    taps_values = [256] if args.smoke else TAPS
    specs = [PolyphaseSpec(taps) for taps in taps_values]

    all_records: list[RunRecord] = []
    print(
        f"Polyphase 320 bench: {len(rates)} rates x {len(specs)} taps x "
        f"{1 if args.smoke else 4} signal(s)"
    )

    for rate in rates:
        print(f"\n[rate {rate} -> {DST_RATE}] Generating sources")
        sources = generate_sources(rate, smoke=args.smoke)
        if args.smoke:
            sources = [item for item in sources if item[0] == "sine1k"]
        for name, path in sources:
            print(f"    {name:8s} {rel(path)} ({path.stat().st_size} bytes)")

        for signal_name, in_path in sources:
            for spec in specs:
                print(f"    run {signal_name:8s} | {spec.label:9s}", end="", flush=True)
                record = run_one(rate, signal_name, spec, in_path, bin_path)
                all_records.append(record)
                write_csv(all_records)
                if record.ok:
                    audio_s = record.output_frames / DST_RATE
                    realtime_x = audio_s / record.duration_s if record.duration_s else math.nan
                    print(f" ok ({record.output_frames} frames, {record.duration_s:.3f}s, {realtime_x:.1f}x)")
                else:
                    print(f" FAILED: {record.error}")

        if not args.no_figures and not args.smoke:
            plot_rate_figures(rate, all_records)

    if not args.no_figures and not args.smoke:
        print("\n[figures] performance summary")
        plot_performance(all_records, FIGS / "performance")

    ok_count = sum(1 for r in all_records if r.ok)
    fail_count = len(all_records) - ok_count
    print(f"\n[done] CSV: {rel(CSV_PATH)}")
    print(f"[done] Logs: {rel(LOGS)}/")
    print(f"[done] Figures: {rel(FIGS)}/")
    print(f"[done] Successful runs: {ok_count}; failed runs: {fail_count}")


if __name__ == "__main__":
    main()
