#!/usr/bin/env python3
"""
Generate timing comparison report for CVBS video captures.

Analyzes horizontal and vertical sync timing from Saleae logic analyzer exports
and generates an HTML report comparing two captures.

Usage:
    python generate_timing_report.py --capture1 ref.csv --capture2 dut.csv --output report.html
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations will be limited")


# NTSC timing constants
NTSC_LINE_FREQ = 15734.264  # Hz
NTSC_LINE_PERIOD = 1.0 / NTSC_LINE_FREQ  # ~63.556 µs
NTSC_FIELD_FREQ = 59.94  # Hz
NTSC_FIELD_PERIOD = 1.0 / NTSC_FIELD_FREQ  # ~16.683 ms
NTSC_LINES_PER_FIELD = 262.5
NTSC_HSYNC_PULSE = 4.7e-6  # 4.7 µs

# Signal validity detection thresholds
# H-period plausibility: 0.4x to 1.6x nominal covers half-lines (~32 µs)
# and normal lines (~63.5 µs) but rejects oscillation junk (1-25 µs, >102 µs)
H_PERIOD_MIN = NTSC_LINE_PERIOD * 0.4    # ~25.4 µs
H_PERIOD_MAX = NTSC_LINE_PERIOD * 1.6    # ~101.7 µs
H_VALIDITY_WINDOW = 200                   # Sliding window size (~12.7 ms, ~¾ field)
H_VALIDITY_THRESHOLD = 0.9               # 90% of periods in window must be plausible
VALIDITY_MERGE_GAP = NTSC_FIELD_PERIOD / 2   # half a field (~8.3 ms, ~131 lines) - merge intervals closer than this
VALIDITY_MIN_DURATION = NTSC_FIELD_PERIOD * 3 # 3 fields (~50 ms) - need at least a bottom/top/bottom sequence
V_RATE_MIN = 15.0                         # Hz - minimum acceptable V rate in valid region
V_RATE_MAX = 240.0                        # Hz - maximum acceptable V rate


def quantize_to_sample_period(times: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """
    Quantize timing values to the sample period.

    This eliminates false precision from timing measurements that cannot
    actually resolve time intervals smaller than the sample period.

    Args:
        times: Array of time values in seconds
        sample_rate_hz: Sample rate in Hz (e.g., 20e6 for 20 MS/s)

    Returns:
        Array of quantized time values
    """
    sample_period = 1.0 / sample_rate_hz
    return np.round(times / sample_period) * sample_period


def get_noise_floor_ns(sample_rate_mhz: float) -> float:
    """
    Calculate the measurement noise floor in nanoseconds.

    For period measurements (difference of two edge times), the uncertainty
    is √2 times the single-edge uncertainty. Using 2× this value as a
    practical threshold for "at noise floor".

    Args:
        sample_rate_mhz: Sample rate in MS/s

    Returns:
        Noise floor threshold in nanoseconds
    """
    sample_period_ns = 1000.0 / sample_rate_mhz
    # Single edge uncertainty = ±½ sample period
    # Period uncertainty = √2 × single edge
    # Use 2× as practical threshold
    return sample_period_ns * np.sqrt(2)


def format_jitter_with_threshold(value_ns: float, threshold_ns: float,
                                  decimals: int = 2, unit: str = 'ns') -> str:
    """
    Format a jitter value, marking it if below measurement threshold.

    Args:
        value_ns: Jitter value in nanoseconds
        threshold_ns: Noise floor threshold in nanoseconds
        decimals: Number of decimal places
        unit: Display unit ('ns' or 'us')

    Returns:
        Formatted string with threshold indicator if applicable
    """
    if value_ns < threshold_ns:
        if unit == 'us':
            return f"<{threshold_ns/1000:.2f} µs*"
        else:
            return f"<{threshold_ns:.0f} ns*"
    else:
        if unit == 'us':
            return f"{value_ns/1000:.{decimals}f} µs"
        else:
            return f"{value_ns:.{decimals}f} ns"


@dataclass
class TimingData:
    """Parsed timing data from a capture file."""
    filename: str
    label: str
    h_periods: np.ndarray  # Horizontal line periods in seconds
    v_periods: np.ndarray  # Vertical field periods in seconds
    h_times: np.ndarray    # Timestamps of HSYNC events
    v_times: np.ndarray    # Timestamps of VSYNC events
    lines_per_field: np.ndarray  # Number of H lines in each field
    duration: float        # Total capture duration


@dataclass
class TimingStats:
    """Statistical summary of timing data."""
    mean: float
    median: float
    std: float
    min: float
    max: float
    p1: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float
    rms_jitter: float      # RMS deviation from median
    pp_jitter: float       # Peak-to-peak jitter
    count: int


@dataclass
class ValidInterval:
    """A contiguous time interval where valid signal is present."""
    start: float   # Start time in seconds
    end: float     # End time in seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SignalValidity:
    """Signal validity analysis results for one capture."""
    valid_intervals: List[ValidInterval]
    total_duration: float
    capture_start: float         # Absolute start time of the capture window
    h_period_mask: np.ndarray    # Boolean mask for h_periods (True = valid)
    v_period_mask: np.ndarray    # Boolean mask for v_periods (True = valid)

    @property
    def valid_duration(self) -> float:
        return sum(iv.duration for iv in self.valid_intervals)

    @property
    def valid_fraction(self) -> float:
        if self.total_duration <= 0:
            return 0.0
        return self.valid_duration / self.total_duration

    @property
    def invalid_duration(self) -> float:
        return self.total_duration - self.valid_duration

    @property
    def num_valid_regions(self) -> int:
        return len(self.valid_intervals)


@dataclass
class CaptureResult:
    """All computed data for a single capture."""
    label: str                    # Display label for this capture
    filepath: str                 # Source file path
    data: TimingData              # Filtered timing data
    raw_data: TimingData          # Pre-filter data (after trim)
    validity: SignalValidity      # Signal validity info
    stats_h: TimingStats          # Horizontal stats (normal lines only)
    stats_v: TimingStats          # Vertical stats
    valid_fraction: float         # Proportion of valid signal (0.0 to 1.0)
    h_deviation: float            # Mean H deviation from nominal (µs)
    v_deviation: float            # Mean V deviation from nominal (ms)


def parse_timing_csv(filepath: str, label: str = None,
                     sample_rate_mhz: float = None) -> TimingData:
    """
    Parse Saleae timing CSV export.

    Format:
        Time [s],Channel 0,Channel 1
        0.000000000,1,1
        ...

    Channel 0 = HSOUT (1->0 = HBI start)
    Channel 1 = VSOUT (1->0 = VBI start)

    Args:
        filepath: Path to CSV file
        label: Label for this capture (default: filename)
        sample_rate_mhz: If provided, quantize timing values to this sample rate
                         to eliminate false precision from measurements
    """
    if label is None:
        label = Path(filepath).stem

    # Use numpy for fast CSV loading (much faster than csv.reader for large files)
    # Skip header row, load all columns
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1,
                         dtype=[('time', 'f8'), ('ch0', 'i1'), ('ch1', 'i1')])

    times = data['time']
    h_vals = data['ch0']
    v_vals = data['ch1']

    # Quantize timing values if sample rate provided
    if sample_rate_mhz is not None:
        times = quantize_to_sample_period(times, sample_rate_mhz * 1e6)

    # Find HSYNC events (1->0 transitions on Channel 0)
    h_edges = np.where((h_vals[:-1] == 1) & (h_vals[1:] == 0))[0] + 1
    h_times = times[h_edges]
    h_periods = np.diff(h_times)

    # Find VSYNC events (1->0 transitions on Channel 1)
    v_edges = np.where((v_vals[:-1] == 1) & (v_vals[1:] == 0))[0] + 1
    v_times = times[v_edges]
    v_periods = np.diff(v_times)

    # Calculate lines per field using searchsorted (O(n log n) instead of O(n*m))
    # For each v_time, find how many h_times fall before it
    h_counts_at_v = np.searchsorted(h_times, v_times)
    lines_per_field = np.diff(h_counts_at_v)

    duration = times[-1] - times[0] if len(times) > 0 else 0

    return TimingData(
        filename=filepath,
        label=label,
        h_periods=h_periods,
        v_periods=v_periods,
        h_times=h_times,
        v_times=v_times,
        lines_per_field=lines_per_field,
        duration=duration
    )


def trim_data(data: TimingData, skip_fields: int = 16) -> TimingData:
    """
    Trim data to remove transients at start/end.

    Removes the first and last `skip_fields` fields worth of data.
    """
    if skip_fields <= 0:
        return data

    n_fields = len(data.v_times)
    if n_fields <= 2 * skip_fields:
        print(f"Warning: Not enough fields to skip {skip_fields} at each end")
        return data

    # Find time boundaries
    start_time = data.v_times[skip_fields]
    end_time = data.v_times[-(skip_fields + 1)]

    # Trim H data
    h_mask = (data.h_times >= start_time) & (data.h_times <= end_time)
    h_times = data.h_times[h_mask]

    # Recalculate H periods from trimmed times
    h_periods = np.diff(h_times)

    # Trim V data
    v_times = data.v_times[skip_fields:-(skip_fields)]
    v_periods = np.diff(v_times)

    # Trim lines per field
    lines_per_field = data.lines_per_field[skip_fields:-(skip_fields)]

    return TimingData(
        filename=data.filename,
        label=data.label,
        h_periods=h_periods,
        v_periods=v_periods,
        h_times=h_times,
        v_times=v_times,
        lines_per_field=lines_per_field,
        duration=end_time - start_time
    )


def filter_half_lines(h_periods: np.ndarray,
                      nominal: float = NTSC_LINE_PERIOD,
                      tolerance: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate normal lines from half-lines.

    Half-lines occur at field boundaries in interlaced video.
    They are approximately half the normal line period.

    Returns:
        (normal_lines, half_lines)
    """
    half_threshold = nominal * (1 - tolerance)

    normal_mask = h_periods >= half_threshold
    half_mask = h_periods < half_threshold

    return h_periods[normal_mask], h_periods[half_mask]


def _points_in_intervals(times: np.ndarray,
                          intervals: List[ValidInterval]) -> np.ndarray:
    """Return boolean mask: True where time falls within any valid interval."""
    mask = np.zeros(len(times), dtype=bool)
    for iv in intervals:
        mask |= (times >= iv.start) & (times <= iv.end)
    return mask


def detect_signal_validity(data: TimingData) -> SignalValidity:
    """
    Detect regions of valid CVBS signal in the capture.

    When no valid signal is present, the LM1881/LM1980 sync separator
    oscillates rapidly, producing H periods of 1-25 µs and rapid V toggling.
    Valid CVBS has H periods near 63.556 µs (NTSC).

    Algorithm:
        1. Mark each H period as plausible if within [H_PERIOD_MIN, H_PERIOD_MAX]
        2. Sliding window: mark regions where >= 90% of H periods in any
           window of 200 are plausible (tolerates sparse bad pulses)
        3. Convert passing window regions to time intervals
        4. Merge intervals with gaps < VALIDITY_MERGE_GAP
        5. Remove intervals shorter than VALIDITY_MIN_DURATION
        6. Cross-validate V event rate within each interval
        7. Generate period-level boolean masks
    """
    total_duration = data.duration
    capture_start = data.h_times[0] if len(data.h_times) > 0 else 0.0

    if len(data.h_periods) == 0:
        return SignalValidity(
            valid_intervals=[],
            total_duration=total_duration,
            capture_start=capture_start,
            h_period_mask=np.array([], dtype=bool),
            v_period_mask=np.array([], dtype=bool),
        )

    # Step 1: H-period plausibility
    h_plausible = (data.h_periods >= H_PERIOD_MIN) & (data.h_periods <= H_PERIOD_MAX)

    # Step 2: Sliding-window density test
    # A window of W periods passes if >= threshold fraction are plausible.
    # This tolerates sparse bad pulses (e.g. 1-in-50 missed syncs on VHS)
    # that would break a consecutive-run requirement.
    W = H_VALIDITY_WINDOW
    n = len(h_plausible)

    if n < W:
        # Too few periods for even one window; check if mostly plausible
        if np.mean(h_plausible) >= H_VALIDITY_THRESHOLD:
            intervals = [ValidInterval(start=data.h_times[0],
                                       end=data.h_times[-1])]
        else:
            intervals = []
    else:
        cumsum = np.concatenate([[0], np.cumsum(h_plausible)])
        window_sums = cumsum[W:] - cumsum[:n - W + 1]  # length n-W+1
        passing = np.where(window_sums >= W * H_VALIDITY_THRESHOLD)[0]

        # Step 3: Convert passing windows to time intervals
        # Each passing window at index i covers periods [i, i+W).
        # Contiguous runs of passing windows form merged regions.
        intervals = []
        if len(passing) > 0:
            gaps = np.diff(passing)
            breaks = np.where(gaps > 1)[0]
            run_starts = np.concatenate([[0], breaks + 1])
            run_ends = np.concatenate([breaks + 1, [len(passing)]])

            for rs, re in zip(run_starts, run_ends):
                first = passing[rs]
                last = passing[re - 1]
                # Union of all periods covered: [first, last + W)
                # Corresponding times: h_times[first] to h_times[last + W]
                end_idx = min(last + W, len(data.h_times) - 1)
                intervals.append(ValidInterval(
                    start=data.h_times[first],
                    end=data.h_times[end_idx]
                ))

    if not intervals:
        return SignalValidity(
            valid_intervals=[],
            total_duration=total_duration,
            capture_start=capture_start,
            h_period_mask=np.zeros(len(data.h_periods), dtype=bool),
            v_period_mask=np.zeros(len(data.v_periods), dtype=bool),
        )

    # Step 4: Merge nearby intervals (VHS head-switch tolerance)
    merged = [ValidInterval(start=intervals[0].start, end=intervals[0].end)]
    for iv in intervals[1:]:
        if iv.start - merged[-1].end < VALIDITY_MERGE_GAP:
            merged[-1] = ValidInterval(start=merged[-1].start,
                                       end=max(merged[-1].end, iv.end))
        else:
            merged.append(ValidInterval(start=iv.start, end=iv.end))

    # Step 5: Remove short intervals
    merged = [iv for iv in merged if iv.duration >= VALIDITY_MIN_DURATION]

    if not merged:
        return SignalValidity(
            valid_intervals=[],
            total_duration=total_duration,
            capture_start=capture_start,
            h_period_mask=np.zeros(len(data.h_periods), dtype=bool),
            v_period_mask=np.zeros(len(data.v_periods), dtype=bool),
        )

    # Step 6: V-rate cross-validation
    validated = []
    for iv in merged:
        v_in_interval = np.sum(
            (data.v_times >= iv.start) & (data.v_times <= iv.end))
        if v_in_interval >= 2:
            v_rate = (v_in_interval - 1) / iv.duration
            if V_RATE_MIN <= v_rate <= V_RATE_MAX:
                validated.append(iv)
        elif iv.duration < 1.0 / V_RATE_MIN:
            # Interval too short to expect V events; keep if H-valid
            validated.append(iv)
    merged = validated

    # Step 7: Generate period-level masks
    h_time_valid = _points_in_intervals(data.h_times, merged)
    h_period_mask = h_time_valid[:-1] & h_time_valid[1:]

    v_time_valid = _points_in_intervals(data.v_times, merged)
    v_period_mask = v_time_valid[:-1] & v_time_valid[1:]

    return SignalValidity(
        valid_intervals=merged,
        total_duration=total_duration,
        capture_start=capture_start,
        h_period_mask=h_period_mask,
        v_period_mask=v_period_mask,
    )


def apply_validity_filter(data: TimingData,
                           validity: SignalValidity) -> TimingData:
    """
    Create a new TimingData containing only valid-signal periods.

    Filters h_periods/v_periods directly using masks (NOT via np.diff on
    filtered times) to avoid spurious large periods at segment boundaries
    when valid intervals are non-contiguous.
    """
    h_mask = validity.h_period_mask
    v_mask = validity.v_period_mask

    h_periods = data.h_periods[h_mask]
    v_periods = data.v_periods[v_mask]

    # Keep h_times where at least one adjacent period is valid
    h_time_keep = np.zeros(len(data.h_times), dtype=bool)
    if len(h_mask) > 0:
        h_time_keep[:-1] |= h_mask
        h_time_keep[1:] |= h_mask
    h_times = data.h_times[h_time_keep]

    v_time_keep = np.zeros(len(data.v_times), dtype=bool)
    if len(v_mask) > 0:
        v_time_keep[:-1] |= v_mask
        v_time_keep[1:] |= v_mask
    v_times = data.v_times[v_time_keep]

    # Recompute lines_per_field for valid data
    if len(v_times) >= 2 and len(h_times) > 0:
        h_counts_at_v = np.searchsorted(h_times, v_times)
        lines_per_field = np.diff(h_counts_at_v)
    else:
        lines_per_field = np.array([], dtype=int)

    return TimingData(
        filename=data.filename,
        label=data.label,
        h_periods=h_periods,
        v_periods=v_periods,
        h_times=h_times,
        v_times=v_times,
        lines_per_field=lines_per_field,
        duration=validity.valid_duration,
    )


def load_and_process_capture(filepath: str, label: str, sample_rate_mhz: float,
                              skip_fields: int, do_validity_filter: bool,
                              verbose: bool = True) -> CaptureResult:
    """
    Load, trim, filter, and compute stats for a single capture.

    This consolidates all processing steps into one function that returns
    a CaptureResult with all computed data.

    Args:
        filepath: Path to CSV file
        label: Display label for this capture
        sample_rate_mhz: Sample rate for timing quantization
        skip_fields: Fields to trim from start/end
        do_validity_filter: Whether to apply signal validity filtering
        verbose: Whether to print progress messages

    Returns:
        CaptureResult with all processed data and statistics
    """
    if verbose:
        print(f"Loading {filepath}...")

    # Parse data with timing quantization
    data = parse_timing_csv(filepath, label, sample_rate_mhz=sample_rate_mhz)
    if verbose:
        print(f"  {len(data.h_periods):,} line periods, {len(data.v_periods):,} field periods")

    # Trim data
    if skip_fields > 0:
        data = trim_data(data, skip_fields)
        if verbose:
            print(f"  After trim: {len(data.h_periods):,} lines, {len(data.v_periods):,} fields")

    # Signal validity detection
    raw_data = data
    if do_validity_filter:
        validity = detect_signal_validity(data)
        if verbose:
            print(f"  Signal validity: {validity.valid_duration:.2f}s "
                  f"({validity.valid_fraction*100:.1f}%), "
                  f"{validity.num_valid_regions} region(s)")
        if validity.num_valid_regions == 0:
            print(f"  WARNING: No valid signal detected in {label}")
        data = apply_validity_filter(data, validity)
    else:
        # Create a dummy validity that marks everything as valid
        validity = SignalValidity(
            valid_intervals=[ValidInterval(
                start=data.h_times[0] if len(data.h_times) > 0 else 0,
                end=data.h_times[-1] if len(data.h_times) > 0 else 0
            )] if len(data.h_times) > 0 else [],
            total_duration=data.duration,
            capture_start=data.h_times[0] if len(data.h_times) > 0 else 0,
            h_period_mask=np.ones(len(data.h_periods), dtype=bool),
            v_period_mask=np.ones(len(data.v_periods), dtype=bool),
        )

    # Calculate statistics on normal lines (filter out half-lines)
    h_normal, _ = filter_half_lines(data.h_periods)
    stats_h = calculate_stats(h_normal, NTSC_LINE_PERIOD)
    stats_v = calculate_stats(data.v_periods, NTSC_FIELD_PERIOD)

    # Calculate deviation metrics
    valid_fraction = validity.valid_fraction
    h_deviation = abs(stats_h.mean - NTSC_LINE_PERIOD) * 1e6  # µs
    v_deviation = abs(stats_v.mean - NTSC_FIELD_PERIOD) * 1e3  # ms

    return CaptureResult(
        label=label,
        filepath=filepath,
        data=data,
        raw_data=raw_data,
        validity=validity,
        stats_h=stats_h,
        stats_v=stats_v,
        valid_fraction=valid_fraction,
        h_deviation=h_deviation,
        v_deviation=v_deviation,
    )


def calculate_stats(periods: np.ndarray, nominal: float = None) -> TimingStats:
    """Calculate comprehensive timing statistics."""
    if len(periods) == 0:
        return TimingStats(
            mean=0, median=0, std=0, min=0, max=0,
            p1=0, p5=0, p25=0, p75=0, p95=0, p99=0,
            rms_jitter=0, pp_jitter=0, count=0
        )

    median = np.median(periods)
    deviations = periods - median

    return TimingStats(
        mean=np.mean(periods),
        median=median,
        std=np.std(periods),
        min=np.min(periods),
        max=np.max(periods),
        p1=np.percentile(periods, 1),
        p5=np.percentile(periods, 5),
        p25=np.percentile(periods, 25),
        p75=np.percentile(periods, 75),
        p95=np.percentile(periods, 95),
        p99=np.percentile(periods, 99),
        rms_jitter=np.sqrt(np.mean(deviations ** 2)),
        pp_jitter=np.max(periods) - np.min(periods),
        count=len(periods)
    )


def format_time(seconds: float, unit: str = 'auto') -> str:
    """Format time value with appropriate unit."""
    if unit == 'auto':
        if abs(seconds) < 1e-6:
            return f"{seconds * 1e9:.3f} ns"
        elif abs(seconds) < 1e-3:
            return f"{seconds * 1e6:.3f} µs"
        elif abs(seconds) < 1:
            return f"{seconds * 1e3:.3f} ms"
        else:
            return f"{seconds:.3f} s"
    elif unit == 'us':
        return f"{seconds * 1e6:.3f} µs"
    elif unit == 'ms':
        return f"{seconds * 1e3:.3f} ms"
    elif unit == 'ns':
        return f"{seconds * 1e9:.3f} ns"
    else:
        return f"{seconds:.6f} s"


def create_timing_heatmap(data: TimingData, output_path: str, title: str,
                          nominal_h: float = NTSC_LINE_PERIOD,
                          x_range: Tuple[float, float] = None,
                          sample_period_us: float = 0.05) -> Tuple[str, float, float]:
    """
    Create a 2D histogram heatmap of timing deviations by line number.

    Y-axis: Line number within field (1-262)
    X-axis: Timing deviation from nominal (µs)
    Color: Density of measurements at each line/deviation cell (darker = more frequent)

    Each row is a histogram of timing deviations for that line number.
    The entire heatmap is normalized globally, so lines with concentrated
    timing appear darker than lines with spread-out timing.

    Data outside the x_range is shown with blue edge indicators.

    Args:
        x_range: Optional (min, max) tuple in microseconds for consistent scaling
        sample_period_us: Sample period in µs, used as minimum histogram bin width

    Returns:
        Tuple of (output_path, clip_percentage, bin_width_ns) where:
        - clip_percentage is the fraction of data outside display range (0.0 to 100.0)
        - bin_width_ns is the histogram bin width in nanoseconds
    """
    if not HAS_MATPLOTLIB:
        return None, 0.0, 0.0

    # Get normal lines (filter out half-lines)
    normal_h, _ = filter_half_lines(data.h_periods, nominal_h)

    if len(normal_h) == 0:
        return None, 0.0, 0.0

    # Calculate deviations in microseconds
    deviations_us = (normal_h - nominal_h) * 1e6

    # Assign each measurement to a line number within field
    n_total = len(deviations_us)
    lines_in_field = 262
    line_numbers = np.arange(n_total) % lines_in_field + 1

    # Determine X-axis range from data if not provided (IQR-based)
    if x_range is None:
        q1 = np.percentile(deviations_us, 25)
        q3 = np.percentile(deviations_us, 75)
        iqr = q3 - q1
        x_min = q1 - 3 * iqr
        x_max = q3 + 3 * iqr
        min_half_range = 0.2
        center = (x_min + x_max) / 2
        half_range = max((x_max - x_min) / 2, min_half_range)
        x_range = (center - half_range * 1.1, center + half_range * 1.1)

    # Count clipped data
    n_clipped_low = np.sum(deviations_us < x_range[0])
    n_clipped_high = np.sum(deviations_us > x_range[1])
    n_clipped = n_clipped_low + n_clipped_high
    clip_pct = 100.0 * n_clipped / n_total if n_total > 0 else 0.0

    # Choose bin width: at least sample_period_us to avoid sparse columns
    # due to measurement quantization. Aim for ~50-100 bins across the range.
    range_width = x_range[1] - x_range[0]
    n_bins_max = int(range_width / sample_period_us)  # max bins at sample period width
    n_bins = max(10, min(100, n_bins_max))  # clamp to reasonable range
    bin_width_us = range_width / n_bins
    bin_width_ns = bin_width_us * 1000
    bin_edges = np.linspace(x_range[0], x_range[1], n_bins + 1)

    # Build 2D histogram: rows = line numbers (1-262), columns = deviation bins
    # Also track per-line clip counts for edge indicators
    heatmap = np.zeros((lines_in_field, n_bins))
    clipped_low_per_line = np.zeros(lines_in_field)
    clipped_high_per_line = np.zeros(lines_in_field)

    for line in range(1, lines_in_field + 1):
        mask = line_numbers == line
        if np.sum(mask) > 0:
            line_devs = deviations_us[mask]
            # Count clipped values per line
            clipped_low_per_line[line - 1] = np.sum(line_devs < x_range[0])
            clipped_high_per_line[line - 1] = np.sum(line_devs > x_range[1])
            # Only histogram the in-range values
            in_range = (line_devs >= x_range[0]) & (line_devs <= x_range[1])
            if np.sum(in_range) > 0:
                hist, _ = np.histogram(line_devs[in_range], bins=bin_edges)
                heatmap[line - 1, :] = hist

    # Global normalization: concentrated lines appear darker than spread-out lines
    global_max = heatmap.max()
    if global_max == 0:
        global_max = 1
    heatmap_norm = heatmap / global_max

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display as image: use 'Greys' colormap (white=0, black=1)
    # extent = [x_min, x_max, y_max, y_min] (y inverted so line 1 at top)
    im = ax.imshow(heatmap_norm, aspect='auto', cmap='Greys',
                   extent=[x_range[0], x_range[1], lines_in_field + 0.5, 0.5],
                   interpolation='nearest', vmin=0, vmax=1)

    # Show clipped data indicators as colored edge markers
    # Normalize clip counts for opacity
    max_clip_low = clipped_low_per_line.max() if clipped_low_per_line.max() > 0 else 1
    max_clip_high = clipped_high_per_line.max() if clipped_high_per_line.max() > 0 else 1

    edge_width = range_width * 0.015  # Width of edge indicator region
    for line in range(1, lines_in_field + 1):
        # Left edge (clipped low)
        if clipped_low_per_line[line - 1] > 0:
            alpha = 0.3 + 0.7 * (clipped_low_per_line[line - 1] / max_clip_low)
            ax.fill_betweenx([line - 0.5, line + 0.5],
                            x_range[0], x_range[0] + edge_width,
                            color='blue', alpha=alpha, linewidth=0)
        # Right edge (clipped high)
        if clipped_high_per_line[line - 1] > 0:
            alpha = 0.3 + 0.7 * (clipped_high_per_line[line - 1] / max_clip_high)
            ax.fill_betweenx([line - 0.5, line + 0.5],
                            x_range[1] - edge_width, x_range[1],
                            color='blue', alpha=alpha, linewidth=0)

    # Reference line at 0 deviation
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Labels and title
    ax.set_xlabel('Timing Deviation (µs)')
    ax.set_ylabel('Line Number in Field')
    ax.set_title(title)

    # Remove top and right spines (Tufte style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, clip_pct, bin_width_ns


def get_h_deviation_range(captures: List[CaptureResult],
                          nominal_h: float = NTSC_LINE_PERIOD) -> Optional[Tuple[float, float]]:
    """
    Calculate combined X-axis range for horizontal timing heatmaps.

    Uses IQR-based outlier detection to determine range: extends 3*IQR beyond
    Q1/Q3. This robustly handles heavy-tailed distributions like VHS timing
    where missed syncs create +63 µs spikes that would dominate percentile-based
    approaches.

    Ensures a minimum range so very stable signals still show detail.

    Args:
        captures: List of CaptureResult objects
        nominal_h: Nominal horizontal period in seconds
    """
    all_dev_arrays = []

    for cap in captures:
        h_normal, _ = filter_half_lines(cap.data.h_periods, nominal_h)
        if len(h_normal) > 0:
            dev_us = (h_normal - nominal_h) * 1e6
            all_dev_arrays.append(dev_us)

    if not all_dev_arrays:
        return None

    all_devs = np.concatenate(all_dev_arrays)

    # IQR-based range: Q1 - 3*IQR to Q3 + 3*IQR
    # This is robust against heavy tails (missed syncs, transients)
    q1 = np.percentile(all_devs, 25)
    q3 = np.percentile(all_devs, 75)
    iqr = q3 - q1
    x_min = q1 - 3 * iqr
    x_max = q3 + 3 * iqr

    # Ensure minimum range of ±0.2 µs for very stable signals
    min_half_range = 0.2
    center = (x_min + x_max) / 2
    half_range = max((x_max - x_min) / 2, min_half_range)
    x_min = center - half_range
    x_max = center + half_range

    # Add 10% padding
    padding = (x_max - x_min) * 0.1
    return (x_min - padding, x_max + padding)


def create_histogram_comparison(captures: List[CaptureResult],
                                output_path: str, timing_type: str = 'horizontal',
                                noise_floor_us: float = None) -> str:
    """
    Create overlaid histogram comparing timing distributions for all sources.

    Args:
        captures: List of CaptureResult objects
        output_path: Path to save the image
        timing_type: 'horizontal' or 'vertical'
        noise_floor_us: If provided, show shaded band around nominal indicating
                        measurement uncertainty
    """
    if not HAS_MATPLOTLIB:
        return None

    # Collect period data from all captures
    periods_list = []
    labels = []
    for cap in captures:
        if timing_type == 'horizontal':
            periods, _ = filter_half_lines(cap.data.h_periods)
        else:
            periods = cap.data.v_periods
        if len(periods) > 0:
            periods_list.append(periods)
            labels.append(cap.label)

    if not periods_list:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if timing_type == 'horizontal':
        nominal = NTSC_LINE_PERIOD
        xlabel = 'Line Period (µs)'
        title = 'Horizontal Timing Distribution'
        scale = 1e6
    else:
        nominal = NTSC_FIELD_PERIOD
        xlabel = 'Field Period (ms)'
        title = 'Vertical Timing Distribution'
        scale = 1e3

    nominal_plot = nominal * scale

    # Convert noise floor
    if noise_floor_us is not None:
        if timing_type == 'horizontal':
            noise_floor_plot = noise_floor_us
        else:
            noise_floor_plot = noise_floor_us / 1000
    else:
        noise_floor_plot = None

    # Convert to plot units
    periods_plot_list = [p * scale for p in periods_list]

    # Calculate bins from all data
    all_periods = np.concatenate(periods_plot_list)
    bins = np.linspace(np.percentile(all_periods, 0.5),
                       np.percentile(all_periods, 99.5), 100)

    # Add noise floor shading if provided
    if noise_floor_plot is not None:
        ax.axvspan(nominal_plot - noise_floor_plot, nominal_plot + noise_floor_plot,
                   color='yellow', alpha=0.2, zorder=0,
                   label='Measurement uncertainty')

    # Color cycle for multiple sources (tab20 supports up to 20)
    colors = plt.cm.tab20.colors

    # Plot histograms
    for i, (periods_plot, label) in enumerate(zip(periods_plot_list, labels)):
        ax.hist(periods_plot, bins=bins, alpha=0.5, label=label, density=True,
                color=colors[i % len(colors)])

    # Reference line
    ax.axvline(nominal_plot, color='red', linestyle='--', linewidth=1,
               label=f'Nominal ({nominal_plot:.3f})')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    # Tufte style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_field_stability_plot(captures: List[CaptureResult],
                                output_path: str,
                                noise_floor_us: float = None) -> str:
    """
    Plot field-to-field timing stability over time with consistent Y-axis.

    Args:
        captures: List of CaptureResult objects
        output_path: Path to save the image
        noise_floor_us: If provided, show shaded band indicating measurement uncertainty
    """
    if not HAS_MATPLOTLIB:
        return None

    # Filter to captures with V data
    valid_captures = [cap for cap in captures
                      if len(cap.data.v_periods) > 0]

    if not valid_captures:
        return None

    n = len(valid_captures)
    # Dynamic figure height: ~3 inches per capture
    fig_height = max(6, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=(12, fig_height))

    # Handle single capture case (axes won't be array)
    if n == 1:
        axes = [axes]

    # Calculate shared Y-axis limits using all datasets
    all_devs_list = []
    for cap in valid_captures:
        dev = (cap.data.v_periods - NTSC_FIELD_PERIOD) * 1e6  # µs
        all_devs_list.append(dev)

    all_devs = np.concatenate(all_devs_list)
    y_min = np.percentile(all_devs, 0.5)
    y_max = np.percentile(all_devs, 99.5)
    padding = (y_max - y_min) * 0.1
    y_limits = (y_min - padding, y_max + padding)

    # Color cycle (tab20 supports up to 20)
    colors = plt.cm.tab20.colors

    for i, (ax, cap) in enumerate(zip(axes, valid_captures)):
        dev = all_devs_list[i]
        fields = np.arange(len(dev))

        # Add noise floor shading if provided
        if noise_floor_us is not None:
            ax.axhspan(-noise_floor_us, noise_floor_us,
                       color='yellow', alpha=0.15, zorder=0,
                       label='Measurement uncertainty')

        ax.plot(fields, dev, color=colors[i % len(colors)],
                linewidth=0.5, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Field Period Deviation (µs)')
        ax.set_title(f'{cap.label} - Field Timing Stability')
        ax.set_ylim(y_limits)
        ax.set_xlabel('Field Number')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def create_signal_timeline(captures: List[CaptureResult],
                            output_path: str) -> str:
    """
    Create a timeline visualization showing valid/invalid signal regions.

    N horizontal bars (one per capture) with green for valid signal
    and light red for invalid/absent signal.
    """
    if not HAS_MATPLOTLIB:
        return None

    n = len(captures)
    # Dynamic figure height: ~0.8 inches per capture, minimum 2 inches
    fig_height = max(2.0, 0.8 * n + 0.5)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_height),
                              gridspec_kw={'hspace': 0.6})

    # Handle single capture case (axes won't be array)
    if n == 1:
        axes = [axes]

    # Find max duration for consistent x-axis
    max_duration = max(cap.validity.total_duration for cap in captures)

    for ax, cap in zip(axes, captures):
        validity = cap.validity
        total = validity.total_duration
        t0 = validity.capture_start

        # Background: invalid (light red)
        ax.barh(0, total, height=0.6, color='#ffcccc', edgecolor='#ddaaaa',
                left=0, linewidth=0.5)

        # Valid regions (green), converted to relative time
        for iv in validity.valid_intervals:
            ax.barh(0, iv.duration, height=0.6, color='#88cc88',
                    edgecolor='#66aa66', left=iv.start - t0, linewidth=0.5)

        ax.set_xlim(0, max_duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(cap.label, fontsize=8, rotation=0, ha='right', va='center')

        # Tufte style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Signal Validity Timeline', fontsize=11, y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def title_from_filename(output_path: str) -> str:
    """
    Generate a human-readable title from the output filename.

    Strips 'timing_report' prefix, replaces underscores/hyphens with spaces,
    and title-cases the result.

    Examples:
        timing_report_tbc_comparison.html -> "TBC Comparison"
        timing_report_rp_playback_all.html -> "RP Playback All"
        timing_report_all_tbc.html -> "All TBC"
    """
    stem = Path(output_path).stem
    # Remove common prefixes
    for prefix in ['timing_report_', 'timing_report-', 'report_', 'report-']:
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break
    # Replace separators with spaces and title-case
    title = stem.replace('_', ' ').replace('-', ' ')
    return title.title()


def generate_html_report(captures: List[CaptureResult],
                         output_path: str,
                         image_dir: str,
                         skip_fields: int,
                         sample_rate_mhz: float = 20.0,
                         reference_idx: Optional[int] = None,
                         ranked_indices: List[int] = None,
                         title: str = None,
                         command_line: str = None) -> str:
    """
    Generate comprehensive HTML report for N sources.

    Args:
        captures: List of CaptureResult objects
        output_path: Path for output HTML file
        image_dir: Directory for generated images
        skip_fields: Number of fields skipped at start/end
        sample_rate_mhz: Logic analyzer sample rate
        reference_idx: Index of reference capture (None = stack ranking mode)
        ranked_indices: List of capture indices sorted by rank
        title: Report title (default: derived from output filename)
        command_line: Command line used to generate this report
    """
    # Resolve title
    if title is None:
        title = title_from_filename(output_path)
    report_title = f"Video Timing Report: {title}"

    # Create image directory
    os.makedirs(image_dir, exist_ok=True)

    n = len(captures)

    # Calculate noise floor for measurement uncertainty display
    noise_floor_ns = get_noise_floor_ns(sample_rate_mhz)
    noise_floor_us = noise_floor_ns / 1000.0  # Convert to µs for charts

    # Generate visualizations
    images = {}
    heatmap_clips = {}  # {index: clip_pct}
    bin_width_ns = 0.0

    if HAS_MATPLOTLIB:
        # Signal validity timeline
        img_path = os.path.join(image_dir, 'signal_timeline.png')
        if create_signal_timeline(captures, img_path):
            images['signal_timeline'] = os.path.basename(img_path)

        # Calculate shared X-axis range for heatmaps
        h_range = get_h_deviation_range(captures)
        sample_period_us = 1.0 / sample_rate_mhz

        # Generate heatmaps for all captures
        for i, cap in enumerate(captures):
            img_path = os.path.join(image_dir, f'heatmap_{i+1}.png')
            result = create_timing_heatmap(
                cap.data, img_path,
                f'{cap.label} - Line Timing Distribution',
                x_range=h_range, sample_period_us=sample_period_us
            )
            if result[0]:
                images[f'heatmap_{i+1}'] = os.path.basename(img_path)
                heatmap_clips[i] = result[1]
                if bin_width_ns == 0.0:
                    bin_width_ns = result[2]

        # Histogram comparisons with noise floor shading
        img_path = os.path.join(image_dir, 'hist_horizontal.png')
        if create_histogram_comparison(captures, img_path, 'horizontal',
                                        noise_floor_us=noise_floor_us):
            images['hist_h'] = os.path.basename(img_path)

        img_path = os.path.join(image_dir, 'hist_vertical.png')
        if create_histogram_comparison(captures, img_path, 'vertical',
                                        noise_floor_us=noise_floor_us):
            images['hist_v'] = os.path.basename(img_path)

        # Field stability with noise floor shading
        img_path = os.path.join(image_dir, 'field_stability.png')
        if create_field_stability_plot(captures, img_path,
                                         noise_floor_us=noise_floor_us):
            images['field_stability'] = os.path.basename(img_path)

    # Get relative path for images
    output_dir = os.path.dirname(output_path)
    rel_image_dir = os.path.relpath(image_dir, output_dir) if output_dir else os.path.basename(image_dir)

    # Determine validity filtering status
    validity_filtering = captures[0].validity.num_valid_regions > 0 or captures[0].valid_fraction < 1.0
    validity_note = " Signal validity filtering applied." if validity_filtering else " Signal validity filtering disabled."

    # Build captures summary
    captures_summary = ""
    for i, cap in enumerate(captures):
        ref_marker = " (reference)" if i == reference_idx else ""
        captures_summary += f"        <strong>{i+1}.</strong> {cap.label} ({cap.data.duration:.2f}s, {cap.stats_h.count:,} lines, {cap.stats_v.count:,} fields){ref_marker}<br>\n"

    # Build summary ranking table
    ranking_rows = ""
    for rank, idx in enumerate(ranked_indices, 1):
        cap = captures[idx]
        ref_class = " style='background-color: #f0f8ff;'" if idx == reference_idx else ""
        ranking_rows += f"""        <tr{ref_class}>
            <td>{rank}</td>
            <td>{cap.label}</td>
            <td>{cap.valid_fraction*100:.1f}%</td>
            <td>{cap.h_deviation:.4f} µs</td>
            <td>{cap.v_deviation:.4f} ms</td>
        </tr>
"""

    # Build signal presence table with N columns
    signal_presence_headers = "".join(f"<th>{cap.label}</th>" for cap in captures)

    def signal_presence_row(metric_name, values):
        cells = "".join(f"<td>{v}</td>" for v in values)
        return f"        <tr><td>{metric_name}</td>{cells}</tr>\n"

    signal_presence_rows = ""
    signal_presence_rows += signal_presence_row(
        "Total Duration (after trim)",
        [f"{cap.validity.total_duration:.2f} s" for cap in captures]
    )
    signal_presence_rows += signal_presence_row(
        "Valid Signal Duration",
        [f"{cap.validity.valid_duration:.2f} s ({cap.valid_fraction*100:.1f}%)" for cap in captures]
    )
    signal_presence_rows += signal_presence_row(
        "Invalid/Absent Signal",
        [f"{cap.validity.invalid_duration:.2f} s" for cap in captures]
    )
    signal_presence_rows += signal_presence_row(
        "Valid Regions",
        [str(cap.validity.num_valid_regions) for cap in captures]
    )
    signal_presence_rows += signal_presence_row(
        "H Periods (raw → filtered)",
        [f"{len(cap.raw_data.h_periods):,} → {len(cap.data.h_periods):,}" for cap in captures]
    )
    signal_presence_rows += signal_presence_row(
        "V Periods (raw → filtered)",
        [f"{len(cap.raw_data.v_periods):,} → {len(cap.data.v_periods):,}" for cap in captures]
    )

    # Build interval details
    interval_rows = ""
    for cap in captures:
        t0 = cap.validity.capture_start
        for i, iv in enumerate(cap.validity.valid_intervals, 1):
            interval_rows += (
                f"        <tr><td>{cap.label}</td><td>{i}</td>"
                f"<td>{iv.start - t0:.3f} s</td><td>{iv.end - t0:.3f} s</td>"
                f"<td>{iv.duration:.3f} s</td></tr>\n"
            )

    timeline_img = ""
    if 'signal_timeline' in images:
        timeline_img = (
            f"    <div class='image-container'><img src='"
            f"{rel_image_dir}/{images['signal_timeline']}' "
            f"alt='Signal Validity Timeline'></div>"
        )

    # Build horizontal timing table with N columns
    h_headers = "".join(f"<th>{cap.label}</th>" for cap in captures)

    def h_timing_row(metric_name, format_fn, nominal=None):
        cells = "".join(f"<td>{format_fn(cap.stats_h)}</td>" for cap in captures)
        nom_cell = f"<td>{nominal}</td>" if nominal else "<td>—</td>"
        return f"        <tr><td>{metric_name}</td>{cells}{nom_cell}</tr>\n"

    h_timing_rows = ""
    h_timing_rows += h_timing_row("Median", lambda s: f"{s.median*1e6:.4f} µs", f"{NTSC_LINE_PERIOD*1e6:.4f} µs")
    h_timing_rows += h_timing_row("Mean", lambda s: f"{s.mean*1e6:.4f} µs")
    h_timing_rows += h_timing_row("Std Dev", lambda s: format_jitter_with_threshold(s.std*1e9, noise_floor_ns))
    h_timing_rows += h_timing_row("RMS Jitter", lambda s: format_jitter_with_threshold(s.rms_jitter*1e9, noise_floor_ns))
    h_timing_rows += h_timing_row("Peak-to-Peak Jitter", lambda s: format_jitter_with_threshold(s.pp_jitter*1e9, noise_floor_ns * 2))
    h_timing_rows += h_timing_row("Min", lambda s: f"{s.min*1e6:.4f} µs")
    h_timing_rows += h_timing_row("Max", lambda s: f"{s.max*1e6:.4f} µs")
    h_timing_rows += h_timing_row("1st Percentile", lambda s: f"{s.p1*1e6:.4f} µs")
    h_timing_rows += h_timing_row("5th Percentile", lambda s: f"{s.p5*1e6:.4f} µs")
    h_timing_rows += h_timing_row("25th Percentile", lambda s: f"{s.p25*1e6:.4f} µs")
    h_timing_rows += h_timing_row("50th Percentile", lambda s: f"{s.median*1e6:.4f} µs")
    h_timing_rows += h_timing_row("75th Percentile", lambda s: f"{s.p75*1e6:.4f} µs")
    h_timing_rows += h_timing_row("95th Percentile", lambda s: f"{s.p95*1e6:.4f} µs")
    h_timing_rows += h_timing_row("99th Percentile", lambda s: f"{s.p99*1e6:.4f} µs")
    h_timing_rows += h_timing_row("Sample Count", lambda s: f"{s.count:,}")

    # Build heatmap grid (2-3 per row)
    heatmap_html = ""
    cols_per_row = 2 if n <= 4 else 3
    for i, cap in enumerate(captures):
        key = f'heatmap_{i+1}'
        if key in images:
            heatmap_html += f"        <div class='image-container'><img src='{rel_image_dir}/{images[key]}' alt='{cap.label} Heatmap'></div>\n"

    # Build clip percentages message
    clip_msgs = []
    for i, cap in enumerate(captures):
        if i in heatmap_clips and heatmap_clips[i] > 0.1:
            clip_msgs.append(f"{cap.label}: {heatmap_clips[i]:.1f}%")
    clip_msg_html = f"<p><em>Outliers beyond display range: {', '.join(clip_msgs)}</em></p>" if clip_msgs else ""

    # Build vertical timing table with N columns
    v_headers = "".join(f"<th>{cap.label}</th>" for cap in captures)

    def v_timing_row(metric_name, format_fn, nominal=None):
        cells = "".join(f"<td>{format_fn(cap.stats_v)}</td>" for cap in captures)
        nom_cell = f"<td>{nominal}</td>" if nominal else "<td>—</td>"
        return f"        <tr><td>{metric_name}</td>{cells}{nom_cell}</tr>\n"

    v_timing_rows = ""
    v_timing_rows += v_timing_row("Median", lambda s: f"{s.median*1e3:.4f} ms", f"{NTSC_FIELD_PERIOD*1e3:.4f} ms")
    v_timing_rows += v_timing_row("Mean", lambda s: f"{s.mean*1e3:.4f} ms")
    v_timing_rows += v_timing_row("Std Dev", lambda s: format_jitter_with_threshold(s.std*1e9, noise_floor_ns, unit='us'))
    v_timing_rows += v_timing_row("RMS Jitter", lambda s: format_jitter_with_threshold(s.rms_jitter*1e9, noise_floor_ns, unit='us'))
    v_timing_rows += v_timing_row("Peak-to-Peak Jitter", lambda s: format_jitter_with_threshold(s.pp_jitter*1e9, noise_floor_ns * 2, unit='us'))
    v_timing_rows += v_timing_row("Min", lambda s: f"{s.min*1e3:.4f} ms")
    v_timing_rows += v_timing_row("Max", lambda s: f"{s.max*1e3:.4f} ms")
    v_timing_rows += v_timing_row("1st Percentile", lambda s: f"{s.p1*1e3:.4f} ms")
    v_timing_rows += v_timing_row("5th Percentile", lambda s: f"{s.p5*1e3:.4f} ms")
    v_timing_rows += v_timing_row("25th Percentile", lambda s: f"{s.p25*1e3:.4f} ms")
    v_timing_rows += v_timing_row("50th Percentile", lambda s: f"{s.median*1e3:.4f} ms")
    v_timing_rows += v_timing_row("75th Percentile", lambda s: f"{s.p75*1e3:.4f} ms")
    v_timing_rows += v_timing_row("95th Percentile", lambda s: f"{s.p95*1e3:.4f} ms")
    v_timing_rows += v_timing_row("99th Percentile", lambda s: f"{s.p99*1e3:.4f} ms")
    v_timing_rows += v_timing_row("Sample Count", lambda s: f"{s.count:,}")

    # Build lines per field table
    lpf_headers = "".join(f"<th>{cap.label}</th>" for cap in captures)

    def lpf_row(metric_name, format_fn):
        cells = "".join(f"<td>{format_fn(cap)}</td>" for cap in captures)
        return f"        <tr><td>{metric_name}</td>{cells}</tr>\n"

    lpf_rows = ""
    lpf_rows += lpf_row("Mean Lines/Field", lambda c: f"{np.mean(c.data.lines_per_field):.2f}" if len(c.data.lines_per_field) > 0 else "N/A")
    lpf_rows += lpf_row("Min", lambda c: str(np.min(c.data.lines_per_field)) if len(c.data.lines_per_field) > 0 else "N/A")
    lpf_rows += lpf_row("Max", lambda c: str(np.max(c.data.lines_per_field)) if len(c.data.lines_per_field) > 0 else "N/A")

    # Build horizontal ranking appendix (sorted by H deviation)
    h_ranked = sorted(range(n), key=lambda i: captures[i].h_deviation)
    h_ranking_rows = ""
    for rank, idx in enumerate(h_ranked, 1):
        cap = captures[idx]
        ref_class = " style='background-color: #f0f8ff;'" if idx == reference_idx else ""
        h_ranking_rows += f"""        <tr{ref_class}>
            <td>{rank}</td>
            <td>{cap.label}</td>
            <td>{cap.h_deviation:.4f} µs</td>
            <td>{format_jitter_with_threshold(cap.stats_h.rms_jitter*1e9, noise_floor_ns)}</td>
        </tr>
"""

    # Build vertical ranking appendix (sorted by V deviation)
    v_ranked = sorted(range(n), key=lambda i: captures[i].v_deviation)
    v_ranking_rows = ""
    for rank, idx in enumerate(v_ranked, 1):
        cap = captures[idx]
        ref_class = " style='background-color: #f0f8ff;'" if idx == reference_idx else ""
        v_ranking_rows += f"""        <tr{ref_class}>
            <td>{rank}</td>
            <td>{cap.label}</td>
            <td>{cap.v_deviation:.4f} ms</td>
            <td>{format_jitter_with_threshold(cap.stats_v.rms_jitter*1e9, noise_floor_ns, unit='us')}</td>
        </tr>
"""

    # Mode indicator
    mode_desc = "Reference Mode" if reference_idx is not None else "Stack Ranking Mode"

    # Build command-line reproduction appendix
    if command_line is not None:
        import html as html_module
        escaped_cmd = html_module.escape(command_line)
        file_list_rows = ""
        for i, cap in enumerate(captures):
            abs_path = os.path.abspath(cap.filepath)
            ref_note = " (reference)" if i == reference_idx else ""
            file_list_rows += (
                f"        <tr><td>{cap.label}{ref_note}</td>"
                f"<td><code>{html_module.escape(abs_path)}</code></td></tr>\n"
            )
        command_line_appendix = f"""<h2>Appendix C: Reproduction</h2>

    <p>Command used to generate this report:</p>
    <pre style="background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px;"><code>{escaped_cmd}</code></pre>

    <p>Source files:</p>
    <table>
        <tr><th>Label</th><th>File Path</th></tr>
{file_list_rows}    </table>"""
    else:
        command_line_appendix = ""

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: right;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: 600;
        }}
        td:first-child, th:first-child {{
            text-align: left;
        }}
        .metric-good {{ color: #27ae60; }}
        .metric-warning {{ color: #f39c12; }}
        .metric-bad {{ color: #e74c3c; }}
        .summary-box {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .ranking-box {{
            background: #fff9e6;
            border-left: 4px solid #f4c430;
            padding: 15px;
            margin: 20px 0;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat({cols_per_row}, 1fr);
            gap: 20px;
        }}
        @media (max-width: 800px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .footnote {{
            font-size: 0.9em;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
        details {{
            margin: 10px 0;
        }}
        summary {{
            cursor: pointer;
            font-weight: bold;
            padding: 5px;
        }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>

    <div class="summary-box">
        <strong>Mode:</strong> {mode_desc}<br>
        <strong>Captures Compared ({n}):</strong><br>
{captures_summary}        <br>
        <strong>Analysis Settings:</strong> First and last {skip_fields} fields trimmed from each capture.{validity_note}
    </div>

    <h2>Summary Ranking</h2>

    <div class="ranking-box">
        <p>Sources ranked by valid signal proportion (higher is better), with H deviation from nominal as first tiebreaker (lower is better) and V deviation as second tiebreaker (lower is better).</p>
    </div>

    <table>
        <tr>
            <th>Rank</th>
            <th>Source</th>
            <th>Valid Signal %</th>
            <th>H Deviation</th>
            <th>V Deviation</th>
        </tr>
{ranking_rows}    </table>

    <h2>Signal Presence</h2>

    <div class="summary-box">
        <strong>Signal validity detection</strong> identifies regions where the sync separator
        is receiving a valid CVBS signal vs. free-running oscillation (no signal input).
        Only valid-signal regions are included in timing statistics below.
    </div>

    <table>
        <tr>
            <th>Metric</th>
            {signal_presence_headers}
        </tr>
{signal_presence_rows}    </table>

    {timeline_img}

    <details>
    <summary>Valid Signal Intervals</summary>
    <table>
        <tr><th>Capture</th><th>#</th><th>Start</th><th>End</th><th>Duration</th></tr>
{interval_rows}    </table>
    </details>

    <h2>Horizontal Timing (Line Period)</h2>

    <p>NTSC nominal line period: <strong>{NTSC_LINE_PERIOD*1e6:.4f} µs</strong> ({NTSC_LINE_FREQ:.3f} Hz)</p>

    <table>
        <tr>
            <th>Metric</th>
            {h_headers}
            <th>NTSC Nominal</th>
        </tr>
{h_timing_rows}    </table>

    <p><em>* Values marked with asterisk are below the measurement noise floor (~{noise_floor_ns:.0f} ns) and cannot be reliably distinguished from zero.</em></p>

    {"<h3>Horizontal Timing Distribution</h3>" if 'hist_h' in images else ""}
    {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('hist_h', '') + "' alt='Horizontal Histogram'></div>" if 'hist_h' in images else ""}

    <h3>Line-by-Line Timing Distribution</h3>
    <p>Each row shows the timing distribution for that line number within a field as a histogram
    (bin width: {bin_width_ns:.0f} ns, sample period: {1000.0/sample_rate_mhz:.0f} ns).
    Darker regions indicate more measurements at that deviation.
    The red dashed line indicates zero deviation from nominal. Blue edge shading indicates clipped outliers.</p>

    <div class="comparison-grid">
{heatmap_html}    </div>
    {clip_msg_html}

    <h2>Vertical Timing (Field Period)</h2>

    <p>NTSC nominal field period: <strong>{NTSC_FIELD_PERIOD*1e3:.4f} ms</strong> ({NTSC_FIELD_FREQ:.2f} Hz)</p>

    <table>
        <tr>
            <th>Metric</th>
            {v_headers}
            <th>NTSC Nominal</th>
        </tr>
{v_timing_rows}    </table>

    <p><em>* Values marked with asterisk are below the measurement noise floor (~{noise_floor_ns:.0f} ns) and cannot be reliably distinguished from zero.</em></p>

    {"<h3>Vertical Timing Distribution</h3>" if 'hist_v' in images else ""}
    {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('hist_v', '') + "' alt='Vertical Histogram'></div>" if 'hist_v' in images else ""}

    {"<h3>Field-to-Field Stability Over Time</h3>" if 'field_stability' in images else ""}
    {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('field_stability', '') + "' alt='Field Stability'></div>" if 'field_stability' in images else ""}

    <h2>Lines Per Field</h2>

    <p>NTSC nominal: <strong>262.5 lines</strong> per field (alternating 262 and 263)</p>

    <table>
        <tr>
            <th>Metric</th>
            {lpf_headers}
        </tr>
{lpf_rows}    </table>

    <h2>Measurement Uncertainty</h2>

    <p>Logic analyzer sample rate: <strong>{sample_rate_mhz:.1f} MS/s</strong></p>

    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Notes</th>
        </tr>
        <tr>
            <td>Sample Period</td>
            <td>{1000.0/sample_rate_mhz:.1f} ns</td>
            <td>Minimum time resolution between edges</td>
        </tr>
        <tr>
            <td>Timing Uncertainty (1σ)</td>
            <td>±{1000.0/sample_rate_mhz/2:.1f} ns</td>
            <td>Each edge can be off by ±½ sample period</td>
        </tr>
        <tr>
            <td>Period Uncertainty (1σ)</td>
            <td>±{1000.0/sample_rate_mhz/2*1.414:.1f} ns</td>
            <td>Two edges: √2 × single edge uncertainty</td>
        </tr>
        <tr>
            <td>Quantization Noise Floor</td>
            <td>{1000.0/sample_rate_mhz/np.sqrt(12):.1f} ns RMS</td>
            <td>Uniform distribution: σ = Δt/√12</td>
        </tr>
    </table>

    <p><em>Jitter values below ~{noise_floor_ns:.0f} ns may be dominated by measurement quantization rather than actual source jitter.</em></p>

    <h2>Appendix A: Horizontal Timing Rankings</h2>

    <p>Sources ranked by mean H period deviation from NTSC nominal (lower is better, indicating more accurate timing).</p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Source</th>
            <th>Mean H Deviation</th>
            <th>H RMS Jitter</th>
        </tr>
{h_ranking_rows}    </table>

    <h2>Appendix B: Vertical Timing Rankings</h2>

    <p>Sources ranked by mean V period deviation from NTSC nominal (lower is better, indicating more accurate timing).</p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Source</th>
            <th>Mean V Deviation</th>
            <th>V RMS Jitter</th>
        </tr>
{v_ranking_rows}    </table>

    {command_line_appendix}

    <div class="footnote">
        <p><strong>Methodology Notes:</strong></p>
        <ul>
            <li>Horizontal timing calculated from HSYNC pulse intervals (Channel 0, 1→0 transitions)</li>
            <li>Vertical timing calculated from VSYNC pulse intervals (Channel 1, 1→0 transitions)</li>
            <li>Half-lines at field boundaries are excluded from horizontal statistics</li>
            <li>First and last {skip_fields} fields trimmed to avoid transient effects</li>
            <li>RMS jitter calculated as deviation from median (not nominal)</li>
            <li>Sample rate: {sample_rate_mhz:.1f} MS/s ({1000.0/sample_rate_mhz:.1f} ns resolution)</li>
            <li>Timing values quantized to sample period to eliminate false precision</li>
            <li><strong>*</strong> Values marked with asterisk are below the measurement noise floor (~{noise_floor_ns:.0f} ns) and cannot be reliably distinguished from zero</li>
            <li>Signal validity detection excludes sync separator free-run oscillation (H periods outside {H_PERIOD_MIN*1e6:.0f}–{H_PERIOD_MAX*1e6:.0f} µs, sliding window of {H_VALIDITY_WINDOW} periods with {H_VALIDITY_THRESHOLD*100:.0f}% plausibility threshold, gaps &lt;{VALIDITY_MERGE_GAP*1e3:.1f} ms merged)</li>
        </ul>
        <p>Generated by <code>generate_timing_report.py</code></p>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def rank_captures(captures: List[CaptureResult],
                  reference_idx: Optional[int] = None) -> List[int]:
    """
    Rank captures by signal validity with tiebreakers.

    Primary: valid signal % (descending - higher is better)
    Tiebreaker 1: H deviation from nominal (ascending - lower is better)
    Tiebreaker 2: V deviation from nominal (ascending - lower is better)

    Args:
        captures: List of CaptureResult objects
        reference_idx: If provided, this capture is excluded from ranking

    Returns:
        List of indices sorted by rank (best first)
    """
    # Build list of (index, valid_fraction, h_deviation, v_deviation) for ranking
    candidates = []
    for i, cap in enumerate(captures):
        if reference_idx is not None and i == reference_idx:
            continue
        candidates.append((i, cap.valid_fraction, cap.h_deviation, cap.v_deviation))

    # Sort by valid_fraction descending, then h_deviation ascending, then v_deviation ascending.
    # Round valid_fraction to 0.1% (matching display precision) so that floating-point
    # noise in the 6th+ decimal place doesn't defeat the tiebreakers.
    candidates.sort(key=lambda x: (-round(x[1], 3), x[2], x[3]))

    return [idx for idx, _, _, _ in candidates]


def main():
    parser = argparse.ArgumentParser(
        description='Generate video timing comparison report for N sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Stack ranking mode (default): compare N files, ranked by timing stability
    %(prog)s file1.csv file2.csv file3.csv -o report.html

    # Reference mode: compare against a reference
    %(prog)s --reference ref.csv file1.csv file2.csv -o report.html

    # With custom labels
    %(prog)s --labels "Leitch,RP1,RP2" file1.csv file2.csv file3.csv -o report.html

    # Backward compatible: 2 sources with old-style arguments
    %(prog)s -1 ref.csv -2 dut.csv -o report.html

    # Disable validity filtering
    %(prog)s file1.csv file2.csv -o report.html --no-validity-filter
        """
    )

    # New positional arguments for N sources
    parser.add_argument('captures', nargs='*', metavar='FILE',
                        help='Capture files to compare (at least 2)')

    # Reference mode
    parser.add_argument('--reference', '-r', metavar='FILE',
                        help='Reference capture file (compare others against this)')

    # Labels
    parser.add_argument('--labels', '-l', metavar='LABELS',
                        help='Comma-separated labels for captures '
                             '(in order: reference first if specified, then positional files)')

    # Legacy arguments for backward compatibility
    parser.add_argument('--capture1', '-1', metavar='FILE',
                        help='(Legacy) First capture file')
    parser.add_argument('--capture2', '-2', metavar='FILE',
                        help='(Legacy) Second capture file')
    parser.add_argument('--label1', metavar='LABEL',
                        help='(Legacy) Label for capture 1')
    parser.add_argument('--label2', metavar='LABEL',
                        help='(Legacy) Label for capture 2')

    # Common options
    parser.add_argument('--output', '-o', required=True,
                        help='Output HTML report path')
    parser.add_argument('--title', '-t', metavar='TITLE',
                        help='Report title (default: derived from output filename)')
    parser.add_argument('--skip-fields', type=int, default=16,
                        help='Fields to skip at start/end (default: 16)')
    parser.add_argument('--sample-rate', type=float, default=20.0,
                        help='Logic analyzer sample rate in MS/s (default: 20.0)')
    parser.add_argument('--no-validity-filter', action='store_true',
                        help='Disable signal validity detection and include all data')

    args = parser.parse_args()

    # Resolve capture files from new or legacy arguments
    capture_files = []
    labels = []
    reference_idx = None

    if args.capture1 and args.capture2:
        # Legacy mode: -1 and -2 arguments
        capture_files = [args.capture1, args.capture2]
        labels = [args.label1, args.label2]
    elif args.captures:
        # New mode: positional arguments
        if args.reference:
            capture_files = [args.reference] + args.captures
            reference_idx = 0
        else:
            capture_files = args.captures

        # Parse labels
        if args.labels:
            labels = [l.strip() for l in args.labels.split(',')]
        else:
            labels = [None] * len(capture_files)
    else:
        print("Error: Must provide at least 2 capture files")
        print("  Use positional arguments: file1.csv file2.csv ...")
        print("  Or legacy arguments: -1 file1.csv -2 file2.csv")
        sys.exit(1)

    # Validate we have at least 2 files
    if len(capture_files) < 2:
        print("Error: Must provide at least 2 capture files")
        sys.exit(1)

    # Pad or truncate labels to match file count
    while len(labels) < len(capture_files):
        labels.append(None)
    labels = labels[:len(capture_files)]

    # Set default labels from filenames
    for i, (filepath, label) in enumerate(zip(capture_files, labels)):
        if label is None:
            labels[i] = Path(filepath).stem

    # Validate all files exist
    for filepath in capture_files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)

    # Load and process all captures
    print(f"Processing {len(capture_files)} capture files...")
    print(f"  Sample rate: {args.sample_rate} MS/s")
    print(f"  Skip fields: {args.skip_fields}")
    print(f"  Validity filter: {'disabled' if args.no_validity_filter else 'enabled'}")
    print()

    captures: List[CaptureResult] = []
    for filepath, label in zip(capture_files, labels):
        cap = load_and_process_capture(
            filepath=filepath,
            label=label,
            sample_rate_mhz=args.sample_rate,
            skip_fields=args.skip_fields,
            do_validity_filter=not args.no_validity_filter,
        )
        captures.append(cap)
        print()  # Blank line between captures

    # Rank captures
    ranked_indices = rank_captures(captures, reference_idx)

    # Print ranking summary
    print("Ranking (by valid signal %, then H deviation, then V deviation):")
    for rank, idx in enumerate(ranked_indices, 1):
        cap = captures[idx]
        ref_marker = " (reference)" if idx == reference_idx else ""
        print(f"  {rank}. {cap.label}: {cap.valid_fraction*100:.1f}%, "
              f"H dev={cap.h_deviation:.4f} µs, V dev={cap.v_deviation:.4f} ms{ref_marker}")
    print()

    # Generate report
    output_path = args.output
    output_dir = os.path.dirname(output_path) or '.'
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    image_dir = os.path.join(output_dir, f"{base_name}_images")

    # Reconstruct command line for reproduction appendix
    import shlex
    command_line = ' '.join(shlex.quote(a) for a in sys.argv)

    # Resolve title
    report_title = args.title  # None means auto-derive from filename

    print("Generating report...")
    os.makedirs(output_dir, exist_ok=True)

    generate_html_report(
        captures=captures,
        output_path=output_path,
        image_dir=image_dir,
        skip_fields=args.skip_fields,
        sample_rate_mhz=args.sample_rate,
        reference_idx=reference_idx,
        ranked_indices=ranked_indices,
        title=report_title,
        command_line=command_line,
    )

    print(f"\nReport generated: {output_path}")
    if HAS_MATPLOTLIB:
        print(f"Images saved to: {image_dir}/")


if __name__ == '__main__':
    main()
