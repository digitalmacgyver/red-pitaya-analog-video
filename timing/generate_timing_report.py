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

    times = []
    h_vals = []
    v_vals = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                times.append(float(row[0]))
                h_vals.append(int(row[1]))
                v_vals.append(int(row[2]))

    times = np.array(times)
    h_vals = np.array(h_vals)
    v_vals = np.array(v_vals)

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

    # Calculate lines per field
    lines_per_field = []
    for i in range(len(v_times) - 1):
        # Count H events between consecutive V events
        mask = (h_times >= v_times[i]) & (h_times < v_times[i + 1])
        lines_per_field.append(np.sum(mask))
    lines_per_field = np.array(lines_per_field)

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
                          noise_floor_us: float = None) -> str:
    """
    Create a Tufte-style timing distribution visualization.

    Y-axis: Line number within field (1-262)
    X-axis: Timing deviation from nominal
    Shows density of timing at each line position.

    Args:
        x_range: Optional (min, max) tuple in microseconds for consistent scaling
        noise_floor_us: If provided, show shaded band indicating measurement uncertainty
    """
    if not HAS_MATPLOTLIB:
        return None

    # Get normal lines (filter out half-lines)
    normal_h, _ = filter_half_lines(data.h_periods, nominal_h)

    # Calculate deviations in microseconds
    deviations_us = (normal_h - nominal_h) * 1e6

    # Assign each measurement to a line number within field
    # Approximate: cycle through 262 lines
    n_lines = len(deviations_us)
    lines_in_field = 262
    line_numbers = np.arange(n_lines) % lines_in_field + 1

    # Create figure with minimal chrome (Tufte style)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add noise floor shading if provided
    if noise_floor_us is not None:
        ax.axvspan(-noise_floor_us, noise_floor_us,
                   color='yellow', alpha=0.15, zorder=0,
                   label=f'±{noise_floor_us*1e3:.0f} ns measurement uncertainty')

    # Calculate statistics per line
    line_stats = []
    for line in range(1, lines_in_field + 1):
        mask = line_numbers == line
        if np.sum(mask) > 0:
            line_devs = deviations_us[mask]
            line_stats.append({
                'line': line,
                'median': np.median(line_devs),
                'p5': np.percentile(line_devs, 5),
                'p25': np.percentile(line_devs, 25),
                'p75': np.percentile(line_devs, 75),
                'p95': np.percentile(line_devs, 95),
                'min': np.min(line_devs),
                'max': np.max(line_devs)
            })

    # Plot range bars for each line
    for stat in line_stats:
        y = stat['line']
        # Full range (thin line)
        ax.hlines(y, stat['min'], stat['max'], colors='lightgray', linewidths=0.5)
        # 5-95 percentile (medium line)
        ax.hlines(y, stat['p5'], stat['p95'], colors='steelblue', linewidths=1, alpha=0.5)
        # 25-75 percentile (thick line)
        ax.hlines(y, stat['p25'], stat['p75'], colors='steelblue', linewidths=2)
        # Median (dot)
        ax.plot(stat['median'], y, 'o', color='darkblue', markersize=2)

    # Reference line at 0 deviation
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set consistent X-axis range if provided
    if x_range is not None:
        ax.set_xlim(x_range)

    # Labels and title
    ax.set_xlabel('Timing Deviation (µs)')
    ax.set_ylabel('Line Number in Field')
    ax.set_title(title)

    # Invert Y axis so line 1 is at top
    ax.invert_yaxis()

    # Remove top and right spines (Tufte style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def get_h_deviation_range(data1: TimingData, data2: TimingData,
                          nominal_h: float = NTSC_LINE_PERIOD) -> Tuple[float, float]:
    """Calculate combined X-axis range for horizontal timing heatmaps."""
    h1_normal, _ = filter_half_lines(data1.h_periods, nominal_h)
    h2_normal, _ = filter_half_lines(data2.h_periods, nominal_h)

    dev1_us = (h1_normal - nominal_h) * 1e6
    dev2_us = (h2_normal - nominal_h) * 1e6

    all_devs = np.concatenate([dev1_us, dev2_us])
    # Use 0.1 and 99.9 percentiles to avoid extreme outliers dominating
    x_min = np.percentile(all_devs, 0.1)
    x_max = np.percentile(all_devs, 99.9)

    # Add 5% padding
    padding = (x_max - x_min) * 0.05
    return (x_min - padding, x_max + padding)


def create_histogram_comparison(data1: TimingData, data2: TimingData,
                                output_path: str, timing_type: str = 'horizontal',
                                noise_floor_us: float = None) -> str:
    """
    Create overlaid histogram comparing timing distributions.

    Args:
        noise_floor_us: If provided, show shaded band around nominal indicating
                        measurement uncertainty
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if timing_type == 'horizontal':
        periods1, _ = filter_half_lines(data1.h_periods)
        periods2, _ = filter_half_lines(data2.h_periods)
        nominal = NTSC_LINE_PERIOD
        xlabel = 'Line Period (µs)'
        title = 'Horizontal Timing Distribution'
    else:
        periods1 = data1.v_periods
        periods2 = data2.v_periods
        nominal = NTSC_FIELD_PERIOD
        xlabel = 'Field Period (ms)'
        title = 'Vertical Timing Distribution'

    # Convert to appropriate units
    if timing_type == 'horizontal':
        periods1_plot = periods1 * 1e6
        periods2_plot = periods2 * 1e6
        nominal_plot = nominal * 1e6
        # Convert noise floor from µs to the plot unit (µs)
        noise_floor_plot = noise_floor_us if noise_floor_us else None
    else:
        periods1_plot = periods1 * 1e3
        periods2_plot = periods2 * 1e3
        nominal_plot = nominal * 1e3
        # Convert noise floor from µs to ms
        noise_floor_plot = noise_floor_us / 1000 if noise_floor_us else None

    # Calculate bins
    all_periods = np.concatenate([periods1_plot, periods2_plot])
    bins = np.linspace(np.percentile(all_periods, 0.5),
                       np.percentile(all_periods, 99.5), 100)

    # Add noise floor shading if provided
    if noise_floor_plot is not None:
        ax.axvspan(nominal_plot - noise_floor_plot, nominal_plot + noise_floor_plot,
                   color='yellow', alpha=0.2, zorder=0,
                   label='Measurement uncertainty')

    # Plot histograms
    ax.hist(periods1_plot, bins=bins, alpha=0.5, label=data1.label, density=True)
    ax.hist(periods2_plot, bins=bins, alpha=0.5, label=data2.label, density=True)

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


def create_field_stability_plot(data1: TimingData, data2: TimingData,
                                output_path: str,
                                noise_floor_us: float = None) -> str:
    """
    Plot field-to-field timing stability over time with consistent Y-axis.

    Args:
        noise_floor_us: If provided, show shaded band indicating measurement uncertainty
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Convert to µs deviation from nominal
    dev1 = (data1.v_periods - NTSC_FIELD_PERIOD) * 1e6  # µs
    dev2 = (data2.v_periods - NTSC_FIELD_PERIOD) * 1e6

    # Calculate shared Y-axis limits using both datasets
    all_devs = np.concatenate([dev1, dev2])
    y_min = np.percentile(all_devs, 0.5)
    y_max = np.percentile(all_devs, 99.5)
    padding = (y_max - y_min) * 0.1
    y_limits = (y_min - padding, y_max + padding)

    fields1 = np.arange(len(dev1))
    fields2 = np.arange(len(dev2))

    # Add noise floor shading if provided
    if noise_floor_us is not None:
        ax1.axhspan(-noise_floor_us, noise_floor_us,
                    color='yellow', alpha=0.15, zorder=0,
                    label='Measurement uncertainty')
        ax2.axhspan(-noise_floor_us, noise_floor_us,
                    color='yellow', alpha=0.15, zorder=0,
                    label='Measurement uncertainty')

    ax1.plot(fields1, dev1, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('Field Period Deviation (µs)')
    ax1.set_title(f'{data1.label} - Field Timing Stability')
    ax1.set_ylim(y_limits)
    ax1.set_xlabel('Field Number')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.plot(fields2, dev2, 'g-', linewidth=0.5, alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Field Number')
    ax2.set_ylabel('Field Period Deviation (µs)')
    ax2.set_title(f'{data2.label} - Field Timing Stability')
    ax2.set_ylim(y_limits)  # Same Y-axis limits as first plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_html_report(data1: TimingData, data2: TimingData,
                         stats1_h: TimingStats, stats2_h: TimingStats,
                         stats1_v: TimingStats, stats2_v: TimingStats,
                         output_path: str, image_dir: str,
                         skip_fields: int,
                         sample_rate_mhz: float = 20.0) -> str:
    """Generate comprehensive HTML report."""

    # Create image directory
    os.makedirs(image_dir, exist_ok=True)

    # Calculate noise floor for measurement uncertainty display
    noise_floor_ns = get_noise_floor_ns(sample_rate_mhz)
    noise_floor_us = noise_floor_ns / 1000.0  # Convert to µs for charts

    # Generate visualizations
    images = {}

    if HAS_MATPLOTLIB:
        # Calculate shared X-axis range for heatmaps
        h_range = get_h_deviation_range(data1, data2)

        # Timing heatmaps with consistent X-axis and noise floor shading
        img_path = os.path.join(image_dir, 'heatmap_1.png')
        create_timing_heatmap(data1, img_path, f'{data1.label} - Line Timing Distribution',
                              x_range=h_range, noise_floor_us=noise_floor_us)
        images['heatmap_1'] = os.path.basename(img_path)

        img_path = os.path.join(image_dir, 'heatmap_2.png')
        create_timing_heatmap(data2, img_path, f'{data2.label} - Line Timing Distribution',
                              x_range=h_range, noise_floor_us=noise_floor_us)
        images['heatmap_2'] = os.path.basename(img_path)

        # Histogram comparisons with noise floor shading
        img_path = os.path.join(image_dir, 'hist_horizontal.png')
        create_histogram_comparison(data1, data2, img_path, 'horizontal',
                                    noise_floor_us=noise_floor_us)
        images['hist_h'] = os.path.basename(img_path)

        img_path = os.path.join(image_dir, 'hist_vertical.png')
        create_histogram_comparison(data1, data2, img_path, 'vertical',
                                    noise_floor_us=noise_floor_us)
        images['hist_v'] = os.path.basename(img_path)

        # Field stability with noise floor shading
        img_path = os.path.join(image_dir, 'field_stability.png')
        create_field_stability_plot(data1, data2, img_path, noise_floor_us=noise_floor_us)
        images['field_stability'] = os.path.basename(img_path)

    # Get relative path for images
    output_dir = os.path.dirname(output_path)
    rel_image_dir = os.path.relpath(image_dir, output_dir) if output_dir else os.path.basename(image_dir)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Video Timing Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
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
            padding: 12px;
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
            grid-template-columns: 1fr 1fr;
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
    </style>
</head>
<body>
    <h1>Video Timing Comparison Report</h1>

    <div class="summary-box">
        <strong>Captures Compared:</strong><br>
        <strong>1.</strong> {data1.label} ({data1.duration:.2f}s, {stats1_h.count:,} lines, {stats1_v.count:,} fields)<br>
        <strong>2.</strong> {data2.label} ({data2.duration:.2f}s, {stats2_h.count:,} lines, {stats2_v.count:,} fields)<br>
        <br>
        <strong>Analysis Settings:</strong> First and last {skip_fields} fields trimmed from each capture.
    </div>

    <h2>Horizontal Timing (Line Period)</h2>

    <p>NTSC nominal line period: <strong>{NTSC_LINE_PERIOD*1e6:.4f} µs</strong> ({NTSC_LINE_FREQ:.3f} Hz)</p>

    <table>
        <tr>
            <th>Metric</th>
            <th>{data1.label}</th>
            <th>{data2.label}</th>
            <th>NTSC Nominal</th>
        </tr>
        <tr>
            <td>Median</td>
            <td>{stats1_h.median*1e6:.4f} µs</td>
            <td>{stats2_h.median*1e6:.4f} µs</td>
            <td>{NTSC_LINE_PERIOD*1e6:.4f} µs</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>{stats1_h.mean*1e6:.4f} µs</td>
            <td>{stats2_h.mean*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Std Dev</td>
            <td>{format_jitter_with_threshold(stats1_h.std*1e9, noise_floor_ns)}</td>
            <td>{format_jitter_with_threshold(stats2_h.std*1e9, noise_floor_ns)}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>RMS Jitter</td>
            <td>{format_jitter_with_threshold(stats1_h.rms_jitter*1e9, noise_floor_ns)}</td>
            <td>{format_jitter_with_threshold(stats2_h.rms_jitter*1e9, noise_floor_ns)}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Peak-to-Peak Jitter</td>
            <td>{format_jitter_with_threshold(stats1_h.pp_jitter*1e9, noise_floor_ns * 2)}</td>
            <td>{format_jitter_with_threshold(stats2_h.pp_jitter*1e9, noise_floor_ns * 2)}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Min</td>
            <td>{stats1_h.min*1e6:.4f} µs</td>
            <td>{stats2_h.min*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Max</td>
            <td>{stats1_h.max*1e6:.4f} µs</td>
            <td>{stats2_h.max*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>1st Percentile</td>
            <td>{stats1_h.p1*1e6:.4f} µs</td>
            <td>{stats2_h.p1*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>5th Percentile</td>
            <td>{stats1_h.p5*1e6:.4f} µs</td>
            <td>{stats2_h.p5*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>25th Percentile</td>
            <td>{stats1_h.p25*1e6:.4f} µs</td>
            <td>{stats2_h.p25*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>50th Percentile</td>
            <td>{stats1_h.median*1e6:.4f} µs</td>
            <td>{stats2_h.median*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>75th Percentile</td>
            <td>{stats1_h.p75*1e6:.4f} µs</td>
            <td>{stats2_h.p75*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>95th Percentile</td>
            <td>{stats1_h.p95*1e6:.4f} µs</td>
            <td>{stats2_h.p95*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>99th Percentile</td>
            <td>{stats1_h.p99*1e6:.4f} µs</td>
            <td>{stats2_h.p99*1e6:.4f} µs</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Sample Count</td>
            <td>{stats1_h.count:,}</td>
            <td>{stats2_h.count:,}</td>
            <td>—</td>
        </tr>
    </table>

    <p><em>* Values marked with asterisk are below the measurement noise floor (~{noise_floor_ns:.0f} ns) and cannot be reliably distinguished from zero.</em></p>

    {"<h3>Horizontal Timing Distribution</h3>" if 'hist_h' in images else ""}
    {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('hist_h', '') + "' alt='Horizontal Histogram'></div>" if 'hist_h' in images else ""}

    <h3>Line-by-Line Timing Distribution</h3>
    <p>Each row shows the timing distribution for that line number within a field.
    The thin gray line shows full range (min-max), blue shows 5th-95th percentile,
    thick blue shows interquartile range (25th-75th), and the dot marks the median.
    The red dashed line indicates zero deviation from nominal.</p>

    <div class="comparison-grid">
        {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('heatmap_1', '') + "' alt='Heatmap 1'></div>" if 'heatmap_1' in images else ""}
        {"<div class='image-container'><img src='" + rel_image_dir + "/" + images.get('heatmap_2', '') + "' alt='Heatmap 2'></div>" if 'heatmap_2' in images else ""}
    </div>

    <h2>Vertical Timing (Field Period)</h2>

    <p>NTSC nominal field period: <strong>{NTSC_FIELD_PERIOD*1e3:.4f} ms</strong> ({NTSC_FIELD_FREQ:.2f} Hz)</p>

    <table>
        <tr>
            <th>Metric</th>
            <th>{data1.label}</th>
            <th>{data2.label}</th>
            <th>NTSC Nominal</th>
        </tr>
        <tr>
            <td>Median</td>
            <td>{stats1_v.median*1e3:.4f} ms</td>
            <td>{stats2_v.median*1e3:.4f} ms</td>
            <td>{NTSC_FIELD_PERIOD*1e3:.4f} ms</td>
        </tr>
        <tr>
            <td>Mean</td>
            <td>{stats1_v.mean*1e3:.4f} ms</td>
            <td>{stats2_v.mean*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Std Dev</td>
            <td>{format_jitter_with_threshold(stats1_v.std*1e9, noise_floor_ns, unit='us')}</td>
            <td>{format_jitter_with_threshold(stats2_v.std*1e9, noise_floor_ns, unit='us')}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>RMS Jitter</td>
            <td>{format_jitter_with_threshold(stats1_v.rms_jitter*1e9, noise_floor_ns, unit='us')}</td>
            <td>{format_jitter_with_threshold(stats2_v.rms_jitter*1e9, noise_floor_ns, unit='us')}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Peak-to-Peak Jitter</td>
            <td>{format_jitter_with_threshold(stats1_v.pp_jitter*1e9, noise_floor_ns * 2, unit='us')}</td>
            <td>{format_jitter_with_threshold(stats2_v.pp_jitter*1e9, noise_floor_ns * 2, unit='us')}</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Min</td>
            <td>{stats1_v.min*1e3:.4f} ms</td>
            <td>{stats2_v.min*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Max</td>
            <td>{stats1_v.max*1e3:.4f} ms</td>
            <td>{stats2_v.max*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>1st Percentile</td>
            <td>{stats1_v.p1*1e3:.4f} ms</td>
            <td>{stats2_v.p1*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>5th Percentile</td>
            <td>{stats1_v.p5*1e3:.4f} ms</td>
            <td>{stats2_v.p5*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>25th Percentile</td>
            <td>{stats1_v.p25*1e3:.4f} ms</td>
            <td>{stats2_v.p25*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>50th Percentile</td>
            <td>{stats1_v.median*1e3:.4f} ms</td>
            <td>{stats2_v.median*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>75th Percentile</td>
            <td>{stats1_v.p75*1e3:.4f} ms</td>
            <td>{stats2_v.p75*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>95th Percentile</td>
            <td>{stats1_v.p95*1e3:.4f} ms</td>
            <td>{stats2_v.p95*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>99th Percentile</td>
            <td>{stats1_v.p99*1e3:.4f} ms</td>
            <td>{stats2_v.p99*1e3:.4f} ms</td>
            <td>—</td>
        </tr>
        <tr>
            <td>Sample Count</td>
            <td>{stats1_v.count:,}</td>
            <td>{stats2_v.count:,}</td>
            <td>—</td>
        </tr>
    </table>

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
            <th>{data1.label}</th>
            <th>{data2.label}</th>
        </tr>
        <tr>
            <td>Mean Lines/Field</td>
            <td>{np.mean(data1.lines_per_field):.2f}</td>
            <td>{np.mean(data2.lines_per_field):.2f}</td>
        </tr>
        <tr>
            <td>Min</td>
            <td>{np.min(data1.lines_per_field)}</td>
            <td>{np.min(data2.lines_per_field)}</td>
        </tr>
        <tr>
            <td>Max</td>
            <td>{np.max(data1.lines_per_field)}</td>
            <td>{np.max(data2.lines_per_field)}</td>
        </tr>
    </table>

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
        </ul>
        <p>Generated by <code>generate_timing_report.py</code></p>
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate video timing comparison report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --capture1 leitch.csv --capture2 redpitaya.csv --output report.html
    %(prog)s -1 ref.csv -2 dut.csv -o /tmp/report.html --skip-fields 32
        """
    )

    parser.add_argument('--capture1', '-1', required=True,
                        help='First capture file (reference)')
    parser.add_argument('--capture2', '-2', required=True,
                        help='Second capture file (device under test)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output HTML report path')
    parser.add_argument('--label1', default=None,
                        help='Label for capture 1 (default: filename)')
    parser.add_argument('--label2', default=None,
                        help='Label for capture 2 (default: filename)')
    parser.add_argument('--skip-fields', type=int, default=16,
                        help='Fields to skip at start/end (default: 16)')
    parser.add_argument('--sample-rate', type=float, default=20.0,
                        help='Logic analyzer sample rate in MS/s (default: 20.0)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.capture1):
        print(f"Error: File not found: {args.capture1}")
        sys.exit(1)
    if not os.path.exists(args.capture2):
        print(f"Error: File not found: {args.capture2}")
        sys.exit(1)

    # Parse data (with timing quantization to sample period)
    print(f"Loading {args.capture1}...")
    print(f"  Quantizing timing values to {args.sample_rate} MS/s sample period")
    data1 = parse_timing_csv(args.capture1, args.label1, sample_rate_mhz=args.sample_rate)
    print(f"  {len(data1.h_periods):,} line periods, {len(data1.v_periods):,} field periods")

    print(f"Loading {args.capture2}...")
    data2 = parse_timing_csv(args.capture2, args.label2, sample_rate_mhz=args.sample_rate)
    print(f"  {len(data2.h_periods):,} line periods, {len(data2.v_periods):,} field periods")

    # Trim data
    if args.skip_fields > 0:
        print(f"\nTrimming first and last {args.skip_fields} fields...")
        data1 = trim_data(data1, args.skip_fields)
        data2 = trim_data(data2, args.skip_fields)
        print(f"  Capture 1: {len(data1.h_periods):,} lines, {len(data1.v_periods):,} fields")
        print(f"  Capture 2: {len(data2.h_periods):,} lines, {len(data2.v_periods):,} fields")

    # Filter half-lines for horizontal stats
    print("\nCalculating statistics...")
    h1_normal, h1_half = filter_half_lines(data1.h_periods)
    h2_normal, h2_half = filter_half_lines(data2.h_periods)

    print(f"  Capture 1: {len(h1_normal):,} normal lines, {len(h1_half):,} half-lines")
    print(f"  Capture 2: {len(h2_normal):,} normal lines, {len(h2_half):,} half-lines")

    # Calculate statistics
    stats1_h = calculate_stats(h1_normal, NTSC_LINE_PERIOD)
    stats2_h = calculate_stats(h2_normal, NTSC_LINE_PERIOD)
    stats1_v = calculate_stats(data1.v_periods, NTSC_FIELD_PERIOD)
    stats2_v = calculate_stats(data2.v_periods, NTSC_FIELD_PERIOD)

    # Generate report
    output_path = args.output
    output_dir = os.path.dirname(output_path) or '.'
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    image_dir = os.path.join(output_dir, f"{base_name}_images")

    print(f"\nGenerating report...")
    os.makedirs(output_dir, exist_ok=True)

    generate_html_report(
        data1, data2,
        stats1_h, stats2_h,
        stats1_v, stats2_v,
        output_path, image_dir,
        args.skip_fields,
        args.sample_rate
    )

    print(f"\nReport generated: {output_path}")
    if HAS_MATPLOTLIB:
        print(f"Images saved to: {image_dir}/")


if __name__ == '__main__':
    main()
