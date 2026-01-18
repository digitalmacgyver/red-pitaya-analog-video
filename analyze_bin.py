#!/usr/bin/env python3
"""
Analyze CVBS timing from a Red Pitaya capture file.

Detects horizontal (HBI) and vertical (VBI) blanking intervals and measures
timing statistics to verify capture quality and identify timing errors.

Usage:
    python analyze_bin.py <capture.bin> [max_seconds]

Arguments:
    capture.bin     Input capture file (int8 samples from rpsa_client)
    max_seconds     Optional: limit analysis to first N seconds

Examples:
    # Analyze entire capture
    python analyze_bin.py capture.bin

    # Analyze first 2 seconds only (faster)
    python analyze_bin.py capture.bin 2

Output:
    - VBI (field) timing: interval between vertical blanking events
    - HBI (line) timing: interval between horizontal sync pulses
    - Statistics: min, max, mean, median, std, jitter
    - Timing relationship analysis (samples per line, fractional errors)

Detection Method:
    - Sync pulses detected at threshold halfway between blanking and sync tip
    - VBI identified by broad pulses (>27µs width)
    - HBI measured only in active video regions (excluding VBI)
    - Outliers filtered (intervals outside ±10% of expected)

NTSC Reference:
    - Line frequency: 15734.264 Hz (63.556 µs period)
    - Field frequency: 59.94 Hz (16683 µs period)
    - 4×fsc = 14.31818 MHz gives exactly 910 samples/line
    - 125/8 MHz = 15.625 MHz gives 993.056 samples/line (fractional!)
"""

import numpy as np
import sys
import os
import argparse

# NTSC timing constants (exact values)
NTSC_COLOR_SUBCARRIER = 3579545.0  # Hz (exactly 315/88 MHz)
NTSC_LINE_FREQ = NTSC_COLOR_SUBCARRIER * 2 / 455  # 15734.264 Hz
NTSC_LINE_PERIOD_US = 1e6 / NTSC_LINE_FREQ  # 63.5556 µs
NTSC_FIELD_FREQ = NTSC_LINE_FREQ / 262.5  # 59.94 Hz
NTSC_FIELD_PERIOD_US = 1e6 / NTSC_FIELD_FREQ  # 16683.17 µs

NTSC_HSYNC_WIDTH_US = 4.7
NTSC_BROAD_WIDTH_US = 27.1
NTSC_EQUALIZING_WIDTH_US = 2.3


def detect_header_size(filepath, check_bytes=256):
    """Detect header by finding where CVBS-like data starts."""
    data = np.fromfile(filepath, dtype=np.int8, count=check_bytes)
    for i in range(0, len(data) - 64, 8):
        chunk = data[i:i+64]
        if np.std(chunk) > 10 and np.abs(chunk).mean() > 5:
            return (i // 64) * 64
    return 0


def find_sync_edges(data, threshold):
    """Find falling edges where signal crosses threshold."""
    below = data < threshold
    edges = np.diff(below.astype(np.int8))
    falling = np.where(edges == 1)[0]
    rising = np.where(edges == -1)[0]
    return falling, rising


def analyze_cvbs_timing(filepath, sample_rate=15.625e6, max_seconds=None):
    """Analyze CVBS timing from a capture file."""

    print(f"=== CVBS Timing Analysis (v2) ===")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Sample rate: {sample_rate/1e6:.6f} MS/s")

    # Calculate expected values in samples
    samples_per_line = NTSC_LINE_PERIOD_US * sample_rate / 1e6
    samples_per_field = NTSC_FIELD_PERIOD_US * sample_rate / 1e6
    samples_per_hsync = NTSC_HSYNC_WIDTH_US * sample_rate / 1e6
    samples_per_broad = NTSC_BROAD_WIDTH_US * sample_rate / 1e6
    samples_per_eq = NTSC_EQUALIZING_WIDTH_US * sample_rate / 1e6

    print(f"\nNTSC timing reference:")
    print(f"  Line freq: {NTSC_LINE_FREQ:.3f} Hz")
    print(f"  Line period: {NTSC_LINE_PERIOD_US:.4f} µs = {samples_per_line:.3f} samples")
    print(f"  Field period: {NTSC_FIELD_PERIOD_US:.3f} µs = {samples_per_field:.2f} samples")
    print(f"  H-sync: {NTSC_HSYNC_WIDTH_US} µs = {samples_per_hsync:.2f} samples")
    print(f"  Broad: {NTSC_BROAD_WIDTH_US} µs = {samples_per_broad:.2f} samples")

    # Check sample rate vs 4fsc
    fsc4 = NTSC_COLOR_SUBCARRIER * 4
    print(f"\n  4×fsc: {fsc4/1e6:.6f} MHz")
    print(f"  Our rate: {sample_rate/1e6:.6f} MHz")
    print(f"  Ratio: {sample_rate/fsc4:.6f}")

    # Load data
    header_size = detect_header_size(filepath)
    with open(filepath, 'rb') as f:
        f.seek(header_size)
        data = np.frombuffer(f.read(), dtype=np.int8)

    if max_seconds:
        max_samples = int(max_seconds * sample_rate)
        if len(data) > max_samples:
            data = data[:max_samples]

    total_duration = len(data) / sample_rate
    print(f"\nLoaded {len(data):,} samples ({total_duration:.3f} s)")

    # Find signal levels
    sync_tip = np.percentile(data, 1)  # 1st percentile to avoid noise
    blanking = np.percentile(data, 10)  # Blanking level
    threshold = (blanking + sync_tip) / 2

    print(f"\nSignal levels:")
    print(f"  Sync tip: {sync_tip:.1f}")
    print(f"  Blanking: {blanking:.1f}")
    print(f"  Threshold: {threshold:.1f}")

    # Find sync edges
    falling, rising = find_sync_edges(data, threshold)
    print(f"\nFound {len(falling):,} sync pulses")

    # Measure pulse widths
    widths = []
    for i, f in enumerate(falling):
        # Find next rising edge
        r_after = rising[rising > f]
        if len(r_after) > 0:
            widths.append(r_after[0] - f)
    widths = np.array(widths)

    # Classify pulses
    # Normal H-sync: 4.7µs = ~73 samples
    # Broad pulse: 27.1µs = ~423 samples
    # Equalizing: 2.3µs = ~36 samples
    # Half-line sync (during VBI): same as H-sync but at half-line intervals

    hsync_max = samples_per_hsync * 2  # Up to ~147 samples
    broad_min = samples_per_broad * 0.7  # At least ~296 samples

    is_hsync = widths < hsync_max
    is_broad = widths > broad_min

    print(f"\nPulse classification:")
    print(f"  H-sync (<{hsync_max:.0f} samples): {np.sum(is_hsync):,}")
    print(f"  Broad (>{broad_min:.0f} samples): {np.sum(is_broad):,}")

    # Find VBI regions by looking for broad pulses
    broad_indices = np.where(is_broad)[0]

    # Group broad pulses into VBI events
    vbi_events = []
    if len(broad_indices) > 0:
        current_vbi_start_idx = broad_indices[0]
        current_vbi_start_sample = falling[current_vbi_start_idx]

        for i in range(1, len(broad_indices)):
            idx = broad_indices[i]
            prev_idx = broad_indices[i-1]

            # If gap > 20 pulses, it's a new VBI
            if idx - prev_idx > 20:
                vbi_events.append({
                    'start_idx': current_vbi_start_idx,
                    'end_idx': prev_idx,
                    'start_sample': current_vbi_start_sample,
                    'end_sample': falling[prev_idx]
                })
                current_vbi_start_idx = idx
                current_vbi_start_sample = falling[idx]

        # Add last VBI
        vbi_events.append({
            'start_idx': current_vbi_start_idx,
            'end_idx': broad_indices[-1],
            'start_sample': current_vbi_start_sample,
            'end_sample': falling[broad_indices[-1]]
        })

    print(f"\nDetected {len(vbi_events)} VBI events")

    # Analyze VBI intervals
    if len(vbi_events) >= 2:
        vbi_times = [v['start_sample'] for v in vbi_events]
        vbi_intervals = np.diff(vbi_times)
        vbi_intervals_us = vbi_intervals / sample_rate * 1e6

        print(f"\n=== VBI (Field) Timing ===")
        print(f"  Expected: {NTSC_FIELD_PERIOD_US:.3f} µs")
        print(f"  Count: {len(vbi_intervals)}")
        print(f"  Min: {vbi_intervals_us.min():.3f} µs")
        print(f"  Max: {vbi_intervals_us.max():.3f} µs")
        print(f"  Mean: {vbi_intervals_us.mean():.3f} µs")
        print(f"  Median: {np.median(vbi_intervals_us):.3f} µs")
        print(f"  Std: {vbi_intervals_us.std():.4f} µs")
        print(f"  Mean error: {vbi_intervals_us.mean() - NTSC_FIELD_PERIOD_US:+.3f} µs ({(vbi_intervals_us.mean() - NTSC_FIELD_PERIOD_US)/NTSC_FIELD_PERIOD_US*1e6:+.1f} ppm)")

    # Analyze HBI intervals - only in active video regions (between VBIs)
    print(f"\n=== HBI (Line) Timing ===")
    print(f"  (Excluding VBI regions)")

    all_hbi_intervals = []

    for v in range(len(vbi_events) - 1):
        # Define active region: 30 pulses after VBI end to 30 pulses before next VBI start
        start_idx = vbi_events[v]['end_idx'] + 30
        end_idx = vbi_events[v+1]['start_idx'] - 30

        if end_idx <= start_idx:
            continue

        # Get H-sync pulses in this region
        hsync_indices = []
        for i in range(start_idx, end_idx):
            if i < len(is_hsync) and is_hsync[i]:
                hsync_indices.append(i)

        if len(hsync_indices) >= 2:
            times = falling[hsync_indices]
            intervals = np.diff(times)
            all_hbi_intervals.extend(intervals)

    if len(all_hbi_intervals) > 0:
        hbi_intervals = np.array(all_hbi_intervals)
        hbi_intervals_us = hbi_intervals / sample_rate * 1e6

        # Filter outliers (keep only intervals within 10% of expected)
        expected_min = NTSC_LINE_PERIOD_US * 0.9
        expected_max = NTSC_LINE_PERIOD_US * 1.1
        valid = (hbi_intervals_us >= expected_min) & (hbi_intervals_us <= expected_max)

        hbi_clean = hbi_intervals_us[valid]
        hbi_samples = hbi_intervals[valid]

        print(f"  Total intervals: {len(hbi_intervals_us)}")
        print(f"  Valid (within 10%): {len(hbi_clean)}")
        print(f"  Rejected: {len(hbi_intervals_us) - len(hbi_clean)}")

        if len(hbi_clean) > 0:
            print(f"\n  Expected: {NTSC_LINE_PERIOD_US:.4f} µs = {samples_per_line:.3f} samples")
            print(f"  Min: {hbi_clean.min():.4f} µs ({hbi_samples.min():.2f} samples)")
            print(f"  Max: {hbi_clean.max():.4f} µs ({hbi_samples.max():.2f} samples)")
            print(f"  Mean: {hbi_clean.mean():.4f} µs ({hbi_samples.mean():.3f} samples)")
            print(f"  Median: {np.median(hbi_clean):.4f} µs ({np.median(hbi_samples):.3f} samples)")
            print(f"  Std: {hbi_clean.std():.4f} µs ({hbi_samples.std():.3f} samples)")
            print(f"  Jitter (std/mean): {hbi_clean.std()/hbi_clean.mean()*100:.4f}%")

            error = hbi_clean.mean() - NTSC_LINE_PERIOD_US
            error_ppm = error / NTSC_LINE_PERIOD_US * 1e6
            print(f"\n  Mean error: {error:+.4f} µs ({error_ppm:+.1f} ppm)")
            print(f"  Error per field (262.5 lines): {error * 262.5:+.2f} µs")
            print(f"  Error per second (~15734 lines): {error * 15734:+.1f} µs")

            # Check for drift
            if len(hbi_clean) > 200:
                first = hbi_clean[:100]
                last = hbi_clean[-100:]
                drift = last.mean() - first.mean()
                print(f"\n  Drift check (first 100 vs last 100):")
                print(f"    First 100 mean: {first.mean():.4f} µs")
                print(f"    Last 100 mean: {last.mean():.4f} µs")
                print(f"    Drift: {drift:+.4f} µs")

    # Summary: timing relationship
    print(f"\n=== Timing Relationship ===")
    print(f"  Sample rate: {sample_rate:,.0f} Hz")
    print(f"  NTSC line rate: {NTSC_LINE_FREQ:,.3f} Hz")
    print(f"  Samples per line (exact): {sample_rate / NTSC_LINE_FREQ:.6f}")
    print(f"  Samples per line (rounded): {round(sample_rate / NTSC_LINE_FREQ)}")
    print(f"  Fractional error: {sample_rate / NTSC_LINE_FREQ - round(sample_rate / NTSC_LINE_FREQ):.6f} samples/line")

    frac_error_per_line = sample_rate / NTSC_LINE_FREQ - round(sample_rate / NTSC_LINE_FREQ)
    frac_error_per_field = frac_error_per_line * 262.5
    frac_error_per_second = frac_error_per_line * NTSC_LINE_FREQ
    print(f"  Cumulative fractional error per field: {frac_error_per_field:.2f} samples")
    print(f"  Cumulative fractional error per second: {frac_error_per_second:.1f} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CVBS timing from a Red Pitaya capture file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s capture.bin              # Analyze entire capture
    %(prog)s capture.bin 2            # Analyze first 2 seconds
    %(prog)s capture.bin --rate 4M    # Specify different sample rate

Output includes:
    - VBI (field) timing statistics
    - HBI (line) timing statistics
    - Jitter and drift analysis
    - Sample rate vs NTSC timing relationship
        """
    )

    parser.add_argument('input', help='Input capture file (.bin)')
    parser.add_argument('max_seconds', nargs='?', type=float, default=None,
                        help='Limit analysis to first N seconds')
    parser.add_argument('--rate', default='15.625M',
                        help='Sample rate (default: 15.625M)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Parse sample rate
    rate_str = args.rate.strip().upper()
    if rate_str.endswith('M'):
        sample_rate = float(rate_str[:-1]) * 1e6
    else:
        sample_rate = float(rate_str)

    analyze_cvbs_timing(args.input, sample_rate=sample_rate, max_seconds=args.max_seconds)


if __name__ == '__main__':
    main()
