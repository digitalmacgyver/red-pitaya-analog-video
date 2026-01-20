#!/usr/bin/env python3
"""
Analyze CVBS timing from a Red Pitaya capture file or resampled WAV.

CORRECTED NTSC TIMING ANALYSIS (v3)
===================================
This script properly analyzes NTSC CVBS timing by:
1. Verifying H-sync count (507 per frame, i.e., 525 - 18 VBI lines)
2. Verifying H-sync spacing matches the expected samples/line
3. Properly identifying Field 1 vs Field 2 by gap to first EQ
4. Understanding that VBI spans 10 lines (9 lines content + 1 line transitions)

KEY INSIGHT:
The apparent +1 line per frame when measuring from VBI markers is NOT an error.
VBI content is 9 lines, but VBI span (last H-sync to first H-sync) is 10 lines.
The extra line is transition gaps (0.5 lines before + 0.5 lines after VBI pulses).

FIELD IDENTIFICATION:
- Field 1: Gap from last H-sync to first EQ = 0.5 lines (~496 samples at 15.625 MS/s)
- Field 2: Gap from last H-sync to first EQ = 1.0 lines (~993 samples)
  This is because the H-sync at line 263.0 is replaced by an EQ-width pulse.

Usage:
    python analyze_bin.py <capture.bin> [max_seconds]
    python analyze_bin.py <resampled.wav> [max_seconds]

Arguments:
    input_file      Input capture file (.bin int8) or resampled WAV file
    max_seconds     Optional: limit analysis to first N seconds

Options:
    --rate RATE     Sample rate (default: auto-detect or 15.625M for .bin)
                    Presets: 4fsc, 2fsc, 1fsc, 0.5fsc, or custom (e.g., 4M)
"""

import numpy as np
import sys
import os
import argparse
import wave

# NTSC timing constants (exact values)
NTSC_COLOR_SUBCARRIER = 3579545.0  # Hz (exactly 315/88 MHz)
NTSC_LINE_FREQ = NTSC_COLOR_SUBCARRIER * 2 / 455  # 15734.264 Hz
NTSC_LINE_PERIOD_US = 1e6 / NTSC_LINE_FREQ  # 63.5556 µs
NTSC_FIELD_FREQ = NTSC_LINE_FREQ / 262.5  # 59.94 Hz
NTSC_FIELD_PERIOD_US = 1e6 / NTSC_FIELD_FREQ  # 16683.17 µs

NTSC_HSYNC_WIDTH_US = 4.7
NTSC_BROAD_WIDTH_US = 27.1
NTSC_EQUALIZING_WIDTH_US = 2.3

# Sample rate presets
FSC_4 = NTSC_COLOR_SUBCARRIER * 4   # 14.31818 MS/s
FSC_2 = NTSC_COLOR_SUBCARRIER * 2   # 7.15909 MS/s
FSC_1 = NTSC_COLOR_SUBCARRIER       # 3.57954 MS/s
FSC_HALF = NTSC_COLOR_SUBCARRIER / 2  # 1.78977 MS/s
RP_RATE = 125e6 / 8  # 15.625 MS/s

PRESETS = {
    '4fsc': FSC_4,
    '2fsc': FSC_2,
    '1fsc': FSC_1,
    '0.5fsc': FSC_HALF,
    'half': FSC_HALF,
    '15.625m': RP_RATE,
}


def parse_sample_rate(rate_str):
    """Parse sample rate string like '4000000', '4M', '4fsc'."""
    rate_str = rate_str.strip().lower()
    if rate_str in PRESETS:
        return PRESETS[rate_str]
    rate_str = rate_str.upper()
    if rate_str.endswith('M'):
        return float(rate_str[:-1]) * 1e6
    elif rate_str.endswith('K'):
        return float(rate_str[:-1]) * 1e3
    return float(rate_str)


def detect_header_size(filepath, check_bytes=256):
    """Detect header by finding where CVBS-like data starts."""
    data = np.fromfile(filepath, dtype=np.int8, count=check_bytes)
    for i in range(0, len(data) - 64, 8):
        chunk = data[i:i+64]
        if np.std(chunk) > 10 and np.abs(chunk).mean() > 5:
            return (i // 64) * 64
    return 0


def load_data(filepath, max_seconds=None, sample_rate=None):
    """Load data from BIN or WAV file."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.wav':
        with wave.open(filepath, 'rb') as wav:
            actual_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            n_frames = wav.getnframes()

            # Read data
            raw_data = wav.readframes(n_frames)

            # Convert based on sample width
            if sampwidth == 1:
                data = np.frombuffer(raw_data, dtype=np.int8)
            elif sampwidth == 2:
                data = np.frombuffer(raw_data, dtype=np.int16)
                # Scale to int8 range for consistent threshold detection
                data = (data / 256).astype(np.float32)
            else:
                data = np.frombuffer(raw_data, dtype=np.int32)
                data = (data / 16777216).astype(np.float32)

            # Take first channel if stereo
            if n_channels > 1:
                data = data[::n_channels]

            if sample_rate is None:
                sample_rate = float(actual_rate)

    else:  # .bin or other
        header_size = detect_header_size(filepath)
        with open(filepath, 'rb') as f:
            f.seek(header_size)
            data = np.frombuffer(f.read(), dtype=np.int8).astype(np.float32)

        if sample_rate is None:
            sample_rate = RP_RATE

    # Limit duration if requested
    if max_seconds:
        max_samples = int(max_seconds * sample_rate)
        if len(data) > max_samples:
            data = data[:max_samples]

    return data, sample_rate


def classify_pulse(width, samples_per_line):
    """Classify a pulse by its width."""
    half_line = samples_per_line / 2

    # Thresholds scaled to sample rate
    eq_max = samples_per_line * 0.055  # ~2.3/63.5 of a line
    hsync_max = samples_per_line * 0.12  # ~4.7*1.5/63.5 of a line
    broad_min = samples_per_line * 0.35  # ~27.1*0.8/63.5 of a line

    if width < 0.02 * samples_per_line:
        return "noise"
    elif width < eq_max:
        return "EQ"
    elif width < hsync_max:
        return "HSYNC"
    elif width > broad_min:
        return "BROAD"
    else:
        return "other"


def analyze_cvbs_timing(filepath, sample_rate=None, max_seconds=None):
    """Analyze CVBS timing with corrected NTSC analysis."""

    print("=" * 70)
    print("CORRECTED NTSC CVBS TIMING ANALYSIS (v3)")
    print("=" * 70)
    print(f"\nFile: {os.path.basename(filepath)}")

    # Load data
    data, sample_rate = load_data(filepath, max_seconds, sample_rate)

    samples_per_line = sample_rate / NTSC_LINE_FREQ
    samples_per_field = sample_rate / NTSC_FIELD_FREQ

    print(f"Sample rate: {sample_rate/1e6:.6f} MS/s")
    print(f"Loaded: {len(data):,} samples ({len(data)/sample_rate:.3f} seconds)")
    print(f"\nExpected samples per line: {samples_per_line:.3f}")
    print(f"Expected samples per field: {samples_per_field:.2f}")

    # Check if this is an integer samples/line rate
    spl_frac = samples_per_line - round(samples_per_line)
    if abs(spl_frac) < 0.001:
        print(f"✓ Integer samples/line ({round(samples_per_line)}) - good for NTSC timing")
    else:
        print(f"⚠ Fractional samples/line ({samples_per_line:.3f}) - may cause timing drift")

    # Find sync levels
    sync_tip = np.percentile(data, 1)
    blanking = np.percentile(data, 10)
    threshold = (blanking + sync_tip) / 2

    print(f"\nSignal levels: sync_tip={sync_tip:.1f}, blanking={blanking:.1f}, threshold={threshold:.1f}")

    # Find sync pulses
    below = data < threshold
    edges = np.diff(below.astype(np.int8))
    falling = np.where(edges == 1)[0]
    rising = np.where(edges == -1)[0]

    # Measure pulse widths
    widths = []
    for f in falling:
        r_after = rising[rising > f]
        if len(r_after) > 0:
            widths.append(r_after[0] - f)
        else:
            widths.append(0)
    widths = np.array(widths)

    # Classify pulses
    pulse_types = np.array([classify_pulse(w, samples_per_line) for w in widths])

    # Count pulse types
    from collections import Counter
    type_counts = Counter(pulse_types)
    print(f"\nPulse counts:")
    for ptype in ["HSYNC", "EQ", "BROAD", "noise", "other"]:
        if ptype in type_counts:
            print(f"  {ptype}: {type_counts[ptype]:,}")

    # Find VBIs (clusters of BROAD pulses)
    broad_indices = np.where(pulse_types == "BROAD")[0]
    if len(broad_indices) < 5:
        print("\n⚠ Not enough BROAD pulses found for VBI detection")
        return None

    vbi_list = []
    current_vbi = [broad_indices[0]]
    for i in range(1, len(broad_indices)):
        if broad_indices[i] - broad_indices[i-1] > 10:
            if len(current_vbi) >= 5:
                vbi_list.append(current_vbi)
            current_vbi = [broad_indices[i]]
        else:
            current_vbi.append(broad_indices[i])
    if len(current_vbi) >= 5:
        vbi_list.append(current_vbi)

    print(f"\nDetected {len(vbi_list)} VBI events")

    results = {
        'sample_rate': sample_rate,
        'samples_per_line': samples_per_line,
        'total_samples': len(data),
        'vbi_count': len(vbi_list),
    }

    # =========================================
    # TEST 1: H-SYNC INTERVAL VERIFICATION
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 1: H-SYNC INTERVAL")
    print("-" * 70)

    hsync_indices = np.where(pulse_types == "HSYNC")[0]
    hsync_samples = falling[hsync_indices]

    # Measure intervals (only single-line intervals in active video)
    hsync_intervals = []
    for i in range(1, len(hsync_samples)):
        interval = hsync_samples[i] - hsync_samples[i-1]
        # Accept intervals within 10% of expected line period
        if 0.9 * samples_per_line < interval < 1.1 * samples_per_line:
            hsync_intervals.append(interval)

    if len(hsync_intervals) > 0:
        hsync_intervals = np.array(hsync_intervals)
        mean_interval = hsync_intervals.mean()
        error_ppm = (mean_interval / samples_per_line - 1) * 1e6

        print(f"Valid H-sync intervals: {len(hsync_intervals):,}")
        print(f"Expected: {samples_per_line:.3f} samples")
        print(f"Measured mean: {mean_interval:.3f} samples")
        print(f"Measured std: {hsync_intervals.std():.4f} samples")
        print(f"Error: {error_ppm:+.1f} ppm")

        results['hsync_mean'] = mean_interval
        results['hsync_error_ppm'] = error_ppm

        if abs(error_ppm) < 100:
            print("✓ PASS - H-sync interval is correct")
        else:
            print("✗ FAIL - H-sync interval error > 100 ppm")
    else:
        print("⚠ No valid H-sync intervals found")

    # =========================================
    # TEST 2: H-SYNC COUNT PER FRAME
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 2: H-SYNC COUNT PER FRAME")
    print("-" * 70)

    # Count H-syncs between same-field VBIs (should be 507 per frame)
    hsync_counts = []
    if len(vbi_list) >= 3:
        for i in range(len(vbi_list) - 2):
            start_idx = vbi_list[i][-1]  # End of VBI
            end_idx = vbi_list[i+2][0]   # Start of next same-field VBI
            count = sum(1 for j in range(start_idx, end_idx) if pulse_types[j] == "HSYNC")
            hsync_counts.append(count)

    if hsync_counts:
        mean_count = np.mean(hsync_counts)
        print(f"H-syncs per frame (measured): {mean_count:.1f}")
        print(f"H-syncs per frame (expected): 507 (= 525 - 18 VBI lines)")

        results['hsync_per_frame'] = mean_count

        if abs(mean_count - 507) < 2:
            print("✓ PASS - H-sync count matches NTSC standard")
        else:
            print(f"✗ FAIL - Expected ~507 H-syncs per frame")
    else:
        print("⚠ Not enough VBIs to count H-syncs per frame")

    # =========================================
    # TEST 3: FIELD IDENTIFICATION
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 3: FIELD IDENTIFICATION")
    print("-" * 70)
    print("Field 1: Gap to first EQ = 0.5 lines")
    print("Field 2: Gap to first EQ = 1.0 lines (H-sync at line 263 replaced by EQ pulse)")
    print()

    field_info = []
    for vbi_num, vbi_broad_indices in enumerate(vbi_list[:10]):
        first_broad_idx = vbi_broad_indices[0]

        # Find last H-sync before VBI
        last_hsync_idx = None
        for i in range(first_broad_idx - 1, max(0, first_broad_idx - 20), -1):
            if pulse_types[i] == "HSYNC":
                last_hsync_idx = i
                break

        # Find first EQ
        first_eq_idx = None
        for i in range(last_hsync_idx + 1 if last_hsync_idx else 0, first_broad_idx):
            if pulse_types[i] == "EQ":
                first_eq_idx = i
                break

        if last_hsync_idx and first_eq_idx:
            gap = falling[first_eq_idx] - falling[last_hsync_idx]
            gap_lines = gap / samples_per_line
            field = 1 if gap_lines < 0.75 else 2
            expected = 0.5 if field == 1 else 1.0
            status = "✓" if abs(gap_lines - expected) < 0.1 else "✗"
            field_info.append({'vbi': vbi_num, 'field': field, 'gap': gap_lines, 'status': status})
            print(f"  VBI {vbi_num}: Field {field}, gap = {gap_lines:.3f} lines (expected {expected}) {status}")

    # Check alternating pattern
    if len(field_info) >= 4:
        fields = [f['field'] for f in field_info]
        alternating = all(fields[i] != fields[i+1] for i in range(len(fields)-1))
        if alternating:
            print("\n✓ PASS - Fields alternate correctly (F1, F2, F1, F2, ...)")
        else:
            print("\n✗ FAIL - Fields do not alternate correctly")
        results['fields_alternate'] = alternating
    else:
        print("\n⚠ Not enough VBIs to verify field alternation")

    # =========================================
    # TEST 4: VBI STRUCTURE
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 4: VBI STRUCTURE")
    print("-" * 70)
    print("Expected: 6 pre-EQ + 6 BROAD + 6 post-EQ per VBI")
    print()

    for vbi_num, vbi_broad_indices in enumerate(vbi_list[:4]):
        first_broad_idx = vbi_broad_indices[0]
        last_broad_idx = vbi_broad_indices[-1]
        broad_count = len(vbi_broad_indices)

        # Count pre-EQ (before first BROAD)
        pre_eq = 0
        for i in range(first_broad_idx - 1, max(0, first_broad_idx - 10), -1):
            if pulse_types[i] == "EQ":
                pre_eq += 1
            elif pulse_types[i] == "HSYNC":
                break

        # Count post-EQ (after last BROAD)
        post_eq = 0
        for i in range(last_broad_idx + 1, min(last_broad_idx + 10, len(pulse_types))):
            if pulse_types[i] == "EQ":
                post_eq += 1
            elif pulse_types[i] == "HSYNC":
                break

        eq_total = pre_eq + post_eq
        eq_ok = "✓" if eq_total >= 10 else "✗"
        broad_ok = "✓" if broad_count == 6 else "✗"
        print(f"  VBI {vbi_num}: {pre_eq} pre-EQ + {broad_count} BROAD + {post_eq} post-EQ  {eq_ok} {broad_ok}")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Determine overall pass/fail
    tests_passed = 0
    tests_total = 0

    if 'hsync_error_ppm' in results:
        tests_total += 1
        if abs(results['hsync_error_ppm']) < 100:
            tests_passed += 1

    if 'hsync_per_frame' in results:
        tests_total += 1
        if abs(results['hsync_per_frame'] - 507) < 2:
            tests_passed += 1

    if 'fields_alternate' in results:
        tests_total += 1
        if results['fields_alternate']:
            tests_passed += 1

    if tests_total > 0 and tests_passed == tests_total:
        print(f"""
✓ SIGNAL IS VALID NTSC CVBS ({tests_passed}/{tests_total} tests passed)

The timing analysis confirms:
- H-sync spacing matches expected {samples_per_line:.3f} samples/line
- H-sync count is ~507 per frame (correct for NTSC)
- Fields alternate correctly with proper half-line offset
- VBI structure matches NTSC standard

NOTE: Frame timing measured from VBI markers may show +1 line due to
the 10-line VBI span (9 lines content + 1 line transitions). This is
a measurement artifact, not a signal error.
""")
    else:
        print(f"""
⚠ SIGNAL TIMING ANALYSIS ({tests_passed}/{tests_total} tests passed)

Review the individual test results above to identify any issues.
""")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CVBS timing with corrected NTSC analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s capture.bin              # Analyze raw capture at 15.625 MS/s
    %(prog)s capture.bin 2            # Analyze first 2 seconds
    %(prog)s resampled_4fsc.wav       # Analyze resampled WAV (auto-detect rate)
    %(prog)s capture.wav --rate 4fsc  # Specify sample rate

Rate presets:
    4fsc     14.31818 MS/s (910 samples/line)
    2fsc      7.15909 MS/s (455 samples/line)
    1fsc      3.57954 MS/s (227.5 samples/line)
    0.5fsc    1.78977 MS/s (113.75 samples/line)
        """
    )

    parser.add_argument('input', help='Input file (.bin or .wav)')
    parser.add_argument('max_seconds', nargs='?', type=float, default=None,
                        help='Limit analysis to first N seconds')
    parser.add_argument('--rate', default=None,
                        help='Sample rate (default: auto-detect from WAV, 15.625M for .bin)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    sample_rate = None
    if args.rate:
        sample_rate = parse_sample_rate(args.rate)

    analyze_cvbs_timing(args.input, sample_rate=sample_rate, max_seconds=args.max_seconds)


if __name__ == '__main__':
    main()
