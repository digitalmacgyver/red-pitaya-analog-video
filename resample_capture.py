#!/usr/bin/env python3
"""
Resample a CVBS capture file for Red Pitaya DAC streaming.

Converts Red Pitaya streaming captures (.bin, int8) to WAV format at a
specified sample rate using high-quality sinc interpolation (polyphase FIR).

Usage:
    python resample_capture.py <input.bin> <target_rate> [options]

Arguments:
    input.bin       Input capture file (int8 samples from rpsa_client at 15.625 MS/s)
    target_rate     Target sample rate or preset:
                    - Presets: 4fsc, 2fsc, 1fsc, 0.5fsc
                    - Custom: 4M, 4000000, 5.2e6, etc.

Options:
    -o, --output FILE       Output WAV file (default: <input>_<rate>.wav)
    --original-rate RATE    Source sample rate (default: 15.625M)
    --gain DB               Apply gain in dB (e.g., 5.7 to compensate for pad loss)
    --center                Center signal at 0V before gain (maximizes headroom)
    --header N              Manual header size in bytes
    --no-skip-header        Don't auto-detect/skip file header

Presets (based on NTSC color subcarrier fsc = 3.579545 MHz):
    4fsc    14.31818 MS/s   910 samples/line (standard CVBS digitization)
    2fsc     7.15909 MS/s   455 samples/line (recommended for streaming)
    1fsc     3.57954 MS/s   227.5 samples/line
    0.5fsc   1.78977 MS/s   113.75 samples/line

Examples:
    # Resample to 2fsc for DAC streaming
    python resample_capture.py capture.bin 2fsc

    # Resample with 5.7 dB gain to compensate for impedance matching pad
    python resample_capture.py capture.bin 2fsc --gain 5.7

    # Resample with gain and DC centering for maximum headroom
    python resample_capture.py capture.bin 2fsc --gain 5.7 --center

Resampling Method:
    Uses scipy.signal.resample_poly() which implements a polyphase FIR filter
    with Kaiser-windowed sinc interpolation. This provides high-quality
    anti-aliasing and reconstruction filtering in a single pass.

Output:
    - 16-bit mono WAV file at the specified sample rate
    - Data aligned to 128 bytes (Red Pitaya DAC streaming requirement)
"""

import sys
import os
import argparse
import numpy as np
import wave
from scipy import signal
from fractions import Fraction

# NTSC timing constants
NTSC_FSC = 3579545.0  # Color subcarrier frequency (exactly 315/88 MHz)
NTSC_LINE_FREQ = 15734.264  # Hz

# Sample rate presets
FSC_4 = NTSC_FSC * 4      # 14.31818 MS/s - 910 samples/line
FSC_2 = NTSC_FSC * 2      # 7.15909 MS/s  - 455 samples/line
FSC_1 = NTSC_FSC          # 3.57954 MS/s  - 227.5 samples/line
FSC_HALF = NTSC_FSC / 2   # 1.78977 MS/s  - 113.75 samples/line

# Standard Red Pitaya capture rate
RP_RATE = 125e6 / 8  # 15.625 MS/s

PRESETS = {
    '4fsc': FSC_4,
    '2fsc': FSC_2,
    '1fsc': FSC_1,
    '0.5fsc': FSC_HALF,
    'half': FSC_HALF,
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
    else:
        return float(rate_str)


def detect_header_size(filepath, check_bytes=256):
    """Detect header by finding where CVBS-like data starts."""
    data = np.fromfile(filepath, dtype=np.int8, count=check_bytes)
    for i in range(0, len(data) - 64, 8):
        chunk = data[i:i+64]
        if np.std(chunk) > 10 and np.abs(chunk).mean() > 5:
            return (i // 64) * 64
    return 0


def db_to_linear(db):
    """Convert dB to linear gain factor."""
    return 10 ** (db / 20.0)


def resample_direct(data, original_rate, target_rate, max_denominator=10000):
    """
    Resample data using polyphase FIR filter (sinc interpolation).

    Uses scipy.signal.resample_poly which implements:
    - Upsampling by integer factor
    - FIR lowpass filter (Kaiser-windowed sinc)
    - Downsampling by integer factor

    This is a high-quality single-pass resampling method.
    """
    ratio = target_rate / original_rate
    frac = Fraction(ratio).limit_denominator(max_denominator)

    print(f"\nResampling: {original_rate/1e6:.6f} MS/s -> {target_rate/1e6:.6f} MS/s")
    print(f"  Ratio: {ratio:.10f}")
    print(f"  Rational approximation: {frac.numerator}/{frac.denominator}")
    print(f"  Approximation error: {abs(ratio - frac.numerator/frac.denominator) * 1e6:.3f} ppm")

    if frac.numerator == frac.denominator:
        print("  No resampling needed (1:1 ratio)")
        return data

    print(f"  Applying polyphase FIR filter (Kaiser-windowed sinc)...")

    # resample_poly handles the anti-aliasing filter automatically
    resampled = signal.resample_poly(data, frac.numerator, frac.denominator)

    print(f"  Input: {len(data):,} samples")
    print(f"  Output: {len(resampled):,} samples")

    return resampled


def analyze_signal(data, name="Signal"):
    """Analyze signal levels for CVBS."""
    p1 = np.percentile(data, 1)    # Sync tip
    p10 = np.percentile(data, 10)  # ~Blanking
    p50 = np.percentile(data, 50)  # Mid
    p99 = np.percentile(data, 99)  # White

    print(f"\n{name} levels (int8 units, ±127 = ±1V):")
    print(f"  Sync tip (1%%):   {p1:+7.1f}  ({p1/127:+.3f}V)")
    print(f"  Blanking (10%%):  {p10:+7.1f}  ({p10/127:+.3f}V)")
    print(f"  Mid (50%%):       {p50:+7.1f}  ({p50/127:+.3f}V)")
    print(f"  White (99%%):     {p99:+7.1f}  ({p99/127:+.3f}V)")
    print(f"  Peak-to-peak:    {p99-p1:7.1f}  ({(p99-p1)/127:.3f}V)")
    print(f"  Center:          {(p1+p99)/2:+7.1f}  ({(p1+p99)/2/127:+.3f}V)")

    return {'sync': p1, 'blank': p10, 'mid': p50, 'white': p99}


def resample_capture(input_path, target_rate, output_path=None,
                     original_rate=RP_RATE, header_size=None,
                     skip_header=True, gain_db=0.0, center_signal=False):
    """
    Resample capture file with optional gain and centering.
    """
    # Auto-generate output path if not specified
    if output_path is None:
        for name, rate in PRESETS.items():
            if abs(target_rate - rate) < 1:
                rate_str = name
                break
        else:
            rate_str = f"{target_rate/1e6:.1f}M".replace('.0M', 'M')

        base = os.path.splitext(input_path)[0]
        if gain_db != 0:
            output_path = f"{base}_{rate_str}_{gain_db:+.1f}dB.wav"
        else:
            output_path = f"{base}_{rate_str}.wav"

    # Detect header
    if header_size is None and skip_header:
        header_size = detect_header_size(input_path)
        if header_size > 0:
            print(f"Auto-detected header: {header_size} bytes")
    elif header_size is None:
        header_size = 0

    # Read input file
    print(f"Reading {input_path}...")
    with open(input_path, 'rb') as f:
        f.seek(header_size)
        data = np.frombuffer(f.read(), dtype=np.int8)

    print(f"Loaded {len(data):,} samples ({len(data)/1024/1024:.2f} MB)")
    print(f"Original rate: {original_rate/1e6:.6f} MS/s")
    print(f"Target rate: {target_rate/1e6:.6f} MS/s")

    # Show NTSC timing info
    samples_per_line_orig = original_rate / NTSC_LINE_FREQ
    samples_per_line_target = target_rate / NTSC_LINE_FREQ
    print(f"\nNTSC samples per line:")
    print(f"  Original: {samples_per_line_orig:.3f}")
    print(f"  Target: {samples_per_line_target:.3f}")

    # Analyze input signal
    levels = analyze_signal(data, "Input")

    # Convert to float for processing
    data_float = data.astype(np.float64)

    # Apply centering if requested (before gain for maximum headroom)
    if center_signal:
        center = (levels['sync'] + levels['white']) / 2
        print(f"\nCentering signal: shifting by {-center:+.1f} ({-center/127:+.3f}V)")
        data_float = data_float - center

        # Show new levels
        new_min = levels['sync'] - center
        new_max = levels['white'] - center
        print(f"  New range: {new_min/127:+.3f}V to {new_max/127:+.3f}V")

    # Apply gain if specified
    if gain_db != 0:
        gain_linear = db_to_linear(gain_db)
        print(f"\nApplying gain: {gain_db:+.1f} dB ({gain_linear:.3f}x linear)")
        data_float = data_float * gain_linear

        # Check for clipping
        max_val = np.max(np.abs(data_float))
        if max_val > 127:
            headroom_db = 20 * np.log10(127 / max_val)
            print(f"  WARNING: Signal will clip! Peak: {max_val:.1f} (limit: 127)")
            print(f"  Headroom needed: {-headroom_db:.1f} dB")
            print(f"  Clipping {np.sum(np.abs(data_float) > 127):,} samples")
        else:
            headroom_db = 20 * np.log10(127 / max_val)
            print(f"  Peak after gain: {max_val:.1f} (headroom: {headroom_db:.1f} dB)")

    # Resample
    data_resampled = resample_direct(data_float, original_rate, target_rate)

    # Convert to int16 for WAV output
    # Scale: int8 range (±127) -> int16 range (±32767)
    # Factor: 32767/127 ≈ 258
    scale_factor = 32767.0 / 127.0
    data_16 = data_resampled * scale_factor

    # Clip to int16 range and convert
    clipped = np.sum(np.abs(data_16) > 32767)
    if clipped > 0:
        print(f"\nClipping {clipped:,} samples to int16 range")
    data_16 = np.clip(data_16, -32768, 32767).astype(np.int16)

    # Truncate to multiple of 64 samples (128 bytes) for Red Pitaya compatibility
    original_len = len(data_16)
    aligned_len = (original_len // 64) * 64
    if aligned_len != original_len:
        data_16 = data_16[:aligned_len]
        print(f"\n128-byte alignment: {original_len:,} -> {aligned_len:,} samples")

    duration = len(data_16) / target_rate
    print(f"\nOutput: {len(data_16):,} samples ({duration:.3f} seconds)")

    # Write WAV file
    print(f"Writing {output_path}...")
    with wave.open(output_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(int(round(target_rate)))
        wav.writeframes(data_16.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Saved: {output_path}")
    print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Resample CVBS capture for Red Pitaya DAC streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
    4fsc     14.31818 MS/s   910 samples/line (standard)
    2fsc      7.15909 MS/s   455 samples/line (streaming)
    1fsc      3.57954 MS/s   227.5 samples/line
    0.5fsc    1.78977 MS/s   113.75 samples/line

Examples:
    %(prog)s capture.bin 2fsc                    # For streaming
    %(prog)s capture.bin 2fsc --gain 5.7         # Compensate for pad loss
    %(prog)s capture.bin 2fsc --gain 5.7 --center  # With DC centering
        """
    )

    parser.add_argument('input', help='Input capture file (.bin)')
    parser.add_argument('target_rate',
                        help='Target rate: 4fsc, 2fsc, 1fsc, 0.5fsc, or custom')
    parser.add_argument('-o', '--output', help='Output WAV file')
    parser.add_argument('--original-rate', default='15.625M',
                        help='Original sample rate (default: 15.625M)')
    parser.add_argument('--gain', type=float, default=0.0,
                        help='Gain in dB (e.g., 5.7 for pad compensation)')
    parser.add_argument('--center', action='store_true',
                        help='Center signal at 0V before gain (maximizes headroom)')
    parser.add_argument('--no-skip-header', action='store_true',
                        help='Do not skip file header')
    parser.add_argument('--header', type=int, default=None,
                        help='Manual header size in bytes')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    target_rate = parse_sample_rate(args.target_rate)
    original_rate = parse_sample_rate(args.original_rate)

    output_path = resample_capture(
        args.input,
        target_rate,
        output_path=args.output,
        original_rate=original_rate,
        header_size=args.header,
        skip_header=not args.no_skip_header,
        gain_db=args.gain,
        center_signal=args.center
    )

    print(f"\nTo stream to Red Pitaya:")
    print(f"  python dac_stream.py \"{output_path}\"")


if __name__ == '__main__':
    main()
