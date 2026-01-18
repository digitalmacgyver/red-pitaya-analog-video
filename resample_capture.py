#!/usr/bin/env python3
"""
Resample a CVBS capture file with proper NTSC timing alignment.

Converts Red Pitaya streaming captures (.bin, int8) to WAV format at a
specified sample rate. First resamples to 4×fsc (14.31818 MHz) for proper
NTSC timing alignment, then to the target rate.

Usage:
    python resample_capture.py <input.bin> <target_rate> [-o output.wav]

Arguments:
    input.bin       Input capture file (int8 samples from rpsa_client at 15.625 MS/s)
    target_rate     Target sample rate or preset:
                    - Presets: 4fsc, 2fsc, 1fsc, 0.5fsc
                    - Custom: 4M, 4000000, 5.2e6, etc.

Options:
    -o, --output FILE       Output WAV file (default: <input>_<rate>.wav)
    --original-rate RATE    Source sample rate (default: 15.625M)
    --header N              Manual header size in bytes
    --no-skip-header        Don't auto-detect/skip file header
    --direct                Skip 4fsc intermediate step (not recommended)

Presets (based on NTSC color subcarrier fsc = 3.579545 MHz):
    4fsc    14.31818 MS/s   910 samples/line (standard CVBS digitization)
    2fsc     7.15909 MS/s   455 samples/line
    1fsc     3.57954 MS/s   227.5 samples/line
    0.5fsc   1.78977 MS/s   113.75 samples/line

Examples:
    # Resample to 4×fsc (best quality, standard CVBS rate)
    python resample_capture.py capture.bin 4fsc

    # Resample to 2×fsc (half bandwidth)
    python resample_capture.py capture.bin 2fsc

    # Resample to custom rate (e.g., for streaming)
    python resample_capture.py capture.bin 4M

Why 4×fsc matters:
    The Red Pitaya captures at 15.625 MS/s, which gives 993.056 samples per
    NTSC line - a fractional number that causes timing drift. Resampling to
    4×fsc (14.31818 MS/s) gives exactly 910 samples per line, eliminating
    the fractional accumulation error.

Output:
    - 16-bit mono WAV file at the specified sample rate
    - Data aligned to 128 bytes (Red Pitaya DAC streaming requirement)
    - Viewable in Audacity for waveform analysis
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
FSC_4 = NTSC_FSC * 4  # 14.31818 MHz
FSC_2 = NTSC_FSC * 2  # 7.15909 MHz
FSC_1 = NTSC_FSC      # 3.57954 MHz
FSC_HALF = NTSC_FSC / 2  # 1.78977 MHz

# Standard Red Pitaya capture rate
RP_RATE = 125e6 / 8  # 15.625 MS/s

# Preset rates
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

    # Check presets first
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


def resample_via_4fsc(data, original_rate, target_rate):
    """
    Resample data via 4×fsc intermediate for proper NTSC timing.

    1. Resample from original_rate to 4×fsc
    2. Resample from 4×fsc to target_rate

    This ensures proper alignment with NTSC timing (910 samples/line at 4fsc).
    """
    print(f"\nResampling via 4×fsc intermediate:")
    print(f"  {original_rate/1e6:.6f} MS/s -> {FSC_4/1e6:.6f} MS/s -> {target_rate/1e6:.6f} MS/s")

    # Step 1: Resample to 4×fsc
    ratio1 = FSC_4 / original_rate
    frac1 = Fraction(ratio1).limit_denominator(10000)
    print(f"\n  Step 1: {original_rate/1e6:.3f} -> {FSC_4/1e6:.6f} MS/s (ratio {frac1.numerator}/{frac1.denominator})")

    if frac1.numerator != frac1.denominator:
        print(f"    Applying lowpass filter and resampling...")
        data_4fsc = signal.resample_poly(data, frac1.numerator, frac1.denominator)
    else:
        data_4fsc = data
    print(f"    Result: {len(data_4fsc):,} samples")

    # Step 2: Resample from 4×fsc to target
    if abs(target_rate - FSC_4) < 1:  # Target is 4fsc
        print(f"\n  Step 2: Target is 4×fsc, no further resampling needed")
        return data_4fsc

    ratio2 = target_rate / FSC_4
    frac2 = Fraction(ratio2).limit_denominator(10000)
    print(f"\n  Step 2: {FSC_4/1e6:.6f} -> {target_rate/1e6:.6f} MS/s (ratio {frac2.numerator}/{frac2.denominator})")

    if frac2.numerator != frac2.denominator:
        print(f"    Applying lowpass filter and resampling...")
        data_target = signal.resample_poly(data_4fsc, frac2.numerator, frac2.denominator)
    else:
        data_target = data_4fsc

    print(f"    Result: {len(data_target):,} samples")
    return data_target


def resample_direct(data, original_rate, target_rate):
    """Resample directly without 4×fsc intermediate (not recommended for NTSC)."""
    ratio = target_rate / original_rate
    frac = Fraction(ratio).limit_denominator(10000)
    print(f"\nDirect resampling: ratio {frac.numerator}/{frac.denominator}")
    print(f"  WARNING: Direct resampling may cause timing drift with NTSC")

    if frac.numerator != frac.denominator:
        return signal.resample_poly(data, frac.numerator, frac.denominator)
    return data


def resample_capture(input_path, target_rate, output_path=None,
                     original_rate=RP_RATE, header_size=None,
                     skip_header=True, use_direct=False):
    """
    Resample capture file with proper NTSC timing alignment.
    """
    # Auto-generate output path if not specified
    if output_path is None:
        # Format rate for filename
        for name, rate in PRESETS.items():
            if abs(target_rate - rate) < 1:
                rate_str = name
                break
        else:
            rate_str = f"{target_rate/1e6:.1f}M".replace('.0M', 'M')
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_{rate_str}.wav"

    # Detect header
    if header_size is None and skip_header:
        header_size = detect_header_size(input_path)
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
    samples_per_line_orig = original_rate / 15734.264
    samples_per_line_target = target_rate / 15734.264
    print(f"\nNTSC samples per line:")
    print(f"  Original: {samples_per_line_orig:.3f}")
    print(f"  Target: {samples_per_line_target:.3f}")

    # Convert to float for processing
    data_float = data.astype(np.float32)

    # Resample
    if use_direct:
        data_resampled = resample_direct(data_float, original_rate, target_rate)
    else:
        data_resampled = resample_via_4fsc(data_float, original_rate, target_rate)

    # Convert to int16 (scale int8 range to int16 range)
    data_16 = (data_resampled * 256).astype(np.float32)
    data_16 = np.clip(data_16, -32768, 32767).astype(np.int16)

    # Truncate to multiple of 64 samples (128 bytes) for Red Pitaya compatibility
    original_len = len(data_16)
    aligned_len = (original_len // 64) * 64
    if aligned_len != original_len:
        data_16 = data_16[:aligned_len]
        print(f"\nAligned: {original_len:,} -> {aligned_len:,} samples")

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
        description='Resample CVBS capture with NTSC timing alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
    4fsc     14.31818 MS/s   910 samples/line (standard)
    2fsc      7.15909 MS/s   455 samples/line
    1fsc      3.57954 MS/s   227.5 samples/line
    0.5fsc    1.78977 MS/s   113.75 samples/line

Examples:
    %(prog)s capture.bin 4fsc                 # Standard CVBS rate
    %(prog)s capture.bin 2fsc -o output.wav   # Half rate
    %(prog)s capture.bin 4M                   # Custom rate
        """
    )

    parser.add_argument('input', help='Input capture file (.bin)')
    parser.add_argument('target_rate',
                        help='Target rate: 4fsc, 2fsc, 1fsc, 0.5fsc, or custom (4M, 4000000)')
    parser.add_argument('-o', '--output', help='Output WAV file')
    parser.add_argument('--original-rate', default='15.625M',
                        help='Original sample rate (default: 15.625M)')
    parser.add_argument('--no-skip-header', action='store_true',
                        help='Do not skip file header')
    parser.add_argument('--header', type=int, default=None,
                        help='Manual header size in bytes')
    parser.add_argument('--direct', action='store_true',
                        help='Direct resampling without 4fsc intermediate')

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
        use_direct=args.direct
    )

    print(f"\nTo view in Audacity: Open {output_path}")
    print(f"To stream to Red Pitaya:")
    print(f"  rpsa_client.exe -o -h 192.168.0.6 -f wav -d \"{output_path}\" -r inf")


if __name__ == '__main__':
    main()
