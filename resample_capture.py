#!/usr/bin/env python3
"""
Resample a capture file to a new sample rate with lowpass filtering.

Converts Red Pitaya streaming captures (.bin, int8) to WAV format at a
specified sample rate, applying an anti-aliasing lowpass filter to prevent
frequency folding artifacts during downsampling.

Usage:
    python resample_capture.py <input.bin> <target_rate> [-o output.wav]

Arguments:
    input.bin       Input capture file (int8 samples from rpsa_client)
    target_rate     Target sample rate: 4M, 4000000, 5.2e6, etc.

Options:
    -o, --output FILE       Output WAV file (default: <input>_<rate>.wav)
    --original-rate RATE    Source sample rate (default: 15.625M)
    --header N              Manual header size in bytes
    --no-skip-header        Don't auto-detect/skip file header

Examples:
    # Resample to 4 MS/s (below streaming limit for continuous playback)
    python resample_capture.py capture.bin 4M

    # Resample to 5.208 MS/s (125/24, 1/3 of original rate)
    python resample_capture.py capture.bin 5.208M

    # Specify output file
    python resample_capture.py capture.bin 4M -o downsampled.wav

    # Use Hz instead of M suffix
    python resample_capture.py capture.bin 4000000

Output:
    - 16-bit mono WAV file at the specified sample rate
    - Data aligned to 128 bytes (Red Pitaya DAC streaming requirement)
    - Viewable in Audacity for waveform analysis
    - Streamable via: rpsa_client.exe -o -h 192.168.0.6 -f wav -d file.wav -r inf

Notes:
    - Original captures are typically 15.625 MS/s (125 MS/s / 8 decimation)
    - Red Pitaya DAC streaming is limited to ~5 MS/s continuous
    - Downsampling to 4-5 MS/s enables continuous playback but loses high frequencies
    - Lowpass filter cutoff is automatically set to Nyquist of target rate
"""

import sys
import os
import argparse
import numpy as np
import wave
from scipy import signal
from math import gcd


def parse_sample_rate(rate_str):
    """Parse sample rate string like '4000000', '4M', '4.5M', '5.2e6'."""
    rate_str = rate_str.strip().upper()

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


def find_rational_approximation(ratio, max_factor=1000):
    """
    Find integer up/down factors for a resampling ratio.

    For scipy.signal.resample_poly(x, up, down):
    - output_rate = input_rate * up / down
    - So for target/original = ratio, we need up/down = ratio

    For downsampling (ratio < 1): up < down
    For upsampling (ratio > 1): up > down
    """
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(max_factor)
    return frac.numerator, frac.denominator  # up, down


def resample_capture(input_path, target_rate, output_path=None,
                     original_rate=15.625e6, header_size=None, skip_header=True):
    """
    Resample capture file with lowpass filtering.

    Args:
        input_path: Path to input .bin file (int8 samples)
        target_rate: Target sample rate in Hz
        output_path: Output .wav path (default: input_rate.wav)
        original_rate: Original sample rate (default: 15.625 MS/s)
        header_size: Header size in bytes (None = auto-detect)
        skip_header: Whether to skip file header

    Returns:
        Path to output file
    """
    # Auto-generate output path if not specified
    if output_path is None:
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

    # Calculate resampling ratio
    ratio = target_rate / original_rate

    if abs(ratio - 1.0) < 1e-9:
        print("Target rate equals original rate, no resampling needed")
        data_resampled = data.astype(np.float32)
    elif ratio > 1.0:
        print("Warning: Upsampling (target > original) - this adds no information")
        up, down = find_rational_approximation(ratio)
        print(f"Resampling ratio: {up}/{down}")
        data_resampled = signal.resample_poly(data.astype(np.float32), up, down)
    else:
        # Downsampling - need lowpass filter
        up, down = find_rational_approximation(ratio)
        print(f"Resampling ratio: {up}/{down} (up by {up}, down by {down})")

        # Calculate effective cutoff
        nyquist_target = target_rate / 2
        print(f"Lowpass cutoff: {nyquist_target/1e6:.3f} MHz (Nyquist of target rate)")

        print("Applying lowpass filter and resampling...")
        data_resampled = signal.resample_poly(data.astype(np.float32), up, down)

    # Convert to int16 (scale int8 range to int16 range)
    data_16 = (data_resampled * 256).astype(np.float32)
    data_16 = np.clip(data_16, -32768, 32767).astype(np.int16)

    # Truncate to multiple of 64 samples (128 bytes) for Red Pitaya compatibility
    original_len = len(data_16)
    aligned_len = (original_len // 64) * 64
    if aligned_len != original_len:
        data_16 = data_16[:aligned_len]
        print(f"Aligned: {original_len:,} -> {aligned_len:,} samples (truncated {original_len - aligned_len})")

    duration = len(data_16) / target_rate
    print(f"Output: {len(data_16):,} samples ({duration:.3f} seconds)")

    # Write WAV file
    print(f"Writing {output_path}...")
    with wave.open(output_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(int(target_rate))
        wav.writeframes(data_16.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Saved: {output_path}")
    print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # Verify alignment
    data_size = file_size - 44  # WAV header is 44 bytes
    if data_size % 128 != 0:
        print(f"Warning: Data size {data_size} is not a multiple of 128 bytes")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Resample capture file with lowpass filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s capture.bin 4M                    # Resample to 4 MS/s
    %(prog)s capture.bin 5.2M -o output.wav    # Specify output file
    %(prog)s capture.bin 4000000               # Use Hz instead of M
    %(prog)s capture.bin 4M --no-skip-header   # Don't skip file header
    %(prog)s capture.bin 4M --original-rate 125M  # Different original rate
        """
    )

    parser.add_argument('input', help='Input capture file (.bin)')
    parser.add_argument('target_rate', help='Target sample rate (e.g., 4M, 4000000, 5.2e6)')
    parser.add_argument('-o', '--output', help='Output WAV file (default: auto-generated)')
    parser.add_argument('--original-rate', default='15.625M',
                        help='Original sample rate (default: 15.625M)')
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
        skip_header=not args.no_skip_header
    )

    print(f"\nTo view in Audacity: Open {output_path}")
    print(f"To stream to Red Pitaya:")
    print(f"  rpsa_client.exe -o -h 192.168.0.6 -f wav -d \"{output_path}\" -r inf")


if __name__ == '__main__':
    main()
