#!/usr/bin/env python3
"""Scan capture file for periodic glitches (spikes/discontinuities)."""

import sys
import numpy as np
import wave

def scan_wav(filepath, chunk_samples=1_000_000):
    """Scan WAV file for glitches."""
    with wave.open(filepath, 'rb') as w:
        sample_rate = w.getframerate()
        total_frames = w.getnframes()
        sample_width = w.getsampwidth()

        print(f"File: {filepath}")
        print(f"Sample rate: {sample_rate:,} Hz")
        print(f"Total frames: {total_frames:,}")
        print(f"Sample width: {sample_width} bytes")
        print(f"Duration: {total_frames/sample_rate:.3f} seconds")
        print()

        # Look for spikes to max value
        max_val = 32767 if sample_width == 2 else 127
        spike_threshold = int(max_val * 0.95)  # 95% of max

        glitch_positions = []
        offset = 0

        while offset < total_frames:
            chunk_size = min(chunk_samples, total_frames - offset)
            raw = w.readframes(chunk_size)

            if sample_width == 2:
                data = np.frombuffer(raw, dtype=np.int16)
            else:
                data = np.frombuffer(raw, dtype=np.int8)

            # Find samples near max
            spikes = np.where(data >= spike_threshold)[0]

            for idx in spikes:
                abs_pos = offset + idx
                # Skip if too close to previous glitch (same event)
                if glitch_positions and abs_pos - glitch_positions[-1][0] < 1000:
                    continue
                glitch_positions.append((abs_pos, data[idx]))

            offset += chunk_size

        print(f"Found {len(glitch_positions)} spike events (>= {spike_threshold})")
        print()

        if len(glitch_positions) > 1:
            # Calculate intervals
            intervals = []
            for i in range(1, len(glitch_positions)):
                interval = glitch_positions[i][0] - glitch_positions[i-1][0]
                intervals.append(interval)

            intervals = np.array(intervals)

            print("Interval statistics (in samples):")
            print(f"  Mean: {np.mean(intervals):,.0f}")
            print(f"  Median: {np.median(intervals):,.0f}")
            print(f"  Std: {np.std(intervals):,.0f}")
            print(f"  Min: {np.min(intervals):,}")
            print(f"  Max: {np.max(intervals):,}")
            print()

            # Convert to bytes (assuming original was 8-bit capture)
            mean_bytes = np.mean(intervals) * 1  # 1 byte per sample in original
            print(f"Mean interval in bytes (8-bit): {mean_bytes:,.0f}")
            print(f"  As MB: {mean_bytes/1024/1024:.3f} MB")
            print(f"  As power of 2: 2^{np.log2(mean_bytes):.2f}")
            print()

            # Time interval
            mean_time = np.mean(intervals) / sample_rate
            print(f"Mean interval in time: {mean_time:.4f} seconds")
            print(f"  As frames (29.97 fps): {mean_time * 29.97:.1f} frames")
            print()

            # Show first 10 glitches
            print("First 10 glitch positions:")
            for i, (pos, val) in enumerate(glitch_positions[:10]):
                time_sec = pos / sample_rate
                byte_pos = pos * 1  # 8-bit original
                print(f"  {i+1}: sample {pos:,} = {time_sec:.4f}s = byte {byte_pos:,} ({byte_pos/1024/1024:.3f} MB), value={val}")


def scan_bin(filepath, chunk_size=64*1024*1024):
    """Scan binary file for glitches."""
    import os
    file_size = os.path.getsize(filepath)

    print(f"File: {filepath}")
    print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

    # Assume 8-bit signed samples at 15.625 MS/s
    sample_rate = 15_625_000
    total_samples = file_size
    print(f"Duration (assuming 8-bit): {total_samples/sample_rate:.3f} seconds")
    print()

    spike_threshold = 120  # ~94% of 127

    glitch_positions = []
    offset = 0

    with open(filepath, 'rb') as f:
        while offset < file_size:
            read_size = min(chunk_size, file_size - offset)
            raw = f.read(read_size)
            data = np.frombuffer(raw, dtype=np.int8)

            # Find samples near max
            spikes = np.where(data >= spike_threshold)[0]

            for idx in spikes:
                abs_pos = offset + idx
                if glitch_positions and abs_pos - glitch_positions[-1][0] < 1000:
                    continue
                glitch_positions.append((abs_pos, data[idx]))

            offset += read_size

    print(f"Found {len(glitch_positions)} spike events (>= {spike_threshold})")
    print()

    if len(glitch_positions) > 1:
        intervals = []
        for i in range(1, len(glitch_positions)):
            interval = glitch_positions[i][0] - glitch_positions[i-1][0]
            intervals.append(interval)

        intervals = np.array(intervals)

        print("Interval statistics (in bytes/samples):")
        print(f"  Mean: {np.mean(intervals):,.0f}")
        print(f"  Median: {np.median(intervals):,.0f}")
        print(f"  Std: {np.std(intervals):,.0f}")
        print(f"  Min: {np.min(intervals):,}")
        print(f"  Max: {np.max(intervals):,}")
        print()

        mean_bytes = np.mean(intervals)
        print(f"Mean interval: {mean_bytes:,.0f} bytes")
        print(f"  As MB: {mean_bytes/1024/1024:.3f} MB")
        print(f"  As power of 2: 2^{np.log2(mean_bytes):.2f}")
        print()

        mean_time = np.mean(intervals) / sample_rate
        print(f"Mean interval in time: {mean_time:.4f} seconds")
        print(f"  As frames (29.97 fps): {mean_time * 29.97:.1f} frames")
        print()

        print("First 10 glitch positions:")
        for i, (pos, val) in enumerate(glitch_positions[:10]):
            time_sec = pos / sample_rate
            print(f"  {i+1}: byte {pos:,} ({pos/1024/1024:.3f} MB) = {time_sec:.4f}s, value={val}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_glitches.py <file.wav|file.bin>")
        sys.exit(1)

    filepath = sys.argv[1]

    if filepath.endswith('.wav'):
        scan_wav(filepath)
    else:
        scan_bin(filepath)
