#!/usr/bin/env python3
"""
Convert 8-bit streaming capture to float32 format for DMG playback.

Usage:
    python convert_to_playback.py input.bin [output.f32]

The output file can then be copied to Red Pitaya for playback.
"""

import sys
import os
import numpy as np

def convert_capture(input_path, output_path=None):
    """Convert 8-bit int capture to float32 for DMG playback."""

    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.f32'

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Load int8 data
    raw = np.fromfile(input_path, dtype=np.int8)
    print(f"\nLoaded {len(raw):,} samples ({raw.nbytes/1024/1024:.2f} MB)")
    print(f"Input range: {raw.min()} to {raw.max()}")

    # Convert to float32 normalized to -1.0 to 1.0
    waveform = raw.astype(np.float32) / 128.0

    # Align to 128 bytes (32 float32 samples) - required by DMG
    aligned_len = (len(waveform) // 32) * 32
    if aligned_len != len(waveform):
        print(f"Aligning from {len(waveform)} to {aligned_len} samples")
        waveform = waveform[:aligned_len]

    print(f"\nOutput: {len(waveform):,} samples ({waveform.nbytes/1024/1024:.2f} MB)")
    print(f"Output range: {waveform.min():.3f} to {waveform.max():.3f}")

    # Check DMG memory limit (128 MB typical)
    DMG_LIMIT = 128 * 1024 * 1024
    if waveform.nbytes > DMG_LIMIT:
        max_samples = DMG_LIMIT // 4
        max_samples = (max_samples // 32) * 32
        print(f"\nWARNING: Output exceeds 128 MB DMG limit!")
        print(f"  Truncating to {max_samples:,} samples ({max_samples*4/1024/1024:.2f} MB)")
        waveform = waveform[:max_samples]

    # Calculate duration
    sample_rate = 15.625e6  # Decimation 8 from 125 MS/s
    duration = len(waveform) / sample_rate
    print(f"\nPlayback duration: {duration:.3f} seconds @ 15.625 MS/s")

    # Save
    waveform.tofile(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {os.path.getsize(output_path)/1024/1024:.2f} MB")

    return output_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_path = convert_capture(input_path, output_path)

    print(f"\nTo copy to Red Pitaya:")
    print(f"  scp {output_path} root@192.168.0.6:/home/jupyter/cvbs_project/cvbs_captures/")


if __name__ == "__main__":
    main()
