#!/usr/bin/env python3
"""
Visualize CVBS capture data from Red Pitaya.

Usage:
    python visualize_capture.py cvbs_capture.bin
    python visualize_capture.py cvbs_capture.bin cvbs_capture_meta.txt
"""

import sys
import os
import numpy as np
from matplotlib import pyplot as plt


def load_metadata(meta_path):
    """Load metadata from text file."""
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
    return metadata


def load_capture(bin_path, meta_path=None):
    """Load capture data from binary file."""
    # Load raw binary data as int16
    data = np.fromfile(bin_path, dtype='<i2')  # Little-endian int16

    # Load metadata
    if meta_path is None:
        meta_path = bin_path.replace('.bin', '_meta.txt')
    metadata = load_metadata(meta_path)

    # Use metadata or defaults
    sample_rate = metadata.get('sample_rate', 15.625e6)

    return data, metadata, sample_rate


def plot_capture(data, sample_rate, title="CVBS Capture"):
    """Plot the captured data."""
    # Calculate samples per video line (~63.5µs for NTSC)
    samples_per_line = int(63.5e-6 * sample_rate)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: First 16 video lines
    preview_samples = samples_per_line * 16
    t_us = np.arange(min(preview_samples, len(data))) / sample_rate * 1e6
    axes[0].plot(t_us, data[:len(t_us)])
    axes[0].set_xlabel('Time (µs)')
    axes[0].set_ylabel('ADC Value (raw)')
    axes[0].set_title(f'First ~16 video lines ({len(t_us)} samples)')
    axes[0].grid(True, alpha=0.3)

    # Add voltage scale on right axis
    ax0_v = axes[0].twinx()
    ax0_v.set_ylim(axes[0].get_ylim()[0] / 8192.0, axes[0].get_ylim()[1] / 8192.0)
    ax0_v.set_ylabel('Voltage (V)')

    # Plot 2: Full capture overview (decimated)
    display_points = 10000
    decimate_factor = max(1, len(data) // display_points)
    t_ms = np.arange(0, len(data), decimate_factor) / sample_rate * 1000
    axes[1].plot(t_ms, data[::decimate_factor])
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('ADC Value (raw)')
    duration = len(data) / sample_rate
    axes[1].set_title(f'Full capture overview ({duration:.2f}s, decimated {decimate_factor}x)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Histogram
    axes[2].hist(data, bins=200, edgecolor='none', alpha=0.7)
    axes[2].set_xlabel('ADC Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Value Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def print_stats(data, metadata, sample_rate):
    """Print capture statistics."""
    print("\n" + "="*50)
    print("CAPTURE STATISTICS")
    print("="*50)

    print(f"\nFile info:")
    print(f"  Samples: {len(data):,}")
    print(f"  Duration: {len(data) / sample_rate:.3f} seconds")
    print(f"  Sample rate: {sample_rate/1e6:.3f} MS/s")

    print(f"\nADC values:")
    print(f"  Min: {data.min()}")
    print(f"  Max: {data.max()}")
    print(f"  Mean: {data.mean():.1f}")
    print(f"  Std: {data.std():.1f}")

    print(f"\nVoltage (assuming LV mode ±1V):")
    print(f"  Min: {data.min() / 8192.0:.3f} V")
    print(f"  Max: {data.max() / 8192.0:.3f} V")
    print(f"  Mean: {data.mean() / 8192.0:.3f} V")

    # CVBS signal analysis
    print(f"\nCVBS signal analysis:")
    sync_threshold = -2000  # Approximate sync tip level
    blank_level = 0
    white_level = 5000

    sync_samples = np.sum(data < sync_threshold)
    print(f"  Samples below sync threshold ({sync_threshold}): {sync_samples:,} ({100*sync_samples/len(data):.1f}%)")

    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bin_path = sys.argv[1]
    meta_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(bin_path):
        print(f"Error: File not found: {bin_path}")
        sys.exit(1)

    print(f"Loading {bin_path}...")
    data, metadata, sample_rate = load_capture(bin_path, meta_path)

    print_stats(data, metadata, sample_rate)

    print("\nGenerating plots...")
    fig = plot_capture(data, sample_rate, title=os.path.basename(bin_path))

    # Save plot
    plot_path = bin_path.replace('.bin', '_plot.png')
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")

    plt.show()


if __name__ == '__main__':
    main()
