#!/usr/bin/env python3
"""
Optimized CVBS DMG Playback - streams int8 capture directly to DMG.

Eliminates intermediate float32 file and minimizes Python memory usage.
Reads capture in chunks, converts to float32, writes with offset.

Usage (on Red Pitaya):
    python3 cvbs_playback_optimized.py /path/to/capture.bin

The capture file should be raw int8 samples from the streaming app.
If file has a header, use --skip-header to auto-detect and skip it.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, '/opt/redpitaya/lib/python')

def detect_header_size(filepath, check_bytes=256):
    """Detect header by finding where CVBS-like data starts."""
    import numpy as np
    data = np.fromfile(filepath, dtype=np.int8, count=check_bytes)

    # Look for where we get consistent varying data (not mostly zeros)
    for i in range(0, len(data) - 64, 8):
        chunk = data[i:i+64]
        # CVBS data has values spread across range, not clustered near zero
        if np.std(chunk) > 10 and np.abs(chunk).mean() > 5:
            # Align to 64 bytes
            return (i // 64) * 64
    return 0

def playback(filepath, skip_header=False, header_size=None, chunk_mb=8, play=True, loops=0):
    """Stream int8 capture to DMG with minimal memory usage."""
    import numpy as np
    from rp_overlay import overlay
    import rp

    # Initialize
    print("Initializing Red Pitaya...")
    fpga = overlay()
    rp.rp_Init()
    CH = rp.RP_CH_1
    DECIMATION = 8
    SAMPLE_RATE = 125e6 / DECIMATION

    # Get DMG memory
    mem_result = rp.rp_GenAxiGetMemoryRegion()
    MEM_START = mem_result[1]
    MEM_SIZE = mem_result[2]
    max_samples = MEM_SIZE // 2  # int16 storage

    print(f"DMG: {MEM_SIZE/1024/1024:.0f} MB = {max_samples:,} samples = {max_samples/SAMPLE_RATE:.2f}s max")

    # Detect or set header size
    if header_size is None:
        if skip_header:
            header_size = detect_header_size(filepath)
            print(f"Auto-detected header: {header_size} bytes")
        else:
            header_size = 0

    # Calculate file info
    file_size = os.path.getsize(filepath)
    data_size = file_size - header_size
    total_samples = data_size  # int8 = 1 byte per sample

    # Truncate to DMG limit and alignment
    total_samples = min(total_samples, max_samples)
    total_samples = (total_samples // 64) * 64  # 128-byte alignment (64 int16 samples)

    duration = total_samples / SAMPLE_RATE
    print(f"File: {filepath}")
    print(f"  Size: {file_size:,} bytes, header: {header_size}, data: {data_size:,}")
    print(f"  Loading: {total_samples:,} samples = {duration:.3f}s")

    # Reserve DMG memory
    rp.rp_GenReset()
    waveform_bytes = total_samples * 2  # int16 storage
    mem_end = MEM_START + waveform_bytes

    result = rp.rp_GenAxiReserveMemory(CH, MEM_START, mem_end)
    if result != 0:
        print(f"ERROR: Failed to reserve memory: {result}")
        rp.rp_Release()
        return False

    rp.rp_GenAxiSetDecimationFactor(CH, DECIMATION)

    # Stream file to DMG in chunks
    chunk_samples = chunk_mb * 1024 * 1024  # chunk_mb MB as int8
    chunk_samples = (chunk_samples // 64) * 64  # Align

    print(f"Streaming with {chunk_mb} MB chunks...")
    t0 = time.time()

    offset = 0
    chunks = 0
    with open(filepath, 'rb') as f:
        f.seek(header_size)  # Skip header

        while offset < total_samples:
            remaining = total_samples - offset
            read_size = min(chunk_samples, remaining)

            # Read int8 chunk
            int8_data = np.fromfile(f, dtype=np.int8, count=read_size)
            if len(int8_data) == 0:
                break

            # Convert to float32 (-1.0 to 1.0)
            float_data = int8_data.astype(np.float32) / 128.0
            del int8_data  # Free immediately

            # Write with offset
            result = rp.rp_GenAxiWriteWaveformOffset(CH, offset, float_data)
            if result != 0:
                print(f"WARNING: Write at offset {offset} returned {result}")

            offset += len(float_data)
            chunks += 1
            del float_data  # Free immediately

    t1 = time.time()
    print(f"Loaded {offset:,} samples in {chunks} chunks ({t1-t0:.2f}s, {offset*2/(t1-t0)/1024/1024:.1f} MB/s)")

    if not play:
        print("Skipping playback (--no-play)")
        rp.rp_GenAxiReleaseMemory(CH)
        rp.rp_Release()
        return True

    # Playback
    print(f"\nPlaying {duration:.3f}s...")
    rp.rp_GenOutEnable(CH)

    try:
        loop_count = 0
        while True:
            rp.rp_GenAxiSetEnable(CH, True)
            time.sleep(duration)
            loop_count += 1

            if loops == 0:
                break
            elif loops > 0 and loop_count > loops:
                break
            # loops < 0 means infinite

            if loops != 0:
                print(f"  Loop {loop_count}...")

    except KeyboardInterrupt:
        print("\nInterrupted")

    # Cleanup
    rp.rp_GenAxiSetEnable(CH, False)
    rp.rp_GenOutDisable(CH)
    rp.rp_GenAxiReleaseMemory(CH)
    rp.rp_Release()
    print("Done")
    return True


def main():
    parser = argparse.ArgumentParser(description='Optimized CVBS DMG Playback')
    parser.add_argument('file', help='Input capture file (int8 raw samples)')
    parser.add_argument('--skip-header', action='store_true',
                        help='Auto-detect and skip file header')
    parser.add_argument('--header', type=int, default=None,
                        help='Manual header size in bytes')
    parser.add_argument('--chunk-mb', type=int, default=8,
                        help='Chunk size in MB (default: 8)')
    parser.add_argument('--no-play', action='store_true',
                        help='Load only, do not play')
    parser.add_argument('--loop', type=int, default=0,
                        help='Loop count: 0=once, N=N+1 times, -1=infinite')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    success = playback(
        args.file,
        skip_header=args.skip_header,
        header_size=args.header,
        chunk_mb=args.chunk_mb,
        play=not args.no_play,
        loops=args.loop
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
