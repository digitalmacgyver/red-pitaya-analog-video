#!/usr/bin/env python3
"""
Strip block headers from Red Pitaya streaming capture files.

The rpsa_client streaming capture inserts 112-byte headers at the start
of the file and after every ~8MB block boundary. This script removes
these headers to produce clean continuous signal data.

Header structure (112 bytes):
  Bytes 0-5:   01 00 00 00 00 00
  Byte 6:      80 (marker)
  Bytes 7-21:  zeros
  Byte 22:     80 (marker)
  Bytes 23-35: zeros
  Bytes 36-37: 29 7F (spike - this causes visible glitches)
  Bytes 38-71: zeros
  Bytes 72-74: 28 6B EE (some identifier?)
  Bytes 75-105: zeros
  Bytes 106:   80 (marker)
  Bytes 108-109: 29 7F (another spike)
  Bytes 110-111: zeros/signal start

Before each header (except file start), there's ~14 bytes of FF padding.

Usage:
    python strip_headers.py <input.bin> [-o output.bin]
"""

import argparse
import os
import sys
import numpy as np

# Header signature bytes
HEADER_SIGNATURE = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00])
HEADER_SIZE = 112
BLOCK_SIZE = 8_388_608  # 8 MB

# Padding before headers (0xFF bytes)
PADDING_BYTE = 0xFF


def find_headers(filepath, max_check=None):
    """Find all header positions in the file by scanning entire file."""
    file_size = os.path.getsize(filepath)
    headers = []

    print(f"  Scanning {file_size:,} bytes for header signatures...")

    chunk_size = 64 * 1024 * 1024  # 64 MB chunks
    offset = 0

    with open(filepath, 'rb') as f:
        while offset < file_size:
            # Read chunk with overlap to catch signatures at boundaries
            read_size = min(chunk_size + len(HEADER_SIGNATURE), file_size - offset)
            data = f.read(read_size)

            # Search for header signature in this chunk
            search_pos = 0
            while True:
                pos = data.find(HEADER_SIGNATURE, search_pos)
                if pos < 0 or pos >= chunk_size:
                    break

                abs_pos = offset + pos
                headers.append(abs_pos)
                search_pos = pos + 1

            # Move to next chunk (minus overlap)
            offset += chunk_size
            f.seek(offset)

    # Sort and deduplicate
    headers = sorted(set(headers))

    # Show first few and last few
    if len(headers) <= 10:
        for h in headers:
            print(f"  Found header at byte {h:,}")
    else:
        for h in headers[:3]:
            print(f"  Found header at byte {h:,}")
        print(f"  ... ({len(headers) - 6} more) ...")
        for h in headers[-3:]:
            print(f"  Found header at byte {h:,}")

    return headers


def find_padding_start(data, header_offset):
    """Find where the FF padding starts before a header."""
    # Search backwards from header for continuous FF bytes
    padding_start = header_offset
    for i in range(header_offset - 1, max(0, header_offset - 50), -1):
        if data[i] == PADDING_BYTE:
            padding_start = i
        else:
            break
    return padding_start


def strip_headers(input_path, output_path):
    """Strip all headers and padding from capture file."""
    file_size = os.path.getsize(input_path)

    print(f"Input file: {input_path}")
    print(f"Input size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print()

    # First pass: find all headers
    print("Scanning for block headers...")
    header_positions = find_headers(input_path)

    if not header_positions:
        print("No headers found - file may already be clean")
        return False

    print(f"\nFound {len(header_positions)} headers")
    print()

    # Second pass: copy data, skipping headers and padding
    print("Stripping headers and writing clean data...")

    bytes_written = 0
    bytes_skipped = 0

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        # Read entire file (for simplicity - could chunk for very large files)
        data = fin.read()

        # Sort header positions
        header_positions = sorted(header_positions)

        # Build list of (start, end) ranges to skip
        skip_ranges = []

        for i, header_pos in enumerate(header_positions):
            if header_pos == 0:
                # File start - just skip the header
                skip_ranges.append((0, HEADER_SIZE))
            else:
                # Find padding start
                padding_start = find_padding_start(data, header_pos)
                skip_end = header_pos + HEADER_SIZE
                skip_ranges.append((padding_start, skip_end))

        # Copy data, skipping the ranges
        pos = 0
        for skip_start, skip_end in skip_ranges:
            # Write data before this skip range
            if pos < skip_start:
                fout.write(data[pos:skip_start])
                bytes_written += skip_start - pos

            # Skip the header/padding
            bytes_skipped += skip_end - max(pos, skip_start)
            pos = skip_end

        # Write remaining data after last header
        if pos < len(data):
            fout.write(data[pos:])
            bytes_written += len(data) - pos

    print(f"\nBytes written: {bytes_written:,}")
    print(f"Bytes skipped: {bytes_skipped:,}")
    print(f"Output file: {output_path}")
    print(f"Output size: {os.path.getsize(output_path):,} bytes")

    return True


def verify_clean(filepath):
    """Verify the output file has no headers."""
    print(f"\nVerifying {filepath}...")

    with open(filepath, 'rb') as f:
        # Check start
        data = f.read(len(HEADER_SIGNATURE))
        if data == HEADER_SIGNATURE:
            print("  WARNING: File still has header at start!")
            return False

        # Check for any remaining header signatures
        f.seek(0)
        data = f.read()

        pos = data.find(HEADER_SIGNATURE)
        if pos >= 0:
            print(f"  WARNING: Found header signature at byte {pos}")
            return False

    print("  OK - no headers found")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Strip block headers from Red Pitaya streaming captures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The streaming capture includes 112-byte block headers that cause:
  - Spikes to max value (41 127 = 0x29 0x7F)
  - ~7 microsecond timing shifts at each 8MB boundary

This tool removes these headers to produce clean continuous data.

Examples:
  %(prog)s capture.bin
  %(prog)s capture.bin -o clean_capture.bin
        """
    )

    parser.add_argument("input", help="Input capture file (.bin)")
    parser.add_argument("-o", "--output", help="Output file (default: input_clean.bin)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify, don't strip")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.verify_only:
        headers = find_headers(args.input)
        if headers:
            print(f"\nFile has {len(headers)} headers that should be stripped")
        else:
            print("\nFile is clean")
        sys.exit(0)

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_clean{ext}"

    # Strip headers
    success = strip_headers(args.input, output_path)

    if success:
        # Verify
        verify_clean(output_path)
        print(f"\nDone! Clean file: {output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
