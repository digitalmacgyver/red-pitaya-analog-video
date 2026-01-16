#!/usr/bin/env python3
"""
Red Pitaya CVBS Streaming Receiver

This script connects to the Red Pitaya streaming server and saves
continuous ADC data to a file. Supports unlimited capture duration
(limited only by disk space).

Usage:
    python stream_receiver.py [options]

Requirements:
    - Red Pitaya with streaming server running (OS 2.07-43+ recommended)
    - Network connection to Red Pitaya
"""

import socket
import struct
import numpy as np
import time
import argparse
import sys
from pathlib import Path


class RedPitayaStreamReceiver:
    """Receives streaming data from Red Pitaya over TCP."""

    # Default ports for OS 2.07-43+
    CMD_PORT = 5000      # Command/control port
    DATA_PORT = 18900    # ADC streaming data port

    def __init__(self, host: str, timeout: float = 10.0):
        self.host = host
        self.timeout = timeout
        self.data_socket = None
        self.cmd_socket = None

    def connect(self):
        """Connect to the streaming data port."""
        print(f"Connecting to {self.host}:{self.DATA_PORT}...")
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.settimeout(self.timeout)
        self.data_socket.connect((self.host, self.DATA_PORT))
        self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Increase receive buffer for high throughput
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        print("Connected!")

    def disconnect(self):
        """Close the connection."""
        if self.data_socket:
            self.data_socket.close()
            self.data_socket = None
        if self.cmd_socket:
            self.cmd_socket.close()
            self.cmd_socket = None

    def receive_data(self, output_file: str, duration_sec: float, sample_rate: float = 15.625e6):
        """
        Receive streaming data and save to file.

        Args:
            output_file: Path to output .npy file
            duration_sec: Duration to capture in seconds
            sample_rate: Expected sample rate in Hz
        """
        expected_bytes = int(duration_sec * sample_rate * 2)  # 16-bit samples
        chunk_size = 1024 * 1024  # 1 MB chunks

        print(f"Capturing {duration_sec}s of data (~{expected_bytes / 1024 / 1024:.1f} MB)...")
        print(f"Sample rate: {sample_rate / 1e6:.3f} MS/s")
        print(f"Output file: {output_file}")
        print()

        received_bytes = 0
        chunks = []
        start_time = time.time()
        last_report = start_time

        try:
            while received_bytes < expected_bytes:
                # Calculate remaining bytes
                remaining = expected_bytes - received_bytes
                to_read = min(chunk_size, remaining)

                # Receive data
                data = b''
                while len(data) < to_read:
                    try:
                        chunk = self.data_socket.recv(to_read - len(data))
                        if not chunk:
                            print("\nConnection closed by server")
                            break
                        data += chunk
                    except socket.timeout:
                        print("\nTimeout waiting for data")
                        break

                if not data:
                    break

                chunks.append(data)
                received_bytes += len(data)

                # Progress report every second
                now = time.time()
                if now - last_report >= 1.0:
                    elapsed = now - start_time
                    rate = received_bytes / elapsed / 1024 / 1024
                    pct = received_bytes / expected_bytes * 100
                    remaining_sec = (expected_bytes - received_bytes) / (received_bytes / elapsed) if elapsed > 0 else 0
                    print(f"\r  {pct:5.1f}% | {received_bytes / 1024 / 1024:8.1f} MB | {rate:5.1f} MB/s | ETA: {remaining_sec:5.1f}s", end='', flush=True)
                    last_report = now

        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user")

        elapsed = time.time() - start_time
        print(f"\n\nReceived {received_bytes:,} bytes in {elapsed:.1f}s ({received_bytes / elapsed / 1024 / 1024:.1f} MB/s)")

        # Convert to numpy array
        print("Converting to numpy array...")
        all_data = b''.join(chunks)
        samples = np.frombuffer(all_data, dtype=np.int16)

        # Save to file
        print(f"Saving {len(samples):,} samples to {output_file}...")
        np.save(output_file, samples)

        # Save metadata
        metadata = {
            'sample_rate': sample_rate,
            'num_samples': len(samples),
            'duration_sec': len(samples) / sample_rate,
            'capture_time': elapsed,
            'timestamp': time.time(),
            'dtype': 'int16',
            'bits': 14,
            'source': f'stream://{self.host}:{self.DATA_PORT}'
        }
        metadata_file = output_file.replace('.npy', '_metadata.npy')
        np.save(metadata_file, metadata)

        print(f"Done! Saved to {output_file}")
        print(f"Actual capture duration: {len(samples) / sample_rate:.3f}s")

        return samples


def start_streaming_server_instructions():
    """Print instructions for starting the streaming server."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RED PITAYA STREAMING SERVER SETUP                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Option 1: Web Interface                                                     ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  1. Open http://192.168.1.169/ in browser                                    ║
║  2. Click on "Streaming" application                                         ║
║  3. Configure:                                                               ║
║     - Mode: Network                                                          ║
║     - Protocol: TCP                                                          ║
║     - Channels: CH1 (or both)                                                ║
║     - Decimation: 8 (for 15.625 MS/s)                                        ║
║     - Data format: RAW (16-bit)                                              ║
║  4. Click "RUN"                                                              ║
║                                                                              ║
║  Option 2: Command Line (via SSH)                                            ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  ssh root@192.168.1.169                                                      ║
║  /opt/redpitaya/bin/streaming-server \\                                       ║
║      --decimation=8 \\                                                        ║
║      --channels=1 \\                                                          ║
║      --format=raw                                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description='Red Pitaya CVBS Streaming Receiver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d 10                    # Capture 10 seconds
  %(prog)s -d 60 -o long_capture    # Capture 60 seconds to long_capture.npy
  %(prog)s --host 192.168.1.100     # Use different IP address
  %(prog)s --help-server            # Show server setup instructions
        """
    )
    parser.add_argument('--host', '-H', default='192.168.1.169',
                        help='Red Pitaya IP address (default: 192.168.1.169)')
    parser.add_argument('--duration', '-d', type=float, default=2.0,
                        help='Capture duration in seconds (default: 2.0)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output filename (without .npy extension)')
    parser.add_argument('--sample-rate', '-r', type=float, default=15.625e6,
                        help='Sample rate in Hz (default: 15.625e6 for DEC_8)')
    parser.add_argument('--port', '-p', type=int, default=18900,
                        help='Streaming data port (default: 18900)')
    parser.add_argument('--help-server', action='store_true',
                        help='Show streaming server setup instructions')

    args = parser.parse_args()

    if args.help_server:
        start_streaming_server_instructions()
        return 0

    # Generate output filename
    if args.output:
        output_file = args.output if args.output.endswith('.npy') else f"{args.output}.npy"
    else:
        timestamp = int(time.time())
        output_file = f"cvbs_stream_{timestamp}.npy"

    # Print server setup reminder
    print("=" * 70)
    print("Make sure the streaming server is running on Red Pitaya!")
    print("Run with --help-server for setup instructions")
    print("=" * 70)
    print()

    # Connect and receive
    receiver = RedPitayaStreamReceiver(args.host)
    receiver.DATA_PORT = args.port

    try:
        receiver.connect()
        receiver.receive_data(output_file, args.duration, args.sample_rate)
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to {args.host}:{args.port}")
        print("Make sure the streaming server is running on Red Pitaya.")
        start_streaming_server_instructions()
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    finally:
        receiver.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
