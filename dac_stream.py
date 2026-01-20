#!/usr/bin/env python3
"""
DAC Streaming Script for Red Pitaya

Streams a WAV file to Red Pitaya for DAC output via network streaming.

Usage:
    python dac_stream.py <wav_file> [options]

Examples:
    python dac_stream.py /wintmp/analog_video/rpsa_client/output/resampled2.wav
    python dac_stream.py myfile.wav --repeat inf
    python dac_stream.py myfile.wav --rate 7159090
"""

import argparse
import subprocess
import sys
import os
import wave
import json
import time

# Paths
RPSA_CLIENT = "/home/viblio/coding_projects/rfreplay/rp_streaming/cmd/rpsa_client"
DEFAULT_HOST = "192.168.0.6"

# Sample rates
RATE_2FSC = 7159090    # 2× color subcarrier (NTSC)
RATE_4FSC = 14318180   # 4× color subcarrier
RATE_15625 = 15625000  # Red Pitaya decimation 8

# Buffer sizes - CRITICAL for avoiding memory errors
DAC_SIZE = 134_217_728  # 128 MB
BLOCK_SIZE = 8_388_608  # 8 MB
SMALL_SIZE = 787_968    # 768 KB (default, used for inactive buffer)


def run_cmd(args, check=True, capture=True):
    """Run a command and return output."""
    print(f"  $ {' '.join(args)}")
    result = subprocess.run(args, capture_output=capture, text=True)
    if capture and result.stdout:
        print(result.stdout)
    if capture and result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return result


def check_wav_file(filepath):
    """Validate WAV file for DAC streaming."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"WAV file not found: {filepath}")

    with wave.open(filepath, 'rb') as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        frames = w.getnframes()
        duration = frames / sample_rate
        data_bytes = frames * sample_width * channels

    print(f"\nWAV File Info:")
    print(f"  Path: {filepath}")
    print(f"  Channels: {channels}")
    print(f"  Sample width: {sample_width * 8} bits")
    print(f"  Sample rate: {sample_rate:,} Hz")
    print(f"  Frames: {frames:,}")
    print(f"  Duration: {duration:.3f} seconds")
    print(f"  Data size: {data_bytes:,} bytes ({data_bytes/1024/1024:.2f} MB)")

    # Check 128-byte alignment
    if data_bytes % 128 != 0:
        print(f"  WARNING: Data not 128-byte aligned (remainder: {data_bytes % 128})")
        print(f"           This may cause artifacts in the output signal.")
    else:
        print(f"  128-byte aligned: OK")

    # Check sample width
    if sample_width != 2:
        print(f"  WARNING: Expected 16-bit samples, got {sample_width*8}-bit")

    return {
        'channels': channels,
        'sample_width': sample_width,
        'sample_rate': sample_rate,
        'frames': frames,
        'duration': duration,
        'data_bytes': data_bytes
    }


def start_streaming_server_ssh(host):
    """Start the streaming server on Red Pitaya via SSH."""
    print(f"\nStarting streaming server via SSH...")

    ssh_cmd = [
        "ssh", f"root@{host}",
        "cd /opt/redpitaya/bin && "
        "LD_LIBRARY_PATH=/opt/redpitaya/lib /opt/redpitaya/sbin/overlay.sh stream_app && "
        "sleep 1 && "
        "LD_LIBRARY_PATH=/opt/redpitaya/lib ./streaming-server -v &"
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        time.sleep(2)

        # Verify server is running
        check_cmd = ["ssh", f"root@{host}", "pgrep -f streaming-server"]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)

        if check_result.returncode == 0:
            print(f"  Streaming server started successfully")
            return True
        else:
            print(f"  Warning: Could not verify server started")
            return False
    except Exception as e:
        print(f"  Error starting server via SSH: {e}")
        return False


def check_streaming_server(host):
    """Check if streaming server is responding."""
    args = [RPSA_CLIENT, "-c", "-h", host, "-g", "V1"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def configure_memory(host):
    """Configure memory sizes for DAC streaming.

    Sets DAC buffer to 128 MB (needed for streaming) and reduces
    ADC buffer to 768 KB (unused during playback) to save RAM.
    """
    print(f"\nConfiguring memory (dac_size={DAC_SIZE:,}, adc_size={SMALL_SIZE:,})...")

    # Set DAC buffer large for streaming
    args = [RPSA_CLIENT, "-c", "-h", host, "-i", f"dac_size={DAC_SIZE}", "-w"]
    result1 = subprocess.run(args, capture_output=True, text=True, timeout=5)

    # Reduce ADC buffer (not used during playback)
    args = [RPSA_CLIENT, "-c", "-h", host, "-i", f"adc_size={SMALL_SIZE}", "-w"]
    result2 = subprocess.run(args, capture_output=True, text=True, timeout=5)

    return result1.returncode == 0 and result2.returncode == 0


def detect_board(host=None):
    """Detect Red Pitaya board on network."""
    print("\nDetecting Red Pitaya...")

    if host:
        # Try specific host - check if server is responding
        if check_streaming_server(host):
            print(f"  Found board at {host}")
            return host
        else:
            # Try to start server via SSH
            print(f"  Server not responding, attempting SSH start...")
            if start_streaming_server_ssh(host):
                time.sleep(1)
                if check_streaming_server(host):
                    print(f"  Found board at {host}")
                    return host
            print(f"  Board not responding at {host}")
            return None
    else:
        # Broadcast detect
        args = [RPSA_CLIENT, "-d", "-t", "3"]
        result = run_cmd(args, check=False)
        print(f"  Using default host: {DEFAULT_HOST}")
        return DEFAULT_HOST


def get_config(host):
    """Get current streaming configuration."""
    print(f"\nGetting current configuration from {host}...")
    args = [RPSA_CLIENT, "-c", "-h", host, "-g", "V1"]
    result = run_cmd(args, check=False)
    if result.returncode == 0 and result.stdout:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    return None


def set_dac_rate(host, rate):
    """Set DAC output rate."""
    print(f"\nSetting DAC rate to {rate:,} Hz...")
    args = [RPSA_CLIENT, "-c", "-h", host, "-i", f"dac_rate={rate}", "-w"]
    run_cmd(args, check=False)


def set_dac_mode_net(host):
    """Set DAC to network streaming mode."""
    print(f"\nSetting DAC mode to network...")
    args = [RPSA_CLIENT, "-c", "-h", host, "-i", "dac_pass_mode=DAC_NET", "-w"]
    run_cmd(args, check=False)


def start_dac_streaming(host, wav_file, repeat="1", verbose=False):
    """Start DAC streaming."""
    print(f"\nStarting DAC stream...")
    print(f"  File: {wav_file}")
    print(f"  Repeat: {repeat}")

    args = [RPSA_CLIENT, "-o", "-h", host, "-f", "wav", "-d", wav_file, "-r", str(repeat)]
    if verbose:
        args.append("-v")

    # This command will block while streaming
    run_cmd(args, check=False, capture=False)


def stop_dac(host):
    """Stop DAC streaming."""
    print(f"\nStopping DAC...")
    args = [RPSA_CLIENT, "-r", "-h", host, "-m", "stop_dac"]
    run_cmd(args, check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Stream WAV file to Red Pitaya DAC output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sample Rates:
  2fsc (NTSC):  7,159,090 Hz  (recommended for streaming)
  4fsc (NTSC): 14,318,180 Hz
  RP Dec 8:    15,625,000 Hz

Examples:
  %(prog)s resampled2.wav
  %(prog)s resampled2.wav --repeat inf
  %(prog)s resampled2.wav --rate 7159090 --repeat 5
        """
    )

    parser.add_argument("wav_file", help="WAV file to stream")
    parser.add_argument("--host", "-H", default=DEFAULT_HOST,
                        help=f"Red Pitaya IP address (default: {DEFAULT_HOST})")
    parser.add_argument("--rate", "-r", type=int, default=None,
                        help="DAC sample rate in Hz (default: use WAV file rate)")
    parser.add_argument("--repeat", "-n", default="1",
                        help="Repeat count: number or 'inf' (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--stop", action="store_true",
                        help="Stop DAC and exit")
    parser.add_argument("--config", action="store_true",
                        help="Show current config and exit")

    args = parser.parse_args()

    # Check rpsa_client exists
    if not os.path.exists(RPSA_CLIENT):
        print(f"Error: rpsa_client not found at {RPSA_CLIENT}")
        sys.exit(1)

    # Stop mode
    if args.stop:
        stop_dac(args.host)
        return

    # Config mode
    if args.config:
        config = get_config(args.host)
        if config:
            print(json.dumps(config, indent=2))
        return

    # Validate WAV file
    wav_info = check_wav_file(args.wav_file)

    # Determine DAC rate
    dac_rate = args.rate if args.rate else wav_info['sample_rate']

    if dac_rate != wav_info['sample_rate']:
        print(f"\nWARNING: DAC rate ({dac_rate:,} Hz) differs from WAV rate ({wav_info['sample_rate']:,} Hz)")
        print(f"         Playback will be {'faster' if dac_rate > wav_info['sample_rate'] else 'slower'} than recorded.")

    # Check rate limits
    if dac_rate > 10000000:
        print(f"\nWARNING: DAC rate {dac_rate:,} Hz exceeds recommended 10 MS/s limit for streaming.")
        print(f"         You may experience buffer underruns or signal artifacts.")

    # Detect board
    host = detect_board(args.host)
    if not host:
        print("Error: Could not connect to Red Pitaya")
        sys.exit(1)

    # Configure memory (CRITICAL - prevents memory errors)
    configure_memory(host)

    # Configure DAC mode and rate
    set_dac_mode_net(host)
    set_dac_rate(host, dac_rate)

    # Small delay to let config settle
    time.sleep(0.5)

    # Start streaming
    print(f"\n{'='*60}")
    print(f"Starting DAC playback at {dac_rate:,} Hz")
    print(f"Duration: {wav_info['duration']:.3f}s × {args.repeat} repeat(s)")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        start_dac_streaming(host, args.wav_file, args.repeat, args.verbose)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        stop_dac(host)


if __name__ == "__main__":
    main()
