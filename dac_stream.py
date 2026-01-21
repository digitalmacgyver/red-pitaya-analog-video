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


def ssh_command(host, cmd, timeout=10):
    """Run a command on Red Pitaya via SSH."""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", f"root@{host}", cmd]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "SSH timeout"
    except Exception as e:
        return False, "", str(e)


def check_streaming_server_running(host):
    """Check if streaming-server process is running on Red Pitaya."""
    ok, stdout, _ = ssh_command(host, "pgrep -f streaming-server")
    return ok and stdout


def kill_streaming_server(host):
    """Kill any existing streaming server processes."""
    print(f"  Killing existing streaming server...")
    ssh_command(host, "pkill -9 -f streaming-server", timeout=5)
    time.sleep(1)


def start_streaming_server_ssh(host, retries=3):
    """Start the streaming server on Red Pitaya via SSH."""
    print(f"\nStarting streaming server via SSH...")

    for attempt in range(retries):
        if attempt > 0:
            print(f"  Retry {attempt + 1}/{retries}...")

        # Kill any existing server first
        kill_streaming_server(host)

        # Load the stream_app overlay and start server
        cmd = (
            "cd /opt/redpitaya/bin && "
            "LD_LIBRARY_PATH=/opt/redpitaya/lib /opt/redpitaya/sbin/overlay.sh stream_app && "
            "sleep 1 && "
            "LD_LIBRARY_PATH=/opt/redpitaya/lib nohup ./streaming-server -v > /tmp/stream.log 2>&1 &"
        )

        ok, stdout, stderr = ssh_command(host, cmd, timeout=15)
        time.sleep(2)

        # Verify server is running
        if check_streaming_server_running(host):
            print(f"  Streaming server started successfully")
            return True
        else:
            print(f"  Server failed to start")
            # Check log for errors
            ok, log, _ = ssh_command(host, "tail -20 /tmp/stream.log")
            if ok and log:
                print(f"  Server log:\n{log}")

    print(f"  Failed to start streaming server after {retries} attempts")
    return False


def check_streaming_server(host):
    """Check if streaming server is responding via rpsa_client."""
    args = [RPSA_CLIENT, "-c", "-h", host, "-g", "V1"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and result.stdout.strip()
    except:
        return False


def start_dac_server(host):
    """Start the DAC streaming server mode."""
    print(f"\nStarting DAC server mode...")
    args = [RPSA_CLIENT, "-r", "-h", host, "-m", "start_dac", "-t", "1000"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  DAC server started")
            return True
        else:
            print(f"  DAC server start failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error starting DAC server: {e}")
        return False


def stop_dac_server(host):
    """Stop the DAC streaming server mode."""
    args = [RPSA_CLIENT, "-r", "-h", host, "-m", "stop_dac"]
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=5)
    except:
        pass


def ensure_server_ready(host, max_retries=3):
    """Ensure streaming server is running and responsive."""
    print(f"\nChecking streaming server status...")

    for attempt in range(max_retries):
        # First check if rpsa_client can communicate
        if check_streaming_server(host):
            print(f"  Server is responding")
            return True

        print(f"  Server not responding (attempt {attempt + 1}/{max_retries})")

        # Check if process is running
        if check_streaming_server_running(host):
            print(f"  Process running but not responding, restarting...")
            kill_streaming_server(host)
            time.sleep(1)

        # Try to start it
        if start_streaming_server_ssh(host, retries=1):
            time.sleep(2)
            if check_streaming_server(host):
                print(f"  Server now responding")
                return True

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
    """Detect Red Pitaya board on network and ensure server is ready."""
    print("\nDetecting Red Pitaya...")

    target_host = host or DEFAULT_HOST

    # First check basic network connectivity
    print(f"  Checking connectivity to {target_host}...")
    ok, _, _ = ssh_command(target_host, "echo ok", timeout=5)
    if not ok:
        print(f"  ERROR: Cannot SSH to {target_host}")
        print(f"  Check network connection and SSH key setup")
        return None

    print(f"  SSH connection OK")

    # Ensure streaming server is running and responsive
    if not ensure_server_ready(target_host):
        print(f"  ERROR: Could not start streaming server")
        return None

    print(f"  Found board at {target_host}")
    return target_host


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
    """Start DAC streaming with real-time output."""
    print(f"\nStarting DAC stream...")
    print(f"  File: {wav_file}")
    print(f"  Repeat: {repeat}")

    args = [RPSA_CLIENT, "-o", "-h", host, "-f", "wav", "-d", wav_file, "-r", str(repeat)]
    if verbose:
        args.append("-v")

    print(f"  $ {' '.join(args)}\n", flush=True)

    # Use Popen for real-time output streaming
    try:
        # Merge stderr into stdout and stream line by line
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        output_lines = []
        start_time = time.time()

        # Read output line by line in real-time
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(line, flush=True)

        # Wait for process to complete
        returncode = process.wait()
        elapsed = time.time() - start_time

        # Check for errors
        if returncode != 0:
            print(f"\nDAC streaming failed (exit code {returncode})")
            return False

        # If command exited very quickly with no output, something went wrong
        if elapsed < 2.0 and not output_lines:
            print(f"\nWARNING: DAC streaming command exited immediately with no output")
            print(f"This usually means the DAC server is not ready.")
            print(f"Checking server status...")

            if not check_streaming_server(host):
                print(f"  Server stopped responding!")
            return False

        return True

    except KeyboardInterrupt:
        print("\n\nStreaming interrupted by user")
        process.terminate()
        process.wait()
        return True
    except Exception as e:
        print(f"\nError during streaming: {e}")
        return False


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

    # Check basic connectivity first
    target_host = args.host or DEFAULT_HOST
    print(f"\nChecking connectivity to {target_host}...")
    ok, _, _ = ssh_command(target_host, "echo ok", timeout=5)
    if not ok:
        print(f"  ERROR: Cannot SSH to {target_host}")
        sys.exit(1)
    print(f"  SSH connection OK")

    # Step 1: Ensure server is running (needed to configure via rpsa_client)
    if not ensure_server_ready(target_host):
        print("Error: Could not start streaming server for configuration")
        sys.exit(1)

    # Step 2: Configure memory and write to config file
    print(f"\nConfiguring memory settings...")
    configure_memory(target_host)

    # Step 3: Configure DAC mode and rate (writes to config file)
    set_dac_mode_net(target_host)
    set_dac_rate(target_host, dac_rate)

    # Step 4: RESTART server so it picks up the new memory config
    # This is CRITICAL - the server reads memory settings on startup only!
    print(f"\nRestarting streaming server to apply memory settings...")
    kill_streaming_server(target_host)
    time.sleep(2)

    if not start_streaming_server_ssh(target_host, retries=2):
        print("Error: Could not restart streaming server")
        sys.exit(1)

    # Wait for server to be fully ready
    time.sleep(2)
    if not check_streaming_server(target_host):
        print("Error: Server restarted but not responding")
        sys.exit(1)

    host = target_host
    print(f"  Server ready with dac_size={DAC_SIZE:,}")

    # Step 5: Start the DAC server mode
    if not start_dac_server(host):
        print("Warning: DAC server start command failed, attempting anyway...")

    # Small delay to let config settle
    time.sleep(0.5)

    # Start streaming with retry logic
    print(f"\n{'='*60}")
    print(f"Starting DAC playback at {dac_rate:,} Hz")
    print(f"Duration: {wav_info['duration']:.3f}s × {args.repeat} repeat(s)")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}")

    max_retries = 3
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"\n--- Retry {attempt + 1}/{max_retries} ---")
            # Full restart sequence
            print("Restarting streaming infrastructure...")
            stop_dac_server(host)
            time.sleep(1)

            if not ensure_server_ready(host):
                print("Failed to restart server")
                continue

            configure_memory(host)
            set_dac_mode_net(host)
            set_dac_rate(host, dac_rate)
            start_dac_server(host)
            time.sleep(0.5)

        success = start_dac_streaming(host, args.wav_file, args.repeat, args.verbose)

        if success:
            break
        else:
            if attempt < max_retries - 1:
                print(f"\nStreaming failed, will retry...")
            else:
                print(f"\nStreaming failed after {max_retries} attempts")
                print("Try running: python dac_stream.py --stop")
                print("Then try again.")
                sys.exit(1)

    # Clean up
    stop_dac_server(host)


if __name__ == "__main__":
    main()
