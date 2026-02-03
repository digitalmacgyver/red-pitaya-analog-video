#!/usr/bin/env python3
"""
ADC Streaming Capture Script for Red Pitaya

Captures CVBS or other analog signals via network streaming from Red Pitaya.
This script can automatically start the streaming server via SSH.

Usage:
    python adc_capture.py [options]

Examples:
    python adc_capture.py                           # Default: 2 seconds at dec=8
    python adc_capture.py -d 5                      # Capture 5 seconds
    python adc_capture.py -d 10 --decimation 16     # 10 seconds at dec=16
    python adc_capture.py -d 3 -o /tmp/captures     # Save to specific directory
    python adc_capture.py -d 30 -n vhs_test         # Capture with custom filename
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime

# Paths
RPSA_CLIENT = "/home/viblio/coding_projects/rfreplay/rp_streaming/cmd/rpsa_client"
DEFAULT_HOST = "192.168.0.6"
DEFAULT_OUTPUT_DIR = "/wintmp/analog_video/rpsa_client/output"

# Red Pitaya base clock
RP_BASE_CLOCK = 125_000_000  # 125 MHz

# Buffer sizes - CRITICAL for avoiding DMA glitches
# Default adc_size is only 768 KB which causes glitches every ~65K samples
# Must be increased to match block_size requirements
BLOCK_SIZE = 8_388_608      # 8 MB (maximum supported)
ADC_SIZE = 134_217_728      # 128 MB (prevents DMA buffer overflow)
SMALL_SIZE = 787_968        # 768 KB (default, used for inactive buffer)


def run_cmd(args, check=True, capture=True, timeout=None):
    """Run a command and return output."""
    print(f"  $ {' '.join(args)}")
    try:
        result = subprocess.run(args, capture_output=capture, text=True, timeout=timeout)
        if capture and result.stdout:
            for line in result.stdout.strip().split('\n')[:10]:
                print(f"    {line}")
        if capture and result.stderr:
            for line in result.stderr.strip().split('\n')[:5]:
                print(f"    [err] {line}", file=sys.stderr)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed with code {result.returncode}")
        return result
    except subprocess.TimeoutExpired:
        print("    [timeout]")
        return None


def has_connection_error(result):
    """Check if rpsa_client result indicates a connection error.

    rpsa_client returns exit code 0 even on connection failure,
    so we must check stderr for error messages.
    """
    if result is None:
        return True
    if result.returncode != 0:
        return True
    if result.stderr and "Connection refused" in result.stderr:
        return True
    if result.stderr and "Error:" in result.stderr:
        return True
    return False


def get_sample_rate(decimation):
    """Calculate sample rate from decimation factor."""
    return RP_BASE_CLOCK / decimation


def format_rate(rate):
    """Format sample rate for display."""
    if rate >= 1e6:
        return f"{rate/1e6:.3f} MS/s"
    elif rate >= 1e3:
        return f"{rate/1e3:.3f} kS/s"
    else:
        return f"{rate:.0f} S/s"


def start_streaming_server_ssh(host):
    """Start the streaming server on Red Pitaya via SSH."""
    print(f"\nStarting streaming server via SSH...")

    # Command to start streaming server with FPGA overlay
    ssh_cmd = [
        "ssh", f"root@{host}",
        "cd /opt/redpitaya/bin && "
        "LD_LIBRARY_PATH=/opt/redpitaya/lib /opt/redpitaya/sbin/overlay.sh stream_app && "
        "sleep 1 && "
        "LD_LIBRARY_PATH=/opt/redpitaya/lib ./streaming-server -v &"
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
        # Give server time to initialize
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


def stop_streaming_server_ssh(host):
    """Stop the streaming server on Red Pitaya via SSH."""
    print(f"\nStopping streaming server via SSH...")
    ssh_cmd = ["ssh", f"root@{host}", "killall streaming-server 2>/dev/null"]
    try:
        subprocess.run(ssh_cmd, capture_output=True, timeout=5)
        print("  Server stopped")
    except:
        pass


def set_config(host, quiet=False, **kwargs):
    """Set configuration values on Red Pitaya.

    Returns:
        True if all settings succeeded, False if any failed.
    """
    all_success = True
    for name, value in kwargs.items():
        if not quiet:
            print(f"\nSetting {name}={value}...")
        args = [RPSA_CLIENT, "-c", "-h", host, "-i", f"{name}={value}", "-w"]
        result = run_cmd(args, check=False)
        # rpsa_client returns 0 even on failure - check stderr for errors
        if has_connection_error(result):
            all_success = False
    return all_success


def get_config(host):
    """Get current configuration."""
    print(f"\nGetting configuration from {host}...")
    args = [RPSA_CLIENT, "-c", "-h", host, "-g", "V1"]
    result = run_cmd(args, check=False)
    if result and result.returncode == 0:
        return result.stdout
    return None


def check_streaming_server(host):
    """Check if Red Pitaya streaming server is running and responsive.

    Note: rpsa_client returns exit code 0 even on connection failure,
    so we must also check stderr for error messages.
    """
    args = [RPSA_CLIENT, "-c", "-h", host, "-g", "V1"]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        # rpsa_client returns 0 even on failure - check stderr for errors
        if has_connection_error(result):
            return False
        return True
    except:
        return False


def configure_streaming(host, decimation, resolution, channel):
    """
    Configure the Red Pitaya streaming server with all required settings.

    Returns:
        True if all configuration succeeded, False otherwise.
    """
    success = True

    # CRITICAL: Set buffer sizes to prevent DMA glitches
    print("\nSetting memory configuration (prevents DMA glitches)...")
    success &= set_config(host, block_size=BLOCK_SIZE)
    success &= set_config(host, adc_size=ADC_SIZE)
    success &= set_config(host, dac_size=SMALL_SIZE)  # Reduce unused DAC buffer

    # Set streaming mode to network
    success &= set_config(host, adc_pass_mode="NET")

    # Set decimation
    success &= set_config(host, adc_decimation=decimation)

    # Set resolution
    resolution_str = "BIT_8" if resolution == "8" else "BIT_16"
    success &= set_config(host, resolution=resolution_str)

    # Enable the selected channel, disable the other
    ch1_state = "ON" if channel == 1 else "OFF"
    ch2_state = "ON" if channel == 2 else "OFF"
    success &= set_config(host, channel_state_1=ch1_state, channel_state_2=ch2_state)

    return success


def ensure_server_and_configure(host, decimation, resolution, channel, use_ssh=True, max_retries=2):
    """
    Ensure streaming server is running and configured, with automatic retry.

    If configuration fails (e.g., server died), this function will:
    1. Restart the streaming server
    2. Retry the configuration

    Args:
        host: Red Pitaya IP address
        decimation: ADC decimation factor
        resolution: "8" or "16" bit
        channel: 1 or 2
        use_ssh: Whether to use SSH to restart server
        max_retries: Maximum number of retry attempts

    Returns:
        True if server is running and configured, False otherwise.
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"\n*** Configuration failed, restarting server (attempt {attempt + 1}/{max_retries + 1}) ***")
            if use_ssh:
                stop_streaming_server_ssh(host)
                time.sleep(1)
                if not start_streaming_server_ssh(host):
                    print("  Failed to restart server")
                    continue
                time.sleep(1)
            else:
                print("  Cannot restart server (--no-ssh specified)")
                return False

        # Attempt configuration
        if configure_streaming(host, decimation, resolution, channel):
            if attempt > 0:
                print("  Configuration succeeded after restart")
            return True

    return False


def stop_streaming(host):
    """Stop ADC streaming on Red Pitaya."""
    print(f"\nStopping ADC streaming...")
    args = [RPSA_CLIENT, "-r", "-h", host, "-m", "stop"]
    run_cmd(args, check=False)


def capture_streaming(host, output_dir, format_type, duration_sec, verbose=False):
    """
    Capture ADC data via network streaming.

    Uses start_stop mode which handles the full capture cycle:
    1. Start receiver process
    2. Send start command to server
    3. Wait for duration
    4. Send stop command
    5. Collect data

    Args:
        host: Red Pitaya IP address
        output_dir: Directory to save output files
        format_type: Output format (bin, wav, csv, tdms)
        duration_sec: Capture duration in seconds
        verbose: Show verbose output from rpsa_client
    """
    print(f"\nStarting capture...")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {format_type}")
    print(f"  Duration: {duration_sec:.1f} seconds")

    # Build receiver command
    recv_args = [RPSA_CLIENT, "-s", "-h", host, "-f", format_type, "-d", output_dir, "-m", "raw"]
    if verbose:
        recv_args.append("-v")

    print(f"  Receiver: {' '.join(recv_args)}")
    print(f"  Press Ctrl+C to stop early\n")

    # Start receiver process (it will wait for data)
    capture_process = subprocess.Popen(
        recv_args,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None,
        text=True
    )

    # Give receiver time to initialize
    time.sleep(0.5)

    # Calculate timeout in milliseconds for start_stop command
    timeout_ms = int(duration_sec * 1000)

    # Use start_stop mode which handles the full capture cycle
    print(f"  Triggering capture for {duration_sec:.1f} seconds...")
    start_stop_args = [RPSA_CLIENT, "-r", "-h", host, "-m", "start_stop", "-t", str(timeout_ms)]

    try:
        # Run start_stop command (blocks for duration)
        subprocess.run(start_stop_args, capture_output=True, timeout=duration_sec + 10)

        # Give receiver time to finish writing
        time.sleep(1)

        # Check if receiver is still running
        if capture_process.poll() is None:
            capture_process.terminate()
            try:
                capture_process.wait(timeout=3)
            except:
                capture_process.kill()

        return True

    except subprocess.TimeoutExpired:
        print("\n  Capture timeout")
        stop_streaming(host)
        capture_process.terminate()
        return True
    except KeyboardInterrupt:
        print("\n  Capture interrupted by user")
        stop_streaming(host)
        capture_process.terminate()
        return True


def rename_output_files(output_dir, base_name, format_type, start_time):
    """
    Rename output files from auto-generated names to user-specified base name.

    rpsa_client generates files like: data_file_192.168.0.6_2026-01-18_10-10-35.bin
    This function renames them to: <base_name>.bin, <base_name>.log, etc.

    Args:
        output_dir: Directory containing the output files
        base_name: User-specified base name for files
        format_type: File format (bin, wav, csv, tdms)
        start_time: Capture start time (to identify new files)

    Returns:
        dict mapping old filenames to new filenames
    """
    renamed = {}
    extensions = [f'.{format_type}', '.log']  # rpsa_client creates data and log files

    try:
        # Find files created after start_time
        for filename in os.listdir(output_dir):
            filepath = os.path.join(output_dir, filename)
            if not os.path.isfile(filepath):
                continue

            # Check if file was created after capture started
            mtime = os.path.getmtime(filepath)
            if mtime < start_time:
                continue

            # Check if this is one of our output files
            for ext in extensions:
                if filename.endswith(ext):
                    new_filename = f"{base_name}{ext}"
                    new_filepath = os.path.join(output_dir, new_filename)

                    # Handle existing files
                    if os.path.exists(new_filepath) and filepath != new_filepath:
                        # Add timestamp suffix if target exists
                        timestamp = datetime.now().strftime("%H%M%S")
                        backup_name = f"{base_name}_{timestamp}{ext}"
                        backup_path = os.path.join(output_dir, backup_name)
                        os.rename(new_filepath, backup_path)
                        print(f"  Moved existing {new_filename} to {backup_name}")

                    if filepath != new_filepath:
                        os.rename(filepath, new_filepath)
                        renamed[filename] = new_filename
                        print(f"  Renamed: {filename} -> {new_filename}")
                    break

    except Exception as e:
        print(f"  Warning: Error renaming files: {e}")

    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="Capture ADC data from Red Pitaya via network streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Decimation and Sample Rates:
  Dec=1     125.000 MS/s   (too fast for network)
  Dec=8      15.625 MS/s   (default, good for CVBS)
  Dec=16      7.813 MS/s
  Dec=32      3.906 MS/s
  Dec=64      1.953 MS/s

Output Formats:
  bin     Binary int8 (most compact, recommended)
  wav     WAV file format
  csv     Comma-separated values (large files)
  tdms    NI TDMS format

Memory Configuration:
  The script automatically configures:
  - block_size: 8 MB (maximum DMA block)
  - adc_size: 128 MB (prevents buffer overflow glitches)

  Without this, captures have glitches every ~65K samples!

Examples:
  %(prog)s                              # 2 seconds at 15.625 MS/s
  %(prog)s -d 5                         # 5 seconds
  %(prog)s -d 10 --decimation 16        # 10 seconds at 7.8 MS/s
  %(prog)s -d 5 -n vhs_test1            # 5 seconds, named 'vhs_test1.bin'
  %(prog)s -d 10 -o /tmp -n experiment  # To /tmp as 'experiment.bin'
  %(prog)s --stop                       # Stop any running capture
        """
    )

    parser.add_argument("-d", "--duration", type=float, default=2.0,
                        help="Capture duration in seconds (default: 2.0)")
    parser.add_argument("--decimation", type=int, default=8,
                        help="ADC decimation factor (default: 8 = 15.625 MS/s)")
    parser.add_argument("-f", "--format", choices=["bin", "wav", "csv", "tdms"],
                        default="bin", help="Output format (default: bin)")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("-n", "--name", default=None,
                        help="Base name for output files (e.g., 'experiment1_vhs_test'). "
                             "Files will be renamed from auto-generated names after capture.")
    parser.add_argument("-H", "--host", default=DEFAULT_HOST,
                        help=f"Red Pitaya IP address (default: {DEFAULT_HOST})")
    parser.add_argument("--channel", type=int, choices=[1, 2], default=1,
                        help="ADC channel (default: 1)")
    parser.add_argument("--resolution", choices=["8", "16"], default="8",
                        help="ADC resolution in bits (default: 8)")
    parser.add_argument("--no-ssh", action="store_true",
                        help="Don't auto-start streaming server via SSH")
    parser.add_argument("--stop", action="store_true",
                        help="Stop any running capture and exit")
    parser.add_argument("--kill-server", action="store_true",
                        help="Kill streaming server and exit")
    parser.add_argument("--config", action="store_true",
                        help="Show current configuration and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Check rpsa_client exists
    if not os.path.exists(RPSA_CLIENT):
        print(f"Error: rpsa_client not found at {RPSA_CLIENT}")
        sys.exit(1)

    # Kill server mode
    if args.kill_server:
        stop_streaming_server_ssh(args.host)
        return

    # Stop mode
    if args.stop:
        stop_streaming(args.host)
        return

    # Config mode
    if args.config:
        config = get_config(args.host)
        if config:
            print(config)
        return

    # Check if streaming server is running, start via SSH if needed
    print("Checking Red Pitaya streaming server...")
    if not check_streaming_server(args.host):
        if args.no_ssh:
            print(f"\nError: Cannot connect to streaming server at {args.host}")
            print("\nStart the streaming server manually:")
            print("  1. Open http://192.168.0.6/ in a browser")
            print("  2. Click the 'Streaming' application")
            print("  3. Click 'RUN' to start the server")
            print("\nOr remove --no-ssh to auto-start via SSH.")
            sys.exit(1)
        else:
            print("  Server not running, starting via SSH...")
            if not start_streaming_server_ssh(args.host):
                print("\nError: Could not start streaming server")
                print("Check that SSH access to root@{} is configured".format(args.host))
                sys.exit(1)
            # Verify connection
            time.sleep(1)
            if not check_streaming_server(args.host):
                print("\nError: Server started but not responding")
                sys.exit(1)

    print(f"  Connected to {args.host}")

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    # Calculate parameters
    sample_rate = get_sample_rate(args.decimation)
    samples_needed = int(args.duration * sample_rate)
    bytes_per_sample = 2 if args.resolution == "16" else 1
    data_size_mb = (samples_needed * bytes_per_sample) / (1024 * 1024)

    # Display capture info
    print("=" * 60)
    print("ADC STREAMING CAPTURE")
    print("=" * 60)
    print(f"\nCapture Parameters:")
    print(f"  Host: {args.host}")
    print(f"  Channel: CH{args.channel}")
    print(f"  Decimation: {args.decimation}")
    print(f"  Sample rate: {format_rate(sample_rate)}")
    print(f"  Resolution: {args.resolution}-bit")
    print(f"  Duration: {args.duration:.1f} seconds")
    print(f"  Samples: {samples_needed:,}")
    print(f"  Est. file size: {data_size_mb:.1f} MB")
    print(f"  Output format: {args.format}")
    print(f"  Output directory: {args.output_dir}")

    # Configure Red Pitaya (with automatic retry if server dies)
    print("\n" + "-" * 60)
    print("Configuring Red Pitaya...")
    print("-" * 60)

    if not ensure_server_and_configure(
        args.host,
        args.decimation,
        args.resolution,
        args.channel,
        use_ssh=not args.no_ssh
    ):
        print("\nError: Failed to configure Red Pitaya after multiple attempts")
        print("Try manually restarting the streaming server:")
        print(f"  python adc_capture.py --kill-server -H {args.host}")
        print(f"  python adc_capture.py -d {args.duration} ...")
        sys.exit(1)

    # Small delay for config to settle
    time.sleep(0.5)

    # Start capture
    print("\n" + "-" * 60)
    print("Starting Capture...")
    print("-" * 60)

    # Record start time for file identification (used for renaming)
    capture_start_time = time.time()
    start_time = time.time()

    success = capture_streaming(
        args.host,
        args.output_dir,
        args.format,
        args.duration,
        verbose=args.verbose
    )

    elapsed = time.time() - start_time

    # Report results
    print("\n" + "=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    print(f"  Elapsed time: {elapsed:.1f} seconds")
    print(f"  Output directory: {args.output_dir}")

    # Rename output files if --name was specified
    if args.name:
        print(f"\nRenaming output files to '{args.name}'...")
        renamed = rename_output_files(args.output_dir, args.name, args.format, capture_start_time)
        if not renamed:
            print("  No files were renamed (files may not have been created)")

    # List output files (newest first)
    print(f"\nOutput files:")
    try:
        files = sorted(
            [f for f in os.listdir(args.output_dir) if f.endswith('.bin') or f.endswith('.wav')],
            key=lambda f: os.path.getmtime(os.path.join(args.output_dir, f)),
            reverse=True
        )
        for f in files[:5]:
            filepath = os.path.join(args.output_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"  {f}: {size:,} bytes ({size/1024/1024:.2f} MB) - {mtime.strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"  Error listing files: {e}")

    # Find the output file for the hint
    try:
        if args.name:
            # Use the renamed file
            output_file = os.path.join(args.output_dir, f"{args.name}.{args.format}")
        else:
            # Find the newest file
            output_file = max(
                [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir)
                 if f.endswith(f'.{args.format}')],
                key=os.path.getmtime
            )

        if os.path.exists(output_file):
            print(f"\nTo analyze the capture:")
            print(f"  python analyze_bin.py {output_file}")
            print(f"\nTo resample for DAC playback:")
            print(f"  python resample_capture.py {output_file} 15.625M")
    except:
        pass


if __name__ == "__main__":
    main()
