# CVBS Capture and Playback Tools

This document describes the capture and playback tools for recording and replaying CVBS (composite video) signals using a Red Pitaya Stemlab 125-14.

## Setup

### SSH Key Authentication

To avoid password prompts when using `--auto` mode, set up SSH key authentication:

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to Red Pitaya
ssh-copy-id root@192.168.0.6
```

After this, SSH commands will authenticate automatically.

### Hardware Setup

1. Set IN1 jumper to **LV** (±1V range)
2. Connect CVBS source to **IN1** (RF input 1)
3. Optional: Add 75Ω termination for impedance matching

### Network

Ensure your local machine and Red Pitaya are on the same network and your firewall allows incoming connections on port 18900.

---

## Overview

Two capture methods are available:

1. **File-based capture** (`cvbs_capture.ipynb`) - Captures to local file on Red Pitaya, retrieve via SCP
2. **Streaming capture** (`stream_capture.py`) - Streams data over network to local machine

Use file-based capture for quick tests and debugging. Use streaming capture for longer recordings or when you want data directly on your local machine.

---

## stream_capture.py

Local driver script that receives streamed CVBS data from Red Pitaya.

### Synopsis

```
stream_capture.py [--auto | --listen] [OPTIONS]
```

### Description

Starts a TCP server to receive streamed capture data from Red Pitaya. In auto mode, also triggers the capture via SSH.

### Modes

**--auto**
: Automatic mode. Starts receiver, SSHs to Red Pitaya to trigger capture, saves received data. This is the recommended mode for normal operation.

**--listen**
: Listen mode. Starts receiver and waits for manual trigger. Use this when debugging or running the capture notebook manually on Red Pitaya.

### Options

**-d, --duration SECONDS**
: Capture duration in seconds. Default: 2.0

**-o, --output FILE**
: Output filename. Default: cvbs_capture.bin

**-p, --port PORT**
: TCP port to listen on. Default: 18900

### Examples

Basic 2-second capture:
```bash
python stream_capture.py --auto
```

10-second capture to custom file:
```bash
python stream_capture.py --auto -d 10 -o long_capture.bin
```

Listen mode (manual trigger):
```bash
python stream_capture.py --listen -o test.bin
# Then run cvbs_capture_stream.ipynb on Red Pitaya
```

### Output Files

The script creates two files:

- `<output>.bin` - Raw capture data (signed 16-bit little-endian integers)
- `<output>_meta.txt` - Metadata (sample rate, duration, etc.)

### Network Requirements

- Red Pitaya must be able to connect to your local machine on the specified port
- Firewall must allow incoming connections on port 18900 (or specified port)
- Both machines must be on the same network

### Visualization

After capture, visualize with:
```bash
python visualize_capture.py cvbs_capture.bin
```

---

## cvbs_capture.ipynb

Jupyter notebook for file-based capture on Red Pitaya.

### Usage

1. Open http://192.168.0.6:8888/jlab/lab
2. Open `cvbs_capture.ipynb`
3. Run all cells
4. Retrieve files via SCP:
   ```bash
   scp root@192.168.0.6:/tmp/cvbs_capture.bin .
   scp root@192.168.0.6:/tmp/cvbs_capture_meta.txt .
   ```

### Configuration

Edit cell 2 to change capture duration:
```python
CAPTURE_DURATION_SEC = 2.0  # Change this value
```

---

## cvbs_capture_stream.ipynb

Jupyter notebook for streaming capture (manual trigger mode).

### Usage

1. Start receiver on local machine:
   ```bash
   python stream_capture.py --listen
   ```
2. Note the IP address displayed
3. Open `cvbs_capture_stream.ipynb` on Red Pitaya
4. Set `RECEIVER_HOST` to your local IP
5. Run all cells

### Configuration

Edit cell 1:
```python
RECEIVER_HOST = "192.168.1.100"  # Your local machine IP
RECEIVER_PORT = 18900
CAPTURE_DURATION_SEC = 2.0
```

---

## visualize_capture.py

Local script to visualize captured data.

### Synopsis

```
visualize_capture.py <capture.bin> [metadata.txt]
```

### Description

Loads binary capture data and displays:
- Waveform plot (first 16 video lines)
- Full capture overview
- Value histogram
- Statistics (min, max, mean, voltage levels)

### Examples

```bash
python visualize_capture.py cvbs_capture.bin
```

With explicit metadata file:
```bash
python visualize_capture.py capture.bin capture_meta.txt
```

### Output

- Displays interactive plot
- Saves plot to `<input>_plot.png`

---

## Data Format

### Binary Data (.bin)

- Format: Raw signed 16-bit integers (int16)
- Byte order: Little-endian
- Sample rate: 15.625 MS/s (125 MS/s / 8)
- ADC resolution: 14-bit (stored in 16-bit)

### Voltage Conversion

```python
voltage = raw_value / 8192.0  # For LV mode (±1V)
```

Typical CVBS levels:
- Sync tip: ~-0.3V (raw: ~-2400)
- Blanking: ~0V (raw: ~0)
- Peak white: ~0.7V (raw: ~5700)

### Metadata (.txt)

Plain text key=value format:
```
sample_rate=15625000.0
num_samples=31250000
decimation=8
dtype=int16
bits=14
endian=little
```

---

## Troubleshooting

### Connection refused
- Ensure `stream_capture.py --listen` is running before triggering capture
- Check firewall allows port 18900

### SSH password prompt
- Set up SSH key authentication:
  ```bash
  ssh-copy-id root@192.168.0.6
  ```

### Module not found errors on Red Pitaya
- The auto mode includes proper Python path setup
- For manual notebook use, paths are configured automatically

### Slow capture speed
- File-based capture uses ctypes for fast DMA access (~7 MB/s)
- Streaming adds network overhead but should achieve ~5+ MB/s on local network

---

# Playback

## Overview

Playback uses Deep Memory Generation (DMG) to output captured waveforms through the Red Pitaya's RF outputs at the original sample rate (15.625 MS/s).

### Limitations

| Metric | Value |
|--------|-------|
| Max DMG memory | 128 MB |
| Max playback duration | **~4.3 seconds** (67M samples at 15.625 MS/s) |
| Output rate | 15.625 MS/s (decimation 8 from 125 MS/s) |
| Output voltage | ±1V (matches CVBS levels) |

**Note:** DMG stores data as int16 internally (2 bytes/sample), regardless of input format. The API converts float32 input to int16 when writing. This means 128 MB = 64M samples = 4.3 seconds max.

---

## Playback Workflow (Recommended)

The optimized workflow reads int8 captures directly—no intermediate conversion needed.

### Step 1: Capture

Use the Red Pitaya streaming app to capture to a .bin file.

### Step 2: Upload to Red Pitaya

```bash
scp capture.bin root@192.168.0.6:/tmp/
```

### Step 3: Play Back

```bash
ssh root@192.168.0.6 "python3 /home/jupyter/cvbs_project/cvbs_playback_optimized.py --skip-header /tmp/capture.bin"
```

Options:
- `--skip-header` - Auto-detect and skip file header (recommended)
- `--loop N` - Loop N additional times (0=once, -1=infinite)
- `--no-play` - Load only, don't play (for testing)
- `--chunk-mb N` - Chunk size in MB (default: 8)

### Hardware Setup for Playback

1. Connect **RF OUT 1** to your video device
2. Output is ±1V range (standard CVBS levels)

---

## cvbs_playback_optimized.py

**Recommended playback method.** Streams int8 capture directly to DMG with minimal memory usage.

### Synopsis

```
cvbs_playback_optimized.py [OPTIONS] <capture.bin>
```

### Description

Reads int8 capture files directly, converts to float32 in small chunks (~8 MB), and writes to DMG with offset. This minimizes Python memory usage (8 MB vs 120+ MB) and eliminates the need for intermediate float32 files.

### Options

| Option | Description |
|--------|-------------|
| `--skip-header` | Auto-detect and skip file header |
| `--header N` | Manual header size in bytes |
| `--chunk-mb N` | Chunk size in MB (default: 8) |
| `--no-play` | Load to DMG but don't play |
| `--loop N` | 0=once, N=N+1 times, -1=infinite |

### Examples

Basic playback:
```bash
python3 cvbs_playback_optimized.py --skip-header capture.bin
```

Infinite loop:
```bash
python3 cvbs_playback_optimized.py --skip-header --loop -1 capture.bin
```

Load only (no playback):
```bash
python3 cvbs_playback_optimized.py --skip-header --no-play capture.bin
```

### Memory Efficiency

| Method | Python Memory | File on Disk |
|--------|---------------|--------------|
| Old (float32 file) | ~120 MB | ~120 MB (.f32) |
| **New (chunked int8)** | **~8 MB** | Original .bin only |

---

## Legacy: cvbs_dmg_playback.ipynb

Jupyter notebook for interactive DMG playback. Uses float32 files.

### Location

- On Red Pitaya: http://192.168.0.6:8888/lab/tree/cvbs_project/cvbs_dmg_playback.ipynb
- Local copy: `cvbs_dmg_playback.ipynb`

### When to Use

Use the notebook when you want:
- Interactive playback controls (play/stop/loop)
- To experiment with the DMG API
- Pre-converted float32 files

For automated/scripted playback, use `cvbs_playback_optimized.py` instead.

### Workflow (Legacy)

1. Convert capture: `python convert_to_playback.py capture.bin`
2. Upload: `scp capture.f32 root@192.168.0.6:/home/jupyter/cvbs_project/cvbs_captures/`
3. Open notebook and run cells

---

## convert_to_playback.py (Legacy)

Converts 8-bit captures to float32 for use with the notebook.

**Note:** This is no longer needed if using `cvbs_playback_optimized.py`.

### Synopsis

```
convert_to_playback.py <input.bin> [output.f32]
```

### Examples

```bash
python convert_to_playback.py cvbs_capture.bin
# Creates cvbs_capture.f32
```

---

## resample_capture.py

Resample capture files with proper NTSC timing alignment. First resamples to 4×fsc (14.31818 MS/s) for correct timing, then to the target rate.

### Why 4×fsc Matters

The Red Pitaya captures at 15.625 MS/s, which gives **993.056 samples per NTSC line** - a fractional number that causes timing drift over time. Resampling via 4×fsc gives **exactly 910 samples per line**, eliminating the fractional accumulation error.

### Synopsis

```
resample_capture.py <input.bin> <target_rate> [-o output.wav]
```

### Presets

| Preset | Sample Rate | Samples/Line | Notes |
|--------|-------------|--------------|-------|
| `4fsc` | 14.31818 MS/s | 910 | Standard CVBS digitization rate |
| `2fsc` | 7.15909 MS/s | 455 | Half bandwidth |
| `1fsc` | 3.57954 MS/s | 227.5 | Color subcarrier rate |
| `0.5fsc` | 1.78977 MS/s | 113.75 | Minimum for sync preservation |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output WAV file (default: `<input>_<rate>.wav`) |
| `--original-rate RATE` | Source sample rate (default: 15.625M) |
| `--header N` | Manual header size in bytes |
| `--no-skip-header` | Don't auto-detect/skip file header |
| `--direct` | Skip 4fsc intermediate (not recommended) |

### Examples

```bash
# Resample to 4×fsc (best quality, standard CVBS rate)
python resample_capture.py capture.bin 4fsc

# Resample to 2×fsc (half bandwidth, smaller file)
python resample_capture.py capture.bin 2fsc

# Custom rate for DAC streaming (~5 MS/s limit)
python resample_capture.py capture.bin 4M

# Specify output file
python resample_capture.py capture.bin 4fsc -o output.wav
```

### Output

- 16-bit mono WAV file at the specified sample rate
- Data aligned to 128 bytes (Red Pitaya DAC streaming requirement)
- Viewable in Audacity for waveform analysis

### Streaming the Resampled File

```bash
rpsa_client.exe -o -h 192.168.0.6 -f wav -d resampled.wav -r inf
```

---

## analyze_bin.py

Analyze CVBS timing from a capture file to verify capture quality and identify timing errors.

### Synopsis

```
analyze_bin.py <capture.bin> [max_seconds]
```

### What It Measures

- **VBI (Vertical Blanking Interval)**: Field timing (expected: 16683 µs)
- **HBI (Horizontal Blanking Interval)**: Line timing (expected: 63.556 µs)
- **Statistics**: min, max, mean, median, std, jitter
- **Timing Relationship**: Samples per line, fractional errors

### Examples

```bash
# Analyze entire capture
python analyze_bin.py capture.bin

# Analyze first 2 seconds only (faster)
python analyze_bin.py capture.bin 2

# Specify different sample rate
python analyze_bin.py capture.bin --rate 14.318M
```

### Detection Method

- Sync pulses detected at threshold halfway between blanking and sync tip
- VBI identified by broad pulses (>27µs width)
- HBI measured only in active video regions (excluding VBI)
- Outliers filtered (intervals outside ±10% of expected)

### Key Finding

At 15.625 MS/s, each NTSC line is 993.056 samples (fractional). This causes:
- 0.056 samples/line drift
- 14.6 samples/field cumulative error
- 876 samples/second timing slip

This is why resampling to 4×fsc (910 samples/line exactly) is recommended

---

## Technical Details

### DMG Internal Storage

The DMG API accepts float32 input but stores data as **int16** internally:
- Input: float32 array (-1.0 to 1.0)
- Storage: int16 (2 bytes/sample)
- Conversion: Handled by `rp_GenAxiWriteWaveform()`

Memory reservation must use int16 size:
```python
waveform_bytes = num_samples * 2  # NOT * 4
```

### Chunked Writing

The `rp_GenAxiWriteWaveformOffset(channel, offset, data)` function allows writing in chunks:
```python
# Write 1M samples starting at offset 0
rp.rp_GenAxiWriteWaveformOffset(CH, 0, chunk1)
# Write next 1M samples
rp.rp_GenAxiWriteWaveformOffset(CH, 1000000, chunk2)
```

This enables streaming large files with minimal memory.

### Alignment Requirements

DMG requires 128-byte alignment (64 int16 samples). The playback scripts handle this automatically by truncating to aligned boundaries.

---

## Playback Troubleshooting

### "Input buffer size does not match memory size" warning
- Memory reservation used wrong multiplier (×4 instead of ×2)
- Use `cvbs_playback_optimized.py` which handles this correctly

### No output signal
- Verify RF OUT 1 connection
- Check that playback completed without errors
- Try `--loop -1` to play continuously

### Kernel crashes during load
- File too large for available RAM
- Use `cvbs_playback_optimized.py` (chunked loading)

### Playback too short
- DMG hardware limit: 128 MB = 4.3 seconds max
- Longer playback requires continuous DAC streaming (limited to ~5 MS/s)
