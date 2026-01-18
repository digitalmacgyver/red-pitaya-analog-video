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
| Max playback duration | ~8.2 seconds (at 15.625 MS/s with float32) |
| Output rate | 15.625 MS/s (decimation 8 from 125 MS/s) |
| Output voltage | ±1V (matches CVBS levels) |

---

## Playback Workflow

### Step 1: Convert Capture to Playback Format

The Red Pitaya DMG API requires float32 data (-1.0 to 1.0). Convert your 8-bit capture:

```bash
python convert_to_playback.py /path/to/capture.bin
```

This creates a `.f32` file in the same directory.

**Note:** The 8-bit streaming capture is about 30 MB for 2 seconds. The float32 conversion is 4x larger (~120 MB). Files exceeding 128 MB will be automatically truncated to fit DMG memory.

### Step 2: Upload to Red Pitaya

```bash
scp capture.f32 root@192.168.0.6:/home/jupyter/cvbs_project/cvbs_captures/
```

### Step 3: Play Back

Open the playback notebook on Red Pitaya:
- URL: http://192.168.0.6:8888/lab/tree/cvbs_dmg_playback.ipynb

Edit the configuration cell:
```python
WAVEFORM_FILE = "/home/jupyter/cvbs_project/cvbs_captures/capture.f32"
LOOP_COUNT = 0      # 0=once, N=N+1 times, -1=infinite
OUTPUT_CHANNEL = 1  # RF OUT 1
```

Run all cells to play back the waveform.

### Hardware Setup for Playback

1. Connect **RF OUT 1** to your video device
2. Output is ±1V range (standard CVBS levels)

---

## convert_to_playback.py

Local script to convert 8-bit capture files to float32 playback format.

### Synopsis

```
convert_to_playback.py <input.bin> [output.f32]
```

### Description

Converts signed 8-bit integer samples to float32 normalized to -1.0 to 1.0 range. Aligns output to 128-byte boundaries as required by DMG hardware.

### Examples

Convert with automatic output name:
```bash
python convert_to_playback.py cvbs_capture.bin
# Creates cvbs_capture.f32
```

Specify output path:
```bash
python convert_to_playback.py capture.bin /tmp/playback.f32
```

### Output

- Creates `.f32` file (float32, little-endian)
- Prints conversion statistics and duration
- Warns if file exceeds 128 MB DMG limit

---

## cvbs_dmg_playback.ipynb

Jupyter notebook for DMG playback on Red Pitaya.

### Location

- On Red Pitaya: http://192.168.0.6:8888/lab/tree/cvbs_dmg_playback.ipynb
- Local copy: `cvbs_dmg_playback.ipynb`

### Configuration

Edit the first code cell:

```python
WAVEFORM_FILE = "/home/jupyter/cvbs_project/cvbs_captures/cvbs_playback.f32"
DECIMATION = 8           # Output rate: 125/8 = 15.625 MS/s
LOOP_COUNT = 0           # 0=once, N=loop N times, -1=infinite
OUTPUT_CHANNEL = 1       # 1 or 2
```

### Playback Controls

After loading the waveform, use these functions:

```python
play()      # Play once
play(5)     # Play 6 times (1 + 5 loops)
play(-1)    # Play infinitely
stop()      # Stop playback
```

### Memory Management

The notebook:
1. Loads the float32 file into Python memory
2. Writes it to DMG (DDR RAM)
3. Frees Python memory
4. Plays from DMG

This allows playback of files up to 128 MB despite the Red Pitaya's limited RAM.

---

## Playback Data Format

### Float32 Files (.f32)

- Format: Raw float32 (32-bit IEEE 754)
- Byte order: Little-endian (native)
- Value range: -1.0 to 1.0
- Sample rate: 15.625 MS/s (when played with decimation 8)

### Conversion Formula

From 8-bit capture:
```python
float_value = int8_value / 128.0
```

From 16-bit capture:
```python
float_value = int16_value / 32768.0
```

---

## Playback Troubleshooting

### "Write error" or memory issues
- Ensure file size is ≤128 MB
- Restart Jupyter kernel and try again
- Check DMG memory with `rp.rp_GenAxiGetMemoryRegion()`

### No output signal
- Verify RF OUT 1 connection
- Check `rp.rp_GenOutEnable()` was called
- Ensure waveform range is within -1.0 to 1.0

### Kernel crashes during load
- File may be too large for Python to load
- Convert on local machine first (not on Red Pitaya)
- Use pre-converted float32 files only

### Output clipping
- Input values outside -1.0 to 1.0 will clip
- Check source capture levels
- May need to scale: `waveform = waveform * 0.8` (80% amplitude)
