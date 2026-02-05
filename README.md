# CVBS Capture and Playback Tools

Tools for recording and replaying CVBS (composite video) signals using a Red Pitaya Stemlab 125-14.

## Prerequisites

- **Red Pitaya Stemlab 125-14** with OS 2.07-43 or later
- **Python 3.8+** with numpy, scipy
- **Red Pitaya streaming client tools** - install in `./rp_streaming/cmd/`
  - Download from: https://github.com/RedPitaya/RedPitaya/tree/master/apps-tools/streaming_manager
  - Required binaries: `rpsa_client`, `convert_tool`

## Quick Start: End-to-End Workflow

### 1. Capture (30 seconds of CVBS)

```bash
# Capture 30 seconds at 15.625 MS/s with meaningful filename
python adc_capture.py -d 30 -n vhs_tape1

# Or with custom output directory
python adc_capture.py -d 30 -o /path/to/output -n vhs_tape1
```

### 2. Convert for Playback

```bash
# Convert .bin to .wav (automatically strips block headers)
python resample_capture.py vhs_tape1.bin 15.625M

# Or resample to a different rate
python resample_capture.py vhs_tape1.bin 2fsc
```

`resample_capture.py` automatically detects .bin files with block headers and uses `convert_tool` to strip them before processing.

### 3. Play Back via DAC Streaming

```bash
# Stream to Red Pitaya DAC (loops infinitely)
python dac_stream.py /path/to/capture.wav --repeat inf
```

**Note:** Playback at 15.625 MS/s functions but exceeds Red Pitaya's documented 10 MS/s limit for DAC streaming, causing periodic timing glitches (see [Streaming Rate Trade-offs](#streaming-rate-trade-offs)).

---

## Setup

### SSH Key Authentication (Required for adc_capture.py)

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to Red Pitaya
ssh-copy-id root@192.168.0.6
```

### Hardware Setup

**Capture (Input Impedance Matching):**

The Red Pitaya has a **high-impedance input** (1MΩ), not the 75Ω that CVBS expects. To properly terminate the video signal:

1. Set IN1 jumper to **LV** (±1V range)
2. Use a **BNC T-connector** on the CVBS source
3. Connect one leg of the T to **IN1** (RF input 1)
4. Connect a **75Ω terminator** to the other leg of the T

```
[CVBS Source]──[BNC T]──[75Ω Terminator]
                  │
            [Red Pitaya IN1]
```

This ensures the source sees the correct 75Ω load while the high-Z Red Pitaya input passively monitors the signal.

**Playback (Output Impedance Matching):**

The Red Pitaya DAC output is 50Ω, but CVBS equipment expects 75Ω. For proper matching:

1. Connect **RF OUT 1** through a **minimum loss pad** (L-pad)
2. Use a passive BNC barrel adapter: 50Ω to 75Ω impedance transformer
3. This attenuates the signal by **5.7 dB**
4. Compensate in software: `--gain 5.7` when converting

```
[Red Pitaya OUT1]──[50Ω→75Ω L-pad]──[Video Device]
        50Ω              -5.7dB            75Ω
```

The L-pad is a passive resistor network that provides bidirectional impedance matching with minimal signal reflection.

---

## Tools Overview

| Tool | Purpose |
|------|---------|
| `adc_capture.py` | **Recommended capture** - Command-line streaming capture via SSH |
| `resample_capture.py` | **Recommended conversion** - Convert .bin to 16-bit .wav, strip headers, resample, apply gain |
| `dac_stream.py` | **Recommended playback** - DAC streaming for unlimited duration |
| `rp_streaming/cmd/convert_tool` | Internal tool for header stripping (8-bit output, not for direct playback) |
| `strip_headers.py` | Fallback header stripping (if convert_tool unavailable) |
| `scan_glitches.py` | Detect glitch spikes in capture files |
| `analyze_bin.py` | Analyze CVBS timing and quality |
| `visualize_capture.py` | Visualize captured waveforms |

---

## adc_capture.py (Recommended Capture)

Command-line tool for ADC streaming capture. Auto-starts the streaming server via SSH.

### Synopsis

```
adc_capture.py [-d DURATION] [--decimation N] [-f FORMAT] [-o DIR] [-n NAME] [OPTIONS]
```

### Key Features

- **Auto-starts streaming server** via SSH (no web UI needed)
- **Sets correct buffer sizes** to prevent DMA glitches
- **Duration-based capture** with automatic stop

### Critical: Buffer Configuration

The script automatically sets:
- `adc_size = 128 MB` (prevents glitches every ~65K samples)
- `block_size = 8 MB` (maximum DMA block)

**Without this, captures have periodic glitches!** The default `adc_size` of 768 KB is too small.

### Options

| Option | Description |
|--------|-------------|
| `-d, --duration SEC` | Capture duration (default: 2.0) |
| `--decimation N` | Decimation factor (default: 8 = 15.625 MS/s) |
| `-f, --format FMT` | Output format: bin, wav, csv, tdms (default: bin) |
| `-o, --output-dir DIR` | Output directory |
| `-n, --name NAME` | Base name for output files (renames from auto-generated names) |
| `-H, --host IP` | Red Pitaya IP address (default: 192.168.0.6) |
| `--channel {1,2}` | ADC channel (default: 1) |
| `--resolution {8,16}` | Bits per sample (default: 8) |
| `--no-ssh` | Don't auto-start server (use web UI instead) |
| `--stop` | Stop any running capture |
| `--kill-server` | Kill streaming server |
| `--config` | Show current configuration |
| `--skip-restart` | Skip memory buffer configuration for faster startup (see below) |
| `-v, --verbose` | Verbose output |

### Fast Startup with --skip-restart

By default, `adc_capture.py` configures memory buffer sizes (`block_size`, `adc_size`, `dac_size`) on each run. This is necessary because the Red Pitaya streaming server reads these settings on startup.

For repeat captures where the server is already configured correctly, use `--skip-restart` to skip memory configuration and start faster:

```bash
# First capture: full configuration
python adc_capture.py -d 30 -n capture1

# Subsequent captures: fast startup (skips memory config)
python adc_capture.py -d 30 -n capture2 --skip-restart
```

**Use `--skip-restart` when:**
- You've already run `adc_capture.py` recently (server is configured)
- You're doing repeated test captures
- You want faster startup time

**Don't use `--skip-restart` when:**
- First run after Red Pitaya reboot
- After using other streaming applications
- If previous capture had buffer overflow errors (glitches every ~65K samples)

### Examples

```bash
# Default 2-second capture
python adc_capture.py

# 30-second capture
python adc_capture.py -d 30

# Capture with custom filename (outputs vhs_test.bin instead of auto-generated name)
python adc_capture.py -d 30 -n vhs_test

# Capture with custom directory and filename
python adc_capture.py -d 30 -o /path/to/output -n experiment1_source2

# 60 seconds at lower rate (for slower network)
python adc_capture.py -d 60 --decimation 16

# Show current Red Pitaya configuration
python adc_capture.py --config
```

### Decimation and Sample Rates

| Decimation | Sample Rate | Data Rate | Notes |
|------------|-------------|-----------|-------|
| 8 | 15.625 MS/s | ~15 MB/s | Default, good for CVBS |
| 16 | 7.813 MS/s | ~7.5 MB/s | Lower bandwidth |
| 32 | 3.906 MS/s | ~3.75 MB/s | Minimum for CVBS |

---

## convert_tool (Internal - Used by resample_capture.py)

Red Pitaya's official tool for stripping block headers from .bin captures. Located at `rp_streaming/cmd/convert_tool`.

**Note:** convert_tool outputs 8-bit WAV which cannot be played directly via DAC streaming. Use `resample_capture.py` instead, which calls convert_tool internally and converts to proper 16-bit format.

### Synopsis

```
convert_tool <capture.bin> [-f WAV|CSV|TDMS] [-i] [-s start] [-e end]
```

**IMPORTANT:** The format flag must be **UPPERCASE** (`-f WAV` not `-f wav`).

### Key Features

- **Properly strips block headers** from .bin files
- **Preserves original sample rate** (e.g., 15.625 MS/s)
- **Official tool** - robust to future format changes
- **8-bit output** - requires conversion to 16-bit for DAC playback

### Options

| Option | Description |
|--------|-------------|
| `-i` | Show file info (segments, samples, channels) |
| `-f FORMAT` | Output format: WAV, CSV, TDMS (must be uppercase!) |
| `-s N` | Start from segment N |
| `-e N` | End at segment N |

### Examples

```bash
# Show file info
convert_tool capture.bin -i

# Convert to WAV (creates capture.wav)
convert_tool capture.bin -f WAV

# Convert specific segment range
convert_tool capture.bin -f WAV -s 0 -e 10
```

---

## strip_headers.py (Alternative - Fallback Tool)

Python fallback for stripping block headers if convert_tool is unavailable.

### Synopsis

```
strip_headers.py <capture.bin> [-o output.bin] [--verify-only]
```

### When to Use

Use this if:
- convert_tool is not available
- You need to process .bin files without converting to .wav
- You want to verify a file has headers

### Examples

```bash
# Strip headers (creates capture_clean.bin)
python strip_headers.py capture.bin

# Specify output filename
python strip_headers.py capture.bin -o clean.bin

# Verify a file has no headers
python strip_headers.py capture.bin --verify-only
```

---

## dac_stream.py (Recommended Playback)

DAC streaming for playback of any duration. Uses network streaming to Red Pitaya.

### Synopsis

```
dac_stream.py <wav_file> [--repeat N|inf] [--rate HZ] [OPTIONS]
```

### Key Features

- **Unlimited playback duration** (not limited to 4.3s like DMG)
- **Auto-detects sample rate** from WAV header
- **Supports 15.625 MS/s** despite documented 10 MS/s limit

### Options

| Option | Description |
|--------|-------------|
| `-n, --repeat N` | Repeat count (default: 1, use `inf` for infinite) |
| `-r, --rate HZ` | Override DAC rate (default: from WAV file) |
| `-H, --host IP` | Red Pitaya IP (default: 192.168.0.6) |
| `--skip-restart` | Skip server restart for faster startup (see note below) |
| `--stop` | Stop DAC and exit |
| `--config` | Show current DAC configuration |
| `-v, --verbose` | Verbose output |

### Fast Startup with --skip-restart

By default, `dac_stream.py` restarts the streaming server on each run to ensure the `dac_size` and `adc_size` memory buffer settings are applied. This is necessary because the Red Pitaya streaming server only reads these settings on startup.

For repeat runs where the server is already configured correctly, use `--skip-restart` to skip this restart and start playback faster:

```bash
# First run: full startup (configures and restarts server)
python dac_stream.py capture.wav --repeat inf

# Subsequent runs: fast startup (skips restart)
python dac_stream.py capture.wav --repeat inf --skip-restart
```

**Use `--skip-restart` when:**
- You've already run `dac_stream.py` recently (server is configured)
- You're doing repeated test playbacks
- You want faster startup time

**Don't use `--skip-restart` when:**
- First run after Red Pitaya reboot
- After using other streaming applications
- If playback fails with memory errors (run without flag to reconfigure)

### Examples

```bash
# Play once
python dac_stream.py capture.wav

# Loop forever
python dac_stream.py capture.wav --repeat inf

# Play at specific rate
python dac_stream.py capture.wav --rate 7159090

# Stop playback
python dac_stream.py --stop
```

### Sample Rate Notes

| Rate | Status | Notes |
|------|--------|-------|
| 15.625 MS/s | Marginal | Original capture rate; timing glitches (exceeds 10 MS/s limit) |
| 10.74 MS/s (3fsc) | **Recommended** | Within spec; timing-accurate but poor picture quality |
| 7.159 MS/s (2fsc) | Works | Standard for streaming |
| 14.318 MS/s (4fsc) | Marginal | Standard CVBS rate; likely has timing glitches |

---

## resample_capture.py

Convert and resample captures for playback. Supports gain compensation and DC centering.

### Synopsis

```
resample_capture.py <input.bin> <target_rate> [-o output.wav] [OPTIONS]
```

### Presets

| Preset | Sample Rate | Samples/Line | Notes |
|--------|-------------|--------------|-------|
| `4fsc` | 14.31818 MS/s | 910 | Standard CVBS digitization |
| `3fsc` | 10.73864 MS/s | 682.5 | **Recommended for streaming** (close to 10 MS/s limit) |
| `2fsc` | 7.15909 MS/s | 455 | Half bandwidth |
| `1fsc` | 3.57954 MS/s | 227.5 | Color subcarrier rate |
| `15.625M` | 15.625 MS/s | 993 | Original capture rate (no resampling) |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Output WAV file |
| `--original-rate RATE` | Source sample rate (default: 15.625M) |
| `--gain DB` | Apply gain in dB (e.g., 5.7 for pad compensation) |
| `--center` | Center signal at 0V before gain (maximizes headroom) |
| `--no-convert-tool` | Don't use convert_tool for header stripping (use fallback) |

### Examples

```bash
# No resampling (convert to WAV at original rate)
python resample_capture.py capture.bin 15.625M

# Resample to 2fsc for lower bandwidth
python resample_capture.py capture.bin 2fsc

# With gain compensation for impedance matching pad
python resample_capture.py capture.bin 2fsc --gain 5.7 --center

# Resample to standard 4fsc rate
python resample_capture.py capture.bin 4fsc
```

### Gain and Centering

When using a minimum loss pad (50Ω to 75Ω impedance matching), the signal loses 5.7 dB. To compensate:

```bash
# Without centering: may clip if signal isn't centered
python resample_capture.py capture.bin 2fsc --gain 5.7

# With centering: shifts signal to 0V first, then applies gain (recommended)
python resample_capture.py capture.bin 2fsc --gain 5.7 --center
```

---

## analyze_bin.py

Analyze CVBS timing from a capture file.

### Synopsis

```
analyze_bin.py <capture.bin> [max_seconds]
```

### What It Measures

- VBI (Vertical Blanking Interval) timing
- HBI (Horizontal Blanking Interval) timing
- Sync pulse detection and classification
- Field identification (Field 1 vs Field 2)
- Timing statistics and jitter

### Examples

```bash
# Analyze first 3 seconds
python analyze_bin.py capture.bin 3

# Full analysis
python analyze_bin.py capture.bin
```

---

## Data Formats

### BIN File Format (Capture)

The `.bin` format from rpsa_client contains block headers that must be stripped before playback.

**Structure per 8MB segment:**
```
[112-byte BinHeader] [8,388,608 bytes sample data] [12 bytes 0xFF padding]
```

**BinHeader structure (112 bytes):**
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | version | Always 1 |
| 4 | 4 | block_size | 8,388,608 (8MB) |
| 6 | 1 | marker | 0x80 |
| 20-23 | 4 | block_size | Repeated |
| 22 | 1 | marker | 0x80 |
| 36-37 | 2 | spike | 0x29 0x7F (causes glitches!) |
| 72-74 | 3 | identifier | 0x28 0x6B 0xEE |
| 106 | 1 | marker | 0x80 |
| 108-109 | 2 | spike | 0x29 0x7F |

**Sample data:**
- Signed 8-bit integers (int8)
- Sample rate: 15.625 MS/s (125 MS/s / 8)
- Voltage conversion: `voltage = raw_value / 127.0` (±1V range)

**Why headers cause glitches:**
- The `0x29 0x7F` bytes (41, 127 decimal) are max-value spikes
- Each header adds ~7μs timing shift
- Appears as visible white flashes every ~0.5 seconds

Use `convert_tool -f WAV` or `strip_headers.py` to remove headers.

### WAV Format (Playback)

**DAC streaming requires:**
- **16-bit signed PCM** (not 8-bit!)
- Full sample range (±32767) represents ±1V output
- 128-byte alignment for DMA transfers

**From convert_tool (8-bit, not directly playable):**
- 8-bit mono WAV with signed int8 values (non-standard WAV format)
- Original sample rate preserved
- **Cannot be used directly for DAC playback** - must convert to 16-bit first

**From resample_capture.py (16-bit, ready for playback):**
- 16-bit mono WAV, properly scaled for DAC
- Scaling: int8 (±127) → int16 (±32767), factor of 258
- 128-byte aligned for Red Pitaya DAC
- Automatically converts convert_tool's 8-bit output to 16-bit

---

## Troubleshooting

### Capture Issues

**Glitches every ~0.5 seconds (block header glitches):**
- **Cause:** BIN file contains 112-byte headers every 8MB
- **Solution:** Use `convert_tool -f WAV` to convert and strip headers
- **Verify:** `convert_tool capture.bin -i` shows segment count

**Glitches every ~65K samples (buffer overflow):**
- Caused by small default `adc_size` (768 KB)
- `adc_capture.py` fixes this automatically by setting `adc_size=128MB`
- If using web UI, manually set: `adc_size=134217728`

**Connection refused:**
- Streaming server not running
- Use `adc_capture.py` (auto-starts via SSH)
- Or start manually: Web UI → Streaming app → RUN

**SSH permission denied:**
- Set up key authentication: `ssh-copy-id root@192.168.0.6`

### Playback Issues

**No output signal / very weak signal:**
- Verify RF OUT 1 connection
- Check `python dac_stream.py --config` shows correct settings
- Ensure DAC mode is `DAC_NET`
- **Check WAV bit depth:** DAC requires 16-bit WAV, not 8-bit
  - convert_tool outputs 8-bit (won't play correctly)
  - Use `resample_capture.py` which outputs proper 16-bit WAV
  - Verify with: `python -c "import wave; print(wave.open('file.wav').getsampwidth())"`
    - Should show `2` (16-bit), not `1` (8-bit)

**Chroma shimmer/color issues:**
- Resampling affects colorburst phase
- Try playback at original 15.625 MS/s rate (no resampling)

**Signal too weak:**
- Using impedance matching pad? Add `--gain 5.7`
- Use `--center` to maximize headroom before gain

### Buffer overflow errors:

- Reduce capture duration or increase decimation
- Network can handle ~15 MB/s sustained

---

## Technical Notes

### Streaming Rate Trade-offs

Red Pitaya documentation states DAC streaming is limited to ~10 MS/s. Streaming at 15.625 MS/s (the default capture rate) **functions marginally** but exhibits notable recurring pauses in playback, presumed to occur on DMA buffer switches.

**Trade-off summary:**

| Rate | Picture Quality | Timing Stability | Use Case |
|------|-----------------|------------------|----------|
| **15.625 MS/s** | Good | Poor (~1/8 line glitch every ~0.5s) | Visual quality priority |
| **3fsc (10.74 MS/s)** | Poor/marginal | Good (within spec, no glitches) | Timing analysis priority |

**Recommendation:**
- **For timing analysis:** Resample to 3fsc before playback. The picture quality is poor/marginal, but timing relationships are preserved accurately.
- **For picture quality:** Stream at 15.625 MS/s, but be aware the Red Pitaya will cause timebase instability (approximately 1/8th of a line delay, occurring a few times per second).

```bash
# For timing-accurate playback (poor picture)
python resample_capture.py capture.bin 3fsc
python dac_stream.py capture_3fsc.wav --repeat inf

# For picture-quality playback (timing glitches)
python resample_capture.py capture.bin 15.625M
python dac_stream.py capture.wav --repeat inf
```

### Buffer Size Discovery

The streaming application has a critical but undocumented buffer setting:

| Setting | Default | Required |
|---------|---------|----------|
| `adc_size` | 787,968 (768 KB) | 134,217,728 (128 MB) |
| `block_size` | 8,388,608 (8 MB) | (already correct) |

The tiny default `adc_size` causes DMA buffer overflow, resulting in glitches every ~65,660 samples (~64 KB). Setting `adc_size=128MB` eliminates these artifacts.

### NTSC Timing at 15.625 MS/s

| Metric | Value |
|--------|-------|
| Samples per line | 993.056 (fractional) |
| Lines per frame | 525 |
| Samples per frame | ~521,354 |

The fractional samples/line causes timing drift. For perfect NTSC timing, resample to 4fsc (910 samples/line exactly).

### DAC Voltage Scaling

The Red Pitaya DAC expects sample values where the **full range represents ±1V output**:

| Format | Sample Range | Voltage Range | Notes |
|--------|--------------|---------------|-------|
| 8-bit signed | ±127 | ±1V | ADC capture format |
| 16-bit signed | ±32767 | ±1V | **DAC playback format** |

**Conversion:** To convert 8-bit ADC samples to 16-bit DAC samples, multiply by 258 (≈32767/127).

**Why this matters:**
- convert_tool outputs 8-bit values (range ±127)
- If played directly, the DAC interprets ±127 as a tiny fraction of ±32767
- Result: ~0.4% of expected voltage = no visible signal
- Solution: `resample_capture.py` scales 8-bit to 16-bit automatically

### Known Issue: DMA Buffer Boundary Pause (at 15.625 MS/s)

When streaming at 15.625 MS/s (above the documented 10 MS/s limit), we observe **recurring ~7.85 µs pauses** in signal output. These pauses are presumed to occur on DMA buffer switches and may be an artifact of exceeding the documented streaming rate limit.

**Observed behavior (at 15.625 MS/s):**
- Pause duration: ~7.85 µs (approximately 1/8th of a horizontal line)
- Interval: Every 32-33 fields (~0.534 seconds)
- Samples between pauses: ~8,385,000 (≈ 8 MB at 1 byte/sample)

**Impact:**
- One field every ~0.5 seconds is delayed by ~8 µs
- Visible as brief horizontal timing glitch a few times per second
- Horizontal timing max extends to ~8000 ns (vs ~150 ns for reference signal)
- Inflates RMS jitter and std dev statistics for Red Pitaya output

**Mitigation:**
Resample to 3fsc (10.74 MS/s) before playback to stay within the documented streaming limit. This eliminates timing glitches but results in poorer picture quality. See [Streaming Rate Trade-offs](#streaming-rate-trade-offs) for details.

**Presumed root cause:**
Red Pitaya uses ping-pong buffering for DMA transfers. At rates exceeding the documented limit, the system may not complete buffer switches seamlessly, causing brief output pauses.

**Related issues:**
- [OpenDGPS/zynq-axi-dma-sg #4](https://github.com/OpenDGPS/zynq-axi-dma-sg/issues/4) - Discontinuities at DMA descriptor boundaries
- [pavel-demin/red-pitaya-notes #320](https://github.com/pavel-demin/red-pitaya-notes/issues/320) - DMA buffer handling
- [Red Pitaya Streaming Documentation](https://redpitaya.readthedocs.io/en/latest/appsFeatures/applications/streaming/appStreaming.html) - Documents 10 MS/s streaming limit
