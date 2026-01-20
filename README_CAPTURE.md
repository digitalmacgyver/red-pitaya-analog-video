# CVBS Capture and Playback Tools

Tools for recording and replaying CVBS (composite video) signals using a Red Pitaya Stemlab 125-14.

## Quick Start: End-to-End Workflow

### 1. Capture (30 seconds of CVBS)

```bash
# Capture 30 seconds at 15.625 MS/s (no UI required - uses SSH)
python adc_capture.py -d 30 -o /wintmp/analog_video/rpsa_client/output
```

### 2. Prepare for Playback (No Resampling)

```bash
# Convert to WAV at original rate for playback
python resample_capture.py /path/to/capture.bin 15.625M -o playback.wav
```

### 3. Play Back via DAC Streaming

```bash
# Stream to Red Pitaya DAC (loops infinitely)
python dac_stream.py playback.wav --repeat inf
```

**Note:** Playback at 15.625 MS/s works reliably despite Red Pitaya documentation stating a 10 MS/s limit for DAC streaming.

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

**Capture:**
1. Set IN1 jumper to **LV** (±1V range)
2. Connect CVBS source to **IN1** (RF input 1)
3. Optional: Add 75Ω termination for impedance matching

**Playback:**
1. Connect **RF OUT 1** to your video device
2. For proper impedance matching (50Ω to 75Ω), use a minimum loss pad
3. Apply 5.7 dB gain compensation if using the pad: `--gain 5.7`

---

## Tools Overview

| Tool | Purpose |
|------|---------|
| `adc_capture.py` | **Recommended capture** - Command-line streaming capture via SSH |
| `dac_stream.py` | **Recommended playback** - DAC streaming for unlimited duration |
| `resample_capture.py` | Convert/resample captures for playback |
| `analyze_bin.py` | Analyze CVBS timing and quality |
| `visualize_capture.py` | Visualize captured waveforms |

---

## adc_capture.py (Recommended Capture)

Command-line tool for ADC streaming capture. Auto-starts the streaming server via SSH.

### Synopsis

```
adc_capture.py [-d DURATION] [--decimation N] [-f FORMAT] [-o DIR] [OPTIONS]
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
| `--channel {1,2}` | ADC channel (default: 1) |
| `--resolution {8,16}` | Bits per sample (default: 8) |
| `--no-ssh` | Don't auto-start server (use web UI instead) |
| `--stop` | Stop any running capture |
| `--kill-server` | Kill streaming server |
| `--config` | Show current configuration |

### Examples

```bash
# Default 2-second capture
python adc_capture.py

# 30-second capture
python adc_capture.py -d 30

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
| `--repeat N` | Repeat count (default: 1, use `inf` for infinite) |
| `--rate HZ` | Override DAC rate (default: from WAV file) |
| `-H, --host IP` | Red Pitaya IP (default: 192.168.0.6) |
| `--stop` | Stop DAC and exit |
| `--config` | Show current DAC configuration |

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
| 15.625 MS/s | **Works** | Original capture rate |
| 7.159 MS/s (2fsc) | Works | Standard for streaming |
| 14.318 MS/s (4fsc) | Works | Standard CVBS rate |

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
| `--header N` | Manual header size in bytes |
| `--no-skip-header` | Don't auto-detect/skip file header |

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

### Capture Format (.bin)

- **From adc_capture.py**: Signed 8-bit integers (int8)
- **Sample rate**: 15.625 MS/s (125 MS/s / 8)
- **Voltage conversion**: `voltage = raw_value / 127.0` (±1V range)

### Playback Format (.wav)

- **Format**: 16-bit mono WAV
- **Sample rate**: As specified (e.g., 15.625 MS/s, 7.159 MS/s)
- **Alignment**: Data aligned to 128 bytes for Red Pitaya DAC

---

## Troubleshooting

### Capture Issues

**Glitches every ~65K samples:**
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

**No output signal:**
- Verify RF OUT 1 connection
- Check `python dac_stream.py --config` shows correct settings
- Ensure DAC mode is `DAC_NET`

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

### Why 15.625 MS/s Works for DAC Streaming

Red Pitaya documentation states DAC streaming is limited to ~10 MS/s, but testing shows 15.625 MS/s works reliably. This may be:
- Conservative documentation
- Hardware-dependent (newer FPGA/firmware)
- Network-dependent (local gigabit connection)

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
