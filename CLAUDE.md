# RF Replay - CVBS Video Capture Project

## Project Goal

Create a "hello world" RF capture application for the Red Pitaya Stemlab 125-14 that captures 2-3 seconds of CVBS (composite video) analog signal via the fast analog RF input ports, saving the captured data to a file for later retrieval via SCP/FTP.

## Target Hardware

**Device:** Red Pitaya Stemlab 125-14
**Network Address:** 192.168.0.6
**Interface:** Jupyter Notebook server running on the device

### Key Hardware Specifications

- **ADC:** 2-channel, 14-bit, 125 MS/s simultaneous sampling
- **Input Range:** ±1V (LV jumper setting) or ±20V (HV jumper setting)
- **FPGA:** Xilinx Zynq 7010 (dual-core ARM Cortex-A9 + FPGA fabric)
- **DDR RAM:** 512 MB (shared between Linux and DMA buffers)
- **Standard Acquisition Buffer:** 16,384 samples (circular buffer)
- **Deep Memory Acquisition (DMA):** Up to ~412 MB usable (leave 100 MB for Linux)

## CVBS Signal Characteristics

- **Bandwidth:** 4.2 MHz (NTSC) to 5.5 MHz (PAL)
- **Color Subcarrier (fsc):** 3.579545 MHz (NTSC) or 4.43 MHz (PAL)
- **Voltage Level:** ~1V peak-to-peak typical

### Target Sampling Rate: 4fsc

**Required:** 4× color subcarrier = **14.3181818 MS/s** (NTSC)

This is the standard sampling rate for CVBS digitization, providing exactly 910 samples per horizontal line (NTSC) and proper phase relationship with the color subcarrier.

## Capture Strategy

### Why Deep Memory Acquisition (DMA)?

The standard 16,384-sample buffer at 125 MS/s only captures ~131 microseconds. For 2-3 seconds of continuous capture, we need DMA which streams directly to DDR RAM.

### Decimation Constraint

Red Pitaya only supports power-of-2 decimation from 125 MS/s:

| Decimation | Sample Rate | vs 4fsc | Notes |
|------------|-------------|---------|-------|
| RP_DEC_8   | 15.625 MS/s | +9.1%   | Closest available, slightly oversampled |
| RP_DEC_16  | 7.8125 MS/s | -45%    | Too slow, will alias |

**Strategy:** Capture at `RP_DEC_8` (15.625 MS/s), then resample to exactly 4fsc (14.3181818 MS/s) in post-processing if needed for video decoding compatibility.

### Data Size Calculations

At RP_DEC_8 (15.625 MS/s), 14-bit samples stored as 16-bit (2 bytes):
- **2 seconds:** 31.25 million samples = ~62.5 MB
- **3 seconds:** 46.875 million samples = ~93.75 MB

Both fit comfortably within the 412 MB DMA limit.

## Implementation Approach

### Option A: Native Python API (Recommended)

Run Python directly on the Red Pitaya using the `rp` module. This provides:
- Direct hardware access via DMA
- NumPy integration for efficient data handling
- No network latency during capture

```python
import rp
import numpy as np

# Key functions:
# rp.rp_AcqAxiGetMemoryRegion() - get DMA memory bounds
# rp.rp_AcqAxiSetDecimationFactor(dec) - set decimation
# rp.rp_AcqAxiSetBufferSamples(channel, start_addr, num_samples)
# rp.rp_AcqAxiEnable(channel, True/False)
# rp.rp_AcqStart() / rp.rp_AcqStop()
# rp.rp_AcqAxiGetDataRawNP(channel, pos, numpy_array) - fast data retrieval
```

### Option B: SCPI over Network

Send commands from a remote computer via TCP socket. Slower but useful for remote control.

### Option C: Streaming Application

Built-in streaming app supports continuous capture to SD card at ~10 MB/s. Limited throughput but simpler setup.

## File Output

Save captured data as:
- **Binary file (.bin):** Raw 16-bit signed integers, most compact
- **NumPy file (.npy):** Easy to load in Python for post-processing
- **WAV file (.wav):** If interoperability needed (streaming app supports this)

## Hardware Setup Notes

1. **Input Jumpers:** Set to LV (±1V) for CVBS signals (~1Vpp)
2. **Termination:** May need 75Ω termination for proper impedance matching
3. **DC Coupling:** CVBS includes sync pulses that go below 0V; check DC offset handling

## ADC Value to Voltage Conversion

The 14-bit ADC outputs signed 16-bit integers:
- **LV mode (±1V):** `voltage = raw_value / 8192.0`
- **HV mode (±20V):** `voltage = raw_value / 8192.0 * 20.0`

Example CVBS readings (LV mode):
- Raw -2300 → -0.28V (sync tip)
- Raw 0 → 0V (blanking level)
- Raw +5300 → +0.65V (peak white)

## DMA Memory Configuration

**IMPORTANT:** The default DMA region is only **2 MB**, limiting capture to ~67ms!

### Setup Notebook: `setup_dma_memory.ipynb`

**URL:** http://192.168.0.6:8888/jlab/lab/tree/setup_dma_memory.ipynb

This notebook increases the DMA region from 2 MB to 128 MB by:
1. Modifying `/opt/redpitaya/dts/<fpga_model>/dtraw.dts`
2. Changing `reg = <0x1000000 0x200000>` to `reg = <0x1000000 0x8000000>`
3. Recompiling device tree with `dtc`
4. Rebooting

| DMA Size | Hex | Max Duration @ 15.625 MS/s |
|----------|-----|---------------------------|
| 2 MB | 0x200000 | 67 ms (default) |
| 64 MB | 0x4000000 | 2.1 seconds |
| 128 MB | 0x8000000 | 4.2 seconds |

**Run `setup_dma_memory.ipynb` first, then reboot, then run `cvbs_capture.ipynb`**

## Capture Notebook

**File:** `cvbs_capture.ipynb`

**Location on Red Pitaya:** Available at root of Jupyter server

**Access URL:** http://192.168.0.6:8888/jlab/lab/tree/cvbs_capture.ipynb

### What it does:
1. Initializes Red Pitaya FPGA and acquisition hardware
2. Configures DMA for 2-second capture at 15.625 MS/s (RP_DEC_8)
3. Triggers immediate capture (no external trigger needed)
4. Reads captured data from DMA buffer into numpy array
5. Saves to `/tmp/cvbs_capture.npy` (fixed filename, overwrites previous)
6. Displays preview plot of captured waveform

### To retrieve captured data:
```bash
scp root@192.168.0.6:/tmp/cvbs_capture.npy .
```

### Hardware Setup Before Running:
1. Set IN1 jumper to **LV** (±1V range)
2. Connect CVBS video source to **IN1**
3. Optional: Add 75Ω termination

## Playback Notebook

**File:** `cvbs_playback.ipynb`

**Access URL:** http://192.168.0.6:8888/jlab/lab/tree/cvbs_playback.ipynb

### What it does:
1. Loads captured data from `/tmp/cvbs_capture.npy`
2. Converts int16 ADC values to float DAC values (-1.0 to +1.0)
3. Uses Deep Memory Generation (DMG) to output waveform at 15.625 MS/s
4. Outputs through RF OUT 1

### Requirements:
- Red Pitaya OS 2.07-48 or later for DMG support
- Captured data file must exist (`/tmp/cvbs_capture.npy`)

### Hardware Setup:
1. Connect RF OUT 1 to video display/capture device
2. Output range is ±1V (matches standard CVBS levels)

### DMG API Functions Used:
- `rp_GenAxiGetMemoryRegion()` - get available memory
- `rp_GenAxiSetDecimationFactor(8)` - set 15.625 MS/s output rate
- `rp_GenAxiWriteWaveform()` - load waveform to deep memory
- `rp_GenAxiSetEnable()` - start/stop output

## Network Streaming (For Longer Captures)

For captures longer than ~13 seconds (DMA limit), use network streaming.

### Advantages
- **Unlimited duration** - stream for hours (limited by PC disk space)
- **No reboot required** - works with default DMA allocation
- **Real-time** - data streams to PC as captured

### Data Rate Performance

| OS Version | Max Network Rate | Can Handle 15.625 MS/s? |
|------------|------------------|------------------------|
| 2.07-43+   | 62.5 MB/s        | Yes (31.25 MB/s needed) |
| 1.04 - 2.05| 20 MB/s          | No - need higher decimation |

### Files

**Local receiver script:** `stream_receiver.py`
```bash
# Capture 10 seconds of streaming data
python stream_receiver.py -d 10

# Capture 60 seconds to specific file
python stream_receiver.py -d 60 -o my_capture

# Show help
python stream_receiver.py --help-server
```

**Red Pitaya setup notebook:** `setup_streaming.ipynb`
- URL: http://192.168.0.6:8888/jlab/lab/tree/setup_streaming.ipynb

### Quick Start

1. **On Red Pitaya:** Start streaming via web interface
   - Open http://192.168.0.6/
   - Click "Streaming" application
   - Set: Mode=Network, Decimation=8, Channel=CH1, Protocol=TCP
   - Click "RUN"

2. **On your PC:** Run receiver
   ```bash
   python stream_receiver.py -d 30   # Capture 30 seconds
   ```

### Hardware Limits (Why Streaming Matters)

| Resource | Limit | Impact |
|----------|-------|--------|
| RAM | 512 MB (soldered) | ~412 MB max for DMA |
| USB | 2.0 only (~35 MB/s) | Can't stream to USB storage |
| SD Card | ~10 MB/s | Too slow for full rate |
| Ethernet | 62.5 MB/s | ✓ Can handle our 31.25 MB/s |

## ADC Capture via Network Streaming

### Recommended: `adc_capture.py`

Command-line tool that captures via network streaming without requiring the web UI:

```bash
# Capture 30 seconds at 15.625 MS/s
python adc_capture.py -d 30

# Capture with specific output directory
python adc_capture.py -d 30 -o /path/to/output

# Show current configuration
python adc_capture.py --config
```

**Key Features:**
- Auto-starts streaming server via SSH (no web UI needed)
- Automatically sets correct buffer sizes to prevent glitches
- Duration-based capture with automatic stop

### CRITICAL: Buffer Size Configuration

The streaming application has an undocumented buffer setting that causes glitches if too small:

| Setting | Default | Required |
|---------|---------|----------|
| `adc_size` | 787,968 (768 KB) | 134,217,728 (128 MB) |
| `block_size` | 8,388,608 (8 MB) | (already correct) |

**Without this fix, captures have glitches every ~65,660 samples (~64 KB DMA buffer overflow)!**

The `adc_capture.py` script sets these automatically. If using the web UI, set manually:
```bash
./rp_streaming/cmd/rpsa_client -c -h 192.168.0.6 -i "adc_size=134217728" -w
```

### Starting Streaming Server via SSH

```bash
ssh root@192.168.0.6 "cd /opt/redpitaya/bin && \
  LD_LIBRARY_PATH=/opt/redpitaya/lib /opt/redpitaya/sbin/overlay.sh stream_app && \
  sleep 1 && \
  LD_LIBRARY_PATH=/opt/redpitaya/lib ./streaming-server -v &"
```

## DAC Network Streaming (For Longer Playback)

For playback longer than ~4.3 seconds (DMG limit), use DAC network streaming.

### Key Discovery: 15.625 MS/s Works!

Despite Red Pitaya documentation stating DAC streaming is limited to ~10 MS/s, **testing shows 15.625 MS/s works reliably**. This means:
- No resampling required for playback at original capture rate
- Color/chroma issues from resampling are avoided
- Simpler workflow

### DAC Streaming Workflow

1. **Convert capture to WAV (no resampling):**
   ```bash
   python resample_capture.py capture.bin 15.625M -o playback.wav
   ```

2. **Stream to Red Pitaya:**
   ```bash
   python dac_stream.py playback.wav --repeat inf
   ```

### DAC Streaming Script: `dac_stream.py`

```bash
# Stream once
python dac_stream.py /path/to/file.wav

# Stream with infinite loop
python dac_stream.py /path/to/file.wav --repeat inf

# Stop DAC output
python dac_stream.py --stop

# Show current config
python dac_stream.py --config
```

### DAC Streaming Rates (Updated)

| Sample Rate | Status | Notes |
|-------------|--------|-------|
| 15.625 MS/s | **Works** | Original capture rate, no resampling |
| 14.318 MS/s (4fsc) | Works | Standard CVBS rate |
| 7.159 MS/s (2fsc) | Works | Half bandwidth |

### File Requirements

- **Format:** WAV (16-bit signed PCM) or TDMS
- **Alignment:** Data must be multiple of 128 bytes
- **Max size:** 4 GB per WAV file (~268 million samples)

## Current Status (2026-01-20)

### What Works
- **Capture:**
  - `adc_capture.py` - **Recommended** command-line capture via SSH (no web UI needed)
  - `cvbs_capture.ipynb` - File-based DMA capture to Red Pitaya filesystem
  - Red Pitaya Streaming App - Web UI-based streaming (requires manual buffer config)
- **Playback:**
  - `dac_stream.py` - **Recommended** DAC network streaming (unlimited duration, works at 15.625 MS/s)
  - `cvbs_playback_optimized.py` - DMG playback (up to ~4.3 seconds)
- **DMA/DMG:** 128 MB region configured and working
- **Buffer Configuration:** Critical `adc_size=128MB` setting discovered and documented
- **Timing Analysis:** `analyze_bin.py` - Analyze VBI/HBI timing
- **Resampling:** `resample_capture.py` - Direct one-pass sinc resampling with gain/centering options

### End-to-End Workflow (Recommended)

```bash
# 1. Capture 30 seconds
python adc_capture.py -d 30 -o /wintmp/analog_video/rpsa_client/output

# 2. Convert to WAV (no resampling for best quality)
python resample_capture.py /path/to/capture.bin 15.625M -o playback.wav

# 3. Play back
python dac_stream.py playback.wav --repeat inf
```

### Key Discoveries

**DAC Streaming at 15.625 MS/s:** Despite documentation stating ~10 MS/s limit, **15.625 MS/s works reliably**. This eliminates the need for resampling and avoids color/chroma issues.

**Buffer Size Critical:** Default `adc_size` of 768 KB causes glitches every ~65K samples. Must set to 128 MB.

### Known Limitations
1. **DMG memory limit:** 128 MB = ~4.3 seconds max playback
2. **VHS timebase instability:** Captured VHS shows timing jitter (expected without TBC)

### Scratch Space

Use `/wintmp/analog_video/rpsa_client/output/tmp/` for temporary files during development and testing. Do not use `/tmp` as local filesystem space is limited.

### Local Files
| File | Purpose |
|------|---------|
| `adc_capture.py` | **Recommended** command-line ADC streaming capture |
| `dac_stream.py` | **Recommended** DAC streaming playback |
| `resample_capture.py` | Convert/resample captures with gain/centering options |
| `analyze_bin.py` | Analyze CVBS timing (VBI/HBI detection, jitter) |
| `visualize_capture.py` | Visualize captured waveforms |
| `cvbs_playback_optimized.py` | DMG playback script (for short clips) |
| `convert_to_playback.py` | Legacy: Convert 8-bit .bin to float32 .f32 |
| `README_CAPTURE.md` | Capture and playback documentation |

## Next Steps

1. ~~Create capture script using DMA with appropriate decimation~~
2. ~~Create playback script~~
3. ~~Upgrade Red Pitaya OS to 2.07-48+~~
4. ~~Test playback with DMG~~
5. ~~Implement network streaming for longer captures~~
6. ~~Fix DMG memory reservation~~
7. ~~Optimize playback to read int8 directly~~
8. ~~Post-processing to resample to exact 4fsc~~
9. ~~Analyze CVBS timing to identify drift issues~~
10. ~~Investigate continuous DAC streaming for longer playback~~ (works at 15.625 MS/s!)
11. ~~Create command-line capture tool~~ (adc_capture.py)
12. ~~Discover and document critical adc_size buffer setting~~
13. Software timebase correction for VHS captures

## References

- [Red Pitaya Documentation](https://redpitaya.readthedocs.io/)
- [Deep Memory Acquisition](https://redpitaya.readthedocs.io/en/latest/appsFeatures/remoteControl/deepMemoryAcquisition.html)
- [DMA Python Examples](https://redpitaya.readthedocs.io/en/latest/appsFeatures/examples/DMM/deepMemoryAcq.html)
- [Streaming Application](https://redpitaya.readthedocs.io/en/latest/appsFeatures/applications/streaming/appStreaming.html)
- [CVBS Video Standards](https://www.ni.com/docs/en-US/bundle/video-measurement-suite/page/nivms/signals_cvbs.html)
