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

## DAC Network Streaming (For Longer Playback)

For playback longer than ~4.3 seconds (DMG limit), use DAC network streaming with resampled files.

### Why Resampling is Required

The Red Pitaya DAC streaming has rate limits:
- **16-bit mode:** ~5 MS/s max stable rate
- **8-bit mode:** ~10 MS/s max stable rate

Since our capture rate (15.625 MS/s) exceeds these limits, we must resample to a lower rate for streaming playback.

**Recommended:** Resample to **2fsc (7.159 MS/s)** which:
- Fits within the 8-bit streaming limit
- Provides 455 integer samples per NTSC line (clean timing)
- Still exceeds Nyquist for CVBS bandwidth (4.2 MHz)

### DAC Streaming Workflow

1. **Resample capture to 2fsc:**
   ```bash
   python resample_capture.py capture.bin 2fsc
   # Creates: capture_resampled_2fsc.wav
   ```

2. **Stream to Red Pitaya:**
   ```bash
   # Using wrapper script
   python dac_stream.py capture_resampled_2fsc.wav

   # Or using rpsa_client directly
   ./rp_streaming/cmd/rpsa_client -o -h 192.168.0.6 -f wav \
     -d capture_resampled_2fsc.wav -r 1
   ```

### DAC Streaming Script: `dac_stream.py`

```bash
# Stream once
python dac_stream.py /path/to/file.wav

# Stream with infinite loop
python dac_stream.py /path/to/file.wav --repeat inf

# Stream with specific repeat count
python dac_stream.py /path/to/file.wav --repeat 5

# Specify custom sample rate (if different from WAV header)
python dac_stream.py /path/to/file.wav --rate 7159090

# Stop DAC output
python dac_stream.py --stop

# Show current config
python dac_stream.py --config
```

### rpsa_client Direct Usage

```bash
# Stream WAV file once
./rp_streaming/cmd/rpsa_client -o -h 192.168.0.6 -f wav -d file.wav -r 1

# Stream infinitely
./rp_streaming/cmd/rpsa_client -o -h 192.168.0.6 -f wav -d file.wav -r inf

# Configure DAC rate manually
./rp_streaming/cmd/rpsa_client -c -h 192.168.0.6 -i "dac_rate=7159090" -w
./rp_streaming/cmd/rpsa_client -c -h 192.168.0.6 -i "dac_pass_mode=DAC_NET" -w

# Check current configuration
./rp_streaming/cmd/rpsa_client -c -h 192.168.0.6 -g V1
```

### DAC Streaming Limits

| Sample Rate | Bits | Duration Limit | Use Case |
|-------------|------|----------------|----------|
| 2fsc (7.159 MS/s) | 16 | Unlimited | Recommended for CVBS |
| 4fsc (14.318 MS/s) | 16 | Limited/unstable | May have underruns |
| 15.625 MS/s | 16 | Not supported | Use DMG instead |

### File Requirements

- **Format:** WAV (16-bit signed PCM) or TDMS
- **Alignment:** Data must be multiple of 128 bytes
- **Max size:** 4 GB per WAV file (~268 million samples)

## Current Status (2026-01-19)

### What Works
- **Capture:** Multiple methods working:
  - `cvbs_capture.ipynb` - File-based DMA capture to Red Pitaya filesystem
  - Red Pitaya Streaming App + Windows client - Network streaming for longer captures
  - `stream_capture.py` - Custom SSH-triggered streaming (alternative method)
- **Playback:** Two methods available:
  - `cvbs_playback_optimized.py` - DMG playback (up to ~4.3 seconds at 15.625 MS/s)
  - `dac_stream.py` - DAC network streaming (unlimited duration at 2fsc/7.159 MS/s)
- **DMA/DMG:** 128 MB region configured and working for both capture and playback
- **Data Quality:** Captured CVBS shows correct voltage levels
- **Timing Analysis:** `analyze_bin.py` - Analyze VBI/HBI timing and detect drift
- **Resampling:** `resample_capture.py` - Resample via 4×fsc for proper NTSC alignment

### Playback Workflow (Recommended)
1. Capture using streaming app (8-bit, 15.625 MS/s) → .bin file
2. Upload: `scp capture.bin root@192.168.0.6:/tmp/`
3. Play: `ssh root@192.168.0.6 "python3 /home/jupyter/cvbs_project/cvbs_playback_optimized.py --skip-header /tmp/capture.bin"`

No intermediate conversion needed - the optimized script reads int8 directly and converts in chunks.

### Resampling Workflow (for DAC streaming or external players)
1. Resample to desired rate: `python resample_capture.py capture.bin 4fsc`
2. Stream via rpsa_client: `rpsa_client.exe -o -h 192.168.0.6 -f wav -d capture_4fsc.wav -r inf`

Available presets: 4fsc (14.318 MHz), 2fsc (7.159 MHz), 1fsc (3.579 MHz), 0.5fsc (1.790 MHz)

### NTSC Timing Analysis Findings

**Key Discovery:** Red Pitaya captures at 15.625 MS/s, which gives **993.056 samples per NTSC line** - a fractional number causing timing drift:
- 0.056 samples/line drift
- 14.6 samples/field cumulative error
- 876 samples/second timing slip

**Solution:** Resample via 4×fsc (14.31818 MS/s) which gives exactly **910 samples/line**, eliminating fractional accumulation.

### DMG Technical Details
- **API input:** float32 (-1.0 to 1.0)
- **Internal storage:** int16 (2 bytes/sample) - API converts automatically
- **Memory reservation:** Must use `num_samples * 2` bytes (not * 4)
- **Max duration:** 128 MB / 2 bytes = 64M samples = **4.3 seconds**
- **Chunked writes:** Use `rp_GenAxiWriteWaveformOffset(ch, offset, data)` for memory efficiency

### Known Limitations
1. **DMG memory limit:** 128 MB = ~4.3 seconds max playback at 15.625 MS/s
2. **DAC streaming rate limit:** ~5-10 MS/s max - requires resampling to 2fsc (7.159 MS/s) for streaming playback
3. **VHS timebase instability:** Captured VHS shows timing jitter (expected without TBC)

### Files on Red Pitaya Jupyter Server
| File | Purpose | Status |
|------|---------|--------|
| `cvbs_capture.ipynb` | Capture CVBS to file | Working |
| `cvbs_playback_optimized.py` | Optimized DMG playback | **Recommended** |
| `cvbs_dmg_playback.ipynb` | DMG playback (legacy) | Working |
| `setup_dma_memory.ipynb` | Expand DMA region | Working |
| `/home/jupyter/cvbs_project/cvbs_captures/` | Directory for playback files | Active |

### Local Files
| File | Purpose |
|------|---------|
| `analyze_bin.py` | Analyze CVBS timing (VBI/HBI detection, jitter, drift) |
| `resample_capture.py` | Resample captures via 4×fsc intermediate for NTSC alignment |
| `dac_stream.py` | Stream WAV files to Red Pitaya DAC via network |
| `cvbs_playback_optimized.py` | Optimized playback script (reads int8 directly) |
| `convert_to_playback.py` | Legacy: Convert 8-bit .bin to float32 .f32 |
| `visualize_capture.py` | Visualize captured data |
| `cvbs_capture.ipynb` | Capture notebook (local copy) |
| `cvbs_dmg_playback.ipynb` | Playback notebook (local copy) |
| `README_CAPTURE.md` | Capture and playback documentation |
| `.credentials` | Red Pitaya login (root/root) - gitignored |

## Next Steps

1. ~~Create capture script using DMA with appropriate decimation~~
2. ~~Create playback script~~
3. ~~Upgrade Red Pitaya OS to 2.07-48+~~
4. ~~Test playback with DMG~~
5. ~~Implement network streaming for longer captures~~
6. ~~Fix DMG memory reservation (was using float32 size, now uses int16)~~
7. ~~Optimize playback to read int8 directly (no intermediate float32 file)~~
8. ~~Post-processing to resample to exact 4fsc~~ (resample_capture.py with 4fsc intermediate)
9. ~~Analyze CVBS timing to identify drift issues~~ (analyze_bin.py)
10. Software timebase correction for VHS captures
11. ~~Investigate continuous DAC streaming for longer playback~~ (working at 2fsc via dac_stream.py)

## References

- [Red Pitaya Documentation](https://redpitaya.readthedocs.io/)
- [Deep Memory Acquisition](https://redpitaya.readthedocs.io/en/latest/appsFeatures/remoteControl/deepMemoryAcquisition.html)
- [DMA Python Examples](https://redpitaya.readthedocs.io/en/latest/appsFeatures/examples/DMM/deepMemoryAcq.html)
- [Streaming Application](https://redpitaya.readthedocs.io/en/latest/appsFeatures/applications/streaming/appStreaming.html)
- [CVBS Video Standards](https://www.ni.com/docs/en-US/bundle/video-measurement-suite/page/nivms/signals_cvbs.html)
