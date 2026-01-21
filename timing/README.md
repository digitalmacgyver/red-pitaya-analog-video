# Video Timing Analysis Tools

Tools for analyzing horizontal and vertical sync timing stability of CVBS video signals.

## Data Collection Setup

Timing data is collected using:
- **Logic Analyzer:** Saleae Logic Pro at 20 MS/s sampling rate
- **Sync Separator:** TI LM1881 or LM1980 evaluation board
- **Test Points:** HSOUT (horizontal sync) and VSOUT (vertical sync/composite sync)

### Connection Diagram

```
[Video Source] ──> [LM1881/LM1980 Eval Board] ──> [Saleae Logic Analyzer]
                          │                              │
                          ├─ HSOUT (pin X) ────> Channel 0
                          └─ VSOUT (pin Y) ────> Channel 1
```

## CSV File Format

Exported from Saleae Logic software as CSV:

```csv
Time [s],Channel 0,Channel 1
0.000000000,1,1
0.000053100,0,1
0.000055450,1,1
...
```

| Column | Description |
|--------|-------------|
| Time [s] | Timestamp in seconds (float) |
| Channel 0 | HSOUT signal (1=high, 0=low) |
| Channel 1 | VSOUT signal (1=high, 0=low) |

### Signal Interpretation

- **HBI (Horizontal Blanking Interval):** Starts when Channel 0 transitions 1→0
- **VBI (Vertical Blanking Interval):** Starts when Channel 1 transitions 1→0

The file contains every edge transition, not sampled data.

## NTSC Timing Standards (Reference)

| Parameter | Nominal Value | Tolerance |
|-----------|---------------|-----------|
| Line frequency (H) | 15,734.264 Hz | ±0.003% |
| Line period | 63.5555... µs | |
| Field frequency | 59.94 Hz | |
| Field period | 16.6833... ms | |
| Lines per frame | 525 | |
| Lines per field | 262.5 | |
| Horizontal sync pulse | 4.7 µs | ±0.1 µs |
| Vertical sync (serrated) | 3H (3 lines) | |

### Half-Line Handling

NTSC uses interlaced scanning with half-lines at field boundaries:
- Field 1 ends with a half-line (line 263)
- Field 2 starts with a half-line (line 1)

The analysis tools automatically detect and handle these half-lines to avoid
artificially inflating timing variance statistics.

## Analysis Tools

### generate_timing_report.py

Generates a comprehensive HTML timing comparison report.

```bash
python timing/generate_timing_report.py \
    --capture1 reference.csv \
    --capture2 device_under_test.csv \
    --output /path/to/report.html
```

#### Options

| Option | Description |
|--------|-------------|
| `--capture1 FILE` | First capture file (typically reference/known-good) |
| `--capture2 FILE` | Second capture file (device under test) |
| `--output FILE` | Output HTML report path |
| `--label1 NAME` | Label for capture 1 (default: filename) |
| `--label2 NAME` | Label for capture 2 (default: filename) |
| `--skip-fields N` | Fields to skip at start/end (default: 16) |

#### Data Trimming

By default, the first and last 16 fields of each capture are discarded to avoid
transient effects at recording start/stop. This is configurable via `--skip-fields`.

### Output

The report includes:

1. **Summary Statistics Table**
   - Median, mean, std dev for H and V timing
   - Percentiles (1, 5, 25, 50, 75, 95, 99)
   - Min/max values

2. **Jitter Metrics**
   - RMS jitter
   - Peak-to-peak jitter
   - Time Interval Error (TIE) distribution

3. **Visualizations**
   - Line-by-line timing distribution (heatmap/density plot)
   - Timing deviation histogram
   - Field-to-field stability plot

## Metrics Definitions

| Metric | Definition |
|--------|------------|
| **Line Period** | Time between successive HSYNC pulses |
| **Field Period** | Time between successive VSYNC pulses |
| **Jitter (RMS)** | Root mean square of timing deviations from median |
| **Jitter (P-P)** | Peak-to-peak timing deviation (max - min) |
| **TIE** | Time Interval Error: deviation from ideal nominal timing |

## Example Workflow

```bash
# Compare Leitch signal generator to Red Pitaya playback
python timing/generate_timing_report.py \
    --capture1 from_leitch.csv \
    --capture2 from_red_pitaya.csv \
    --label1 "Leitch Generator" \
    --label2 "Red Pitaya" \
    --output timing_comparison.html
```
