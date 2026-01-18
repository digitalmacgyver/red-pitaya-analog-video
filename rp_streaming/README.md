# Red Pitaya Streaming Client Setup

## Overview

The Red Pitaya streaming application allows continuous capture of ADC data over the network, bypassing DMA memory limits.

## Windows Client Setup

### 1. Download the Client

Download the Windows streaming client from Red Pitaya's website (rpsa_client for your OS version).

### 2. Firewall Configuration

Windows Defender Firewall must allow the client to access the network:

1. When first running the client, a popup will offer to allow **public network** access - accept this
2. For **private network** access (which the Red Pitaya is likely on), manually configure:
   - Open Windows Defender Firewall
   - Go to "Allow an app through firewall"
   - Find the rpsa_client binary
   - Check both "Private" and "Public" network boxes

### 3. Red Pitaya Web UI Configuration

Configure streaming via the Red Pitaya web interface (http://192.168.0.6/ -> Streaming app):

| Setting | Value | Notes |
|---------|-------|-------|
| IP | 192.168.0.6 | This is the Red Pitaya's own IP, NOT the destination |
| Rate | 15.625 MS/s | Decimation 8 from 125 MS/s (closest to 4fsc for CVBS) |
| Resolution | 8-bit | Sufficient for VHS; halves data rate |
| Calibration | On | Applies factory ADC calibration |

**Note:** The Rate field only accepts power-of-2 decimation values from 125 MS/s. If you enter an arbitrary value like 14.318 MS/s, it will round down to the nearest valid rate.

### 4. Running the Client

1. Start the Windows streaming client
2. The client will auto-discover the Red Pitaya on the network
3. The streaming interface appears pre-populated with settings from the Red Pitaya web UI
4. Click "Start ADC" to begin capture

## Linux Client

The Linux client (rpsa_client_qt) requires glibc 2.38+ and GLIBCXX 3.4.32+. Ubuntu 22.04 has older versions (glibc 2.35, GLIBCXX 3.4.30), so the pre-built binary won't run.

**Alternatives for Linux:**
- Use the Windows client on a Windows host
- Use the custom `stream_capture.py` script (SSH-triggered, different protocol)
- Build the client from source with matching library versions

## Network Setup

For streaming to work, the client PC must be on the same network as the Red Pitaya:

| Device | IP |
|--------|-----|
| Red Pitaya | 192.168.0.6 |
| Windows PC (if using Windows client) | 192.168.0.51 |
| Linux VM (if using custom scripts) | 192.168.0.100 |

## Data Rates

At 15.625 MS/s with 8-bit resolution:
- Data rate: 15.625 MB/s
- Network limit: 62.5 MB/s (OS 2.07+)
- Headroom: 77%

This allows reliable continuous capture for extended durations.
