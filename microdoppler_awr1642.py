#!/usr/bin/env python3
"""
AWR1642 Range-Time Visualization for Vital Signs
=================================================
This script captures radar data and creates visualizations for:
- Range profile over time (for vital signs monitoring)
- Motion signatures at different ranges

Since the AWR1642 demo firmware outputs Range Profile (not Range-Doppler heatmap),
this script focuses on range-time analysis which is ideal for vital signs.

Usage:
    python microdoppler_awr1642.py --mode live --duration 30
    python microdoppler_awr1642.py --mode analyze --frames 200
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from awr1642_driver import AWR1642, load_config_from_file, DEFAULT_CONFIG_FILE


def plot_range_time_live(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1', duration=30):
    """
    Real-time range-time visualization.
    
    This shows the range profile evolution over time, which is useful for
    detecting breathing and heartbeat patterns at a fixed distance.
    """
    print("="*70)
    print("AWR1642 Range-Time Live Visualization")
    print("="*70)
    
    # Setup radar
    radar = AWR1642(cli_port=cli_port, data_port=data_port)
    
    try:
        radar.connect()
        print("✓ Connected to radar")
        
        # Load and send configuration
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        config_cmds = load_config_from_file(config_file)
        radar.configure_sensor(config_cmds)
        print("✓ Radar configured")
        
        # Get parameters
        range_res = radar.config_params.get('rangeResolutionMeters', 0.044)
        num_range_bins = radar.config_params.get('numRangeBins', 256)
        
        # Data collection parameters
        buffer_size = 200  # Number of frames to keep in buffer
        range_profile_buffer = []
        
        # Create figure
        plt.ion()  # Interactive mode
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('AWR1642 Range-Time Analysis (Live)', fontsize=14)
        
        # Initialize plots
        range_axis = np.arange(num_range_bins) * range_res
        
        print(f"\n✓ Starting data collection ({duration} seconds)")
        print("  Stand 0.5-1.5m from radar for vital signs detection")
        print("  Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration:
            # Read frame
            data_ok, frame_num, det_obj, tlv_data = radar.read_tlv_packet(timeout=0.5)
            
            if data_ok and 'range_profile' in tlv_data:
                frame_count += 1
                
                # Range profile is already in dB from the driver
                range_profile = tlv_data['range_profile']
                range_profile_buffer.append(range_profile)
                
                # Keep buffer size limited
                if len(range_profile_buffer) > buffer_size:
                    range_profile_buffer.pop(0)
                
                # Update plots every 5 frames
                if frame_count % 5 == 0 and len(range_profile_buffer) > 10:
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()
                    
                    # 1. Current range profile
                    ax1.plot(range_axis[:len(range_profile)], range_profile, 'b-', linewidth=1)
                    ax1.set_xlabel('Range (m)')
                    ax1.set_ylabel('Magnitude (dB)')
                    ax1.set_title(f'Current Range Profile (Frame {frame_count})')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlim([0, range_axis[-1]])
                    
                    # 2. Range-Time heatmap
                    rp_array = np.array(range_profile_buffer).T  # (range_bins, time)
                    num_bins, num_time = rp_array.shape
                    time_axis = np.arange(num_time) / 20.0  # Assuming 20 FPS
                    
                    im2 = ax2.imshow(rp_array, aspect='auto', cmap='viridis', origin='lower',
                                    extent=[0, time_axis[-1], 0, range_axis[num_bins-1]],
                                    vmin=np.percentile(rp_array, 5), 
                                    vmax=np.percentile(rp_array, 95))
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('Range (m)')
                    ax2.set_title('Range-Time Intensity Map (dB)')
                    
                    # 3. Time series at specific ranges (for vital signs)
                    target_ranges = [0.5, 0.75, 1.0, 1.25]  # meters
                    for target_m in target_ranges:
                        bin_idx = int(target_m / range_res)
                        if 0 <= bin_idx < num_bins:
                            # Subtract mean to see variations (vital signs)
                            signal_at_range = rp_array[bin_idx, :] - np.mean(rp_array[bin_idx, :])
                            ax3.plot(time_axis, signal_at_range, label=f'{target_m:.2f} m')
                    
                    ax3.set_xlabel('Time (s)')
                    ax3.set_ylabel('Magnitude Variation (dB)')
                    ax3.set_title('Signal Variation at Target Ranges (Breathing/Heartbeat)')
                    ax3.legend(loc='upper right')
                    ax3.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.pause(0.01)
                
                # Print progress
                if frame_count % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frames: {frame_count:4d} | Time: {elapsed:5.1f}s | FPS: {fps:.1f} | Buffer: {len(range_profile_buffer):3d}")
        
        print(f"\n✓ Collection complete: {frame_count} frames captured")
        
        # Keep plot open
        plt.ioff()
        plt.show()
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        radar.disconnect()
        print("✓ Radar disconnected")


def capture_and_analyze(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1', num_frames=200):
    """
    Capture data and create comprehensive analysis plots.
    """
    print("="*70)
    print("AWR1642 Range-Time Analysis")
    print("="*70)
    
    # Setup radar
    radar = AWR1642(cli_port=cli_port, data_port=data_port)
    
    try:
        radar.connect()
        
        # Load and send configuration
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        config_cmds = load_config_from_file(config_file)
        radar.configure_sensor(config_cmds)
        
        # Get parameters
        range_res = radar.config_params.get('rangeResolutionMeters', 0.044)
        num_range_bins = radar.config_params.get('numRangeBins', 256)
        frame_rate = 20  # Approximate
        
        print(f"\nCapturing {num_frames} frames...")
        print("Stand 0.5-1.5m from radar for vital signs analysis\n")
        
        # Capture frames
        range_profiles = []
        detected_objects = []
        
        start_time = time.time()
        for i in range(num_frames):
            data_ok, frame_num, det_obj, tlv_data = radar.read_tlv_packet(timeout=1.0)
            
            if data_ok:
                if 'range_profile' in tlv_data:
                    range_profiles.append(tlv_data['range_profile'])
                if det_obj and det_obj.get('numObj', 0) > 0:
                    detected_objects.append(det_obj)
                
                if (i+1) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Captured {i+1}/{num_frames} frames ({elapsed:.1f}s)")
        
        elapsed = time.time() - start_time
        print(f"✓ Captured {len(range_profiles)} frames in {elapsed:.1f}s ({len(range_profiles)/elapsed:.1f} FPS)")
        
        if len(range_profiles) < 20:
            print("✗ Not enough data for analysis")
            return
        
        # Convert to array
        rp_array = np.array(range_profiles).T  # (range_bins, time)
        num_bins, num_time = rp_array.shape
        range_axis = np.arange(num_bins) * range_res
        time_axis = np.arange(num_time) / frame_rate
        
        # Create comprehensive plots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('AWR1642 Range-Time Analysis', fontsize=14)
        
        # 1. Range-Time heatmap
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(rp_array, aspect='auto', cmap='viridis', origin='lower',
                        extent=[0, time_axis[-1], 0, range_axis[-1]],
                        vmin=np.percentile(rp_array, 5), vmax=np.percentile(rp_array, 95))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Range (m)')
        ax1.set_title('Range-Time Intensity (dB)')
        plt.colorbar(im1, ax=ax1, label='dB')
        
        # 2. Average range profile
        ax2 = plt.subplot(2, 3, 2)
        avg_profile = np.mean(rp_array, axis=1)
        std_profile = np.std(rp_array, axis=1)
        ax2.plot(range_axis, avg_profile, 'b-', label='Mean')
        ax2.fill_between(range_axis, avg_profile - std_profile, avg_profile + std_profile, 
                        alpha=0.3, label='±1 std')
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Average Range Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Find range bin with most variation (likely target)
        ax3 = plt.subplot(2, 3, 3)
        variation_per_bin = np.std(rp_array, axis=1)
        
        # Focus on typical vital signs range (0.3 - 2.0m)
        min_bin = int(0.3 / range_res)
        max_bin = int(2.0 / range_res)
        variation_masked = variation_per_bin.copy()
        variation_masked[:min_bin] = 0
        variation_masked[max_bin:] = 0
        
        peak_bin = np.argmax(variation_masked)
        peak_range = peak_bin * range_res
        
        ax3.plot(range_axis, variation_per_bin, 'b-')
        ax3.axvline(x=peak_range, color='r', linestyle='--', label=f'Peak at {peak_range:.2f}m')
        ax3.axvspan(0.3, 2.0, alpha=0.2, color='green', label='Target zone')
        ax3.set_xlabel('Range (m)')
        ax3.set_ylabel('Temporal Variation (dB)')
        ax3.set_title('Variation per Range Bin')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Time series at peak range (for vital signs)
        ax4 = plt.subplot(2, 3, 4)
        signal_at_peak = rp_array[peak_bin, :] - np.mean(rp_array[peak_bin, :])
        ax4.plot(time_axis, signal_at_peak, 'b-', linewidth=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnitude Variation (dB)')
        ax4.set_title(f'Signal at {peak_range:.2f}m (DC removed)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Spectrogram of signal at peak range
        ax5 = plt.subplot(2, 3, 5)
        nperseg = min(64, len(signal_at_peak) // 4)
        if nperseg > 8:
            f, t, Sxx = signal.spectrogram(signal_at_peak, fs=frame_rate, 
                                           nperseg=nperseg, noverlap=nperseg//2)
            im5 = ax5.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            ax5.set_ylabel('Frequency (Hz)')
            ax5.set_xlabel('Time (s)')
            ax5.set_title(f'Spectrogram at {peak_range:.2f}m')
            
            # Mark typical vital signs frequencies
            ax5.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='Breathing (~0.2Hz)')
            ax5.axhline(y=1.2, color='r', linestyle='--', alpha=0.5, label='Heart (~1.2Hz)')
            ax5.legend(loc='upper right', fontsize=8)
            plt.colorbar(im5, ax=ax5, label='Power (dB)')
        else:
            ax5.text(0.5, 0.5, 'Not enough data for spectrogram', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # 6. FFT of signal at peak range
        ax6 = plt.subplot(2, 3, 6)
        # Apply window and compute FFT
        windowed_signal = signal_at_peak * np.hanning(len(signal_at_peak))
        fft_result = np.abs(np.fft.rfft(windowed_signal))
        fft_freq = np.fft.rfftfreq(len(signal_at_peak), d=1/frame_rate)
        
        # Convert to BPM for vital signs
        fft_bpm = fft_freq * 60
        
        ax6.plot(fft_bpm, 20*np.log10(fft_result + 1e-10), 'b-')
        ax6.axvline(x=12, color='g', linestyle='--', alpha=0.5, label='Breathing (12 BPM)')
        ax6.axvline(x=72, color='r', linestyle='--', alpha=0.5, label='Heart (72 BPM)')
        ax6.set_xlabel('Rate (BPM)')
        ax6.set_ylabel('Magnitude (dB)')
        ax6.set_title(f'Frequency Spectrum at {peak_range:.2f}m')
        ax6.set_xlim([0, 150])
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'range_time_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Analysis plot saved as '{output_file}'")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        radar.disconnect()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AWR1642 Range-Time Visualization')
    parser.add_argument('--cli', default='/dev/ttyACM0', help='CLI port')
    parser.add_argument('--data', default='/dev/ttyACM1', help='Data port')
    parser.add_argument('--mode', choices=['live', 'analyze'], default='live',
                       help='Visualization mode: live (real-time) or analyze (capture then plot)')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration in seconds (for live mode)')
    parser.add_argument('--frames', type=int, default=200,
                       help='Number of frames to capture (for analyze mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        plot_range_time_live(args.cli, args.data, args.duration)
    else:
        capture_and_analyze(args.cli, args.data, args.frames)


if __name__ == '__main__':
    main()
