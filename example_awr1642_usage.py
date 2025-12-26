#!/usr/bin/env python3
"""
Example script demonstrating AWR1642 radar driver usage.

This script shows how to:
1. Connect to the AWR1642 radar
2. Configure it for object detection
3. Capture TLV-formatted data
4. Process and visualize detected objects

Usage:
    python example_awr1642_usage.py [--cli /dev/ttyACM0] [--data /dev/ttyACM1]
"""

from awr1642_driver import AWR1642, load_config_from_file, DEFAULT_CONFIG_FILE
import numpy as np
import time
import sys
import os


def simple_test(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1'):
    """Simple test: Capture and display basic statistics."""
    print("=" * 70)
    print("AWR1642 Simple Test")
    print("=" * 70)
    
    NUM_FRAMES = 40  # Capture 2 seconds at 20 FPS
    
    print(f"\nConfiguration:")
    print(f"  CLI Port: {cli_port}")
    print(f"  Data Port: {data_port}")
    print(f"  Frames to capture: {NUM_FRAMES}")
    
    try:
        # Connect and configure
        print("\n[1/4] Connecting to radar...")
        radar = AWR1642(cli_port=cli_port, data_port=data_port)
        radar.connect()
        
        print("\n[2/4] Configuring sensor...")
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        config_cmds = load_config_from_file(config_file)
        radar.configure_sensor(config_cmds)
        
        # Capture data
        print(f"\n[3/4] Capturing {NUM_FRAMES} frames...")
        frames = radar.capture_frames(num_frames=NUM_FRAMES, return_raw=True, verbose=True)
        
        # Disconnect
        radar.disconnect()
        
        # Display statistics
        print("\n" + "=" * 70)
        print("Results:")
        print("=" * 70)
        print(f"Frames captured: {len(frames)}")
        
        if frames:
            # Analyze detected objects across frames
            total_objects = 0
            frames_with_objects = 0
            ranges = []
            
            for frame in frames:
                if frame['detected_objects']:
                    num_obj = frame['detected_objects']['numObj']
                    if num_obj > 0:
                        total_objects += num_obj
                        frames_with_objects += 1
                        ranges.extend(frame['detected_objects']['range'].tolist())
            
            print(f"\nDetection Statistics:")
            print(f"  - Frames with objects: {frames_with_objects}/{len(frames)}")
            print(f"  - Total objects detected: {total_objects}")
            print(f"  - Average objects/frame: {total_objects/len(frames):.2f}")
            
            if ranges:
                print(f"  - Range min: {np.min(ranges):.3f} m")
                print(f"  - Range max: {np.max(ranges):.3f} m")
                print(f"  - Range mean: {np.mean(ranges):.3f} m")
            
            # Sample frame data
            print(f"\nSample Frame (#{frames[0]['frame_number']}):")
            det_obj = frames[0]['detected_objects']
            if det_obj and det_obj['numObj'] > 0:
                print(f"  Objects: {det_obj['numObj']}")
                for i in range(min(5, det_obj['numObj'])):
                    print(f"    [{i}] Range: {det_obj['range'][i]:.3f} m, "
                          f"X: {det_obj['x'][i]:.3f} m, Y: {det_obj['y'][i]:.3f} m")
            else:
                print("  No objects detected")
            
            # Check for range profiles
            if 'range_profile' in frames[0]['tlv_data']:
                profile = frames[0]['tlv_data']['range_profile']
                print(f"\nRange Profile:")
                print(f"  - Length: {len(profile)} bins")
                print(f"  - Max value: {np.max(profile)}")
                print(f"  - Mean value: {np.mean(profile):.2f}")
        
        print("\n✓ Test completed successfully!")
        
        return frames
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check device ports: ls /dev/ttyACM*")
        print("  2. Verify permissions: sudo usermod -a -G dialout $USER")
        print("  3. Ensure AWR1642 is powered and connected")
        print("  4. Try different ports (AWR1642 usually uses /dev/ttyACM*)")
        import traceback
        traceback.print_exc()
        return None


def continuous_monitoring(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1', duration_seconds=10):
    """
    Continuously monitor and display detected objects.
    
    Parameters
    ----------
    cli_port : str
        CLI port path
    data_port : str
        Data port path
    duration_seconds : int
        How long to monitor (seconds)
    """
    print("=" * 70)
    print(f"AWR1642 Continuous Monitoring ({duration_seconds}s)")
    print("=" * 70)
    
    try:
        # Setup
        radar = AWR1642(cli_port=cli_port, data_port=data_port)
        radar.connect()
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        config_cmds = load_config_from_file(config_file)
        radar.configure_sensor(config_cmds, verbose=False)
        
        print(f"\nMonitoring started (press Ctrl+C to stop)...")
        print(f"{'Frame':<8} {'Objects':<10} {'Closest Range (m)':<20}")
        print("-" * 70)
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            data_ok, frame_number, detected_objects, tlv_data = radar.read_tlv_packet(timeout=0.1)
            
            if data_ok and detected_objects:
                num_obj = detected_objects['numObj']
                if num_obj > 0:
                    closest_range = np.min(detected_objects['range'])
                    print(f"{frame_number:<8} {num_obj:<10} {closest_range:<20.3f}")
                else:
                    print(f"{frame_number:<8} {num_obj:<10} {'N/A':<20}")
                
                frame_count += 1
            
            time.sleep(0.01)
        
        radar.disconnect()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Monitoring complete: {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.1f} FPS)")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        try:
            radar.disconnect()
        except:
            pass
    
    except Exception as e:
        print(f"\n✗ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()


def visualize_range_time(cli_port='/dev/ttyACM0', data_port='/dev/ttyACM1', num_frames=100):
    """
    Capture frames and create a range-time intensity plot.
    Range profile values are in dB scale.
    """
    print("=" * 70)
    print(f"AWR1642 Range-Time Visualization")
    print("=" * 70)
    
    try:
        import matplotlib.pyplot as plt
        
        radar = AWR1642(cli_port=cli_port, data_port=data_port)
        radar.connect()
        config_file = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        config_cmds = load_config_from_file(config_file)
        radar.configure_sensor(config_cmds)
        
        # Get range resolution for axis labels
        range_res = radar.config_params.get('rangeResolutionMeters', 0.044)
        
        print(f"\nCapturing {num_frames} frames...")
        range_data = radar.capture_frames(num_frames=num_frames, return_raw=False, verbose=True)
        
        radar.disconnect()
        
        if range_data is not None:
            num_range_bins, actual_frames = range_data.shape
            
            # Calculate range axis in meters
            range_axis = np.arange(num_range_bins) * range_res
            frame_rate = 20  # Approximate frame rate
            time_axis = np.arange(actual_frames) / frame_rate
            
            # Create range-time plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Range-time heatmap (already in dB)
            im1 = ax1.imshow(range_data, aspect='auto', cmap='viridis', origin='lower',
                            extent=[time_axis[0], time_axis[-1], range_axis[0], range_axis[-1]],
                            vmin=np.percentile(range_data, 5), vmax=np.percentile(range_data, 95))
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Range (m)')
            ax1.set_title('Range-Time Intensity Map (AWR1642) - dB Scale')
            plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
            
            # Plot 2: Range profile over time (specific ranges)
            target_ranges_m = [0.5, 1.0, 1.5, 2.0]  # Ranges to monitor in meters
            for target_m in target_ranges_m:
                bin_idx = int(target_m / range_res)
                if 0 <= bin_idx < num_range_bins:
                    ax2.plot(time_axis, range_data[bin_idx, :], label=f'{target_m:.1f} m')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.set_title('Temporal Evolution at Different Ranges')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_file = 'awr1642_range_time.png'
            plt.savefig(output_file, dpi=150)
            print(f"✓ Visualization saved to: {output_file}")
            
            plt.show()
        
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\nAWR1642 Radar Driver - Example Usage")
    print("=" * 70)
    
    # Parse command-line arguments
    cli_port = '/dev/ttyACM0'
    data_port = '/dev/ttyACM1'
    
    if '--cli' in sys.argv:
        idx = sys.argv.index('--cli')
        if idx + 1 < len(sys.argv):
            cli_port = sys.argv[idx + 1]
    
    if '--data' in sys.argv:
        idx = sys.argv.index('--data')
        if idx + 1 < len(sys.argv):
            data_port = sys.argv[idx + 1]
    
    print(f"Port Configuration:")
    print(f"  CLI: {cli_port}")
    print(f"  Data: {data_port}")
    print("=" * 70)
    
    if '--monitor' in sys.argv:
        # Continuous monitoring mode
        duration = 10
        if '--duration' in sys.argv:
            idx = sys.argv.index('--duration')
            if idx + 1 < len(sys.argv):
                duration = int(sys.argv[idx + 1])
        
        continuous_monitoring(cli_port, data_port, duration_seconds=duration)
    
    elif '--viz' in sys.argv:
        # Visualization mode
        num_frames = 100
        if '--frames' in sys.argv:
            idx = sys.argv.index('--frames')
            if idx + 1 < len(sys.argv):
                num_frames = int(sys.argv[idx + 1])
        
        visualize_range_time(cli_port, data_port, num_frames=num_frames)
    
    else:
        # Simple test mode (default)
        print("\nAvailable modes:")
        print("  python example_awr1642_usage.py              # Simple test (default)")
        print("  python example_awr1642_usage.py --monitor    # Continuous monitoring")
        print("  python example_awr1642_usage.py --viz        # Range-time visualization")
        print("  Add --cli PORT and --data PORT to specify custom ports")
        print("=" * 70)
        
        simple_test(cli_port, data_port)
