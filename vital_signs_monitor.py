#!/usr/bin/env python3
"""
VitalFlow-Radar: Real-Time Vital Signs Monitor
===============================================

This script performs real-time monitoring of vital signs (heart rate and
breathing rate) using the AWR1642 mmWave radar sensor.

Usage:
    python vital_signs_monitor.py --duration 60
    python vital_signs_monitor.py --live --plot
    python vital_signs_monitor.py --analyze --frames 300

Author: VitalFlow-Radar Project
Date: December 2024
"""

import argparse
import sys
import time
import numpy as np
from collections import deque

# Try to import PyQtGraph for fast real-time plotting
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets, QtCore
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

try:
    from awr1642_driver import AWR1642, DEFAULT_CONFIG_FILE, load_config_from_file
    from vital_signs_processor import VitalSignsProcessor, VitalSignsWavelet
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the VitalFlow-Radar directory")
    sys.exit(1)


class VitalSignsMonitor:
    """
    Real-time vital signs monitoring using AWR1642 radar.
    
    This class provides a high-level interface for:
    - Continuous radar data acquisition
    - Real-time vital signs processing
    - Live visualization
    - Results logging
    """
    
    def __init__(self, config_file=None, fps=10.0):
        """
        Initialize the vital signs monitor.
        
        Parameters
        ----------
        config_file : str or None
            Path to radar configuration file. If None, uses default.
        fps : float
            Expected frame rate (used for signal processing)
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.fps = fps
        
        # Initialize radar driver
        self.radar = AWR1642()
        
        # Initialize signal processor
        self.processor = VitalSignsProcessor(
            fps=fps,
            hr_limits=(45, 150),
            br_limits=(8, 30),
            range_min_m=0.3,
            range_max_m=1.5
        )
        
        # Data buffers
        self.phase_history = deque(maxlen=int(60 * fps))  # 60 seconds
        self.range_profile_history = deque(maxlen=int(30 * fps))
        
        # Results history
        self.hr_history = deque(maxlen=100)
        self.br_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
        # State
        self.is_running = False
        self.selected_bin = None
        self.range_resolution = 0.044  # Default, updated from config
        
    def connect(self):
        """Connect to radar and configure sensor."""
        print("=" * 60)
        print("VitalFlow-Radar Vital Signs Monitor")
        print("=" * 60)
        
        self.radar.connect()
        
        # Load and send configuration
        print(f"Loading configuration: {self.config_file}")
        try:
            config_commands = load_config_from_file(self.config_file)
            self.radar.configure_sensor(config_commands)
            
            # Update parameters from config
            self.range_resolution = self.radar.config_params.get('rangeIdxToMeters', 0.044)
            actual_fps = 1000 / self.radar.config_params.get('framePeriodicity', 100)
            if abs(actual_fps - self.fps) > 1:
                print(f"Note: Adjusting FPS from {self.fps} to {actual_fps}")
                self.fps = actual_fps
                self.processor.fps = actual_fps
                self.processor.fs = actual_fps
                
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration")
    
    def disconnect(self):
        """Disconnect from radar."""
        self.is_running = False
        if self.radar:
            self.radar.disconnect()
    
    def _select_range_bin(self, range_profiles, num_frames=20):
        """
        Select optimal range bin based on signal variance.
        
        Uses variance-based selection similar to MATLAB implementation:
        1. Compute variance of magnitude over time for each bin
        2. Restrict to expected human distance (0.3-1.5m)
        3. Apply power threshold to reject noise bins
        4. Select bin with maximum variance
        
        Parameters
        ----------
        range_profiles : list or ndarray
            Array of range profiles (frames x bins), can be dB or linear
        num_frames : int
            Number of frames to use for selection
            
        Returns
        -------
        int
            Selected range bin index
        """
        profiles_list = list(range_profiles)
        if len(profiles_list) < num_frames:
            # Not enough data, return center of expected range
            expected_bin = int(0.7 / self.range_resolution)
            if len(profiles_list) > 0:
                return min(expected_bin, len(profiles_list[0]) - 1)
            return expected_bin
        
        # Use last num_frames profiles
        profiles = np.array(profiles_list[-num_frames:])
        
        # Handle both dB and linear values
        # If values are mostly negative or around typical dB range, convert to linear
        mean_val = np.mean(profiles)
        if mean_val < 50:  # Likely dB values
            # Convert from dB to linear for variance calculation
            magnitudes = 10 ** (profiles / 20.0)
        else:
            magnitudes = np.abs(profiles)
        
        # Compute variance per bin
        var_per_bin = np.var(magnitudes, axis=0)
        
        # Create distance mask
        num_bins = var_per_bin.shape[0]
        distances = np.arange(num_bins) * self.range_resolution
        valid_mask = (distances >= self.processor.range_min_m) & \
                     (distances <= self.processor.range_max_m)
        
        # Apply power threshold (in linear domain)
        mean_power = np.mean(magnitudes, axis=0)
        max_power = np.max(mean_power)
        if max_power > 0:
            power_thresh = max_power * 0.01  # -40dB equivalent
            power_mask = mean_power > power_thresh
        else:
            power_mask = np.ones(num_bins, dtype=bool)
        
        # Combine masks
        combined_mask = valid_mask & power_mask
        
        if not np.any(combined_mask):
            # Fallback to max variance in valid range
            var_masked = np.where(valid_mask, var_per_bin, 0)
        else:
            var_masked = np.where(combined_mask, var_per_bin, 0)
        
        selected_bin = np.argmax(var_masked)
        
        return selected_bin
    
    def monitor_live(self, duration=60, plot=False, verbose=True):
        """
        Run live vital signs monitoring.
        
        Uses amplitude variations in range profile as proxy for displacement.
        The AWR1642 out-of-box demo provides magnitude (dB) range profiles,
        not complex I/Q data, so we use amplitude-based analysis.
        
        Parameters
        ----------
        duration : float
            Monitoring duration in seconds
        plot : bool
            Show live plot
        verbose : bool
            Print updates to console
            
        Returns
        -------
        dict
            Final results with averages and history
        """
        self.is_running = True
        start_time = time.time()
        frame_count = 0
        last_update_time = start_time
        update_interval = 3.0  # Update estimates every 3 seconds
        
        # Initialize PyQtGraph plot if requested
        app = None
        win = None
        plots = {}
        
        if plot:
            if not PYQTGRAPH_AVAILABLE:
                print("⚠ PyQtGraph not available, install with: pip install pyqtgraph PyQt5")
                print("  Continuing without plotting...\n")
                plot = False
            else:
                # Initialize Qt application
                app = QtWidgets.QApplication.instance()
                if app is None:
                    app = QtWidgets.QApplication([])
                
                # Create window with dark theme
                pg.setConfigOptions(antialias=True)
                win = pg.GraphicsLayoutWidget(title="VitalFlow-Radar: Real-Time Vital Signs")
                win.resize(1200, 800)
                win.show()
                
                # Plot 1: Vital signs over time
                p1 = win.addPlot(row=0, col=0, title="Vital Signs Over Time")
                p1.setLabel('left', 'Rate (BPM)')
                p1.setLabel('bottom', 'Time (s)')
                p1.addLegend()
                p1.setYRange(0, 150)
                plots['hr_line'] = p1.plot([], [], pen=pg.mkPen('r', width=2), name='Heart Rate')
                plots['br_line'] = p1.plot([], [], pen=pg.mkPen('b', width=2), name='Breathing Rate')
                
                # Plot 2: Filtered signals  
                p2 = win.addPlot(row=1, col=0, title="Filtered Signals")
                p2.setLabel('left', 'Amplitude')
                p2.setLabel('bottom', 'Time (s)')
                p2.addLegend()
                plots['hr_sig'] = p2.plot([], [], pen=pg.mkPen('r', width=1), name='Heart Signal')
                plots['br_sig'] = p2.plot([], [], pen=pg.mkPen('b', width=1), name='Breathing Signal')
                
                # Plot 3: Current readings as text
                p3 = win.addPlot(row=2, col=0, title="Current Readings")
                p3.hideAxis('left')
                p3.hideAxis('bottom')
                plots['text'] = pg.TextItem(text="HR: -- BPM | BR: -- BPM", color='w', anchor=(0.5, 0.5))
                plots['text'].setFont(pg.QtGui.QFont('Arial', 24, pg.QtGui.QFont.Weight.Bold))
                p3.addItem(plots['text'])
                plots['text'].setPos(0.5, 0.5)
                p3.setXRange(0, 1)
                p3.setYRange(0, 1)
                
                plots['p1'] = p1
                plots['p2'] = p2
            
        print(f"\nMonitoring vital signs for {duration} seconds...")
        print("Stand 0.5-1.5m from radar. Remain still for accurate readings.\n")
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Process Qt events FIRST to keep plot responsive
                if app is not None:
                    app.processEvents()
                
                # Read frame using TLV packet reader (short timeout)
                data_ok, frame_num, det_obj, tlv_data = self.radar.read_tlv_packet(timeout=0.05)
                
                if not data_ok:
                    continue  # Don't sleep, just try again
                
                frame_count += 1
                
                # Get range profile from TLV data
                range_profile = tlv_data.get('range_profile')
                if range_profile is None:
                    continue
                
                self.range_profile_history.append(range_profile)
                
                # Update range bin selection periodically
                if frame_count % 50 == 1 or self.selected_bin is None:
                    self.selected_bin = self._select_range_bin(
                        self.range_profile_history, num_frames=20
                    )
                    if verbose and frame_count == 1:
                        dist = self.selected_bin * self.range_resolution
                        print(f"Selected range bin: {self.selected_bin} ({dist:.2f}m)")
                
                # Extract amplitude from selected bin
                # For vital signs, amplitude variations track chest displacement
                if self.selected_bin < len(range_profile):
                    # Use amplitude (dB or linear) as displacement proxy
                    amplitude = range_profile[self.selected_bin]
                    self.phase_history.append(amplitude)  # Reuse phase buffer for amplitude
                
                # Process vital signs periodically
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    if len(self.phase_history) >= int(10 * self.fps):
                        # Process Qt events before heavy computation
                        if app is not None:
                            app.processEvents()
                        
                        # Use amplitude signal (variations indicate displacement)
                        amplitude_signal = np.array(list(self.phase_history))
                        
                        # Normalize to zero-mean (removes DC offset)
                        amplitude_signal = amplitude_signal - np.mean(amplitude_signal)
                        
                        result = self.processor.extract_vital_signs(amplitude_signal)
                        
                        # Process Qt events after heavy computation
                        if app is not None:
                            app.processEvents()
                        
                        hr = result['hr_bpm']
                        br = result['br_bpm']
                        hr_conf = result['hr_confidence']
                        br_conf = result['br_confidence']
                        
                        if not np.isnan(hr) and not np.isnan(br):
                            self.hr_history.append(hr)
                            self.br_history.append(br)
                            self.time_history.append(current_time - start_time)
                            
                            if verbose:
                                print(f"[{current_time - start_time:6.1f}s] "
                                      f"HR: {hr:5.1f} BPM (conf: {hr_conf:.2f}) | "
                                      f"BR: {br:5.1f} BPM (conf: {br_conf:.2f})")
                            
                            # Update PyQtGraph plot
                            if plot and plots:
                                self._update_pyqtgraph_plot(plots, result)
                    
                    last_update_time = current_time
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            self.is_running = False
            if win is not None:
                win.close()
        
        # Compute final results
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        hr_avg = np.mean(list(self.hr_history)) if self.hr_history else np.nan
        br_avg = np.mean(list(self.br_history)) if self.br_history else np.nan
        hr_std = np.std(list(self.hr_history)) if len(self.hr_history) > 1 else np.nan
        br_std = np.std(list(self.br_history)) if len(self.br_history) > 1 else np.nan
        
        print(f"\n{'='*60}")
        print("Monitoring Complete")
        print(f"{'='*60}")
        print(f"Duration: {elapsed:.1f}s | Frames: {frame_count} | FPS: {actual_fps:.1f}")
        print(f"\nVital Signs Summary:")
        print(f"  Heart Rate:     {hr_avg:.1f} ± {hr_std:.1f} BPM")
        print(f"  Breathing Rate: {br_avg:.1f} ± {br_std:.1f} BPM")
        
        return {
            'hr_avg': hr_avg,
            'hr_std': hr_std,
            'br_avg': br_avg,
            'br_std': br_std,
            'hr_history': list(self.hr_history),
            'br_history': list(self.br_history),
            'time_history': list(self.time_history),
            'elapsed': elapsed,
            'frames': frame_count,
            'fps': actual_fps
        }
    
    def _update_pyqtgraph_plot(self, plots, result):
        """Update PyQtGraph plot - very fast real-time updates."""
        try:
            # Update HR/BR time series (small data, fast)
            if len(self.hr_history) > 0:
                times = np.array(list(self.time_history))
                hr_vals = np.array(list(self.hr_history))
                br_vals = np.array(list(self.br_history))
                
                plots['hr_line'].setData(times, hr_vals)
                plots['br_line'].setData(times, br_vals)
                
                # Auto-scale x-axis only when needed
                if len(times) > 1 and times[-1] > plots['p1'].viewRange()[0][1] - 5:
                    plots['p1'].setXRange(0, times[-1] + 10, padding=0)
            
            # Update filtered signals (downsample if too many points)
            if 'phase_hr' in result and 'phase_br' in result:
                hr_sig = result['phase_hr']
                br_sig = result['phase_br']
                n = len(hr_sig)
                
                # Downsample to max 500 points for smooth rendering
                if n > 500:
                    step = n // 500
                    hr_sig = hr_sig[::step]
                    br_sig = br_sig[::step]
                    t = np.arange(len(hr_sig)) * step / self.fps
                else:
                    t = np.arange(n) / self.fps
                
                plots['hr_sig'].setData(t, hr_sig)
                plots['br_sig'].setData(t, br_sig)
                
                if len(t) > 1:
                    plots['p2'].setXRange(0, t[-1], padding=0)
            
            # Update text display
            if len(self.hr_history) > 0:
                hr_current = self.hr_history[-1]
                br_current = self.br_history[-1]
                plots['text'].setText(f"HR: {hr_current:.0f} BPM  |  BR: {br_current:.0f} BPM")
                
        except Exception as e:
            pass  # Ignore plot errors to keep monitoring running
    
    def capture_and_analyze(self, num_frames=200, save_plot=True):
        """
        Capture data and perform detailed analysis.
        
        Parameters
        ----------
        num_frames : int
            Number of frames to capture
        save_plot : bool
            Save analysis plot to file
            
        Returns
        -------
        dict
            Detailed analysis results
        """
        print(f"\nCapturing {num_frames} frames for analysis...")
        print("Stand 0.5-1.5m from radar. Remain still.\n")
        
        frames = []
        phase_data = []
        range_profiles = []
        
        start_time = time.time()
        frame_count = 0
        
        while frame_count < num_frames:
            data_ok, frame_num, det_obj, tlv_data = self.radar.read_tlv_packet(timeout=0.5)
            
            if data_ok:
                range_profile = tlv_data.get('range_profile')
                if range_profile is not None:
                    range_profiles.append(range_profile)
                    frame_count += 1
                    
                    if frame_count % 50 == 0:
                        print(f"  Captured {frame_count}/{num_frames} frames...")
            
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        print(f"✓ Captured {frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} FPS)")
        
        # Select best range bin
        range_profiles = np.array(range_profiles)
        self.selected_bin = self._select_range_bin(range_profiles, num_frames=min(50, num_frames))
        dist = self.selected_bin * self.range_resolution
        print(f"Selected range bin: {self.selected_bin} ({dist:.2f}m)")
        
        # Extract amplitude signal (not phase, since we have magnitude-only data)
        # Amplitude variations track chest displacement for vital signs
        amplitude_signal = range_profiles[:, self.selected_bin]
        
        # Normalize to zero-mean (removes DC offset)
        amplitude_signal = amplitude_signal - np.mean(amplitude_signal)
        
        # Process vital signs
        print("\nProcessing vital signs...")
        result = self.processor.extract_vital_signs(amplitude_signal)
        
        print(f"\n{'='*60}")
        print("Analysis Results")
        print(f"{'='*60}")
        print(f"Heart Rate:     {result['hr_bpm']:.1f} BPM (confidence: {result['hr_confidence']:.2f})")
        print(f"Breathing Rate: {result['br_bpm']:.1f} BPM (confidence: {result['br_confidence']:.2f})")
        
        if save_plot:
            self._save_analysis_plot(range_profiles, amplitude_signal, result, actual_fps)
        
        return result
    
    def _save_analysis_plot(self, range_profiles, amplitude_signal, result, fps):
        """Generate and save comprehensive analysis plot."""
        fig = plt.figure(figsize=(14, 12))
        
        # Time axis
        t = np.arange(len(amplitude_signal)) / fps
        
        # 1. Range-Time heatmap
        ax1 = fig.add_subplot(3, 2, 1)
        extent = [0, t[-1], 0, range_profiles.shape[1] * self.range_resolution]
        # Handle dB or linear values for display
        display_data = range_profiles.T
        if np.mean(display_data) < 50:  # Likely dB values
            ax1.imshow(display_data, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis')
        else:
            ax1.imshow(20*np.log10(np.abs(display_data) + 1e-6), aspect='auto', origin='lower',
                      extent=extent, cmap='viridis')
        ax1.axhline(y=self.selected_bin * self.range_resolution, color='r', 
                   linestyle='--', label=f'Selected bin')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Range (m)')
        ax1.set_title('Range Profile Over Time')
        ax1.legend()
        
        # 2. Raw amplitude signal
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(t, amplitude_signal, 'b-', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Amplitude Signal (Selected Bin)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Preprocessed phase
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(t, result['phase_hp'], 'k-', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Preprocessed Phase (High-pass Filtered)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Separated signals
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(t, result['phase_br'], 'b-', label='Breathing Band', alpha=0.8)
        ax4.plot(t, result['phase_hr'], 'r-', label='Heart Rate Band', alpha=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Separated Vital Signs Signals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Time-varying estimates
        ax5 = fig.add_subplot(3, 2, 5)
        if len(result['hr_time_axis']) > 0:
            ax5.plot(result['hr_time_axis'], result['hr_time_series'], 'r.-', 
                    label=f'HR ({result["hr_bpm"]:.1f} BPM)', linewidth=2)
        if len(result['br_time_axis']) > 0:
            ax5.plot(result['br_time_axis'], result['br_time_series'], 'b.-', 
                    label=f'BR ({result["br_bpm"]:.1f} BPM)', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Rate (BPM)')
        ax5.set_title('Time-Varying Vital Signs Estimates')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Power spectrum
        ax6 = fig.add_subplot(3, 2, 6)
        from scipy import signal as sig
        f_br, Pxx_br = sig.welch(result['phase_br'], fs=fps, nperseg=min(256, len(result['phase_br'])))
        f_hr, Pxx_hr = sig.welch(result['phase_hr'], fs=fps, nperseg=min(256, len(result['phase_hr'])))
        ax6.semilogy(f_br * 60, Pxx_br, 'b-', label='Breathing Band', alpha=0.8)
        ax6.semilogy(f_hr * 60, Pxx_hr, 'r-', label='Heart Rate Band', alpha=0.8)
        ax6.axvline(x=result['br_bpm'], color='b', linestyle='--', alpha=0.5)
        ax6.axvline(x=result['hr_bpm'], color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Frequency (BPM)')
        ax6.set_ylabel('Power Spectral Density')
        ax6.set_title('Power Spectrum Analysis')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 180)
        
        plt.tight_layout()
        
        filename = 'vital_signs_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✓ Analysis plot saved as '{filename}'")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="VitalFlow-Radar: Real-Time Vital Signs Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vital_signs_monitor.py --live --duration 60
  python vital_signs_monitor.py --live --plot --duration 120
  python vital_signs_monitor.py --analyze --frames 300
  python vital_signs_monitor.py --config vital_signs_awr1642.cfg --live
        """
    )
    
    parser.add_argument('--live', action='store_true',
                       help='Run live monitoring mode')
    parser.add_argument('--analyze', action='store_true',
                       help='Capture and analyze mode')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in seconds (default: 60)')
    parser.add_argument('--frames', type=int, default=200,
                       help='Number of frames for analyze mode (default: 200)')
    parser.add_argument('--plot', action='store_true',
                       help='Show live plot during monitoring')
    parser.add_argument('--config', type=str, default=None,
                       help='Radar configuration file')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Expected frame rate (default: 10)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce console output')
    
    args = parser.parse_args()
    
    if not args.live and not args.analyze:
        parser.print_help()
        print("\nError: Please specify --live or --analyze mode")
        sys.exit(1)
    
    # Create monitor
    monitor = VitalSignsMonitor(config_file=args.config, fps=args.fps)
    
    try:
        # Connect to radar
        monitor.connect()
        
        if args.live:
            # Live monitoring mode
            monitor.monitor_live(
                duration=args.duration,
                plot=args.plot,
                verbose=not args.quiet
            )
        elif args.analyze:
            # Capture and analyze mode
            monitor.capture_and_analyze(
                num_frames=args.frames,
                save_plot=True
            )
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.disconnect()


if __name__ == "__main__":
    main()
