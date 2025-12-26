"""
VitalFlow-Radar: Live Edge Producer with AWR1642 Integration
=============================================================

This module streams LIVE radar data from the AWR1642 sensor to Confluent Cloud/Kafka.
It integrates directly with the radar hardware for real-time vital signs streaming.

Usage:
    # Live radar mode (default)
    python edge_producer_live.py
    
    # With custom config
    python edge_producer_live.py --config vital_signs_awr1642.cfg
    
    # Simulation mode (no hardware)
    python edge_producer_live.py --simulate
"""

import os
import sys
import json
import time
import uuid
import signal
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import numpy as np

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

from confluent_kafka import Producer

# Local imports
from confluent_config import (
    get_producer_config,
    TOPICS,
    print_config_status,
)

# AWR1642 driver
try:
    from awr1642_driver import AWR1642, DEFAULT_CONFIG_FILE, load_config_from_file
    AWR1642_AVAILABLE = True
except ImportError:
    AWR1642_AVAILABLE = False
    print("⚠ AWR1642 driver not available")

# Vital signs processor
try:
    from vital_signs_processor import VitalSignsProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    print("⚠ VitalSignsProcessor not available")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PhaseMessage:
    """Schema-compliant phase/amplitude data message."""
    timestamp: float
    sequence: int
    phase: float           # Amplitude value (displacement proxy)
    range_bin: int
    range_m: float
    signal_quality: float
    device_id: str
    
    def to_json(self) -> Dict[str, Any]:
        return convert_numpy_types(asdict(self))


@dataclass 
class VitalSignsMessage:
    """Vital signs measurement message."""
    timestamp: float
    heart_rate_bpm: float
    heart_rate_confidence: float
    breathing_rate_bpm: float
    breathing_rate_confidence: float
    device_id: str
    window_frames: int
    
    def to_json(self) -> Dict[str, Any]:
        return convert_numpy_types(asdict(self))


# ============================================================================
# LIVE EDGE PRODUCER WITH AWR1642
# ============================================================================

class LiveEdgeProducer:
    """
    Live edge producer streaming from AWR1642 radar to Confluent/Kafka.
    
    This class:
    1. Connects to AWR1642 radar hardware
    2. Reads TLV packets in real-time
    3. Extracts phase/amplitude data from range profiles
    4. Streams to Kafka topics
    5. Periodically computes and streams vital signs
    """
    
    def __init__(
        self,
        config_file: str = None,
        fps: float = 10.0,
        device_id: str = None,
    ):
        """
        Initialize the live edge producer.
        
        Parameters
        ----------
        config_file : str
            Path to radar config file (default: vital_signs_awr1642.cfg)
        fps : float
            Expected frame rate
        device_id : str
            Device identifier (auto-generated if None)
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE if AWR1642_AVAILABLE else None
        self.fps = fps
        self.device_id = device_id or os.environ.get('DEVICE_ID', f"radar-{uuid.uuid4().hex[:8]}")
        
        # Radar components
        self.radar: Optional[AWR1642] = None
        self.processor: Optional[VitalSignsProcessor] = None
        self.range_resolution = 0.044  # Updated from config
        
        # Kafka producer
        self.producer: Optional[Producer] = None
        
        # Data buffers
        self.amplitude_history: deque = deque(maxlen=int(60 * fps))  # 60 sec
        self.range_profile_history: deque = deque(maxlen=int(30 * fps))
        
        # State
        self.selected_bin: Optional[int] = None
        self.is_running = False
        self.frame_count = 0
        self.messages_sent = 0
        
    def connect(self):
        """Connect to radar and Kafka."""
        print("=" * 60)
        print("VitalFlow-Radar Live Edge Producer")
        print("=" * 60)
        
        # Connect to Kafka
        print("\n📡 Connecting to Kafka...")
        print_config_status()
        
        config = get_producer_config(client_id=f'vitalflow-edge-{self.device_id}')
        self.producer = Producer(config)
        print("✓ Kafka producer initialized")
        
        # Connect to AWR1642 radar
        if not AWR1642_AVAILABLE:
            raise RuntimeError("AWR1642 driver not available. Cannot run in live mode.")
        
        print(f"\n📡 Connecting to AWR1642 radar...")
        self.radar = AWR1642()
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
                
        except Exception as e:
            print(f"⚠ Could not load config file: {e}")
            print("  Using default configuration")
        
        # Initialize vital signs processor
        if PROCESSOR_AVAILABLE:
            self.processor = VitalSignsProcessor(
                fps=self.fps,
                hr_limits=(45, 150),
                br_limits=(8, 30),
                range_min_m=0.3,
                range_max_m=1.5
            )
            print("✓ Vital signs processor initialized")
        
        print(f"\n✓ Connected to device: {self.device_id}")
        print(f"  Range resolution: {self.range_resolution:.4f} m")
        print(f"  Frame rate: {self.fps} FPS")
        
        return self
    
    def disconnect(self):
        """Disconnect from radar and flush Kafka."""
        self.is_running = False
        
        if self.radar:
            self.radar.disconnect()
            print("✓ Radar disconnected")
        
        if self.producer:
            remaining = self.producer.flush(timeout=10)
            if remaining > 0:
                print(f"⚠ {remaining} messages still pending")
            print("✓ Kafka producer flushed")
    
    def _select_range_bin(self, num_frames: int = 20) -> int:
        """
        Select optimal range bin based on signal variance.
        
        Same algorithm as vital_signs_monitor.py
        """
        profiles_list = list(self.range_profile_history)
        if len(profiles_list) < num_frames:
            # Not enough data, return center of expected range
            expected_bin = int(0.7 / self.range_resolution)
            if len(profiles_list) > 0:
                return min(expected_bin, len(profiles_list[0]) - 1)
            return expected_bin
        
        # Use last num_frames profiles
        profiles = np.array(profiles_list[-num_frames:])
        
        # Handle dB or linear values
        mean_val = np.mean(profiles)
        if mean_val < 50:  # Likely dB values
            magnitudes = 10 ** (profiles / 20.0)
        else:
            magnitudes = np.abs(profiles)
        
        # Compute variance per bin
        var_per_bin = np.var(magnitudes, axis=0)
        
        # Create distance mask (0.3m to 1.5m)
        num_bins = var_per_bin.shape[0]
        distances = np.arange(num_bins) * self.range_resolution
        valid_mask = (distances >= 0.3) & (distances <= 1.5)
        
        # Apply power threshold
        mean_power = np.mean(magnitudes, axis=0)
        max_power = np.max(mean_power)
        if max_power > 0:
            power_thresh = max_power * 0.01
            power_mask = mean_power > power_thresh
        else:
            power_mask = np.ones(num_bins, dtype=bool)
        
        # Combine masks
        combined_mask = valid_mask & power_mask
        
        if not np.any(combined_mask):
            var_masked = np.where(valid_mask, var_per_bin, 0)
        else:
            var_masked = np.where(combined_mask, var_per_bin, 0)
        
        return np.argmax(var_masked)
    
    def _produce_phase_message(self, amplitude: float, sequence: int):
        """Produce a phase/amplitude message to Kafka."""
        msg = PhaseMessage(
            timestamp=time.time(),
            sequence=sequence,
            phase=float(amplitude),
            range_bin=self.selected_bin or 0,
            range_m=(self.selected_bin or 0) * self.range_resolution,
            signal_quality=0.85,  # Could compute from signal
            device_id=self.device_id,
        )
        
        self.producer.produce(
            topic=TOPICS['phase_stream'],
            key=self.device_id,
            value=json.dumps(msg.to_json()),
        )
        self.producer.poll(0)
        self.messages_sent += 1
    
    def _produce_vital_signs(self, hr: float, hr_conf: float, br: float, br_conf: float):
        """Produce vital signs message to Kafka."""
        msg = VitalSignsMessage(
            timestamp=time.time(),
            heart_rate_bpm=hr,
            heart_rate_confidence=hr_conf,
            breathing_rate_bpm=br,
            breathing_rate_confidence=br_conf,
            device_id=self.device_id,
            window_frames=len(self.amplitude_history),
        )
        
        self.producer.produce(
            topic=TOPICS['vital_signs'],
            key=self.device_id,
            value=json.dumps(msg.to_json()),
        )
        self.producer.poll(0)
    
    def run(self, duration: float = 60, verbose: bool = True):
        """
        Run live streaming from AWR1642 to Kafka.
        
        Parameters
        ----------
        duration : float
            Streaming duration in seconds
        verbose : bool
            Print updates to console
        """
        self.is_running = True
        start_time = time.time()
        last_vital_update = start_time
        vital_update_interval = 3.0  # Compute vital signs every 3 seconds
        
        print(f"\n🔴 LIVE STREAMING from AWR1642 to Kafka")
        print(f"   Duration: {duration}s")
        print(f"   Topic: {TOPICS['phase_stream']}")
        print(f"   Stand 0.5-1.5m from radar. Remain still.\n")
        print("   Press Ctrl+C to stop\n")
        
        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n⚠ Shutdown signal received")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                # Read frame from radar
                data_ok, frame_num, det_obj, tlv_data = self.radar.read_tlv_packet(timeout=0.05)
                
                if not data_ok:
                    continue
                
                self.frame_count += 1
                
                # Get range profile from TLV data
                range_profile = tlv_data.get('range_profile')
                if range_profile is None:
                    continue
                
                self.range_profile_history.append(range_profile)
                
                # Update range bin selection periodically
                if self.frame_count % 50 == 1 or self.selected_bin is None:
                    self.selected_bin = self._select_range_bin(num_frames=20)
                    if verbose and self.frame_count == 1:
                        dist = self.selected_bin * self.range_resolution
                        print(f"  Selected range bin: {self.selected_bin} ({dist:.2f}m)")
                
                # Extract amplitude from selected bin
                if self.selected_bin < len(range_profile):
                    amplitude = range_profile[self.selected_bin]
                    self.amplitude_history.append(amplitude)
                    
                    # Stream to Kafka
                    self._produce_phase_message(amplitude, self.frame_count)
                
                # Compute and stream vital signs periodically
                current_time = time.time()
                if current_time - last_vital_update >= vital_update_interval:
                    last_vital_update = current_time
                    
                    if self.processor and len(self.amplitude_history) >= int(10 * self.fps):
                        # Process amplitude signal
                        amplitude_signal = np.array(list(self.amplitude_history))
                        amplitude_signal = amplitude_signal - np.mean(amplitude_signal)
                        
                        result = self.processor.extract_vital_signs(amplitude_signal)
                        
                        hr = result['hr_bpm']
                        br = result['br_bpm']
                        hr_conf = result['hr_confidence']
                        br_conf = result['br_confidence']
                        
                        if not np.isnan(hr) and not np.isnan(br):
                            # Stream vital signs to Kafka
                            self._produce_vital_signs(hr, hr_conf, br, br_conf)
                            
                            if verbose:
                                elapsed = current_time - start_time
                                print(f"  [{elapsed:6.1f}s] HR: {hr:5.1f} BPM ({hr_conf:.2f}) | "
                                      f"BR: {br:5.1f} BPM ({br_conf:.2f}) | "
                                      f"Msgs: {self.messages_sent}")
        
        finally:
            self._print_summary(start_time)
    
    def _print_summary(self, start_time: float):
        """Print streaming summary."""
        elapsed = time.time() - start_time
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print("Streaming Complete")
        print(f"{'='*60}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames captured: {self.frame_count}")
        print(f"Actual FPS: {actual_fps:.1f}")
        print(f"Messages sent: {self.messages_sent}")
        print(f"Device ID: {self.device_id}")


# ============================================================================
# SIMULATION MODE (when no hardware available)
# ============================================================================

class SimulatedEdgeProducer:
    """Simulated producer for testing without hardware."""
    
    def __init__(self, fps: float = 10.0, device_id: str = None):
        self.fps = fps
        self.device_id = device_id or f"sim-radar-{uuid.uuid4().hex[:8]}"
        self.producer: Optional[Producer] = None
        self.is_running = False
        self.messages_sent = 0
        
    def connect(self):
        print("=" * 60)
        print("VitalFlow-Radar Simulated Edge Producer")
        print("=" * 60)
        
        print_config_status()
        config = get_producer_config(client_id=f'vitalflow-sim-{self.device_id}')
        self.producer = Producer(config)
        print(f"✓ Simulated device: {self.device_id}")
        return self
    
    def disconnect(self):
        if self.producer:
            self.producer.flush(10)
    
    def run(self, duration: float = 60, verbose: bool = True):
        self.is_running = True
        start_time = time.time()
        sequence = 0
        
        print(f"\n📡 SIMULATED STREAMING to Kafka")
        print(f"   Duration: {duration}s | FPS: {self.fps}")
        print("   Press Ctrl+C to stop\n")
        
        # Simulate realistic vital signs
        base_hr = 72  # BPM
        base_br = 15  # breaths/min
        
        try:
            while self.is_running and (time.time() - start_time) < duration:
                t = time.time() - start_time
                
                # Simulate phase/amplitude with breathing + heartbeat components
                breathing = 0.5 * np.sin(2 * np.pi * (base_br/60) * t)
                heartbeat = 0.1 * np.sin(2 * np.pi * (base_hr/60) * t)
                noise = 0.02 * np.random.randn()
                amplitude = breathing + heartbeat + noise
                
                msg = {
                    'timestamp': time.time(),
                    'sequence': sequence,
                    'phase': float(amplitude),
                    'range_bin': 15,
                    'range_m': 0.66,
                    'signal_quality': 0.85,
                    'device_id': self.device_id,
                }
                
                self.producer.produce(
                    topic=TOPICS['phase_stream'],
                    key=self.device_id,
                    value=json.dumps(msg),
                )
                self.producer.poll(0)
                
                sequence += 1
                self.messages_sent += 1
                
                if verbose and sequence % 100 == 0:
                    print(f"  [{t:6.1f}s] Sent {self.messages_sent} messages")
                
                time.sleep(1.0 / self.fps)
                
        except KeyboardInterrupt:
            print("\n⚠ Stopped by user")
        
        print(f"\n✓ Sent {self.messages_sent} messages in {time.time()-start_time:.1f}s")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VitalFlow-Radar Live Edge Producer (AWR1642)'
    )
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Use simulation mode (no hardware)')
    parser.add_argument('--duration', '-d', type=float, default=60,
                        help='Streaming duration in seconds (default: 60)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Radar configuration file')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frame rate (default: 20)')
    parser.add_argument('--device-id', type=str, default=None,
                        help='Device identifier')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Reduce console output')
    
    args = parser.parse_args()
    
    if args.simulate:
        # Simulation mode
        producer = SimulatedEdgeProducer(
            fps=args.fps,
            device_id=args.device_id,
        )
    else:
        # Live radar mode
        if not AWR1642_AVAILABLE:
            print("✗ AWR1642 driver not available.")
            print("  Use --simulate for simulation mode.")
            sys.exit(1)
        
        producer = LiveEdgeProducer(
            config_file=args.config,
            fps=args.fps,
            device_id=args.device_id,
        )
    
    try:
        producer.connect()
        producer.run(duration=args.duration, verbose=not args.quiet)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        producer.disconnect()


if __name__ == '__main__':
    main()
