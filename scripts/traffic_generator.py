#!/usr/bin/env python3
"""
VitalFlow-Radar: Traffic Generator for Confluent Hackathon Demo
================================================================

This script generates realistic vital signs traffic to demonstrate the 
real-time streaming capabilities of VitalFlow-Radar with Confluent Cloud.

Scenarios simulated:
1. Normal vital signs (baseline)
2. Gradual tachycardia (heart rate increase)
3. Bradycardia episode (low heart rate)
4. Apnea event (breathing pause)
5. High stress response (elevated HR + BR)
6. Sleep patterns (low HR, regular BR)
7. Multi-patient concurrent monitoring

Usage:
    # Run all scenarios
    python scripts/traffic_generator.py --scenario all
    
    # Run specific scenario
    python scripts/traffic_generator.py --scenario tachycardia
    
    # Multi-patient mode (3 patients)
    python scripts/traffic_generator.py --multi-patient 3
    
    # Continuous mode for live demos
    python scripts/traffic_generator.py --continuous --duration 300

Environment Variables:
    CONFLUENT_BOOTSTRAP_SERVERS: Kafka bootstrap servers
    CONFLUENT_API_KEY: API key for Confluent Cloud
    CONFLUENT_API_SECRET: API secret for Confluent Cloud
"""

import os
import sys
import json
import time
import uuid
import random
import argparse
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

from confluent_kafka import Producer

# Import config
try:
    from confluent_config import get_producer_config, TOPICS, print_config_status
except ImportError:
    print("❌ confluent_config not found. Please ensure you're in the project directory.")
    sys.exit(1)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PhaseMessage:
    """Raw phase data from radar (simulated)."""
    timestamp: float
    sequence: int
    phase: float
    range_bin: int
    range_m: float
    signal_quality: float
    device_id: str
    patient_id: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class VitalSignsMessage:
    """Computed vital signs."""
    timestamp: float
    heart_rate_bpm: float
    heart_rate_confidence: float
    breathing_rate_bpm: float
    breathing_rate_confidence: float
    device_id: str
    patient_id: str
    scenario: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class AnomalyMessage:
    """Detected anomaly."""
    timestamp: float
    anomaly_type: str
    severity: str
    current_value: float
    normal_range_min: float
    normal_range_max: float
    confidence: float
    device_id: str
    patient_id: str
    description: str
    recommended_action: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ============================================================================
# SCENARIO GENERATORS
# ============================================================================

class ScenarioGenerator:
    """Generates realistic vital signs patterns for different scenarios."""
    
    def __init__(self, device_id: str, patient_id: str):
        self.device_id = device_id
        self.patient_id = patient_id
        self.sequence = 0
        
    def _add_noise(self, value: float, noise_level: float = 0.02) -> float:
        """Add realistic noise to measurements."""
        return value * (1 + np.random.normal(0, noise_level))
    
    def normal_baseline(self, duration_sec: int = 60) -> List[VitalSignsMessage]:
        """Generate normal vital signs baseline."""
        messages = []
        base_hr = 72
        base_br = 14
        
        for t in range(0, duration_sec, 3):  # Every 3 seconds
            msg = VitalSignsMessage(
                timestamp=time.time() + t,
                heart_rate_bpm=self._add_noise(base_hr, 0.03),
                heart_rate_confidence=random.uniform(0.85, 0.95),
                breathing_rate_bpm=self._add_noise(base_br, 0.05),
                breathing_rate_confidence=random.uniform(0.80, 0.92),
                device_id=self.device_id,
                patient_id=self.patient_id,
                scenario="normal_baseline"
            )
            messages.append(msg)
        return messages
    
    def tachycardia_episode(self, duration_sec: int = 90) -> List[VitalSignsMessage]:
        """
        Simulate gradual tachycardia episode.
        HR increases from 75 to 140 BPM over time, then normalizes.
        """
        messages = []
        phases = [
            (30, 75, 110),   # Ramp up phase
            (30, 110, 140),  # Peak phase  
            (30, 140, 85),   # Recovery phase
        ]
        
        current_time = 0
        for phase_duration, start_hr, end_hr in phases:
            for t in range(0, phase_duration, 3):
                progress = t / phase_duration
                hr = start_hr + (end_hr - start_hr) * progress
                
                # Higher HR = lower confidence (harder to track)
                confidence = max(0.6, 0.95 - (hr - 60) / 200)
                
                msg = VitalSignsMessage(
                    timestamp=time.time() + current_time + t,
                    heart_rate_bpm=self._add_noise(hr, 0.04),
                    heart_rate_confidence=confidence,
                    breathing_rate_bpm=self._add_noise(18 + (hr - 75) / 10, 0.06),
                    breathing_rate_confidence=random.uniform(0.75, 0.88),
                    device_id=self.device_id,
                    patient_id=self.patient_id,
                    scenario="tachycardia_episode"
                )
                messages.append(msg)
            current_time += phase_duration
        return messages
    
    def bradycardia_episode(self, duration_sec: int = 90) -> List[VitalSignsMessage]:
        """
        Simulate bradycardia episode.
        HR drops from 65 to 42 BPM, then recovers.
        """
        messages = []
        phases = [
            (30, 65, 48),   # Decline phase
            (30, 48, 42),   # Critical low phase
            (30, 42, 62),   # Recovery phase
        ]
        
        current_time = 0
        for phase_duration, start_hr, end_hr in phases:
            for t in range(0, phase_duration, 3):
                progress = t / phase_duration
                hr = start_hr + (end_hr - start_hr) * progress
                
                msg = VitalSignsMessage(
                    timestamp=time.time() + current_time + t,
                    heart_rate_bpm=self._add_noise(hr, 0.03),
                    heart_rate_confidence=random.uniform(0.82, 0.93),
                    breathing_rate_bpm=self._add_noise(12, 0.05),
                    breathing_rate_confidence=random.uniform(0.78, 0.90),
                    device_id=self.device_id,
                    patient_id=self.patient_id,
                    scenario="bradycardia_episode"
                )
                messages.append(msg)
            current_time += phase_duration
        return messages
    
    def apnea_event(self, duration_sec: int = 60) -> List[VitalSignsMessage]:
        """
        Simulate apnea (breathing pause) event.
        BR drops to near-zero, then resumes with compensatory increase.
        """
        messages = []
        phases = [
            (15, 14, 14),   # Normal baseline
            (15, 14, 4),    # Apnea onset  
            (10, 4, 4),     # Apnea (critical)
            (20, 4, 22),    # Recovery (compensatory hyperventilation)
        ]
        
        current_time = 0
        for phase_duration, start_br, end_br in phases:
            for t in range(0, phase_duration, 3):
                progress = t / phase_duration
                br = start_br + (end_br - start_br) * progress
                
                # Very low BR = low confidence
                confidence = max(0.4, 0.9 - abs(14 - br) / 20)
                
                # HR increases during apnea (stress response)
                hr = 70 + max(0, (14 - br)) * 3
                
                msg = VitalSignsMessage(
                    timestamp=time.time() + current_time + t,
                    heart_rate_bpm=self._add_noise(hr, 0.04),
                    heart_rate_confidence=random.uniform(0.75, 0.88),
                    breathing_rate_bpm=self._add_noise(br, 0.08),
                    breathing_rate_confidence=confidence,
                    device_id=self.device_id,
                    patient_id=self.patient_id,
                    scenario="apnea_event"
                )
                messages.append(msg)
            current_time += phase_duration
        return messages
    
    def stress_response(self, duration_sec: int = 90) -> List[VitalSignsMessage]:
        """
        Simulate stress/anxiety response.
        Both HR and BR elevated with high variability.
        """
        messages = []
        
        for t in range(0, duration_sec, 3):
            # Simulate stress peaks
            stress_factor = 1 + 0.3 * np.sin(t / 10) + 0.1 * np.sin(t / 3)
            
            hr = 85 * stress_factor
            br = 20 * stress_factor
            
            msg = VitalSignsMessage(
                timestamp=time.time() + t,
                heart_rate_bpm=self._add_noise(hr, 0.06),
                heart_rate_confidence=random.uniform(0.70, 0.85),
                breathing_rate_bpm=self._add_noise(br, 0.08),
                breathing_rate_confidence=random.uniform(0.65, 0.82),
                device_id=self.device_id,
                patient_id=self.patient_id,
                scenario="stress_response"
            )
            messages.append(msg)
        return messages
    
    def sleep_pattern(self, duration_sec: int = 120) -> List[VitalSignsMessage]:
        """
        Simulate sleep vital signs pattern.
        Low HR, regular BR with occasional variations.
        """
        messages = []
        
        for t in range(0, duration_sec, 3):
            # Simulate sleep stages with subtle variations
            stage_factor = 1 + 0.05 * np.sin(t / 30)  # Slow oscillation
            
            hr = 56 * stage_factor
            br = 13 * stage_factor
            
            msg = VitalSignsMessage(
                timestamp=time.time() + t,
                heart_rate_bpm=self._add_noise(hr, 0.02),
                heart_rate_confidence=random.uniform(0.88, 0.96),
                breathing_rate_bpm=self._add_noise(br, 0.03),
                breathing_rate_confidence=random.uniform(0.85, 0.94),
                device_id=self.device_id,
                patient_id=self.patient_id,
                scenario="sleep_pattern"
            )
            messages.append(msg)
        return messages
    
    def pediatric_high_hr(self, duration_sec: int = 60) -> List[VitalSignsMessage]:
        """
        Simulate pediatric vital signs (higher baseline HR).
        Children have naturally higher HR: 70-120 BPM for ages 1-10.
        """
        messages = []
        base_hr = 105  # Normal for child
        base_br = 22   # Normal for child
        
        for t in range(0, duration_sec, 3):
            # Children show more HR variability
            hr_variation = np.sin(t / 5) * 8
            
            msg = VitalSignsMessage(
                timestamp=time.time() + t,
                heart_rate_bpm=self._add_noise(base_hr + hr_variation, 0.04),
                heart_rate_confidence=random.uniform(0.78, 0.90),
                breathing_rate_bpm=self._add_noise(base_br, 0.06),
                breathing_rate_confidence=random.uniform(0.75, 0.88),
                device_id=self.device_id,
                patient_id=self.patient_id,
                scenario="pediatric_high_hr"
            )
            messages.append(msg)
        return messages


# ============================================================================
# TRAFFIC GENERATOR
# ============================================================================

class TrafficGenerator:
    """
    Main traffic generator for Confluent hackathon demo.
    Streams simulated vital signs data to Kafka topics.
    """
    
    def __init__(self):
        self.producer = None
        self.running = False
        self.messages_sent = 0
        self.anomalies_generated = 0
        
    def _delivery_callback(self, err, msg):
        """Kafka delivery callback."""
        if err:
            print(f"❌ Delivery failed: {err}")
        else:
            self.messages_sent += 1
            
    def connect(self):
        """Connect to Confluent Cloud."""
        print("\n" + "=" * 60)
        print("VitalFlow-Radar Traffic Generator")
        print("=" * 60)
        print_config_status()
        
        config = get_producer_config()
        self.producer = Producer(config)
        print("✅ Connected to Confluent Cloud")
        
    def _detect_anomalies(self, msg: VitalSignsMessage) -> List[AnomalyMessage]:
        """Detect anomalies in vital signs."""
        anomalies = []
        
        # Heart rate anomalies
        hr = msg.heart_rate_bpm
        if hr < 40:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="severe_bradycardia",
                severity="critical",
                current_value=hr,
                normal_range_min=60,
                normal_range_max=100,
                confidence=msg.heart_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Severely low heart rate detected: {hr:.0f} BPM",
                recommended_action="Immediate medical attention required"
            ))
        elif hr < 60:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="bradycardia",
                severity="medium",
                current_value=hr,
                normal_range_min=60,
                normal_range_max=100,
                confidence=msg.heart_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Low heart rate detected: {hr:.0f} BPM",
                recommended_action="Monitor closely, consider intervention"
            ))
        elif hr > 150:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="severe_tachycardia",
                severity="high",
                current_value=hr,
                normal_range_min=60,
                normal_range_max=100,
                confidence=msg.heart_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Severely elevated heart rate: {hr:.0f} BPM",
                recommended_action="Assess patient, check for arrhythmias"
            ))
        elif hr > 100:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="tachycardia",
                severity="medium",
                current_value=hr,
                normal_range_min=60,
                normal_range_max=100,
                confidence=msg.heart_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Elevated heart rate: {hr:.0f} BPM",
                recommended_action="Monitor for sustained elevation"
            ))
        
        # Breathing rate anomalies
        br = msg.breathing_rate_bpm
        if br < 6:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="apnea",
                severity="critical",
                current_value=br,
                normal_range_min=12,
                normal_range_max=20,
                confidence=msg.breathing_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Possible apnea detected: {br:.0f} breaths/min",
                recommended_action="Check patient breathing, prepare intervention"
            ))
        elif br < 12:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="bradypnea",
                severity="medium",
                current_value=br,
                normal_range_min=12,
                normal_range_max=20,
                confidence=msg.breathing_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Slow breathing rate: {br:.0f} breaths/min",
                recommended_action="Monitor respiratory status"
            ))
        elif br > 25:
            anomalies.append(AnomalyMessage(
                timestamp=msg.timestamp,
                anomaly_type="tachypnea",
                severity="high",
                current_value=br,
                normal_range_min=12,
                normal_range_max=20,
                confidence=msg.breathing_rate_confidence,
                device_id=msg.device_id,
                patient_id=msg.patient_id,
                description=f"Rapid breathing: {br:.0f} breaths/min",
                recommended_action="Assess respiratory distress"
            ))
        
        return anomalies
        
    def run_scenario(self, scenario_name: str, patient_id: str = "patient-001"):
        """Run a specific scenario."""
        device_id = f"radar-{uuid.uuid4().hex[:8]}"
        generator = ScenarioGenerator(device_id, patient_id)
        
        scenarios = {
            "normal": generator.normal_baseline,
            "tachycardia": generator.tachycardia_episode,
            "bradycardia": generator.bradycardia_episode,
            "apnea": generator.apnea_event,
            "stress": generator.stress_response,
            "sleep": generator.sleep_pattern,
            "pediatric": generator.pediatric_high_hr,
        }
        
        if scenario_name == "all":
            for name, gen_func in scenarios.items():
                print(f"\n🎬 Running scenario: {name.upper()}")
                self._stream_messages(gen_func(), name)
                time.sleep(2)  # Brief pause between scenarios
        elif scenario_name in scenarios:
            print(f"\n🎬 Running scenario: {scenario_name.upper()}")
            self._stream_messages(scenarios[scenario_name](), scenario_name)
        else:
            print(f"❌ Unknown scenario: {scenario_name}")
            print(f"Available: {', '.join(scenarios.keys())}, all")
            return
            
    def _stream_messages(self, messages: List[VitalSignsMessage], scenario_name: str):
        """Stream messages to Kafka with realistic timing."""
        start_time = time.time()
        
        for i, msg in enumerate(messages):
            # Update timestamp to now
            msg.timestamp = time.time()
            
            # Send vital signs to Kafka
            self.producer.produce(
                topic=TOPICS['vital_signs'],
                key=msg.patient_id,
                value=msg.to_json(),
                callback=self._delivery_callback
            )
            
            # Check for anomalies
            anomalies = self._detect_anomalies(msg)
            for anomaly in anomalies:
                self.producer.produce(
                    topic=TOPICS['anomalies'],
                    key=msg.patient_id,
                    value=anomaly.to_json(),
                    callback=self._delivery_callback
                )
                self.anomalies_generated += 1
                print(f"  ⚠️  [{anomaly.severity.upper()}] {anomaly.description}")
            
            self.producer.poll(0)
            
            # Progress indicator
            hr = msg.heart_rate_bpm
            br = msg.breathing_rate_bpm
            status = "🟢" if 60 <= hr <= 100 and 12 <= br <= 20 else "🟡" if 50 <= hr <= 120 else "🔴"
            print(f"  {status} HR: {hr:5.1f} BPM | BR: {br:5.1f} bpm | Conf: {msg.heart_rate_confidence:.0%}")
            
            # Wait for realistic timing (3 second intervals)
            time.sleep(3)
        
        self.producer.flush()
        elapsed = time.time() - start_time
        print(f"\n✅ Scenario '{scenario_name}' complete: {len(messages)} messages, {self.anomalies_generated} anomalies in {elapsed:.1f}s")
        
    def run_multi_patient(self, num_patients: int = 3, duration_sec: int = 120):
        """
        Run multi-patient concurrent monitoring simulation.
        Demonstrates scalability of the streaming architecture.
        """
        print(f"\n🏥 Multi-Patient Mode: {num_patients} patients for {duration_sec}s")
        
        # Create patient scenarios
        patient_scenarios = [
            ("patient-001", "normal", "Ward A, Room 101"),
            ("patient-002", "tachycardia", "Ward A, Room 102"),
            ("patient-003", "sleep", "Ward B, Room 201"),
            ("patient-004", "stress", "ICU, Bed 1"),
            ("patient-005", "pediatric", "Pediatrics, Room 301"),
        ][:num_patients]
        
        threads = []
        for patient_id, scenario, location in patient_scenarios:
            device_id = f"radar-{uuid.uuid4().hex[:8]}"
            print(f"  📡 {patient_id}: {scenario} scenario @ {location}")
            
            t = threading.Thread(
                target=self._patient_stream,
                args=(patient_id, device_id, scenario, duration_sec)
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
            
        print(f"\n✅ Multi-patient simulation complete")
        print(f"   Total messages: {self.messages_sent}")
        print(f"   Total anomalies: {self.anomalies_generated}")
        
    def _patient_stream(self, patient_id: str, device_id: str, scenario: str, duration_sec: int):
        """Stream data for a single patient (threaded)."""
        generator = ScenarioGenerator(device_id, patient_id)
        
        scenarios = {
            "normal": generator.normal_baseline,
            "tachycardia": generator.tachycardia_episode,
            "bradycardia": generator.bradycardia_episode,
            "apnea": generator.apnea_event,
            "stress": generator.stress_response,
            "sleep": generator.sleep_pattern,
            "pediatric": generator.pediatric_high_hr,
        }
        
        # Generate messages
        if scenario in scenarios:
            messages = scenarios[scenario](duration_sec)
        else:
            messages = scenarios["normal"](duration_sec)
            
        # Stream with timing
        for msg in messages:
            msg.timestamp = time.time()
            
            self.producer.produce(
                topic=TOPICS['vital_signs'],
                key=msg.patient_id,
                value=msg.to_json(),
                callback=self._delivery_callback
            )
            
            anomalies = self._detect_anomalies(msg)
            for anomaly in anomalies:
                self.producer.produce(
                    topic=TOPICS['anomalies'],
                    key=msg.patient_id,
                    value=anomaly.to_json(),
                    callback=self._delivery_callback
                )
                self.anomalies_generated += 1
                
            self.producer.poll(0)
            time.sleep(3 + random.uniform(-0.5, 0.5))
            
        self.producer.flush()
        
    def run_continuous(self, duration_sec: int = 300):
        """
        Run continuous demo mode cycling through scenarios.
        Ideal for live presentations.
        """
        print(f"\n🔄 Continuous Demo Mode for {duration_sec}s")
        print("   Press Ctrl+C to stop")
        
        self.running = True
        start_time = time.time()
        
        scenarios = ["normal", "tachycardia", "normal", "apnea", "normal", "stress", "normal"]
        scenario_idx = 0
        
        try:
            while self.running and (time.time() - start_time) < duration_sec:
                scenario = scenarios[scenario_idx % len(scenarios)]
                print(f"\n🎬 Demo Scenario: {scenario.upper()}")
                
                self.run_scenario(scenario, "demo-patient-001")
                scenario_idx += 1
                
                if not self.running:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping continuous mode...")
            self.running = False
            
        print(f"\n✅ Continuous demo complete")
        print(f"   Duration: {time.time() - start_time:.1f}s")
        print(f"   Messages sent: {self.messages_sent}")
        print(f"   Anomalies: {self.anomalies_generated}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VitalFlow-Radar Traffic Generator for Confluent Hackathon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios
  python scripts/traffic_generator.py --scenario all
  
  # Run tachycardia scenario
  python scripts/traffic_generator.py --scenario tachycardia
  
  # Multi-patient simulation (3 patients, 2 minutes)
  python scripts/traffic_generator.py --multi-patient 3 --duration 120
  
  # Continuous demo mode (5 minutes)
  python scripts/traffic_generator.py --continuous --duration 300

Available Scenarios:
  normal      - Normal baseline vital signs
  tachycardia - Heart rate elevation episode
  bradycardia - Low heart rate episode  
  apnea       - Breathing pause event
  stress      - High stress/anxiety response
  sleep       - Sleep pattern vitals
  pediatric   - Pediatric high heart rate
  all         - Run all scenarios sequentially
        """
    )
    
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default="all",
        help="Scenario to run (default: all)"
    )
    
    parser.add_argument(
        "--multi-patient", "-m",
        type=int,
        default=0,
        metavar="N",
        help="Run multi-patient mode with N patients"
    )
    
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuous demo mode"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=300,
        help="Duration in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--patient-id", "-p",
        type=str,
        default="patient-001",
        help="Patient ID for single scenarios"
    )
    
    args = parser.parse_args()
    
    # Create and run generator
    generator = TrafficGenerator()
    generator.connect()
    
    if args.multi_patient > 0:
        generator.run_multi_patient(args.multi_patient, args.duration)
    elif args.continuous:
        generator.run_continuous(args.duration)
    else:
        generator.run_scenario(args.scenario, args.patient_id)
        
    print("\n" + "=" * 60)
    print("Traffic Generation Summary")
    print("=" * 60)
    print(f"  Messages sent:     {generator.messages_sent}")
    print(f"  Anomalies created: {generator.anomalies_generated}")
    print("=" * 60)


if __name__ == "__main__":
    main()
