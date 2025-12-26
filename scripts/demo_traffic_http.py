#!/usr/bin/env python3
"""
VitalFlow-Radar: HTTP Demo Traffic Generator
=============================================

Simplified traffic generator that sends vital signs directly to the backend API
via HTTP POST requests. This works WITHOUT Kafka for easy demo/testing.

Use this for:
- Quick demos without Confluent Cloud setup
- Testing the dashboard and AI features
- Hackathon judges who want to test locally

Usage:
    python scripts/demo_traffic_http.py
    python scripts/demo_traffic_http.py --scenario tachycardia
    python scripts/demo_traffic_http.py --continuous --duration 120

For full Kafka streaming demo, use: traffic_generator.py
"""

import os
import sys
import json
import time
import random
import argparse
import requests
from datetime import datetime
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
API_URL = os.environ.get('VITALFLOW_API_URL', 'http://localhost:8000')


class DemoTrafficGenerator:
    """HTTP-based demo traffic generator."""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url.rstrip('/')
        self.messages_sent = 0
        self.running = True
        
    def check_connection(self) -> bool:
        """Check if backend is running."""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connected to VitalFlow Backend")
                print(f"   Mode: {data.get('monitoring_mode', 'unknown')}")
                print(f"   Kafka: {'Connected' if data.get('integrations', {}).get('kafka', {}).get('connected') else 'Offline'}")
                print(f"   AI: {'Active' if data.get('integrations', {}).get('vertex_ai', {}).get('initialized') else 'Offline'}")
                return True
        except Exception as e:
            print(f"❌ Cannot connect to backend at {self.api_url}")
            print(f"   Error: {e}")
            print(f"\n   Make sure the backend is running:")
            print(f"   cd backend && uvicorn main:app --host 0.0.0.0 --port 8000")
            return False
    
    def send_vital_signs(self, patient_id: str, hr: float, br: float, 
                          hr_conf: float = 0.88, br_conf: float = 0.85,
                          status: str = "normal", alerts: list = None):
        """Send vital signs to backend via WebSocket simulation endpoint."""
        data = {
            "patient_id": patient_id,
            "heart_rate": round(hr, 1),
            "heart_rate_confidence": round(hr_conf, 2),
            "breathing_rate": round(br, 1),
            "breathing_rate_confidence": round(br_conf, 2),
            "status": status,
            "timestamp": time.time(),
            "alerts": alerts or []
        }
        
        try:
            # Use the internal broadcast endpoint
            response = requests.post(
                f"{self.api_url}/api/demo/vitals",
                json=data,
                timeout=5
            )
            self.messages_sent += 1
            return response.status_code == 200
        except Exception as e:
            # If demo endpoint doesn't exist, that's okay - data still goes through simulation
            return False
    
    def add_noise(self, value: float, noise_pct: float = 0.03) -> float:
        """Add realistic noise."""
        return value * (1 + np.random.normal(0, noise_pct))
    
    def run_scenario(self, scenario: str, patient_id: str = "default", duration: int = 60):
        """Run a specific vital signs scenario."""
        print(f"\n🎬 Running scenario: {scenario.upper()}")
        print(f"   Patient: {patient_id}")
        print(f"   Duration: {duration}s")
        print("-" * 50)
        
        scenarios = {
            "normal": self._scenario_normal,
            "tachycardia": self._scenario_tachycardia,
            "bradycardia": self._scenario_bradycardia,
            "apnea": self._scenario_apnea,
            "stress": self._scenario_stress,
            "all": self._scenario_all,
        }
        
        if scenario not in scenarios:
            print(f"❌ Unknown scenario: {scenario}")
            print(f"   Available: {', '.join(scenarios.keys())}")
            return
        
        scenarios[scenario](patient_id, duration)
        print(f"\n✅ Scenario complete. Messages sent: {self.messages_sent}")
    
    def _scenario_normal(self, patient_id: str, duration: int):
        """Normal baseline vital signs."""
        base_hr, base_br = 72, 14
        
        for t in range(0, duration, 2):
            if not self.running:
                break
            
            hr = self.add_noise(base_hr + 3 * np.sin(t / 20))
            br = self.add_noise(base_br + 0.5 * np.sin(t / 15))
            
            self._display_and_send(patient_id, hr, br, 0.90, 0.87, "normal")
            time.sleep(2)
    
    def _scenario_tachycardia(self, patient_id: str, duration: int):
        """Tachycardia episode - HR rises to 130+ then recovers."""
        phases = [
            (duration // 3, 75, 130),    # Rise
            (duration // 3, 130, 140),   # Peak
            (duration // 3, 140, 80),    # Recovery
        ]
        
        for phase_dur, start_hr, end_hr in phases:
            for t in range(0, phase_dur, 2):
                if not self.running:
                    return
                
                progress = t / phase_dur
                hr = start_hr + (end_hr - start_hr) * progress
                hr = self.add_noise(hr)
                br = self.add_noise(18 + (hr - 75) * 0.1)
                
                status = "critical" if hr > 120 else "warning" if hr > 100 else "normal"
                alerts = []
                if hr > 100:
                    alerts.append({
                        "type": "heart_rate",
                        "severity": "critical" if hr > 120 else "warning",
                        "message": f"Tachycardia: HR {hr:.0f} BPM"
                    })
                
                self._display_and_send(patient_id, hr, br, 0.85, 0.80, status, alerts)
                time.sleep(2)
    
    def _scenario_bradycardia(self, patient_id: str, duration: int):
        """Bradycardia episode - HR drops to 45 then recovers."""
        phases = [
            (duration // 3, 65, 48),
            (duration // 3, 48, 42),
            (duration // 3, 42, 60),
        ]
        
        for phase_dur, start_hr, end_hr in phases:
            for t in range(0, phase_dur, 2):
                if not self.running:
                    return
                
                progress = t / phase_dur
                hr = start_hr + (end_hr - start_hr) * progress
                hr = self.add_noise(hr)
                br = self.add_noise(12)
                
                status = "critical" if hr < 45 else "warning" if hr < 55 else "normal"
                alerts = []
                if hr < 60:
                    alerts.append({
                        "type": "heart_rate",
                        "severity": "critical" if hr < 45 else "warning",
                        "message": f"Bradycardia: HR {hr:.0f} BPM"
                    })
                
                self._display_and_send(patient_id, hr, br, 0.88, 0.85, status, alerts)
                time.sleep(2)
    
    def _scenario_apnea(self, patient_id: str, duration: int):
        """Apnea event - breathing pauses."""
        phases = [
            (duration // 4, 14, 14),    # Normal
            (duration // 4, 14, 5),     # Drop
            (duration // 4, 5, 5),      # Apnea
            (duration // 4, 5, 20),     # Recovery
        ]
        
        for phase_dur, start_br, end_br in phases:
            for t in range(0, phase_dur, 2):
                if not self.running:
                    return
                
                progress = t / phase_dur
                br = start_br + (end_br - start_br) * progress
                br = max(3, self.add_noise(br, 0.05))
                
                # HR increases during apnea (stress)
                hr = self.add_noise(70 + max(0, (14 - br) * 2))
                
                status = "critical" if br < 8 else "warning" if br < 10 else "normal"
                alerts = []
                if br < 10:
                    alerts.append({
                        "type": "breathing_rate",
                        "severity": "critical" if br < 8 else "warning",
                        "message": f"Apnea/Bradypnea: BR {br:.0f}/min"
                    })
                
                self._display_and_send(patient_id, hr, br, 0.80, 0.60 if br < 8 else 0.75, status, alerts)
                time.sleep(2)
    
    def _scenario_stress(self, patient_id: str, duration: int):
        """Stress response - elevated and variable HR/BR."""
        for t in range(0, duration, 2):
            if not self.running:
                break
            
            stress = 1 + 0.2 * np.sin(t / 8) + 0.1 * random.random()
            hr = self.add_noise(90 * stress, 0.05)
            br = self.add_noise(22 * stress, 0.06)
            
            status = "warning" if hr > 100 or br > 22 else "normal"
            alerts = []
            if hr > 100:
                alerts.append({"type": "heart_rate", "severity": "warning", "message": f"Elevated HR: {hr:.0f} BPM"})
            if br > 22:
                alerts.append({"type": "breathing_rate", "severity": "warning", "message": f"Rapid breathing: {br:.0f}/min"})
            
            self._display_and_send(patient_id, hr, br, 0.78, 0.72, status, alerts)
            time.sleep(2)
    
    def _scenario_all(self, patient_id: str, duration: int):
        """Run all scenarios sequentially."""
        scenario_dur = max(30, duration // 5)
        
        for scenario in ["normal", "tachycardia", "bradycardia", "apnea", "stress"]:
            if not self.running:
                break
            print(f"\n--- {scenario.upper()} ---")
            getattr(self, f"_scenario_{scenario}")(patient_id, scenario_dur)
            time.sleep(2)
    
    def _display_and_send(self, patient_id: str, hr: float, br: float, 
                          hr_conf: float, br_conf: float, status: str, alerts: list = None):
        """Display and send vital signs."""
        icon = "🟢" if status == "normal" else "🟡" if status == "warning" else "🔴"
        print(f"{icon} HR: {hr:5.1f} BPM | BR: {br:5.1f}/min | Status: {status}")
        
        self.send_vital_signs(patient_id, hr, br, hr_conf, br_conf, status, alerts)
    
    def run_continuous(self, patient_id: str = "default", duration: int = 300):
        """Run continuous demo cycling through scenarios."""
        print(f"\n🔄 Continuous Demo Mode for {duration}s")
        print("   Press Ctrl+C to stop")
        
        start = time.time()
        scenarios = ["normal", "tachycardia", "normal", "apnea", "normal", "stress"]
        idx = 0
        
        try:
            while self.running and (time.time() - start) < duration:
                scenario = scenarios[idx % len(scenarios)]
                self.run_scenario(scenario, patient_id, 45)
                idx += 1
        except KeyboardInterrupt:
            print("\n⏹️ Stopped")
        
        print(f"\n✅ Demo complete. Total messages: {self.messages_sent}")


def main():
    parser = argparse.ArgumentParser(
        description="VitalFlow HTTP Demo Traffic Generator (no Kafka required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This generator sends vital signs directly via HTTP to demonstrate the dashboard.
No Kafka setup required - perfect for quick demos and testing.

Examples:
  python scripts/demo_traffic_http.py --scenario normal
  python scripts/demo_traffic_http.py --scenario tachycardia --duration 60
  python scripts/demo_traffic_http.py --scenario all
  python scripts/demo_traffic_http.py --continuous --duration 180
"""
    )
    
    parser.add_argument("--scenario", "-s", default="all", help="Scenario: normal, tachycardia, bradycardia, apnea, stress, all")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--patient-id", "-p", default="default", help="Patient ID")
    parser.add_argument("--continuous", "-c", action="store_true", help="Continuous demo mode")
    parser.add_argument("--api-url", default=API_URL, help="Backend API URL")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("VitalFlow-Radar HTTP Demo Traffic Generator")
    print("=" * 60)
    
    generator = DemoTrafficGenerator(args.api_url)
    
    if not generator.check_connection():
        sys.exit(1)
    
    print()
    
    try:
        if args.continuous:
            generator.run_continuous(args.patient_id, args.duration)
        else:
            generator.run_scenario(args.scenario, args.patient_id, args.duration)
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
