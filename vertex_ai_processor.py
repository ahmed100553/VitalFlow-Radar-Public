"""
VitalFlow-Radar: Google Cloud Vertex AI Anomaly Detection
==========================================================

This module provides AI-powered anomaly detection for vital signs using:
- Google Cloud Vertex AI for ML model hosting
- Gemini API for intelligent health summaries
- Real-time anomaly classification (bradycardia, tachycardia, apnea, etc.)

Environment Variables Required:
    - GOOGLE_CLOUD_PROJECT: GCP project ID
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional)
    - VERTEX_AI_LOCATION: Region (default: us-central1)

Usage:
    from vertex_ai_processor import VitalSignsAnomalyDetector
    
    detector = VitalSignsAnomalyDetector()
    anomalies = detector.detect_anomalies(heart_rate=45, breathing_rate=8)
    summary = detector.generate_health_summary(vital_signs)
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Google Cloud imports
try:
    import google.auth
    from google.cloud import aiplatform
    from google.auth import default as auth_default
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("⚠ Google Cloud SDK not installed. Install with: pip install google-cloud-aiplatform")

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class AnomalyType(Enum):
    """Types of vital sign anomalies."""
    BRADYCARDIA = "bradycardia"           # HR < 60 BPM
    TACHYCARDIA = "tachycardia"           # HR > 100 BPM
    SEVERE_BRADYCARDIA = "severe_bradycardia"  # HR < 40 BPM
    SEVERE_TACHYCARDIA = "severe_tachycardia"  # HR > 150 BPM
    APNEA = "apnea"                       # BR < 8 or pause
    BRADYPNEA = "bradypnea"               # BR < 12
    TACHYPNEA = "tachypnea"               # BR > 20
    IRREGULAR_HR = "irregular_hr"          # High HR variability
    IRREGULAR_BR = "irregular_br"          # High BR variability
    SIGNAL_LOSS = "signal_loss"           # Low confidence signal


class Severity(Enum):
    """Anomaly severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VitalSigns:
    """Current vital signs measurement."""
    timestamp: float
    heart_rate_bpm: float
    heart_rate_confidence: float
    breathing_rate_bpm: float
    breathing_rate_confidence: float
    device_id: str
    subject_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Anomaly:
    """Detected anomaly."""
    timestamp: float
    anomaly_type: str
    severity: str
    current_value: float
    normal_range_min: float
    normal_range_max: float
    confidence: float
    device_id: str
    description: str
    recommended_action: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class HealthSummary:
    """AI-generated health summary."""
    timestamp: float
    vital_signs: VitalSigns
    anomalies: List[Anomaly]
    summary_text: str
    risk_level: str
    recommendations: List[str]
    gemini_model_used: str = "gemini-2.5-flash-lite"


# ============================================================================
# ANOMALY DETECTION RULES
# ============================================================================

# Normal ranges for adults at rest
NORMAL_RANGES = {
    'heart_rate': {
        'min': 60,
        'max': 100,
        'unit': 'BPM',
    },
    'breathing_rate': {
        'min': 12,
        'max': 20,
        'unit': 'breaths/min',
    },
}

# Threshold definitions for anomalies
ANOMALY_THRESHOLDS = {
    AnomalyType.SEVERE_BRADYCARDIA: {
        'metric': 'heart_rate',
        'condition': lambda hr: hr < 40,
        'severity': Severity.CRITICAL,
        'description': "Severely low heart rate detected",
        'action': "Seek immediate medical attention",
    },
    AnomalyType.BRADYCARDIA: {
        'metric': 'heart_rate',
        'condition': lambda hr: 40 <= hr < 60,
        'severity': Severity.MEDIUM,
        'description': "Heart rate below normal resting range",
        'action': "Monitor closely; consult physician if persistent",
    },
    AnomalyType.TACHYCARDIA: {
        'metric': 'heart_rate',
        'condition': lambda hr: 100 < hr <= 150,
        'severity': Severity.MEDIUM,
        'description': "Heart rate above normal resting range",
        'action': "Rest and monitor; check for stress or caffeine",
    },
    AnomalyType.SEVERE_TACHYCARDIA: {
        'metric': 'heart_rate',
        'condition': lambda hr: hr > 150,
        'severity': Severity.HIGH,
        'description': "Dangerously elevated heart rate",
        'action': "Seek medical attention promptly",
    },
    AnomalyType.APNEA: {
        'metric': 'breathing_rate',
        'condition': lambda br: br < 6,
        'severity': Severity.CRITICAL,
        'description': "Possible breathing pause or very slow breathing",
        'action': "Check subject immediately; may need emergency response",
    },
    AnomalyType.BRADYPNEA: {
        'metric': 'breathing_rate',
        'condition': lambda br: 6 <= br < 12,
        'severity': Severity.MEDIUM,
        'description': "Breathing rate below normal",
        'action': "Monitor for respiratory depression",
    },
    AnomalyType.TACHYPNEA: {
        'metric': 'breathing_rate',
        'condition': lambda br: br > 25,
        'severity': Severity.HIGH,
        'description': "Rapid breathing detected",
        'action': "Assess for respiratory distress or anxiety",
    },
}


# ============================================================================
# VERTEX AI ANOMALY DETECTOR
# ============================================================================

class VitalSignsAnomalyDetector:
    """
    AI-powered vital signs anomaly detection using Google Cloud Vertex AI.
    """
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        use_gemini: bool = True,
    ):
        """
        Initialize the anomaly detector.
        
        Parameters
        ----------
        project_id : str, optional
            GCP project ID (from env if not provided)
        location : str
            Vertex AI region
        use_gemini : bool
            Enable Gemini API for natural language summaries
        """
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location = location
        self.use_gemini = use_gemini
        self.gemini_model: Optional[GenerativeModel] = None
        self._initialized = False
        
        # Historical data for trend analysis
        self._hr_history: List[float] = []
        self._br_history: List[float] = []
        self._history_window = 60  # seconds of history
        
    def initialize(self) -> bool:
        """Initialize Vertex AI connection."""
        if self._initialized:
            return True
            
        if not GOOGLE_CLOUD_AVAILABLE:
            print("⚠ Google Cloud SDK not available. Running in offline mode.")
            return False
        
        try:
            if self.project_id:
                print(f"Initializing Vertex AI for project: {self.project_id}")
                aiplatform.init(project=self.project_id, location=self.location)
                
                if self.use_gemini and VERTEX_AI_AVAILABLE:
                    vertexai.init(project=self.project_id, location=self.location)
                    # Use gemini-1.5-flash-002 which is available in all regions
                    self.gemini_model = GenerativeModel("gemini-2.5-flash-lite")
                    print("✓ Gemini model initialized")
                
                self._initialized = True
                print("✓ Vertex AI initialized successfully")
                return True
            else:
                print("⚠ No GOOGLE_CLOUD_PROJECT set. Running in offline mode.")
                return False
                
        except Exception as e:
            print(f"⚠ Could not initialize Vertex AI: {e}")
            print("  Running in offline mode with rule-based detection only.")
            return False
    
    def detect_anomalies(
        self,
        vital_signs: VitalSigns,
        include_history: bool = True,
    ) -> List[Anomaly]:
        """
        Detect anomalies in vital signs.
        
        Parameters
        ----------
        vital_signs : VitalSigns
            Current vital signs measurement
        include_history : bool
            Consider historical trends
            
        Returns
        -------
        List[Anomaly]
            List of detected anomalies
        """
        anomalies = []
        
        hr = vital_signs.heart_rate_bpm
        br = vital_signs.breathing_rate_bpm
        hr_conf = vital_signs.heart_rate_confidence
        br_conf = vital_signs.breathing_rate_confidence
        
        # Update history
        self._hr_history.append(hr)
        self._br_history.append(br)
        
        # Trim history
        max_samples = int(self._history_window * 0.5)  # Assuming ~0.5 Hz updates
        self._hr_history = self._hr_history[-max_samples:]
        self._br_history = self._br_history[-max_samples:]
        
        # Check signal quality
        if hr_conf < 0.5 or br_conf < 0.5:
            anomalies.append(Anomaly(
                timestamp=vital_signs.timestamp,
                anomaly_type=AnomalyType.SIGNAL_LOSS.value,
                severity=Severity.LOW.value,
                current_value=min(hr_conf, br_conf),
                normal_range_min=0.7,
                normal_range_max=1.0,
                confidence=0.9,
                device_id=vital_signs.device_id,
                description="Low signal quality - measurements may be unreliable",
                recommended_action="Ensure subject is within radar range and stationary",
            ))
        
        # Check against thresholds
        for anomaly_type, config in ANOMALY_THRESHOLDS.items():
            metric = config['metric']
            value = hr if metric == 'heart_rate' else br
            
            if config['condition'](value):
                normal_range = NORMAL_RANGES[metric]
                anomalies.append(Anomaly(
                    timestamp=vital_signs.timestamp,
                    anomaly_type=anomaly_type.value,
                    severity=config['severity'].value,
                    current_value=value,
                    normal_range_min=normal_range['min'],
                    normal_range_max=normal_range['max'],
                    confidence=hr_conf if metric == 'heart_rate' else br_conf,
                    device_id=vital_signs.device_id,
                    description=config['description'],
                    recommended_action=config['action'],
                ))
        
        # Check for irregular patterns (variability analysis)
        if include_history and len(self._hr_history) >= 10:
            hr_std = np.std(self._hr_history[-10:])
            if hr_std > 15:  # High variability
                anomalies.append(Anomaly(
                    timestamp=vital_signs.timestamp,
                    anomaly_type=AnomalyType.IRREGULAR_HR.value,
                    severity=Severity.MEDIUM.value,
                    current_value=hr_std,
                    normal_range_min=0,
                    normal_range_max=10,
                    confidence=0.7,
                    device_id=vital_signs.device_id,
                    description=f"Irregular heart rate pattern (variability: {hr_std:.1f} BPM)",
                    recommended_action="Monitor for arrhythmia; consider ECG if persistent",
                ))
        
        return anomalies
    
    def generate_health_summary(
        self,
        vital_signs: VitalSigns,
        anomalies: List[Anomaly],
    ) -> HealthSummary:
        """
        Generate an AI-powered health summary using Gemini.
        
        Parameters
        ----------
        vital_signs : VitalSigns
            Current measurements
        anomalies : List[Anomaly]
            Detected anomalies
            
        Returns
        -------
        HealthSummary
            AI-generated health summary
        """
        # Determine risk level
        if any(a.severity == Severity.CRITICAL.value for a in anomalies):
            risk_level = "CRITICAL"
        elif any(a.severity == Severity.HIGH.value for a in anomalies):
            risk_level = "HIGH"
        elif any(a.severity == Severity.MEDIUM.value for a in anomalies):
            risk_level = "MODERATE"
        elif anomalies:
            risk_level = "LOW"
        else:
            risk_level = "NORMAL"
        
        # Try to generate Gemini summary
        summary_text = self._generate_gemini_summary(vital_signs, anomalies, risk_level)
        
        # Extract recommendations
        recommendations = [a.recommended_action for a in anomalies if a.recommended_action]
        if not recommendations:
            recommendations = ["Continue routine monitoring"]
        
        return HealthSummary(
            timestamp=time.time(),
            vital_signs=vital_signs,
            anomalies=anomalies,
            summary_text=summary_text,
            risk_level=risk_level,
            recommendations=recommendations,
            gemini_model_used="gemini-2.5-flash-lite" if self.gemini_model else "rule-based",
        )
    
    def _generate_gemini_summary(
        self,
        vital_signs: VitalSigns,
        anomalies: List[Anomaly],
        risk_level: str,
    ) -> str:
        """Generate summary using Gemini API."""
        
        # Build prompt
        anomaly_descriptions = "\n".join([
            f"- {a.anomaly_type}: {a.description} (Severity: {a.severity})"
            for a in anomalies
        ]) if anomalies else "No anomalies detected."
        
        prompt = f"""You are a medical monitoring assistant analyzing vital signs from a radar-based contactless monitor.

Current Vital Signs:
- Heart Rate: {vital_signs.heart_rate_bpm:.1f} BPM (confidence: {vital_signs.heart_rate_confidence:.0%})
- Breathing Rate: {vital_signs.breathing_rate_bpm:.1f} breaths/min (confidence: {vital_signs.breathing_rate_confidence:.0%})

Normal Ranges:
- Heart Rate: 60-100 BPM (resting adult)
- Breathing Rate: 12-20 breaths/min (resting adult)

Detected Anomalies:
{anomaly_descriptions}

Risk Level: {risk_level}

Provide a brief (2-3 sentences), clear summary suitable for a caregiver or family member. 
Be reassuring if values are normal, or clearly explain any concerns without causing panic.
Do NOT provide medical advice or diagnosis - only summarize the measurements."""

        # Try Gemini API
        if self.gemini_model and self._initialized:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"⚠ Gemini API error: {e}")
        
        # Fallback to rule-based summary
        return self._generate_fallback_summary(vital_signs, anomalies, risk_level)
    
    def _generate_fallback_summary(
        self,
        vital_signs: VitalSigns,
        anomalies: List[Anomaly],
        risk_level: str,
    ) -> str:
        """Generate summary without AI (fallback)."""
        hr = vital_signs.heart_rate_bpm
        br = vital_signs.breathing_rate_bpm
        
        if risk_level == "NORMAL":
            return (
                f"Vital signs are within normal range. "
                f"Heart rate is {hr:.0f} BPM and breathing rate is {br:.0f} breaths per minute. "
                f"No concerns detected."
            )
        elif risk_level == "CRITICAL":
            critical = [a for a in anomalies if a.severity == Severity.CRITICAL.value]
            return (
                f"⚠️ CRITICAL ALERT: {critical[0].description}. "
                f"Heart rate: {hr:.0f} BPM, Breathing rate: {br:.0f} breaths/min. "
                f"Immediate attention recommended."
            )
        else:
            return (
                f"Monitoring alert: {anomalies[0].description}. "
                f"Current readings: HR {hr:.0f} BPM, BR {br:.0f} breaths/min. "
                f"Risk level: {risk_level}."
            )


# ============================================================================
# STREAM PROCESSOR
# ============================================================================

class VitalSignsStreamProcessor:
    """
    Processes vital signs stream from Kafka with anomaly detection.
    """
    
    def __init__(
        self,
        detector: VitalSignsAnomalyDetector,
        alert_callback: Optional[callable] = None,
    ):
        self.detector = detector
        self.alert_callback = alert_callback
        self._processing = False
        
    def process_vital_signs(self, vital_signs: VitalSigns) -> HealthSummary:
        """Process a single vital signs reading."""
        # Detect anomalies
        anomalies = self.detector.detect_anomalies(vital_signs)
        
        # Generate summary
        summary = self.detector.generate_health_summary(vital_signs, anomalies)
        
        # Trigger alerts for high-severity anomalies
        if self.alert_callback and anomalies:
            critical_anomalies = [
                a for a in anomalies 
                if a.severity in [Severity.HIGH.value, Severity.CRITICAL.value]
            ]
            if critical_anomalies:
                self.alert_callback(summary)
        
        return summary


# ============================================================================
# MAIN (DEMO)
# ============================================================================

def demo():
    """Demonstrate anomaly detection capabilities."""
    print("=" * 60)
    print("VitalFlow-Radar Anomaly Detection Demo")
    print("=" * 60)
    
    detector = VitalSignsAnomalyDetector()
    detector.initialize()
    
    # Test cases
    test_cases = [
        ("Normal", 72, 0.9, 16, 0.85),
        ("Bradycardia", 52, 0.88, 14, 0.82),
        ("Tachycardia", 115, 0.85, 18, 0.80),
        ("Severe Bradycardia", 35, 0.75, 10, 0.70),
        ("Apnea Risk", 68, 0.82, 5, 0.65),
        ("Tachypnea", 88, 0.90, 28, 0.88),
        ("Low Signal", 70, 0.35, 15, 0.40),
    ]
    
    for name, hr, hr_conf, br, br_conf in test_cases:
        print(f"\n--- Test Case: {name} ---")
        
        vital_signs = VitalSigns(
            timestamp=time.time(),
            heart_rate_bpm=hr,
            heart_rate_confidence=hr_conf,
            breathing_rate_bpm=br,
            breathing_rate_confidence=br_conf,
            device_id="demo-radar-001",
        )
        
        anomalies = detector.detect_anomalies(vital_signs)
        summary = detector.generate_health_summary(vital_signs, anomalies)
        
        print(f"HR: {hr} BPM, BR: {br} breaths/min")
        print(f"Risk Level: {summary.risk_level}")
        print(f"Anomalies: {len(anomalies)}")
        for a in anomalies:
            print(f"  - {a.anomaly_type}: {a.description}")
        print(f"Summary: {summary.summary_text}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == '__main__':
    demo()
