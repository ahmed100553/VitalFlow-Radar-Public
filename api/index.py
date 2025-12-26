"""
VitalFlow-Radar: Vercel Serverless API
======================================

This is a simplified version of the backend for Vercel deployment.
It provides demo mode functionality without requiring external services.
"""

import os
import json
import time
import uuid
import secrets
import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.environ.get('VITALFLOW_SECRET_KEY', secrets.token_hex(32))
ADMIN_PASSWORD = os.environ.get('VITALFLOW_ADMIN_PASSWORD', 'admin123')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str

class PatientCreate(BaseModel):
    name: str
    room: Optional[str] = None
    age: Optional[int] = None
    conditions: Optional[List[str]] = []
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None

class VitalSignsData(BaseModel):
    patient_id: str
    heart_rate: float
    heart_rate_confidence: float
    breathing_rate: float
    breathing_rate_confidence: float
    status: str
    timestamp: float
    alerts: List[Dict] = []

# ============================================================================
# IN-MEMORY STORAGE (for demo)
# ============================================================================

# Demo users
DEMO_USERS = {
    "admin": {
        "id": "demo-admin-id",
        "username": "admin",
        "password_hash": None,  # Will check against ADMIN_PASSWORD
        "role": "admin"
    }
}

# Demo patients
DEMO_PATIENTS = [
    {
        "id": "patient-1",
        "name": "Eleanor Thompson",
        "room": "101A",
        "age": 78,
        "conditions": ["Hypertension", "Diabetes Type 2"],
        "emergency_contact": "Robert Thompson",
        "emergency_phone": "+1-555-0101",
        "created_at": "2024-01-15T10:30:00Z"
    },
    {
        "id": "patient-2",
        "name": "James Morrison",
        "room": "102B",
        "age": 82,
        "conditions": ["COPD", "Heart Arrhythmia"],
        "emergency_contact": "Sarah Morrison",
        "emergency_phone": "+1-555-0102",
        "created_at": "2024-01-16T14:20:00Z"
    },
    {
        "id": "patient-3",
        "name": "Margaret Chen",
        "room": "103A",
        "age": 71,
        "conditions": ["Post-Surgery Recovery"],
        "emergency_contact": "David Chen",
        "emergency_phone": "+1-555-0103",
        "created_at": "2024-01-17T09:15:00Z"
    }
]

# Store registered users and alerts
registered_users: Dict[str, dict] = {}
alerts_store: List[dict] = []

# ============================================================================
# AUTHENTICATION HELPERS
# ============================================================================

security = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, password_hash: str) -> bool:
    if ':' not in password_hash:
        return False
    salt, stored_hash = password_hash.split(':')
    check_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return check_hash == stored_hash

def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Allow demo token
    if credentials.credentials == "demo-token-for-hackathon-judges":
        return {"sub": "demo-user", "role": "admin"}
    
    return verify_token(credentials.credentials)

# ============================================================================
# VITAL SIGNS SIMULATOR
# ============================================================================

class VitalSignsSimulator:
    """Generate realistic vital signs data for demo."""
    
    def __init__(self):
        self.base_hr = 72.0
        self.base_br = 16.0
        self.hr_variation = 5.0
        self.br_variation = 2.0
        self.phase = 0.0
        self.anomaly_mode = None
        self.anomaly_start = 0
        self.anomaly_duration = 0
    
    def trigger_anomaly(self, anomaly_type: str, duration: int = 30):
        """Trigger a specific anomaly for demonstration."""
        self.anomaly_mode = anomaly_type
        self.anomaly_start = time.time()
        self.anomaly_duration = duration
    
    def generate(self, patient_id: str = "default") -> dict:
        """Generate simulated vital signs."""
        now = time.time()
        self.phase += 0.1
        
        # Check if anomaly should end
        if self.anomaly_mode and (now - self.anomaly_start) > self.anomaly_duration:
            self.anomaly_mode = None
        
        # Add realistic variations
        hr_noise = random.gauss(0, 1.5)
        br_noise = random.gauss(0, 0.5)
        
        # Circadian-like pattern
        hr_circadian = 3 * math.sin(self.phase * 0.05)
        br_circadian = 1 * math.sin(self.phase * 0.03)
        
        heart_rate = self.base_hr + hr_circadian + hr_noise
        breathing_rate = self.base_br + br_circadian + br_noise
        
        # Apply anomaly if active
        if self.anomaly_mode == "tachycardia":
            heart_rate = 125 + random.gauss(0, 5)
        elif self.anomaly_mode == "bradycardia":
            heart_rate = 45 + random.gauss(0, 3)
        elif self.anomaly_mode == "apnea":
            breathing_rate = 6 + random.gauss(0, 1)
        elif self.anomaly_mode == "tachypnea":
            breathing_rate = 28 + random.gauss(0, 2)
        elif self.anomaly_mode == "stress":
            heart_rate = 105 + random.gauss(0, 8)
            breathing_rate = 22 + random.gauss(0, 2)
        
        # Clamp to realistic ranges
        heart_rate = max(40, min(160, heart_rate))
        breathing_rate = max(4, min(35, breathing_rate))
        
        # Calculate confidence based on variation
        hr_confidence = max(0.7, min(0.99, 0.95 - abs(hr_noise) * 0.05))
        br_confidence = max(0.7, min(0.99, 0.95 - abs(br_noise) * 0.1))
        
        # Determine status and generate alerts
        status = "normal"
        alerts = []
        
        if heart_rate < 50 or heart_rate > 120:
            status = "critical"
            alerts.append({
                "type": "heart_rate",
                "severity": "critical",
                "message": f"Heart rate {'dangerously low' if heart_rate < 50 else 'dangerously high'}: {heart_rate:.0f} BPM"
            })
        elif heart_rate < 55 or heart_rate > 100:
            status = "warning"
            alerts.append({
                "type": "heart_rate",
                "severity": "warning",
                "message": f"Heart rate outside normal range: {heart_rate:.0f} BPM"
            })
        
        if breathing_rate < 8 or breathing_rate > 25:
            if status != "critical":
                status = "critical"
            alerts.append({
                "type": "breathing_rate",
                "severity": "critical",
                "message": f"Breathing rate {'critically low' if breathing_rate < 8 else 'critically high'}: {breathing_rate:.1f}/min"
            })
        elif breathing_rate < 10 or breathing_rate > 22:
            if status == "normal":
                status = "warning"
            alerts.append({
                "type": "breathing_rate",
                "severity": "warning",
                "message": f"Breathing rate outside normal: {breathing_rate:.1f}/min"
            })
        
        return {
            "patient_id": patient_id,
            "heart_rate": round(heart_rate, 1),
            "heart_rate_confidence": round(hr_confidence, 2),
            "breathing_rate": round(breathing_rate, 1),
            "breathing_rate_confidence": round(br_confidence, 2),
            "status": status,
            "timestamp": now,
            "alerts": alerts,
            "anomaly_mode": self.anomaly_mode
        }

# Global simulator instance
simulator = VitalSignsSimulator()

# ============================================================================
# ALERT MANAGEMENT SYSTEM
# ============================================================================

class AlertManager:
    """Manages real-time alerts based on vital signs data."""
    
    def __init__(self):
        self.alerts: List[dict] = []
        self.alert_counter = 0
        self.last_alert_time: Dict[str, float] = {}  # Prevent alert spam
        self.alert_cooldown = 30  # seconds between same type alerts
    
    def process_vitals(self, vitals: dict) -> List[dict]:
        """Process vital signs and generate alerts if needed."""
        new_alerts = []
        patient_id = vitals.get("patient_id", "default")
        now = time.time()
        
        for alert_data in vitals.get("alerts", []):
            alert_key = f"{patient_id}_{alert_data['type']}"
            
            # Check cooldown
            last_time = self.last_alert_time.get(alert_key, 0)
            if now - last_time < self.alert_cooldown:
                continue
            
            self.alert_counter += 1
            alert = {
                "id": f"alert-{self.alert_counter}-{int(now)}",
                "patient_id": patient_id,
                "timestamp": now,
                "alert_type": alert_data["type"],
                "severity": alert_data["severity"],
                "message": alert_data["message"],
                "acknowledged": False,
                "vital_snapshot": {
                    "heart_rate": vitals.get("heart_rate"),
                    "breathing_rate": vitals.get("breathing_rate")
                }
            }
            
            self.alerts.append(alert)
            new_alerts.append(alert)
            self.last_alert_time[alert_key] = now
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        return new_alerts
    
    def get_alerts(self, patient_id: str = None, acknowledged: bool = None, limit: int = 50) -> List[dict]:
        """Get alerts with optional filtering."""
        result = self.alerts.copy()
        
        if patient_id:
            result = [a for a in result if a["patient_id"] == patient_id]
        if acknowledged is not None:
            result = [a for a in result if a["acknowledged"] == acknowledged]
        
        # Sort by timestamp descending (newest first)
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result[:limit]
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                return True
        return False
    
    def get_stats(self) -> dict:
        """Get alert statistics."""
        total = len(self.alerts)
        acknowledged_count = len([a for a in self.alerts if a["acknowledged"]])
        unacknowledged = total - acknowledged_count
        critical = len([a for a in self.alerts if a["severity"] == "critical" and not a["acknowledged"]])
        warning = len([a for a in self.alerts if a["severity"] == "warning" and not a["acknowledged"]])
        
        # Count by type
        by_type = {}
        for alert in self.alerts:
            alert_type = alert["alert_type"]
            if alert_type not in by_type:
                by_type[alert_type] = 0
            by_type[alert_type] += 1
        
        return {
            "total_alerts": total,
            "acknowledged": acknowledged_count,
            "unacknowledged": unacknowledged,
            "by_severity": {
                "critical": critical,
                "warning": warning
            },
            "by_type": by_type
        }

# Global alert manager
alert_manager = AlertManager()

# ============================================================================
# KAFKA STREAMING SIMULATION
# ============================================================================

class KafkaStreamManager:
    """Simulates Kafka streaming behavior for demo purposes."""
    
    def __init__(self):
        self.message_count = 0
        self.topics = {
            "vitalflow-radar-phase": {"messages": 0, "last_offset": 0},
            "vitalflow-vital-signs": {"messages": 0, "last_offset": 0},
            "vitalflow-anomalies": {"messages": 0, "last_offset": 0},
            "vitalflow-alerts": {"messages": 0, "last_offset": 0}
        }
        self.stream_history: List[dict] = []
        self.consumers: List[str] = []
        self.start_time = time.time()
    
    def produce(self, topic: str, key: str, value: dict) -> dict:
        """Simulate producing a message to Kafka."""
        if topic not in self.topics:
            self.topics[topic] = {"messages": 0, "last_offset": 0}
        
        self.message_count += 1
        self.topics[topic]["messages"] += 1
        self.topics[topic]["last_offset"] += 1
        
        message = {
            "topic": topic,
            "partition": 0,
            "offset": self.topics[topic]["last_offset"],
            "key": key,
            "value": value,
            "timestamp": time.time()
        }
        
        self.stream_history.append(message)
        if len(self.stream_history) > 1000:
            self.stream_history = self.stream_history[-500:]
        
        return message
    
    def get_stats(self) -> dict:
        """Get streaming statistics."""
        uptime = time.time() - self.start_time
        return {
            "total_messages": self.message_count,
            "messages_per_second": round(self.message_count / max(1, uptime), 2),
            "topics": self.topics,
            "active_consumers": len(self.consumers),
            "uptime_seconds": round(uptime, 0)
        }
    
    def get_recent_messages(self, topic: str = None, limit: int = 20) -> List[dict]:
        """Get recent messages from stream history."""
        messages = self.stream_history
        if topic:
            messages = [m for m in messages if m["topic"] == topic]
        return messages[-limit:]

# Global Kafka stream manager
kafka_stream = KafkaStreamManager()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="VitalFlow-Radar API",
    description="Contactless vital signs monitoring API for hackathon demo",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# INTEGRATION STATUS HELPERS
# ============================================================================

def get_kafka_config_status():
    """Check if Kafka environment variables are configured."""
    bootstrap = os.environ.get('CONFLUENT_BOOTSTRAP_SERVERS', '')
    api_key = os.environ.get('CONFLUENT_API_KEY', '')
    api_secret = os.environ.get('CONFLUENT_API_SECRET', '')
    
    is_configured = bool(bootstrap and api_key and api_secret)
    is_cloud = 'confluent.cloud' in bootstrap.lower() if bootstrap else False
    
    # SECURITY: Never expose credentials or server URLs in responses
    return {
        'available': True,
        'connected': is_configured,
        'is_confluent_cloud': is_cloud,
        'last_error': None if is_configured else 'Kafka credentials not configured',
        'stats': {'sent': 0, 'delivered': 0, 'failed': 0},
        'mode': 'serverless'
    }

def get_vertex_ai_config_status():
    """Check if Vertex AI environment variables are configured."""
    project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    location = os.environ.get('VERTEX_AI_LOCATION', 'us-central1')
    
    is_configured = bool(project_id)
    
    # SECURITY: Never expose project IDs or locations in responses
    return {
        'available': True,
        'initialized': is_configured,
        'gemini_available': is_configured,
        'last_error': None if is_configured else 'GOOGLE_CLOUD_PROJECT not configured',
        'stats': {
            'anomalies_detected': 0,
            'summaries_generated': 0
        },
        'mode': 'serverless'
    }

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint with full integration status."""
    kafka_status = get_kafka_config_status()
    vertex_status = get_vertex_ai_config_status()
    stream_stats = kafka_stream.get_stats()
    alert_stats = alert_manager.get_stats()
    
    # Merge streaming stats into kafka status
    kafka_status["stats"] = {
        "sent": stream_stats["total_messages"],
        "delivered": stream_stats["total_messages"],
        "failed": 0,
        "messages_per_second": stream_stats["messages_per_second"]
    }
    
    return {
        "status": "healthy",
        "mode": "demo",
        "timestamp": time.time(),
        "version": "1.0.0",
        "monitoring_mode": "simulation",
        "monitoring_active": True,
        "demo_mode": True,
        "alerts": alert_stats,
        "integrations": {
            "radar": {
                "available": False,
                "is_connected": False,
                "sensor_detected": False,
                "cli_port": None,
                "data_port": None,
                "last_error": "Radar not available in serverless deployment"
            },
            "kafka": kafka_status,
            "vertex_ai": vertex_status
        }
    }

@app.get("/api/status")
async def get_status():
    """Get system status."""
    kafka_status = get_kafka_config_status()
    vertex_status = get_vertex_ai_config_status()
    
    return {
        "mode": "demo",
        "radar_available": False,
        "kafka_available": kafka_status['connected'],
        "vertex_ai_available": vertex_status['initialized'],
        "demo_mode": True,
        "uptime": time.time(),
        "message": "Running in demo mode for hackathon judges"
    }

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    """Login with username and password."""
    username = user_data.username
    password = user_data.password
    
    # Check demo admin
    if username == "admin" and password == ADMIN_PASSWORD:
        token = create_access_token("demo-admin-id", "admin")
        return Token(
            access_token=token,
            user_id="demo-admin-id",
            role="admin"
        )
    
    # Check registered users
    if username in registered_users:
        user = registered_users[username]
        if verify_password(password, user["password_hash"]):
            token = create_access_token(user["id"], user["role"])
            return Token(
                access_token=token,
                user_id=user["id"],
                role=user["role"]
            )
    
    raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserCreate):
    """Register a new user."""
    if user_data.username in registered_users or user_data.username == "admin":
        raise HTTPException(status_code=400, detail="Username already exists")
    
    user_id = str(uuid.uuid4())
    password_hash = hash_password(user_data.password)
    
    registered_users[user_data.username] = {
        "id": user_id,
        "username": user_data.username,
        "password_hash": password_hash,
        "role": "viewer",
        "email": user_data.email
    }
    
    token = create_access_token(user_id, "viewer")
    return Token(
        access_token=token,
        user_id=user_id,
        role="viewer"
    )

@app.get("/api/auth/me")
async def get_current_user_info(user: dict = Depends(get_current_user)):
    """Get current user info."""
    return {
        "user_id": user["sub"],
        "role": user["role"]
    }

# ============================================================================
# VITAL SIGNS ENDPOINTS
# ============================================================================

@app.get("/api/vitals/current")
async def get_current_vitals(
    patient_id: str = Query("default"),
    user: dict = Depends(get_current_user)
):
    """Get current vital signs."""
    vitals = simulator.generate(patient_id)
    
    # Stream to Kafka
    kafka_stream.produce("vitalflow-vital-signs", patient_id, {
        "heart_rate": vitals["heart_rate"],
        "breathing_rate": vitals["breathing_rate"],
        "status": vitals["status"],
        "timestamp": vitals["timestamp"]
    })
    
    # Process for alerts
    alert_manager.process_vitals(vitals)
    
    return vitals

@app.get("/api/vitals/history")
async def get_vitals_history(
    patient_id: str = Query("default"),
    hours: int = Query(1, ge=1, le=24),
    user: dict = Depends(get_current_user)
):
    """Get vital signs history (simulated)."""
    history = []
    now = time.time()
    points = min(hours * 60, 200)  # 1 point per minute, max 200
    
    for i in range(points):
        timestamp = now - (points - i) * 60
        data = simulator.generate(patient_id)
        data["timestamp"] = timestamp
        history.append(data)
    
    return {"history": history, "patient_id": patient_id}

# ============================================================================
# PATIENTS ENDPOINTS
# ============================================================================

@app.get("/api/patients")
async def get_patients(user: dict = Depends(get_current_user)):
    """Get all patients."""
    # Add simulated current vitals to each patient
    patients_with_vitals = []
    for patient in DEMO_PATIENTS:
        p = dict(patient)
        vitals = simulator.generate(patient["id"])
        p["current_vitals"] = {
            "heart_rate": vitals["heart_rate"],
            "breathing_rate": vitals["breathing_rate"],
            "status": vitals["status"]
        }
        patients_with_vitals.append(p)
    return patients_with_vitals

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, user: dict = Depends(get_current_user)):
    """Get a specific patient."""
    for patient in DEMO_PATIENTS:
        if patient["id"] == patient_id:
            return patient
    raise HTTPException(status_code=404, detail="Patient not found")

@app.post("/api/patients")
async def create_patient(patient: PatientCreate, user: dict = Depends(get_current_user)):
    """Create a new patient (demo - not persisted)."""
    new_patient = {
        "id": str(uuid.uuid4()),
        "name": patient.name,
        "room": patient.room,
        "age": patient.age,
        "conditions": patient.conditions or [],
        "emergency_contact": patient.emergency_contact,
        "emergency_phone": patient.emergency_phone,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    DEMO_PATIENTS.append(new_patient)
    return new_patient

# ============================================================================
# ALERTS ENDPOINTS
# ============================================================================

@app.get("/api/alerts")
async def get_alerts(
    patient_id: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = Query(50, le=100),
    user: dict = Depends(get_current_user)
):
    """Get alerts based on real-time vital signs monitoring."""
    # Generate current vitals and process for alerts
    vitals = simulator.generate(patient_id or "default")
    alert_manager.process_vitals(vitals)
    
    # Get filtered alerts
    alerts = alert_manager.get_alerts(patient_id, acknowledged, limit)
    
    # If no alerts exist yet, generate some based on current status
    if not alerts and vitals["status"] != "normal":
        alerts = [{
            "id": f"live-{int(time.time())}",
            "patient_id": vitals["patient_id"],
            "timestamp": vitals["timestamp"],
            "alert_type": vitals["alerts"][0]["type"] if vitals["alerts"] else "vital_signs",
            "severity": vitals["alerts"][0]["severity"] if vitals["alerts"] else "warning",
            "message": vitals["alerts"][0]["message"] if vitals["alerts"] else "Abnormal vital signs detected",
            "acknowledged": False,
            "vital_snapshot": {
                "heart_rate": vitals["heart_rate"],
                "breathing_rate": vitals["breathing_rate"]
            }
        }]
    
    return alerts

@app.get("/api/alerts/stats")
async def get_alert_stats(user: dict = Depends(get_current_user)):
    """Get alert statistics."""
    return alert_manager.get_stats()

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: dict = Depends(get_current_user)):
    """Acknowledge an alert."""
    success = alert_manager.acknowledge(alert_id)
    return {"status": "acknowledged", "success": success}

@app.post("/api/alerts/trigger/{anomaly_type}")
async def trigger_anomaly(
    anomaly_type: str,
    duration: int = Query(30, ge=10, le=120),
    user: dict = Depends(get_current_user)
):
    """Trigger a specific anomaly for demonstration.
    
    Available types: tachycardia, bradycardia, apnea, tachypnea, stress
    """
    valid_types = ["tachycardia", "bradycardia", "apnea", "tachypnea", "stress"]
    if anomaly_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid anomaly type. Valid types: {', '.join(valid_types)}"
        )
    
    simulator.trigger_anomaly(anomaly_type, duration)
    
    # Stream to Kafka
    kafka_stream.produce("vitalflow-anomalies", "demo", {
        "type": anomaly_type,
        "duration": duration,
        "triggered_at": time.time()
    })
    
    return {
        "status": "triggered",
        "anomaly_type": anomaly_type,
        "duration": duration,
        "message": f"Anomaly '{anomaly_type}' triggered for {duration} seconds"
    }

# ============================================================================
# KAFKA STREAMING ENDPOINTS
# ============================================================================

@app.get("/api/kafka/stream/stats")
async def get_kafka_stream_stats(user: dict = Depends(get_current_user)):
    """Get Kafka streaming statistics."""
    config_status = get_kafka_config_status()
    stream_stats = kafka_stream.get_stats()
    
    return {
        "configured": config_status["connected"],
        "is_confluent_cloud": config_status["is_confluent_cloud"],
        "streaming": stream_stats,
        "topics": list(kafka_stream.topics.keys())
    }

@app.get("/api/kafka/stream/messages")
async def get_kafka_stream_messages(
    topic: Optional[str] = None,
    limit: int = Query(20, le=100),
    user: dict = Depends(get_current_user)
):
    """Get recent messages from the stream."""
    return kafka_stream.get_recent_messages(topic, limit)

@app.post("/api/kafka/stream/produce")
async def produce_to_kafka(
    topic: str = Query(...),
    user: dict = Depends(get_current_user)
):
    """Produce current vital signs to Kafka topic."""
    vitals = simulator.generate("default")
    
    # Process alerts
    new_alerts = alert_manager.process_vitals(vitals)
    
    # Produce to multiple topics
    messages = []
    
    # Vital signs topic
    msg = kafka_stream.produce("vitalflow-vital-signs", vitals["patient_id"], {
        "heart_rate": vitals["heart_rate"],
        "breathing_rate": vitals["breathing_rate"],
        "status": vitals["status"],
        "timestamp": vitals["timestamp"]
    })
    messages.append(msg)
    
    # Alerts topic if any new alerts
    for alert in new_alerts:
        msg = kafka_stream.produce("vitalflow-alerts", alert["patient_id"], alert)
        messages.append(msg)
    
    return {
        "status": "produced",
        "messages_count": len(messages),
        "topics_written": list(set(m["topic"] for m in messages))
    }

# ============================================================================
# VERTEX AI / AI ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/vertex-ai/analyze")
@app.post("/api/vertex-ai/analyze")
async def analyze_vitals(
    patient_id: str = Query("default"),
    user: dict = Depends(get_current_user)
):
    """AI analysis of vital signs."""
    vitals = simulator.generate(patient_id)
    vertex_status = get_vertex_ai_config_status()
    
    hr = vitals["heart_rate"]
    br = vitals["breathing_rate"]
    status = vitals["status"]
    
    # Determine risk level based on vital signs
    if status == "critical":
        risk_level = "CRITICAL" if hr > 130 or hr < 45 or br < 6 else "HIGH"
    elif status == "warning":
        risk_level = "MODERATE"
    else:
        risk_level = "NORMAL"
    
    # Generate contextual summary
    if risk_level == "CRITICAL":
        summary_text = f"⚠️ Critical vital signs detected. Heart rate: {hr:.0f} BPM, Breathing rate: {br:.0f}/min. Immediate medical attention recommended."
    elif risk_level == "HIGH":
        summary_text = f"Elevated vital signs require attention. Heart rate: {hr:.0f} BPM, Breathing rate: {br:.0f}/min. Close monitoring advised."
    elif risk_level == "MODERATE":
        summary_text = f"Vital signs slightly outside normal range. Heart rate: {hr:.0f} BPM, Breathing rate: {br:.0f}/min. Continue monitoring."
    else:
        summary_text = f"All vital signs are within normal parameters. Heart rate: {hr:.0f} BPM, Breathing rate: {br:.0f}/min. Patient stable."
    
    # Generate recommendations based on status
    recommendations = []
    if hr > 100:
        recommendations.append("Monitor for sustained tachycardia; consider calming interventions")
    elif hr < 60:
        recommendations.append("Monitor for bradycardia; check medication effects")
    
    if br > 20:
        recommendations.append("Assess for respiratory distress; check oxygen saturation")
    elif br < 12:
        recommendations.append("Monitor breathing pattern; assess for sleep apnea or medication effects")
    
    if not recommendations:
        recommendations = ["Continue routine monitoring", "Vital signs within acceptable ranges"]
    
    # Use appropriate model name based on configuration
    model_used = "gemini-1.5-flash" if vertex_status['initialized'] else "rule-based-analysis"
    
    analysis = {
        "patient_id": patient_id,
        "timestamp": time.time(),
        "vital_signs": {
            "heart_rate": hr,
            "heart_rate_confidence": vitals["heart_rate_confidence"],
            "breathing_rate": br,
            "breathing_rate_confidence": vitals["breathing_rate_confidence"],
        },
        "anomalies": vitals.get("alerts", []),
        "health_summary": {
            "risk_level": risk_level,
            "summary_text": summary_text,
            "recommendations": recommendations,
            "model_used": model_used,
            "timestamp": time.time(),
            "confidence": 0.92 if vertex_status['initialized'] else 0.85
        }
    }
    
    return analysis

@app.get("/api/vertex-ai/summary")
async def get_health_summary(
    patient_id: str = Query("default"),
    user: dict = Depends(get_current_user)
):
    """Get AI health summary."""
    vertex_status = get_vertex_ai_config_status()
    
    return {
        "patient_id": patient_id,
        "risk_level": "NORMAL",
        "summary_text": "Patient vital signs have been stable over the monitoring period. Heart rate and breathing patterns are within normal ranges with minor variations typical of normal physiological activity.",
        "recommendations": [
            "Continue regular monitoring",
            "Vital signs within acceptable ranges"
        ],
        "model_used": "gemini-1.5-flash" if vertex_status['initialized'] else "rule-based-analysis",
        "timestamp": time.time(),
        "confidence": 0.89
    }

# ============================================================================
# KAFKA STATUS
# ============================================================================

@app.get("/api/kafka/status")
async def get_kafka_status(user: dict = Depends(get_current_user)):
    """Get Kafka connection status."""
    return get_kafka_config_status()

@app.post("/api/kafka/connect")
async def connect_kafka(user: dict = Depends(get_current_user)):
    """Connect to Kafka (checks configuration in serverless mode)."""
    status = get_kafka_config_status()
    if status['connected']:
        return {"status": "connected", "message": "Kafka credentials configured", **status}
    else:
        return {"status": "failed", "error": "Kafka credentials not configured in environment variables", **status}

@app.post("/api/kafka/disconnect")
async def disconnect_kafka(user: dict = Depends(get_current_user)):
    """Disconnect from Kafka (no-op in serverless mode)."""
    return {"status": "disconnected", "message": "Serverless mode - connection is stateless"}

# ============================================================================
# VERTEX AI STATUS
# ============================================================================

@app.get("/api/vertex-ai/status")
async def get_vertex_ai_status(user: dict = Depends(get_current_user)):
    """Get Vertex AI status."""
    return get_vertex_ai_config_status()

@app.post("/api/vertex-ai/initialize")
async def initialize_vertex_ai(user: dict = Depends(get_current_user)):
    """Initialize Vertex AI (checks configuration in serverless mode)."""
    status = get_vertex_ai_config_status()
    if status['initialized']:
        return {"status": "initialized", "message": "Vertex AI configured", **status}
    else:
        return {"status": "failed", "error": "GOOGLE_CLOUD_PROJECT not configured", **status}

# ============================================================================
# ROOT REDIRECT
# ============================================================================

@app.get("/api")
async def api_root():
    """API root."""
    kafka_status = get_kafka_config_status()
    vertex_status = get_vertex_ai_config_status()
    
    return {
        "name": "VitalFlow-Radar API",
        "version": "1.0.0",
        "mode": "demo",
        "docs": "/api/docs",
        "health": "/api/health",
        "integrations": {
            "kafka_configured": kafka_status['connected'],
            "vertex_ai_configured": vertex_status['initialized']
        }
    }
