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
    
    def generate(self, patient_id: str = "default") -> dict:
        """Generate simulated vital signs."""
        now = time.time()
        self.phase += 0.1
        
        # Add realistic variations
        hr_noise = random.gauss(0, 1.5)
        br_noise = random.gauss(0, 0.5)
        
        # Circadian-like pattern
        hr_circadian = 3 * math.sin(self.phase * 0.05)
        br_circadian = 1 * math.sin(self.phase * 0.03)
        
        heart_rate = self.base_hr + hr_circadian + hr_noise
        breathing_rate = self.base_br + br_circadian + br_noise
        
        # Clamp to realistic ranges
        heart_rate = max(55, min(110, heart_rate))
        breathing_rate = max(10, min(22, breathing_rate))
        
        # Calculate confidence based on variation
        hr_confidence = max(0.7, min(0.99, 0.95 - abs(hr_noise) * 0.05))
        br_confidence = max(0.7, min(0.99, 0.95 - abs(br_noise) * 0.1))
        
        # Determine status
        status = "normal"
        alerts = []
        
        if heart_rate < 50 or heart_rate > 120:
            status = "critical"
            alerts.append({
                "type": "heart_rate",
                "severity": "high",
                "message": f"Heart rate {'too low' if heart_rate < 50 else 'too high'}: {heart_rate:.0f} BPM"
            })
        elif heart_rate < 55 or heart_rate > 100:
            status = "warning"
            alerts.append({
                "type": "heart_rate",
                "severity": "medium",
                "message": f"Heart rate outside normal range: {heart_rate:.0f} BPM"
            })
        
        if breathing_rate < 8 or breathing_rate > 25:
            status = "critical"
            alerts.append({
                "type": "breathing_rate",
                "severity": "high",
                "message": f"Breathing rate {'too low' if breathing_rate < 8 else 'too high'}: {breathing_rate:.1f}/min"
            })
        
        return {
            "patient_id": patient_id,
            "heart_rate": round(heart_rate, 1),
            "heart_rate_confidence": round(hr_confidence, 2),
            "breathing_rate": round(breathing_rate, 1),
            "breathing_rate_confidence": round(br_confidence, 2),
            "status": status,
            "timestamp": now,
            "alerts": alerts
        }

# Global simulator instance
simulator = VitalSignsSimulator()

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
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mode": "demo",
        "timestamp": time.time(),
        "version": "1.0.0",
        "demo_mode": True
    }

@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "mode": "demo",
        "radar_available": False,
        "kafka_available": False,
        "vertex_ai_available": False,
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
    return simulator.generate(patient_id)

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
    user: dict = Depends(get_current_user)
):
    """Get alerts."""
    # Generate some demo alerts
    demo_alerts = [
        {
            "id": "alert-1",
            "patient_id": "patient-2",
            "timestamp": time.time() - 3600,
            "alert_type": "heart_rate",
            "severity": "medium",
            "message": "Elevated heart rate detected: 98 BPM",
            "acknowledged": True
        },
        {
            "id": "alert-2",
            "patient_id": "patient-1",
            "timestamp": time.time() - 1800,
            "alert_type": "breathing_rate",
            "severity": "low",
            "message": "Slight breathing irregularity detected",
            "acknowledged": False
        }
    ]
    
    result = demo_alerts + alerts_store
    
    if patient_id:
        result = [a for a in result if a["patient_id"] == patient_id]
    if acknowledged is not None:
        result = [a for a in result if a["acknowledged"] == acknowledged]
    
    return result

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: dict = Depends(get_current_user)):
    """Acknowledge an alert."""
    for alert in alerts_store:
        if alert["id"] == alert_id:
            alert["acknowledged"] = True
            return {"status": "acknowledged"}
    return {"status": "acknowledged"}  # Demo - always succeed

# ============================================================================
# VERTEX AI / AI ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/api/vertex-ai/analyze")
async def analyze_vitals(
    patient_id: str = Query("default"),
    user: dict = Depends(get_current_user)
):
    """AI analysis of vital signs (simulated for demo)."""
    vitals = simulator.generate(patient_id)
    
    # Generate AI-like analysis
    hr = vitals["heart_rate"]
    br = vitals["breathing_rate"]
    
    analysis = {
        "patient_id": patient_id,
        "timestamp": time.time(),
        "vital_signs": {
            "heart_rate": hr,
            "breathing_rate": br
        },
        "analysis": {
            "overall_status": vitals["status"],
            "heart_rate_assessment": "Normal" if 60 <= hr <= 100 else ("Elevated" if hr > 100 else "Low"),
            "breathing_rate_assessment": "Normal" if 12 <= br <= 20 else ("Elevated" if br > 20 else "Low"),
            "risk_score": random.uniform(0.1, 0.3) if vitals["status"] == "normal" else random.uniform(0.4, 0.7),
        },
        "recommendations": [
            "Continue regular monitoring",
            "Vital signs within acceptable ranges" if vitals["status"] == "normal" else "Review patient condition",
        ],
        "ai_confidence": 0.92,
        "model": "VitalFlow-AI-Demo"
    }
    
    return analysis

@app.get("/api/vertex-ai/summary")
async def get_health_summary(
    patient_id: str = Query("default"),
    user: dict = Depends(get_current_user)
):
    """Get AI health summary."""
    return {
        "patient_id": patient_id,
        "summary": "Patient vital signs have been stable over the monitoring period. Heart rate and breathing patterns are within normal ranges with minor variations typical of normal physiological activity.",
        "trends": {
            "heart_rate": "stable",
            "breathing_rate": "stable"
        },
        "last_updated": time.time(),
        "confidence": 0.89
    }

# ============================================================================
# KAFKA STATUS (DEMO)
# ============================================================================

@app.get("/api/kafka/status")
async def get_kafka_status(user: dict = Depends(get_current_user)):
    """Get Kafka connection status (demo mode)."""
    return {
        "connected": False,
        "mode": "demo",
        "message": "Kafka not available in demo mode - using simulated data",
        "topics": []
    }

# ============================================================================
# ROOT REDIRECT
# ============================================================================

@app.get("/api")
async def api_root():
    """API root."""
    return {
        "name": "VitalFlow-Radar API",
        "version": "1.0.0",
        "mode": "demo",
        "docs": "/api/docs",
        "health": "/api/health"
    }
