#!/usr/bin/env python3
"""
VitalFlow-Radar: FastAPI Backend
================================

Production-ready FastAPI backend for Raspberry Pi deployment.
Provides REST API and WebSocket endpoints for real-time vital signs monitoring.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import time
import uuid
import asyncio
import hashlib
import secrets
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import jwt
import aiosqlite
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# EXTERNAL INTEGRATIONS AVAILABILITY
# ============================================================================

# Local imports (for radar)
RADAR_AVAILABLE = False
PROCESSOR_AVAILABLE = False
KAFKA_AVAILABLE = False
VERTEX_AI_AVAILABLE = False
GEMINI_AVAILABLE = False

try:
    from awr1642_driver import AWR1642, load_config_from_file, DEFAULT_CONFIG_FILE
    RADAR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Radar driver not available: {e}")

try:
    from vital_signs_processor import VitalSignsProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vital signs processor not available: {e}")

# Confluent Kafka
try:
    from confluent_kafka import Producer, Consumer, KafkaError
    from confluent_config import (
        get_producer_config,
        get_consumer_config,
        TOPICS,
        is_confluent_cloud,
        get_bootstrap_servers,
        print_config_status,
    )
    KAFKA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Confluent Kafka not available: {e}")

# Vertex AI / Gemini
try:
    from vertex_ai_processor import (
        VitalSignsAnomalyDetector,
        VitalSigns as VertexVitalSigns,
        Anomaly as VertexAnomaly,
        HealthSummary,
        AnomalyType,
        Severity,
        NORMAL_RANGES,
    )
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vertex AI processor not available: {e}")

try:
    import google.auth
    from google.cloud import aiplatform
    GEMINI_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings:
    SECRET_KEY: str = os.environ.get('VITALFLOW_SECRET_KEY', secrets.token_hex(32))
    ADMIN_PASSWORD: str = os.environ.get('VITALFLOW_ADMIN_PASSWORD', 'admin123')
    DATABASE_PATH: str = os.environ.get('VITALFLOW_DB', './data/vitalflow.db')
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    
    # Radar settings
    RADAR_CLI_PORT: str = os.environ.get('RADAR_CLI_PORT', '/dev/ttyACM0')
    RADAR_DATA_PORT: str = os.environ.get('RADAR_DATA_PORT', '/dev/ttyACM1')
    RADAR_FPS: int = int(os.environ.get('RADAR_FPS', '10'))
    RADAR_CONFIG_FILE: str = os.environ.get('RADAR_CONFIG', 'vital_signs_awr1642.cfg')
    
    # Operating mode: 'auto', 'kafka', 'radar', 'simulation'
    # 'auto' - tries Kafka first, then radar, then simulation
    # 'kafka' - only Kafka consumer mode (waits for edge producer)
    # 'radar' - only direct radar mode
    # 'simulation' - only simulation mode
    VITALFLOW_MODE: str = os.environ.get('VITALFLOW_MODE', 'auto')
    
    # Vital signs thresholds
    HR_MIN: float = float(os.environ.get('HR_MIN', '50'))
    HR_MAX: float = float(os.environ.get('HR_MAX', '120'))
    BR_MIN: float = float(os.environ.get('BR_MIN', '8'))
    BR_MAX: float = float(os.environ.get('BR_MAX', '25'))
    
    # Google Cloud / Vertex AI
    GOOGLE_CLOUD_PROJECT: str = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
    VERTEX_AI_LOCATION: str = os.environ.get('VERTEX_AI_LOCATION', 'us-central1')
    
    # Confluent Cloud
    CONFLUENT_BOOTSTRAP_SERVERS: str = os.environ.get('CONFLUENT_BOOTSTRAP_SERVERS', '')
    CONFLUENT_API_KEY: str = os.environ.get('CONFLUENT_API_KEY', '')
    CONFLUENT_API_SECRET: str = os.environ.get('CONFLUENT_API_SECRET', '')

settings = Settings()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vitalflow')

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

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

class PatientResponse(BaseModel):
    id: str
    name: str
    room: Optional[str]
    age: Optional[int]
    conditions: Optional[List[str]]
    emergency_contact: Optional[str]
    emergency_phone: Optional[str]
    created_at: str

class VitalSignsData(BaseModel):
    patient_id: str
    heart_rate: float
    heart_rate_confidence: float
    breathing_rate: float
    breathing_rate_confidence: float
    status: str
    timestamp: float
    alerts: List[Dict] = []

class AlertResponse(BaseModel):
    id: str
    patient_id: str
    timestamp: float
    alert_type: str
    severity: str
    message: str
    acknowledged: bool

# ============================================================================
# DATABASE
# ============================================================================

async def init_database():
    """Initialize SQLite database."""
    db_dir = Path(settings.DATABASE_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'viewer',
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                room TEXT,
                age INTEGER,
                conditions TEXT,
                emergency_contact TEXT,
                emergency_phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS vital_signs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                heart_rate REAL,
                heart_rate_confidence REAL,
                breathing_rate REAL,
                breathing_rate_confidence REAL,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        ''')
        
        await db.execute('CREATE INDEX IF NOT EXISTS idx_vitals_patient ON vital_signs(patient_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_vitals_time ON vital_signs(timestamp)')
        await db.commit()
        
        # Create admin user if not exists
        cursor = await db.execute('SELECT id FROM users WHERE username = ?', ('admin',))
        if not await cursor.fetchone():
            admin_id = str(uuid.uuid4())
            password_hash = hash_password(settings.ADMIN_PASSWORD)
            await db.execute(
                'INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)',
                (admin_id, 'admin', password_hash, 'admin')
            )
            await db.commit()
            logger.info("Admin user created")

async def get_db():
    """Get database connection."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db

# ============================================================================
# AUTHENTICATION
# ============================================================================

security = HTTPBearer()

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, password_hash: str) -> bool:
    salt, hashed = password_hash.split(':')
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed

def create_token(user_id: str, role: str) -> str:
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(hours=settings.ACCESS_TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_admin_user(user: dict = Depends(get_current_user)):
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, patient_id: str):
        await websocket.accept()
        if patient_id not in self.active_connections:
            self.active_connections[patient_id] = []
        self.active_connections[patient_id].append(websocket)
        logger.info(f"WebSocket connected for patient {patient_id}")
    
    def disconnect(self, websocket: WebSocket, patient_id: str):
        if patient_id in self.active_connections:
            try:
                self.active_connections[patient_id].remove(websocket)
                if not self.active_connections[patient_id]:
                    del self.active_connections[patient_id]
            except ValueError:
                # Websocket already removed
                pass
        logger.info(f"WebSocket disconnected for patient {patient_id}")
    
    async def broadcast(self, patient_id: str, data: dict):
        if patient_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[patient_id]:
                try:
                    await connection.send_json(data)
                except:
                    dead_connections.append(connection)
            for conn in dead_connections:
                self.disconnect(conn, patient_id)

manager = ConnectionManager()

# ============================================================================
# CONFLUENT KAFKA SERVICE
# ============================================================================

class KafkaService:
    """
    Manages Confluent Kafka connections for streaming vital signs data.
    Produces vital signs to Kafka topics and consumes alerts.
    """
    
    def __init__(self):
        self.producer: Optional[Producer] = None
        self.is_connected = False
        self.last_error: Optional[str] = None
        self._delivery_stats = {'sent': 0, 'delivered': 0, 'failed': 0}
    
    def connect(self) -> bool:
        """Connect to Confluent Kafka/Cloud."""
        if not KAFKA_AVAILABLE:
            self.last_error = "confluent_kafka library not installed"
            return False
        
        try:
            config = get_producer_config(client_id='vitalflow-api-producer')
            self.producer = Producer(config)
            self.is_connected = True
            self.last_error = None
            logger.info(f"Connected to Kafka: {get_bootstrap_servers()}")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Kafka."""
        if self.producer:
            try:
                self.producer.flush(timeout=5.0)
            except:
                pass
            self.producer = None
        self.is_connected = False
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery reports."""
        if err is not None:
            self._delivery_stats['failed'] += 1
            logger.warning(f"Kafka delivery failed: {err}")
        else:
            self._delivery_stats['delivered'] += 1
    
    def produce_vital_signs(self, vital_data: dict, patient_id: str):
        """Produce vital signs message to Kafka topic."""
        if not self.is_connected or not self.producer:
            return False
        
        try:
            message = {
                'timestamp': time.time(),
                'patient_id': patient_id,
                'heart_rate_bpm': vital_data.get('heart_rate', 0),
                'heart_rate_confidence': vital_data.get('heart_rate_confidence', 0),
                'breathing_rate_bpm': vital_data.get('breathing_rate', 0),
                'breathing_rate_confidence': vital_data.get('breathing_rate_confidence', 0),
                'status': vital_data.get('status', 'unknown'),
                'device_id': 'vitalflow-edge-01',
            }
            
            self.producer.produce(
                TOPICS.get('vital_signs', 'vitalflow-vital-signs'),
                key=patient_id.encode('utf-8'),
                value=json.dumps(message).encode('utf-8'),
                callback=self._delivery_callback
            )
            self._delivery_stats['sent'] += 1
            
            # Trigger delivery reports
            self.producer.poll(0)
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce message: {e}")
            return False
    
    def produce_anomaly(self, anomaly: dict, patient_id: str):
        """Produce anomaly alert to Kafka topic."""
        if not self.is_connected or not self.producer:
            return False
        
        try:
            self.producer.produce(
                TOPICS.get('anomalies', 'vitalflow-anomalies'),
                key=patient_id.encode('utf-8'),
                value=json.dumps(anomaly).encode('utf-8'),
                callback=self._delivery_callback
            )
            self.producer.poll(0)
            return True
        except Exception as e:
            logger.error(f"Failed to produce anomaly: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get Kafka connection status."""
        return {
            'available': KAFKA_AVAILABLE,
            'connected': self.is_connected,
            'is_confluent_cloud': is_confluent_cloud() if KAFKA_AVAILABLE else False,
            'bootstrap_servers': get_bootstrap_servers() if KAFKA_AVAILABLE else None,
            'last_error': self.last_error,
            'stats': self._delivery_stats
        }

kafka_service = KafkaService()


class KafkaConsumerService:
    """
    Consumes vital signs data from Kafka topics published by edge producers.
    Runs in background to receive phase stream and vital signs messages.
    """
    
    def __init__(self):
        self.consumer: Optional[Consumer] = None
        self.is_running = False
        self.last_error: Optional[str] = None
        self._consume_task = None
        self._phase_buffer: Dict[str, deque] = {}  # Buffer per device
        
    def connect(self) -> bool:
        """Connect Kafka consumer."""
        if not KAFKA_AVAILABLE:
            self.last_error = "confluent_kafka library not installed"
            return False
        
        try:
            config = get_consumer_config(
                group_id='vitalflow-backend-consumer',
                client_id='vitalflow-backend-api'
            )
            # Enable auto-commit for simpler management
            config['enable.auto.commit'] = True
            config['auto.commit.interval.ms'] = 1000
            config['auto.offset.reset'] = 'latest'  # Start from latest messages
            
            self.consumer = Consumer(config)
            
            # Subscribe to topics
            topics = [
                TOPICS.get('phase_stream', 'vitalflow-radar-phase'),
                TOPICS.get('vital_signs', 'vitalflow-vital-signs'),
            ]
            self.consumer.subscribe(topics)
            
            logger.info(f"Kafka consumer subscribed to: {topics}")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to connect Kafka consumer: {e}")
            return False
    
    def disconnect(self):
        """Disconnect consumer."""
        self.is_running = False
        if self.consumer:
            try:
                self.consumer.close()
            except:
                pass
            self.consumer = None
    
    async def start_consuming(self):
        """Start consuming messages in background."""
        if not self.consumer:
            if not self.connect():
                return
        
        self.is_running = True
        self._consume_task = asyncio.create_task(self._consume_loop())
        logger.info("Kafka consumer started")
    
    async def stop_consuming(self):
        """Stop consuming messages."""
        self.is_running = False
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
        self.disconnect()
        logger.info("Kafka consumer stopped")
    
    async def _consume_loop(self):
        """Main consumer loop running in background."""
        while self.is_running:
            try:
                # Poll for messages (non-blocking with timeout)
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, self.consumer.poll, 0.1
                )
                
                if msg is None:
                    await asyncio.sleep(0.01)
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka consumer error: {msg.error()}")
                        continue
                
                # Process message based on topic
                topic = msg.topic()
                key = msg.key().decode('utf-8') if msg.key() else 'unknown'
                
                try:
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    if 'phase_stream' in topic:
                        await self._handle_phase_message(data, key)
                    elif 'vital_signs' in topic:
                        await self._handle_vital_signs_message(data, key)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode message from {topic}")
                
            except Exception as e:
                logger.error(f"Error in Kafka consumer loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _handle_phase_message(self, data: dict, device_id: str):
        """Handle incoming phase/amplitude message."""
        # Buffer phase data for local processing if needed
        if device_id not in self._phase_buffer:
            self._phase_buffer[device_id] = deque(maxlen=600)  # 60 seconds @ 10Hz
        
        self._phase_buffer[device_id].append({
            'timestamp': data.get('timestamp'),
            'phase': data.get('phase'),
            'range_bin': data.get('range_bin'),
            'range_m': data.get('range_m'),
        })
    
    async def _handle_vital_signs_message(self, data: dict, device_id: str):
        """Handle incoming vital signs message."""
        # Extract patient ID from device ID or use default
        patient_id = data.get('patient_id', 'default')
        
        # Create VitalSignsData object
        vital_data = VitalSignsData(
            patient_id=patient_id,
            heart_rate=round(data.get('heart_rate_bpm', 0), 1),
            heart_rate_confidence=round(data.get('heart_rate_confidence', 0), 2),
            breathing_rate=round(data.get('breathing_rate_bpm', 0), 1),
            breathing_rate_confidence=round(data.get('breathing_rate_confidence', 0), 2),
            status='normal',  # Will be re-evaluated
            timestamp=data.get('timestamp', time.time()),
            alerts=[]
        )
        
        # Re-evaluate status based on ranges
        hr = vital_data.heart_rate
        br = vital_data.breathing_rate
        
        if hr < settings.HR_MIN or hr > settings.HR_MAX:
            vital_data.status = 'critical'
            vital_data.alerts.append({
                'type': 'heart_rate',
                'severity': 'critical',
                'message': f'Heart rate {hr} BPM is out of normal range'
            })
        elif br < settings.BR_MIN or br > settings.BR_MAX:
            vital_data.status = 'warning'
            vital_data.alerts.append({
                'type': 'breathing_rate',
                'severity': 'warning',
                'message': f'Breathing rate {br} BPM is unusual'
            })
        
        # Broadcast to WebSocket clients
        await manager.broadcast(patient_id, vital_data.dict())
        
        logger.debug(f"Received vital signs from {device_id}: HR={hr}, BR={br}")

kafka_consumer_service = KafkaConsumerService()


# ============================================================================
# VERTEX AI SERVICE
# ============================================================================

class VertexAIService:
    """
    Manages Google Cloud Vertex AI for anomaly detection and health summaries.
    Uses Gemini for intelligent health insights.
    """
    
    def __init__(self):
        self.detector: Optional[VitalSignsAnomalyDetector] = None
        self.is_initialized = False
        self.last_error: Optional[str] = None
        self._anomaly_count = 0
        self._summary_count = 0
    
    def initialize(self) -> bool:
        """Initialize Vertex AI connection."""
        if not VERTEX_AI_AVAILABLE:
            self.last_error = "vertex_ai_processor module not available"
            return False
        
        try:
            self.detector = VitalSignsAnomalyDetector(
                project_id=settings.GOOGLE_CLOUD_PROJECT or None,
                location=settings.VERTEX_AI_LOCATION,
                use_gemini=True
            )
            
            if self.detector.initialize():
                self.is_initialized = True
                self.last_error = None
                logger.info("Vertex AI initialized successfully")
                return True
            else:
                self.last_error = "Vertex AI initialization returned False (offline mode)"
                logger.warning("Vertex AI running in offline mode")
                # Still mark as initialized - will use rule-based detection
                self.is_initialized = True
                return True
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return False
    
    def detect_anomalies(self, vital_data: dict, device_id: str = "edge-01") -> List[dict]:
        """
        Detect anomalies in vital signs using Vertex AI.
        
        Returns list of anomaly dictionaries.
        """
        if not self.is_initialized or not self.detector:
            # Fallback to simple rule-based detection
            return self._simple_anomaly_detection(vital_data)
        
        try:
            # Create VitalSigns object for Vertex AI
            vital_signs = VertexVitalSigns(
                timestamp=vital_data.get('timestamp', time.time()),
                heart_rate_bpm=vital_data.get('heart_rate', 0),
                heart_rate_confidence=vital_data.get('heart_rate_confidence', 0),
                breathing_rate_bpm=vital_data.get('breathing_rate', 0),
                breathing_rate_confidence=vital_data.get('breathing_rate_confidence', 0),
                device_id=device_id
            )
            
            # Detect anomalies
            anomalies = self.detector.detect_anomalies(vital_signs, include_history=True)
            self._anomaly_count += len(anomalies)
            
            # Convert to dict list
            return [a.to_dict() for a in anomalies]
            
        except Exception as e:
            logger.error(f"Vertex AI anomaly detection error: {e}")
            return self._simple_anomaly_detection(vital_data)
    
    def generate_health_summary(self, vital_data: dict, anomalies: List[dict]) -> Optional[dict]:
        """
        Generate AI-powered health summary using Gemini.
        """
        if not self.is_initialized or not self.detector:
            return self._simple_health_summary(vital_data, anomalies)
        
        try:
            vital_signs = VertexVitalSigns(
                timestamp=vital_data.get('timestamp', time.time()),
                heart_rate_bpm=vital_data.get('heart_rate', 0),
                heart_rate_confidence=vital_data.get('heart_rate_confidence', 0),
                breathing_rate_bpm=vital_data.get('breathing_rate', 0),
                breathing_rate_confidence=vital_data.get('breathing_rate_confidence', 0),
                device_id="edge-01"
            )
            
            # Convert anomaly dicts back to Anomaly objects
            anomaly_objects = []
            for a in anomalies:
                anomaly_objects.append(VertexAnomaly(
                    timestamp=a.get('timestamp', time.time()),
                    anomaly_type=a.get('anomaly_type', 'unknown'),
                    severity=a.get('severity', 'low'),
                    current_value=a.get('current_value', 0),
                    normal_range_min=a.get('normal_range_min', 0),
                    normal_range_max=a.get('normal_range_max', 0),
                    confidence=a.get('confidence', 0),
                    device_id=a.get('device_id', 'edge-01'),
                    description=a.get('description', ''),
                    recommended_action=a.get('recommended_action', '')
                ))
            
            summary = self.detector.generate_health_summary(vital_signs, anomaly_objects)
            self._summary_count += 1
            
            return {
                'timestamp': summary.timestamp,
                'summary_text': summary.summary_text,
                'risk_level': summary.risk_level,
                'recommendations': summary.recommendations,
                'model_used': summary.gemini_model_used
            }
            
        except Exception as e:
            logger.error(f"Gemini health summary error: {e}")
            return self._simple_health_summary(vital_data, anomalies)
    
    def _simple_anomaly_detection(self, vital_data: dict) -> List[dict]:
        """Simple rule-based anomaly detection fallback."""
        anomalies = []
        hr = vital_data.get('heart_rate', 0)
        br = vital_data.get('breathing_rate', 0)
        
        if hr < 40:
            anomalies.append({
                'anomaly_type': 'severe_bradycardia',
                'severity': 'critical',
                'current_value': hr,
                'description': 'Severely low heart rate',
                'recommended_action': 'Seek immediate medical attention'
            })
        elif hr < 60:
            anomalies.append({
                'anomaly_type': 'bradycardia',
                'severity': 'medium',
                'current_value': hr,
                'description': 'Heart rate below normal',
                'recommended_action': 'Monitor closely'
            })
        elif hr > 150:
            anomalies.append({
                'anomaly_type': 'severe_tachycardia',
                'severity': 'high',
                'current_value': hr,
                'description': 'Dangerously elevated heart rate',
                'recommended_action': 'Seek medical attention'
            })
        elif hr > 100:
            anomalies.append({
                'anomaly_type': 'tachycardia',
                'severity': 'medium',
                'current_value': hr,
                'description': 'Heart rate above normal',
                'recommended_action': 'Rest and monitor'
            })
        
        if br < 6:
            anomalies.append({
                'anomaly_type': 'apnea',
                'severity': 'critical',
                'current_value': br,
                'description': 'Possible breathing pause',
                'recommended_action': 'Check subject immediately'
            })
        elif br < 12:
            anomalies.append({
                'anomaly_type': 'bradypnea',
                'severity': 'medium',
                'current_value': br,
                'description': 'Breathing rate below normal',
                'recommended_action': 'Monitor for respiratory depression'
            })
        elif br > 25:
            anomalies.append({
                'anomaly_type': 'tachypnea',
                'severity': 'high',
                'current_value': br,
                'description': 'Rapid breathing detected',
                'recommended_action': 'Assess for respiratory distress'
            })
        
        return anomalies
    
    def _simple_health_summary(self, vital_data: dict, anomalies: List[dict]) -> dict:
        """Simple rule-based health summary fallback."""
        hr = vital_data.get('heart_rate', 0)
        br = vital_data.get('breathing_rate', 0)
        
        if any(a.get('severity') == 'critical' for a in anomalies):
            risk_level = 'CRITICAL'
            summary = f"CRITICAL: Immediate attention required. HR: {hr:.0f} BPM, BR: {br:.0f}/min."
        elif any(a.get('severity') == 'high' for a in anomalies):
            risk_level = 'HIGH'
            summary = f"High risk detected. HR: {hr:.0f} BPM, BR: {br:.0f}/min. Medical review recommended."
        elif any(a.get('severity') == 'medium' for a in anomalies):
            risk_level = 'MODERATE'
            summary = f"Moderate concerns. HR: {hr:.0f} BPM, BR: {br:.0f}/min. Continue monitoring."
        elif anomalies:
            risk_level = 'LOW'
            summary = f"Minor deviations noted. HR: {hr:.0f} BPM, BR: {br:.0f}/min."
        else:
            risk_level = 'NORMAL'
            summary = f"Vital signs normal. HR: {hr:.0f} BPM, BR: {br:.0f}/min. Subject is stable."
        
        return {
            'timestamp': time.time(),
            'summary_text': summary,
            'risk_level': risk_level,
            'recommendations': [a.get('recommended_action', '') for a in anomalies if a.get('recommended_action')],
            'model_used': 'rule-based'
        }
    
    def get_status(self) -> dict:
        """Get Vertex AI status."""
        return {
            'available': VERTEX_AI_AVAILABLE,
            'gemini_available': GEMINI_AVAILABLE,
            'initialized': self.is_initialized,
            'project_id': settings.GOOGLE_CLOUD_PROJECT or None,
            'location': settings.VERTEX_AI_LOCATION,
            'last_error': self.last_error,
            'stats': {
                'anomalies_detected': self._anomaly_count,
                'summaries_generated': self._summary_count
            }
        }

vertex_ai_service = VertexAIService()

# ============================================================================
# VITAL SIGNS SERVICE
# ============================================================================

class RadarService:
    """Manages the AWR1642 radar hardware connection."""
    
    def __init__(self):
        self.radar = None
        self.processor = None
        self.is_connected = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
    
    def detect_sensor(self) -> bool:
        """Check if radar sensor is connected."""
        if not RADAR_AVAILABLE:
            self.last_error = "Radar driver not installed"
            return False
        
        # Check if serial ports exist
        cli_exists = os.path.exists(settings.RADAR_CLI_PORT)
        data_exists = os.path.exists(settings.RADAR_DATA_PORT)
        
        if not cli_exists or not data_exists:
            self.last_error = f"Serial ports not found: CLI={cli_exists}, Data={data_exists}"
            return False
        
        return True
    
    def connect(self) -> bool:
        """Connect to the radar sensor."""
        with self._lock:
            if self.is_connected:
                return True
            
            if not self.detect_sensor():
                return False
            
            try:
                self.radar = AWR1642(
                    cli_port=settings.RADAR_CLI_PORT,
                    data_port=settings.RADAR_DATA_PORT
                )
                self.radar.connect()
                
                # Load and send configuration
                config_path = Path(__file__).parent.parent / settings.RADAR_CONFIG_FILE
                if config_path.exists():
                    commands = load_config_from_file(str(config_path))
                    self.radar.configure_sensor(commands, verbose=False)
                else:
                    logger.warning(f"Config file not found: {config_path}, using default")
                    from awr1642_driver import config_commands_1642
                    self.radar.configure_sensor(config_commands_1642, verbose=False)
                
                # Initialize processor
                if PROCESSOR_AVAILABLE:
                    self.processor = VitalSignsProcessor(fps=settings.RADAR_FPS)
                
                self.is_connected = True
                self.last_error = None
                logger.info("Radar sensor connected and configured")
                return True
                
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Failed to connect to radar: {e}")
                self.disconnect()
                return False
    
    def disconnect(self):
        """Disconnect from radar sensor."""
        with self._lock:
            if self.radar:
                try:
                    self.radar.disconnect()
                except:
                    pass
                self.radar = None
            self.processor = None
            self.is_connected = False
    
    def read_frame(self) -> Optional[dict]:
        """Read a single frame from the radar."""
        if not self.is_connected or not self.radar:
            return None
        
        try:
            data_ok, frame_number, detected_objects, tlv_data = self.radar.read_tlv_packet(timeout=0.2)
            
            if data_ok:
                return {
                    'frame_number': frame_number,
                    'detected_objects': detected_objects,
                    'tlv_data': tlv_data
                }
        except Exception as e:
            logger.error(f"Error reading radar frame: {e}")
        
        return None
    
    def get_status(self) -> dict:
        """Get current radar status."""
        return {
            'driver_available': RADAR_AVAILABLE,
            'processor_available': PROCESSOR_AVAILABLE,
            'sensor_detected': self.detect_sensor() if RADAR_AVAILABLE else False,
            'is_connected': self.is_connected,
            'last_error': self.last_error,
            'cli_port': settings.RADAR_CLI_PORT,
            'data_port': settings.RADAR_DATA_PORT
        }

radar_service = RadarService()


class VitalSignsService:
    def __init__(self):
        self.is_running = False
        self.current_data: Dict[str, VitalSignsData] = {}
        self.data_buffer: Dict[str, deque] = {}
        self._task = None
        self._use_radar = False
        self._phase_buffer: List[float] = []
    
    async def start(self, patient_id: str = "default"):
        if self.is_running:
            return
        
        self.is_running = True
        self.data_buffer[patient_id] = deque(maxlen=3600)
        
        # Intelligent mode selection based on VITALFLOW_MODE setting
        mode = settings.VITALFLOW_MODE.lower()
        
        if mode == 'kafka':
            # Force Kafka consumer mode only
            self._use_radar = False
            self._task = asyncio.create_task(self._kafka_consumer_vitals(patient_id))
            logger.info(f"Vital signs monitoring started (KAFKA CONSUMER MODE) for {patient_id}")
            
        elif mode == 'radar':
            # Force direct radar mode only
            if radar_service.connect():
                self._use_radar = True
                self._task = asyncio.create_task(self._read_radar_vitals(patient_id))
                logger.info(f"Vital signs monitoring started (RADAR MODE) for {patient_id}")
            else:
                # Fallback to simulation if radar unavailable
                self._use_radar = False
                self._task = asyncio.create_task(self._simulate_vitals(patient_id))
                logger.warning(f"Radar unavailable, using SIMULATION MODE for {patient_id}")
                
        elif mode == 'simulation':
            # Force simulation mode only
            self._use_radar = False
            self._task = asyncio.create_task(self._simulate_vitals(patient_id))
            logger.info(f"Vital signs monitoring started (SIMULATION MODE) for {patient_id}")
            
        else:  # mode == 'auto' or invalid
            # Auto mode: try radar first, then Kafka consumer, then simulation
            if radar_service.connect():
                self._use_radar = True
                self._task = asyncio.create_task(self._read_radar_vitals(patient_id))
                logger.info(f"Vital signs monitoring started (RADAR MODE - auto-detected) for {patient_id}")
            elif kafka_consumer_service.is_running:
                self._use_radar = False
                self._task = asyncio.create_task(self._kafka_consumer_vitals(patient_id))
                logger.info(f"Vital signs monitoring started (KAFKA CONSUMER MODE - auto-detected) for {patient_id}")
            else:
                self._use_radar = False
                self._task = asyncio.create_task(self._simulate_vitals(patient_id))
                logger.info(f"Vital signs monitoring started (SIMULATION MODE - fallback) for {patient_id}")
    
    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        radar_service.disconnect()
    
    def get_mode(self) -> str:
        """Return current operating mode."""
        return "radar" if self._use_radar else "simulation"
    
    async def _read_radar_vitals(self, patient_id: str):
        """Read vital signs from actual radar sensor."""
        frame_count = 0
        last_vital_update = time.time()
        vital_update_interval = 3.0  # Compute vital signs every 3 seconds
        
        # Buffers for signal processing
        amplitude_history = deque(maxlen=int(60 * settings.RADAR_FPS))
        range_profile_history = deque(maxlen=int(30 * settings.RADAR_FPS))
        selected_bin = None
        
        # Store last valid readings to avoid sending 0s
        last_valid_hr = None
        last_valid_br = None
        last_valid_hr_conf = 0.0
        last_valid_br_conf = 0.0
        
        # Exponential moving average for smoothing (prevents sudden jumps)
        ema_hr = None
        ema_br = None
        ema_alpha = 0.3  # Smoothing factor (0.3 = 30% new, 70% old)
        
        while self.is_running:
            try:
                frame = radar_service.read_frame()
                
                if frame and 'tlv_data' in frame:
                    tlv_data = frame['tlv_data']
                    
                    # Process range profile for vital signs
                    if 'range_profile' in tlv_data:
                        range_profile = tlv_data['range_profile']
                        range_profile_history.append(range_profile)
                        
                        frame_count += 1
                        
                        # Update range bin selection periodically
                        if frame_count % 50 == 1 or selected_bin is None:
                            selected_bin = self._select_range_bin(
                                range_profile_history,
                                radar_service.radar.config_params.get('rangeResolutionMeters', 0.044)
                            )
                            logger.debug(f"Selected range bin: {selected_bin}")
                        
                        # Extract amplitude from selected bin
                        if selected_bin is not None and selected_bin < len(range_profile):
                            amplitude = range_profile[selected_bin]
                            amplitude_history.append(amplitude)
                        
                        # Compute vital signs periodically
                        current_time = time.time()
                        if current_time - last_vital_update >= vital_update_interval:
                            last_vital_update = current_time
                            
                            if radar_service.processor and len(amplitude_history) >= int(10 * settings.RADAR_FPS):
                                # Process amplitude signal
                                amplitude_signal = np.array(list(amplitude_history))
                                amplitude_signal = amplitude_signal - np.mean(amplitude_signal)
                                
                                # Use enhanced extraction with scoring-based harmonic rejection
                                result = radar_service.processor.extract_vital_signs_enhanced(amplitude_signal)
                                
                                hr = result['hr_bpm']
                                br = result['br_bpm']
                                hr_conf = result['hr_confidence']
                                br_conf = result['br_confidence']
                                
                                # Apply EMA smoothing to prevent sudden jumps
                                if not np.isnan(hr) and hr > 0:
                                    if ema_hr is None:
                                        ema_hr = hr
                                    else:
                                        # Reject outliers: if change > 20 BPM, use smaller alpha
                                        if abs(hr - ema_hr) > 20:
                                            ema_hr = ema_hr * 0.9 + hr * 0.1  # Very slow adaptation
                                        else:
                                            ema_hr = ema_hr * (1 - ema_alpha) + hr * ema_alpha
                                    last_valid_hr = ema_hr
                                    last_valid_hr_conf = hr_conf
                                    
                                if not np.isnan(br) and br > 0:
                                    if ema_br is None:
                                        ema_br = br
                                    else:
                                        # Reject outliers: if change > 5 BPM for BR
                                        if abs(br - ema_br) > 5:
                                            ema_br = ema_br * 0.9 + br * 0.1
                                        else:
                                            ema_br = ema_br * (1 - ema_alpha) + br * ema_alpha
                                    last_valid_br = ema_br
                                    last_valid_br_conf = br_conf
                                
                                # Only broadcast if we have valid readings
                                if last_valid_hr is not None and last_valid_br is not None:
                                    # Determine status based on thresholds
                                    status, alerts = self._evaluate_vitals(
                                        last_valid_hr, last_valid_br, 
                                        last_valid_hr_conf, last_valid_br_conf
                                    )
                                    
                                    data = VitalSignsData(
                                        patient_id=patient_id,
                                        heart_rate=round(last_valid_hr, 1),
                                        heart_rate_confidence=round(last_valid_hr_conf, 2),
                                        breathing_rate=round(last_valid_br, 1),
                                        breathing_rate_confidence=round(last_valid_br_conf, 2),
                                        status=status,
                                        timestamp=time.time(),
                                        alerts=alerts
                                    )
                                    
                                    self.current_data[patient_id] = data
                                    self.data_buffer[patient_id].append(data.dict())
                                    await manager.broadcast(patient_id, data.dict())
                                    
                                    # Store alerts to database
                                    await self._store_alerts(patient_id, alerts)
                                    
                                    logger.info(f"Vital signs: HR={last_valid_hr:.1f} ({last_valid_hr_conf:.2f}), BR={last_valid_br:.1f} ({last_valid_br_conf:.2f})")
                        
                        # Send calibration status while collecting data
                        elif last_valid_hr is None and frame_count % int(settings.RADAR_FPS * 2) == 0:
                            progress = min(100, int(len(amplitude_history) / (10 * settings.RADAR_FPS) * 100))
                            data = VitalSignsData(
                                patient_id=patient_id,
                                heart_rate=0,
                                heart_rate_confidence=0,
                                breathing_rate=0,
                                breathing_rate_confidence=0,
                                status='calibrating',
                                timestamp=time.time(),
                                alerts=[{'type': 'info', 'severity': 'info', 
                                        'message': f'Collecting data... {progress}%'}]
                            )
                            await manager.broadcast(patient_id, data.dict())
                
                await asyncio.sleep(1.0 / settings.RADAR_FPS)
                
            except Exception as e:
                logger.error(f"Error in radar vital signs loop: {e}")
                await asyncio.sleep(1.0)
    
    def _select_range_bin(self, range_profiles, range_resolution):
        """Select optimal range bin based on variance (similar to edge producer)."""
        if len(range_profiles) < 20:
            return 16  # Default bin
        
        profiles = np.array(list(range_profiles)[-20:])
        
        # Convert to linear if in dB
        mean_val = np.mean(profiles)
        if mean_val < 50:
            magnitudes = 10 ** (profiles / 20.0)
        else:
            magnitudes = profiles
        
        # Compute variance per bin
        var_per_bin = np.var(magnitudes, axis=0)
        
        # Distance mask (0.5-1.5m range)
        num_bins = var_per_bin.shape[0]
        distances = np.arange(num_bins) * range_resolution
        valid_mask = (distances >= 0.5) & (distances <= 1.5)
        
        # Power threshold
        mean_power = np.mean(magnitudes, axis=0)
        max_power = np.max(mean_power)
        if max_power > 0:
            power_threshold = max_power * 0.1
            power_mask = mean_power > power_threshold
        else:
            power_mask = np.ones_like(mean_power, dtype=bool)
        
        # Combined mask
        combined_mask = valid_mask & power_mask
        
        if not np.any(combined_mask):
            return int(0.7 / range_resolution)  # Default to 0.7m
        
        var_masked = var_per_bin.copy()
        var_masked[~combined_mask] = 0
        selected_bin = np.argmax(var_masked)
        
        return int(selected_bin)
    
    async def _kafka_consumer_vitals(self, patient_id: str):
        """Wait for vital signs from Kafka consumer (edge producers)."""
        logger.info(f"Waiting for vital signs from Kafka consumer for patient {patient_id}")
        logger.info(f"Start edge producer with: python edge_producer_live.py")
        
        # Send waiting status to frontend
        while self.is_running:
            try:
                data = VitalSignsData(
                    patient_id=patient_id,
                    heart_rate=0,
                    heart_rate_confidence=0,
                    breathing_rate=0,
                    breathing_rate_confidence=0,
                    status='waiting',
                    timestamp=time.time(),
                    alerts=[{
                        'type': 'info',
                        'severity': 'info',
                        'message': 'Waiting for data from edge producer. Run: python edge_producer_live.py'
                    }]
                )
                await manager.broadcast(patient_id, data.dict())
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in Kafka consumer vitals loop: {e}")
                await asyncio.sleep(5.0)
    
    def _evaluate_vitals(self, hr: float, br: float, hr_conf: float, br_conf: float) -> tuple:
        """Evaluate vital signs and generate alerts."""
        status = 'normal'
        alerts = []
        
        # Check heart rate
        if not np.isnan(hr) and hr_conf > 0.5:
            if hr < settings.HR_MIN or hr > settings.HR_MAX:
                status = 'critical'
                alerts.append({
                    'type': 'heart_rate',
                    'severity': 'critical',
                    'message': f'Abnormal HR: {hr:.0f} BPM'
                })
            elif hr < 60 or hr > 100:
                status = 'warning'
                alerts.append({
                    'type': 'heart_rate',
                    'severity': 'warning',
                    'message': f'HR outside normal range: {hr:.0f} BPM'
                })
        
        # Check breathing rate
        if not np.isnan(br) and br_conf > 0.5:
            if br < settings.BR_MIN or br > settings.BR_MAX:
                if status != 'critical':
                    status = 'critical'
                alerts.append({
                    'type': 'breathing_rate',
                    'severity': 'critical',
                    'message': f'Abnormal BR: {br:.0f} breaths/min'
                })
            elif br < 12 or br > 20:
                if status == 'normal':
                    status = 'warning'
                alerts.append({
                    'type': 'breathing_rate',
                    'severity': 'warning',
                    'message': f'BR outside normal range: {br:.0f} breaths/min'
                })
        
        # Check confidence
        if hr_conf < 0.5 or br_conf < 0.5:
            alerts.append({
                'type': 'signal_quality',
                'severity': 'info',
                'message': 'Low signal quality - ensure patient is still'
            })
        
        return status, alerts
    
    async def _store_alerts(self, patient_id: str, alerts: List[dict]):
        """Store critical alerts to database."""
        for alert in alerts:
            if alert['severity'] in ('critical', 'warning'):
                try:
                    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
                        await db.execute('''
                            INSERT INTO alerts (id, patient_id, timestamp, alert_type, severity, message)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            str(uuid.uuid4()),
                            patient_id,
                            time.time(),
                            alert['type'],
                            alert['severity'],
                            alert['message']
                        ))
                        await db.commit()
                except Exception as e:
                    logger.error(f"Failed to store alert: {e}")
    
    async def _simulate_vitals(self, patient_id: str):
        """Simulate vital signs for testing when no radar is connected.
        
        Integrates with Confluent Kafka for streaming and Vertex AI for anomaly detection.
        """
        t = 0
        summary_interval = 30  # Generate health summary every 30 seconds
        last_summary_time = 0
        
        while self.is_running:
            # Simulate realistic vital signs with occasional anomalies
            base_hr = 72
            base_br = 16
            
            # Add occasional anomaly scenarios (5% chance)
            anomaly_scenario = np.random.random()
            if anomaly_scenario < 0.02:  # 2% chance of low HR
                hr = 52 + np.random.normal(0, 3)
            elif anomaly_scenario < 0.04:  # 2% chance of high HR
                hr = 110 + np.random.normal(0, 5)
            else:  # Normal variation
                hr = base_hr + np.random.normal(0, 2) + 5 * np.sin(t / 60)
            
            br = base_br + np.random.normal(0, 0.5) + np.sin(t / 30)
            hr_conf = 0.85 + np.random.uniform(-0.1, 0.1)
            br_conf = 0.80 + np.random.uniform(-0.1, 0.1)
            
            vital_dict = {
                'heart_rate': round(hr, 1),
                'heart_rate_confidence': round(hr_conf, 2),
                'breathing_rate': round(br, 1),
                'breathing_rate_confidence': round(br_conf, 2),
                'timestamp': time.time()
            }
            
            # === VERTEX AI ANOMALY DETECTION ===
            ai_anomalies = vertex_ai_service.detect_anomalies(vital_dict)
            
            # Merge AI anomalies with basic threshold alerts
            status, basic_alerts = self._evaluate_vitals(hr, br, hr_conf, br_conf)
            
            # Convert AI anomalies to alert format
            for anomaly in ai_anomalies:
                severity = anomaly.get('severity', 'low')
                if severity in ('critical', 'high'):
                    status = 'critical'
                elif severity == 'medium' and status == 'normal':
                    status = 'warning'
                    
                basic_alerts.append({
                    'type': anomaly.get('anomaly_type', 'unknown'),
                    'severity': severity,
                    'message': anomaly.get('description', 'Anomaly detected'),
                    'ai_detected': True,
                    'recommended_action': anomaly.get('recommended_action', '')
                })
            
            data = VitalSignsData(
                patient_id=patient_id,
                heart_rate=round(hr, 1),
                heart_rate_confidence=round(hr_conf, 2),
                breathing_rate=round(br, 1),
                breathing_rate_confidence=round(br_conf, 2),
                status=status,
                timestamp=time.time(),
                alerts=basic_alerts
            )
            
            self.current_data[patient_id] = data
            
            if patient_id not in self.data_buffer:
                self.data_buffer[patient_id] = deque(maxlen=3600)
            self.data_buffer[patient_id].append(data.dict())
            
            # === CONFLUENT KAFKA STREAMING ===
            # Produce vital signs to Kafka
            kafka_service.produce_vital_signs(vital_dict, patient_id)
            
            # Produce anomalies to Kafka
            for anomaly in ai_anomalies:
                anomaly['patient_id'] = patient_id
                kafka_service.produce_anomaly(anomaly, patient_id)
            
            # === GEMINI HEALTH SUMMARY (periodic) ===
            current_time = time.time()
            if current_time - last_summary_time >= summary_interval:
                summary = vertex_ai_service.generate_health_summary(vital_dict, ai_anomalies)
                if summary:
                    # Broadcast summary to WebSocket clients
                    await manager.broadcast(patient_id, {
                        'type': 'health_summary',
                        'summary': summary
                    })
                last_summary_time = current_time
            
            # Broadcast vital signs via WebSocket
            await manager.broadcast(patient_id, data.dict())
            
            t += 1
            await asyncio.sleep(1.0)

vital_service = VitalSignsService()

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("VitalFlow-Radar API Starting")
    logger.info("=" * 60)
    
    await init_database()
    
    # Initialize Confluent Kafka
    logger.info("Initializing Confluent Kafka...")
    if kafka_service.connect():
        logger.info("✓ Confluent Kafka connected")
        
        # Start Kafka consumer to receive edge producer data
        await kafka_consumer_service.start_consuming()
        logger.info("✓ Kafka consumer started")
    else:
        logger.warning(f"⚠ Kafka not connected: {kafka_service.last_error}")
    
    # Initialize Vertex AI
    logger.info("Initializing Google Cloud Vertex AI...")
    if vertex_ai_service.initialize():
        logger.info("✓ Vertex AI initialized")
    else:
        logger.warning(f"⚠ Vertex AI not initialized: {vertex_ai_service.last_error}")
    
    # Start vital signs monitoring (will use Kafka consumer data if available)
    await vital_service.start("default")
    
    logger.info("=" * 60)
    logger.info("VitalFlow API Ready")
    logger.info(f"  Mode: {vital_service.get_mode().upper()}")
    logger.info(f"  Kafka: {'Connected' if kafka_service.is_connected else 'Offline'}")
    logger.info(f"  Kafka Consumer: {'Active' if kafka_consumer_service.is_running else 'Offline'}")
    logger.info(f"  Vertex AI: {'Active' if vertex_ai_service.is_initialized else 'Offline'}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    await vital_service.stop()
    await kafka_consumer_service.stop_consuming()
    kafka_service.disconnect()
    logger.info("VitalFlow API stopped")

app = FastAPI(
    title="VitalFlow Radar API",
    description="Contactless vital signs monitoring using mmWave radar",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint with full system status."""
    radar_status = radar_service.get_status()
    kafka_status = kafka_service.get_status()
    vertex_status = vertex_ai_service.get_status()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "monitoring_mode": vital_service.get_mode(),
        "monitoring_active": vital_service.is_running,
        "integrations": {
            "radar": radar_status,
            "kafka": kafka_status,
            "vertex_ai": vertex_status
        }
    }

# === RADAR ENDPOINTS ===

@app.get("/api/radar/status")
async def get_radar_status():
    """Get detailed radar sensor status."""
    return radar_service.get_status()

@app.post("/api/radar/connect")
async def connect_radar(user: dict = Depends(get_admin_user)):
    """Manually attempt to connect to radar sensor."""
    success = radar_service.connect()
    if success:
        # Restart vital signs service with radar
        await vital_service.stop()
        await vital_service.start("default")
        return {"status": "connected", "mode": "radar"}
    else:
        return {"status": "failed", "error": radar_service.last_error}

@app.post("/api/radar/disconnect")
async def disconnect_radar(user: dict = Depends(get_admin_user)):
    """Disconnect from radar sensor and switch to simulation."""
    radar_service.disconnect()
    await vital_service.stop()
    await vital_service.start("default")
    return {"status": "disconnected", "mode": "simulation"}

# === CONFLUENT KAFKA ENDPOINTS ===

@app.get("/api/kafka/status")
async def get_kafka_status():
    """Get Confluent Kafka connection status and metrics."""
    return kafka_service.get_status()

@app.post("/api/kafka/connect")
async def connect_kafka(user: dict = Depends(get_admin_user)):
    """Connect to Confluent Kafka/Cloud."""
    success = kafka_service.connect()
    return {
        "status": "connected" if success else "failed",
        "error": kafka_service.last_error if not success else None,
        **kafka_service.get_status()
    }

@app.post("/api/kafka/disconnect")
async def disconnect_kafka(user: dict = Depends(get_admin_user)):
    """Disconnect from Confluent Kafka."""
    kafka_service.disconnect()
    return {"status": "disconnected"}

# === VERTEX AI / GEMINI ENDPOINTS ===

@app.get("/api/vertex-ai/status")
async def get_vertex_ai_status():
    """Get Google Cloud Vertex AI status and metrics."""
    return vertex_ai_service.get_status()

@app.post("/api/vertex-ai/initialize")
async def initialize_vertex_ai(user: dict = Depends(get_admin_user)):
    """Initialize Vertex AI connection."""
    success = vertex_ai_service.initialize()
    return {
        "status": "initialized" if success else "failed",
        "error": vertex_ai_service.last_error if not success else None,
        **vertex_ai_service.get_status()
    }

@app.post("/api/vertex-ai/analyze")
async def analyze_vitals(patient_id: str = "default"):
    """Run Vertex AI anomaly detection on current vitals."""
    if patient_id not in vital_service.current_data:
        raise HTTPException(status_code=404, detail="Patient not being monitored")
    
    current = vital_service.current_data[patient_id]
    vital_dict = {
        'heart_rate': current.heart_rate,
        'heart_rate_confidence': current.heart_rate_confidence,
        'breathing_rate': current.breathing_rate,
        'breathing_rate_confidence': current.breathing_rate_confidence,
        'timestamp': current.timestamp
    }
    
    anomalies = vertex_ai_service.detect_anomalies(vital_dict)
    summary = vertex_ai_service.generate_health_summary(vital_dict, anomalies)
    
    return {
        "vital_signs": vital_dict,
        "anomalies": anomalies,
        "health_summary": summary
    }

# === AUTH ROUTES ===

@app.post("/api/auth/login", response_model=Token)
async def login(data: UserLogin):
    """User login."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            'SELECT * FROM users WHERE username = ?', (data.username,)
        )
        user = await cursor.fetchone()
        
        if user and verify_password(data.password, user['password_hash']):
            token = create_token(user['id'], user['role'])
            return Token(
                access_token=token,
                user_id=user['id'],
                role=user['role']
            )
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/register", response_model=Token)
async def register(data: UserCreate):
    """User registration."""
    user_id = str(uuid.uuid4())
    password_hash = hash_password(data.password)
    
    try:
        async with aiosqlite.connect(settings.DATABASE_PATH) as db:
            await db.execute(
                'INSERT INTO users (id, username, password_hash, email) VALUES (?, ?, ?, ?)',
                (user_id, data.username, password_hash, data.email)
            )
            await db.commit()
        
        token = create_token(user_id, 'viewer')
        return Token(access_token=token, user_id=user_id, role='viewer')
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")

# Patient routes
@app.get("/api/patients", response_model=List[PatientResponse])
async def get_patients(user: dict = Depends(get_current_user)):
    """Get all patients."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute('SELECT * FROM patients')
        rows = await cursor.fetchall()
        
        patients = []
        for row in rows:
            conditions = json.loads(row['conditions']) if row['conditions'] else []
            patients.append(PatientResponse(
                id=row['id'],
                name=row['name'],
                room=row['room'],
                age=row['age'],
                conditions=conditions,
                emergency_contact=row['emergency_contact'],
                emergency_phone=row['emergency_phone'],
                created_at=row['created_at']
            ))
        return patients

@app.post("/api/patients", response_model=PatientResponse)
async def create_patient(data: PatientCreate, user: dict = Depends(get_admin_user)):
    """Create new patient."""
    patient_id = str(uuid.uuid4())
    
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute('''
            INSERT INTO patients (id, name, room, age, conditions, emergency_contact, emergency_phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id, data.name, data.room, data.age,
            json.dumps(data.conditions), data.emergency_contact, data.emergency_phone
        ))
        await db.commit()
    
    # Start monitoring for this patient
    await vital_service.start(patient_id)
    
    return PatientResponse(
        id=patient_id,
        name=data.name,
        room=data.room,
        age=data.age,
        conditions=data.conditions,
        emergency_contact=data.emergency_contact,
        emergency_phone=data.emergency_phone,
        created_at=datetime.now().isoformat()
    )

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str, user: dict = Depends(get_current_user)):
    """Get single patient with current status."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        row = await cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        result = dict(row)
        result['conditions'] = json.loads(result['conditions']) if result['conditions'] else []
        
        # Add current status
        if patient_id in vital_service.current_data:
            result['current_vitals'] = vital_service.current_data[patient_id].dict()
        
        return result

# Vital signs routes
@app.get("/api/vitals/{patient_id}/current")
async def get_current_vitals(patient_id: str, user: dict = Depends(get_current_user)):
    """Get current vital signs."""
    if patient_id not in vital_service.current_data:
        raise HTTPException(status_code=404, detail="Patient not being monitored")
    
    return vital_service.current_data[patient_id].dict()

@app.get("/api/vitals/{patient_id}/history")
async def get_vital_history(
    patient_id: str,
    duration: int = Query(60, description="Duration in minutes"),
    user: dict = Depends(get_current_user)
):
    """Get vital signs history."""
    if patient_id in vital_service.data_buffer:
        data = list(vital_service.data_buffer[patient_id])
        cutoff = time.time() - (duration * 60)
        return [d for d in data if d['timestamp'] >= cutoff]
    
    return []

# Alert routes
@app.get("/api/alerts", response_model=List[AlertResponse])
async def get_alerts(
    patient_id: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """Get unacknowledged alerts."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        if patient_id:
            cursor = await db.execute(
                'SELECT * FROM alerts WHERE patient_id = ? AND acknowledged = 0 ORDER BY timestamp DESC',
                (patient_id,)
            )
        else:
            cursor = await db.execute(
                'SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC'
            )
        rows = await cursor.fetchall()
        return [AlertResponse(**dict(row)) for row in rows]

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user: dict = Depends(get_current_user)):
    """Acknowledge an alert."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute(
            'UPDATE alerts SET acknowledged = 1 WHERE id = ?', (alert_id,)
        )
        await db.commit()
    return {"message": "Alert acknowledged"}

# Monitoring control
@app.post("/api/monitoring/start/{patient_id}")
async def start_monitoring(patient_id: str, user: dict = Depends(get_admin_user)):
    """Start monitoring for a patient."""
    await vital_service.start(patient_id)
    return {"message": f"Monitoring started for {patient_id}"}

@app.post("/api/monitoring/stop")
async def stop_monitoring(user: dict = Depends(get_admin_user)):
    """Stop all monitoring."""
    await vital_service.stop()
    return {"message": "Monitoring stopped"}

# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    """WebSocket for real-time vital signs streaming."""
    await manager.connect(websocket, patient_id)
    try:
        # Send current data immediately
        if patient_id in vital_service.current_data:
            await websocket.send_json(vital_service.current_data[patient_id].dict())
        
        while True:
            # Keep connection alive and handle client messages with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_text("heartbeat")
                except Exception:
                    break  # Connection is dead
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, patient_id)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
