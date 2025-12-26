"""
VitalFlow-Radar: Confluent Cloud Configuration
===============================================

This module provides centralized configuration for Confluent Cloud integration.
Supports both local Kafka (development) and Confluent Cloud (production).

Environment Variables Required for Confluent Cloud:
    - CONFLUENT_BOOTSTRAP_SERVERS: Confluent Cloud bootstrap server URL
    - CONFLUENT_API_KEY: API key for authentication
    - CONFLUENT_API_SECRET: API secret for authentication
    - CONFLUENT_SCHEMA_REGISTRY_URL: Schema Registry URL (optional)
    - CONFLUENT_SCHEMA_REGISTRY_KEY: Schema Registry API key (optional)
    - CONFLUENT_SCHEMA_REGISTRY_SECRET: Schema Registry API secret (optional)

Usage:
    from confluent_config import get_producer_config, get_consumer_config, TOPICS
    
    producer = Producer(get_producer_config())
    consumer = Consumer(get_consumer_config('my-consumer-group'))
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in the project root
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from {env_path}")
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables only.")


# ============================================================================
# TOPIC DEFINITIONS
# ============================================================================

TOPICS = {
    'phase_stream': 'vitalflow-radar-phase',           # Raw phase data from edge
    'vital_signs': 'vitalflow-vital-signs',            # Processed HR/BR estimates
    'anomalies': 'vitalflow-anomalies',                # Detected health anomalies
    'alerts': 'vitalflow-alerts',                      # Critical alerts for action
}


# ============================================================================
# SCHEMA DEFINITIONS (for Schema Registry)
# ============================================================================

PHASE_SCHEMA = {
    "type": "record",
    "name": "PhaseData",
    "namespace": "com.vitalflow.radar",
    "fields": [
        {"name": "timestamp", "type": "double", "doc": "Unix timestamp in seconds"},
        {"name": "sequence", "type": "int", "doc": "Frame sequence number"},
        {"name": "phase", "type": "double", "doc": "Processed phase value in radians"},
        {"name": "range_bin", "type": "int", "doc": "Selected range bin index"},
        {"name": "range_m", "type": "double", "doc": "Distance to subject in meters"},
        {"name": "signal_quality", "type": "double", "doc": "Signal quality metric 0-1"},
        {"name": "device_id", "type": "string", "doc": "Radar device identifier"},
    ]
}

VITAL_SIGNS_SCHEMA = {
    "type": "record",
    "name": "VitalSigns",
    "namespace": "com.vitalflow.radar",
    "fields": [
        {"name": "timestamp", "type": "double", "doc": "Unix timestamp in seconds"},
        {"name": "window_start", "type": "double", "doc": "Start of analysis window"},
        {"name": "window_end", "type": "double", "doc": "End of analysis window"},
        {"name": "heart_rate_bpm", "type": "double", "doc": "Heart rate in BPM"},
        {"name": "heart_rate_confidence", "type": "double", "doc": "HR confidence 0-1"},
        {"name": "breathing_rate_bpm", "type": "double", "doc": "Breathing rate in BPM"},
        {"name": "breathing_rate_confidence", "type": "double", "doc": "BR confidence 0-1"},
        {"name": "device_id", "type": "string", "doc": "Radar device identifier"},
        {"name": "subject_id", "type": ["null", "string"], "default": None, "doc": "Optional subject ID"},
    ]
}

ANOMALY_SCHEMA = {
    "type": "record",
    "name": "Anomaly",
    "namespace": "com.vitalflow.radar",
    "fields": [
        {"name": "timestamp", "type": "double", "doc": "Detection timestamp"},
        {"name": "anomaly_type", "type": "string", "doc": "Type: bradycardia, tachycardia, apnea, tachypnea, irregular"},
        {"name": "severity", "type": "string", "doc": "Severity: low, medium, high, critical"},
        {"name": "current_value", "type": "double", "doc": "Current vital sign value"},
        {"name": "normal_range_min", "type": "double", "doc": "Normal range minimum"},
        {"name": "normal_range_max", "type": "double", "doc": "Normal range maximum"},
        {"name": "confidence", "type": "double", "doc": "Detection confidence 0-1"},
        {"name": "device_id", "type": "string", "doc": "Radar device identifier"},
        {"name": "description", "type": "string", "doc": "Human-readable description"},
    ]
}


# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def is_confluent_cloud() -> bool:
    """Check if Confluent Cloud credentials are configured."""
    return bool(os.environ.get('CONFLUENT_BOOTSTRAP_SERVERS'))


def get_bootstrap_servers() -> str:
    """Get bootstrap servers - Confluent Cloud or local."""
    return os.environ.get('CONFLUENT_BOOTSTRAP_SERVERS', 'localhost:9092')


def get_producer_config(
    client_id: str = 'vitalflow-producer',
    enable_idempotence: bool = True,
    compression: str = 'snappy',
) -> Dict[str, Any]:
    """
    Get Kafka producer configuration.
    
    Parameters
    ----------
    client_id : str
        Client identifier for tracking
    enable_idempotence : bool
        Enable exactly-once semantics
    compression : str
        Compression type: none, gzip, snappy, lz4, zstd
        
    Returns
    -------
    dict
        Configuration dictionary for confluent_kafka.Producer
    """
    config = {
        'bootstrap.servers': get_bootstrap_servers(),
        'client.id': client_id,
        'compression.type': compression,
        'acks': 'all',  # Wait for all replicas
        'retries': 3,
        'retry.backoff.ms': 100,
        'linger.ms': 5,  # Small batching for low latency
        'batch.size': 16384,
    }
    
    if enable_idempotence:
        config['enable.idempotence'] = True
    
    # Add Confluent Cloud authentication if configured
    if is_confluent_cloud():
        api_key = os.environ.get('CONFLUENT_API_KEY')
        api_secret = os.environ.get('CONFLUENT_API_SECRET')
        
        if api_key and api_secret:
            config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanisms': 'PLAIN',
                'sasl.username': api_key,
                'sasl.password': api_secret,
            })
        else:
            raise ValueError(
                "CONFLUENT_BOOTSTRAP_SERVERS is set but CONFLUENT_API_KEY "
                "and CONFLUENT_API_SECRET are missing"
            )
    
    return config


def get_consumer_config(
    group_id: str,
    client_id: str = 'vitalflow-consumer',
    auto_offset_reset: str = 'latest',
    enable_auto_commit: bool = True,
) -> Dict[str, Any]:
    """
    Get Kafka consumer configuration.
    
    Parameters
    ----------
    group_id : str
        Consumer group identifier
    client_id : str
        Client identifier for tracking
    auto_offset_reset : str
        Where to start: 'earliest', 'latest'
    enable_auto_commit : bool
        Auto commit offsets
        
    Returns
    -------
    dict
        Configuration dictionary for confluent_kafka.Consumer
    """
    config = {
        'bootstrap.servers': get_bootstrap_servers(),
        'group.id': group_id,
        'client.id': client_id,
        'auto.offset.reset': auto_offset_reset,
        'enable.auto.commit': enable_auto_commit,
        'session.timeout.ms': 45000,
        'heartbeat.interval.ms': 15000,
        'max.poll.interval.ms': 300000,
    }
    
    # Add Confluent Cloud authentication if configured
    if is_confluent_cloud():
        api_key = os.environ.get('CONFLUENT_API_KEY')
        api_secret = os.environ.get('CONFLUENT_API_SECRET')
        
        if api_key and api_secret:
            config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanisms': 'PLAIN',
                'sasl.username': api_key,
                'sasl.password': api_secret,
            })
    
    return config


def get_schema_registry_config() -> Optional[Dict[str, Any]]:
    """
    Get Schema Registry configuration if available.
    
    Returns
    -------
    dict or None
        Schema Registry configuration or None if not configured
    """
    url = os.environ.get('CONFLUENT_SCHEMA_REGISTRY_URL')
    if not url:
        return None
    
    config = {'url': url}
    
    key = os.environ.get('CONFLUENT_SCHEMA_REGISTRY_KEY')
    secret = os.environ.get('CONFLUENT_SCHEMA_REGISTRY_SECRET')
    
    if key and secret:
        config['basic.auth.user.info'] = f'{key}:{secret}'
    
    return config


def get_admin_config() -> Dict[str, Any]:
    """
    Get Kafka Admin client configuration for topic management.
    
    Returns
    -------
    dict
        Configuration dictionary for confluent_kafka.admin.AdminClient
    """
    config = {
        'bootstrap.servers': get_bootstrap_servers(),
    }
    
    if is_confluent_cloud():
        api_key = os.environ.get('CONFLUENT_API_KEY')
        api_secret = os.environ.get('CONFLUENT_API_SECRET')
        
        if api_key and api_secret:
            config.update({
                'security.protocol': 'SASL_SSL',
                'sasl.mechanisms': 'PLAIN',
                'sasl.username': api_key,
                'sasl.password': api_secret,
            })
    
    return config


def print_config_status():
    """Print current configuration status for debugging."""
    print("=" * 60)
    print("VitalFlow-Radar Confluent Configuration Status")
    print("=" * 60)
    
    if is_confluent_cloud():
        print("Mode: CONFLUENT CLOUD (Production)")
        print(f"Bootstrap Servers: {get_bootstrap_servers()}")
        print(f"API Key configured: {bool(os.environ.get('CONFLUENT_API_KEY'))}")
        print(f"Schema Registry: {bool(os.environ.get('CONFLUENT_SCHEMA_REGISTRY_URL'))}")
    else:
        print("Mode: LOCAL KAFKA (Development)")
        print(f"Bootstrap Servers: {get_bootstrap_servers()}")
    
    print("\nConfigured Topics:")
    for name, topic in TOPICS.items():
        print(f"  - {name}: {topic}")
    
    print("=" * 60)


if __name__ == '__main__':
    print_config_status()
