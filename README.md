# VitalFlow-Radar 🫀📡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)
![Radar](https://img.shields.io/badge/Radar-AWR1642-blue)
![Streaming](https://img.shields.io/badge/Streaming-Confluent%20Kafka-000000)
![AI](https://img.shields.io/badge/AI-Vertex%20AI-4285F4)

> **🏆 Confluent Challenge Submission - Google Cloud x Confluent Hackathon**
>
> *Unleashing AI on Data in Motion: Real-Time Contactless Vital Signs Monitoring*

**VitalFlow-Radar** is a next-generation healthcare application that combines **77GHz mmWave radar sensing**, **Confluent Cloud real-time streaming**, and **Vertex AI Gemini** to enable contactless vital signs monitoring at scale. The system demonstrates how real-time data streaming unlocks critical healthcare challenges—enabling hospitals to monitor multiple patients simultaneously without any physical contact.

![VitalFlow Dashboard](imgs/front-end.png)

---

## 🎯 Challenge Response

### The Problem We're Solving

**Traditional vital signs monitoring** requires physical contact—pulse oximeters, ECG leads, chest straps. This creates:
- **Infection risk** in hospital settings
- **Discomfort** for long-term monitoring (sleep, pediatrics)
- **Scalability limits** (1 nurse : 4-6 patients)
- **Alert fatigue** from motion artifacts

### Our Real-Time AI Solution

VitalFlow-Radar streams **mmWave radar data** through Confluent Cloud to:

1. **Detect heartbeats and breathing** through clothing, at distance (0.3-1.5m)
2. **Process in real-time** using edge DSP + cloud AI
3. **Predict anomalies** before they become critical
4. **Scale to thousands of patients** with a single cloud backend

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   EDGE DEVICE   │     │   CONFLUENT CLOUD    │     │   VERTEX AI         │
│   AWR1642 Radar │────▶│   Apache Kafka       │────▶│   Gemini 1.5        │
│                 │     │                      │     │                     │
│ • 20 Hz sampling│     │ • vitalflow-phase    │     │ • Anomaly detection │
│ • Range FFT     │     │ • vitalflow-vitals   │     │ • Health summaries  │
│ • Phase extract │     │ • vitalflow-anomaly  │     │ • Predictive alerts │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   REAL-TIME DASHBOARD │
                    │   React + WebSocket   │
                    │   Live vitals display │
                    └──────────────────────┘
```

---

## 🌟 Key Features

### Real-Time Data Streaming with Confluent

| Feature | Implementation |
|---------|----------------|
| **High-Frequency Ingestion** | 20 Hz radar phase data → Kafka |
| **Multi-Topic Architecture** | Phase data, vital signs, anomalies, alerts |
| **Scalable Consumers** | Multiple dashboard instances, AI processors |
| **Low Latency** | End-to-end <100ms radar → dashboard |

### AI-Powered Insights with Vertex AI

- **Gemini 1.5 Flash** for real-time health summaries
- **Anomaly Detection**: Bradycardia, tachycardia, apnea, tachypnea
- **Pediatric-Specific** algorithms (children have different normal ranges)
- **Natural Language Alerts**: "Patient showing signs of respiratory distress"

### Edge Processing

- **FMCW Radar DSP**: Range FFT, MTI filtering, phase extraction
- **Variance-Based Tracking**: Automatically finds the chest signal
- **Motion Artifact Rejection**: Handles patient movement
- **Low Power**: Runs on Raspberry Pi 4

---

## 🚀 Quick Start

### ⚡ Instant Demo (Hackathon Judges)

**No cloud credentials required!** Test the full system with one command:

```bash
git clone https://github.com/ahmed100553/VitalFlow-Radar.git
cd VitalFlow-Radar

# Make the demo script executable
chmod +x scripts/start_demo.sh

# Start everything (backend + frontend + demo traffic)
./scripts/start_demo.sh
```

Then open http://localhost:5173 and watch the live vital signs demo!

- **Demo login**: `admin` / `admin123`
- **API Docs**: http://localhost:8000/docs
- **Demo scenarios**: Normal → Tachycardia → Apnea → Recovery

---

### Full Setup (With Confluent Cloud)

#### Prerequisites

1. **[Confluent Cloud Account](https://www.confluent.io/confluent-cloud/tryfree/)** (Free trial with code: `CONFLUENTDEV1`)
2. **[Google Cloud Account](https://cloud.google.com/)** with Vertex AI enabled
3. **Python 3.10+**
4. **Node.js 18+** (for frontend)

#### 1. Clone & Configure

```bash
git clone https://github.com/ahmed100553/VitalFlow-Radar.git
cd VitalFlow-Radar

# Copy environment template
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Confluent Cloud (Required for streaming)
CONFLUENT_BOOTSTRAP_SERVERS=pkc-xxxxx.us-central1.gcp.confluent.cloud:9092
CONFLUENT_API_KEY=your-api-key
CONFLUENT_API_SECRET=your-api-secret

# Google Cloud Vertex AI (Required for AI summaries)
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_AI_LOCATION=us-central1

# Optional: Schema Registry
CONFLUENT_SCHEMA_REGISTRY_URL=https://psrc-xxxxx.us-central1.gcp.confluent.cloud
```

#### 2. Create Kafka Topics

In Confluent Cloud Console, create these topics:

| Topic Name | Partitions | Description |
|------------|------------|-------------|
| `vitalflow-radar-phase` | 6 | Raw phase data from radar |
| `vitalflow-vital-signs` | 6 | Computed HR/BR |
| `vitalflow-anomalies` | 3 | Detected anomalies |
| `vitalflow-alerts` | 3 | Critical alerts |

#### 3. Install Dependencies

```bash
# Backend
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

#### 4. Run the System

**Option A: With Kafka Traffic Generator (Recommended)**

```bash
# Terminal 1: Start backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend  
cd frontend && npm run dev

# Terminal 3: Generate Kafka traffic
python scripts/traffic_generator.py --scenario all
```

**Option B: With Real Radar Hardware**

```bash
# Terminal 1: Start backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend && npm run dev

# Terminal 3: Run edge producer (connects to AWR1642)
python edge_producer_live.py
```

#### 5. Access Dashboard

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **Login**: `admin` / `admin123`

---

## 📊 Traffic Generator (Hackathon Demo)

The traffic generator simulates realistic vital signs scenarios to demonstrate the system:

```bash
# Run all scenarios sequentially
python scripts/traffic_generator.py --scenario all

# Specific scenario
python scripts/traffic_generator.py --scenario tachycardia

# Multi-patient concurrent monitoring (scalability demo)
python scripts/traffic_generator.py --multi-patient 5 --duration 120

# Continuous demo mode for presentations
python scripts/traffic_generator.py --continuous --duration 300
```

### Available Scenarios

| Scenario | Description | Anomalies Generated |
|----------|-------------|---------------------|
| `normal` | Baseline healthy vitals | None |
| `tachycardia` | Heart rate 75→140→85 BPM | Tachycardia alerts |
| `bradycardia` | Heart rate drops to 42 BPM | Bradycardia alerts |
| `apnea` | Breathing pause event | Apnea critical alerts |
| `stress` | Elevated HR + BR | Warning alerts |
| `sleep` | Low resting vitals | None |
| `pediatric` | Child-appropriate higher HR | None (adjusted norms) |

---

## 🏗️ Architecture Deep Dive

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VitalFlow-Radar Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │   AWR1642 Radar  │  77GHz FMCW mmWave                                   │
│  │   (TI EVM)       │  • 10 FPS frame rate                                 │
│  └────────┬─────────┘  • 256 ADC samples                                   │
│           │            • 16 chirps/frame                                    │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │  Edge Producer   │  edge_producer_live.py                               │
│  │  (Raspberry Pi)  │                                                       │
│  │                  │  DSP Pipeline:                                        │
│  │  • Range FFT     │  1. ADC → Range bins                                 │
│  │  • MTI Filter    │  2. Remove static clutter                            │
│  │  • Phase Extract │  3. Find chest bin (0.3-1.5m)                        │
│  └────────┬─────────┘  4. Extract phase signal                             │
│           │                                                                  │
│           ▼                                                                  │
│  ╔══════════════════════════════════════════════════════════════╗          │
│  ║              CONFLUENT CLOUD (Apache Kafka)                   ║          │
│  ║                                                               ║          │
│  ║  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ ║          │
│  ║  │ vitalflow-      │  │ vitalflow-      │  │ vitalflow-   │ ║          │
│  ║  │ radar-phase     │  │ vital-signs     │  │ anomalies    │ ║          │
│  ║  │                 │  │                 │  │              │ ║          │
│  ║  │ 20 Hz raw data  │  │ HR/BR every 3s  │  │ AI-detected  │ ║          │
│  ║  └─────────────────┘  └─────────────────┘  └──────────────┘ ║          │
│  ╚══════════════════════════════════════════════════════════════╝          │
│           │                      │                    │                     │
│           ▼                      ▼                    ▼                     │
│  ┌──────────────────┐   ┌──────────────────┐  ┌─────────────────┐         │
│  │  Cloud Processor │   │   FastAPI        │  │   Vertex AI     │         │
│  │                  │   │   Backend        │  │   Gemini 1.5    │         │
│  │  • STFT analysis │   │                  │  │                 │         │
│  │  • Vital compute │   │  • Kafka consume │  │  • Anomaly AI   │         │
│  │  • Trend detect  │   │  • WebSocket     │  │  • Health sums  │         │
│  └──────────────────┘   │  • REST API      │  │  • Predictions  │         │
│                         └────────┬─────────┘  └─────────────────┘         │
│                                  │                                         │
│                                  ▼                                         │
│                         ┌──────────────────┐                              │
│                         │   React Frontend │                              │
│                         │                  │                              │
│                         │  • Real-time     │                              │
│                         │    charts        │                              │
│                         │  • AI insights   │                              │
│                         │  • Alert history │                              │
│                         └──────────────────┘                              │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Message Schemas

**Phase Data (Edge → Cloud)**
```json
{
  "timestamp": 1703260800.123,
  "sequence": 42,
  "phase": 0.0523,
  "range_bin": 15,
  "range_m": 0.66,
  "signal_quality": 0.85,
  "device_id": "radar-a1b2c3d4",
  "patient_id": "patient-001"
}
```

**Vital Signs (Processed)**
```json
{
  "timestamp": 1703260830.456,
  "heart_rate_bpm": 72.5,
  "heart_rate_confidence": 0.89,
  "breathing_rate_bpm": 14.2,
  "breathing_rate_confidence": 0.85,
  "device_id": "radar-a1b2c3d4",
  "patient_id": "patient-001"
}
```

**Anomaly Alert**
```json
{
  "timestamp": 1703260860.789,
  "anomaly_type": "tachycardia",
  "severity": "medium",
  "current_value": 112.3,
  "normal_range_min": 60,
  "normal_range_max": 100,
  "description": "Elevated heart rate detected",
  "recommended_action": "Monitor for sustained elevation",
  "ai_summary": "Patient showing signs of elevated cardiac activity..."
}
```

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Hardware** | TI AWR1642 | 77GHz FMCW mmWave radar |
| **Edge** | Python, NumPy | DSP processing, Kafka producer |
| **Streaming** | Confluent Cloud | Apache Kafka managed service |
| **Backend** | FastAPI, WebSocket | REST API, real-time updates |
| **AI** | Vertex AI Gemini | Anomaly detection, health summaries |
| **Frontend** | React, TypeScript, TailwindCSS | Real-time dashboard |
| **Database** | SQLite | Alerts, patient data |

---

## 📁 Project Structure

```
VitalFlow-Radar/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── contexts/        # State management
│   │   └── pages/           # Dashboard, Patients, Alerts
│   └── package.json
├── scripts/
│   ├── traffic_generator.py # 🆕 Hackathon demo traffic
│   ├── start_dev.sh         # Development server
│   └── setup_raspberry_pi.sh
├── awr1642_driver.py        # Radar TLV parser
├── vital_signs_processor.py # DSP algorithms
├── edge_producer_live.py    # Edge → Kafka streaming
├── confluent_config.py      # Kafka configuration
├── vertex_ai_processor.py   # AI anomaly detection
└── vital_signs_awr1642.cfg  # Radar parameters
```

---

## 🧪 Running Tests

```bash
# Test Confluent connectivity
python -c "from confluent_config import print_config_status; print_config_status()"

# Test Vertex AI
python -c "from vertex_ai_processor import VitalSignsAnomalyDetector; d = VitalSignsAnomalyDetector(); d.initialize(); print('✅ Vertex AI ready')"

# Run traffic generator (tests full pipeline)
python scripts/traffic_generator.py --scenario normal
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **End-to-End Latency** | <100ms |
| **Frame Rate** | 10-20 FPS |
| **Heart Rate Accuracy** | ±2 BPM (vs reference) |
| **Breathing Rate Accuracy** | ±1 BPM |
| **Effective Range** | 0.3 - 1.5m |
| **Kafka Throughput** | 1000+ messages/sec |

---

## 🎥 Video Demo

[Watch my demo for AI Partner Catalyst](https://www.youtube.com/watch?v=wP1_eyVGeQI)

---

## 📚 Resources

- [Confluent Cloud Documentation](https://docs.confluent.io/cloud/current/)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [TI AWR1642 mmWave Radar](https://www.ti.com/tool/AWR1642BOOST)
- [Build AI with Confluent](https://docs.confluent.io/cloud/current/ai/overview.html)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Confluent** for the streaming platform and hackathon challenge
- **Google Cloud** for Vertex AI and Gemini
- **Texas Instruments** for AWR1642 mmWave SDK

---

<p align="center">
  <b>Built with ❤️ for the Google Cloud x Confluent Hackathon</b>
  <br>
  <i>Real-time AI on Data in Motion for Healthcare</i>
</p>
