-- VitalFlow-Radar: Flink SQL Stream Processing
-- ================================================
-- Real-time vital signs analytics using Confluent Cloud Flink SQL
-- 
-- This file contains Flink SQL statements for:
-- 1. Windowed aggregations (rolling averages)
-- 2. Anomaly detection rules
-- 3. Patient trend analysis
-- 4. Multi-patient alerting
--
-- To use: Copy these statements into Confluent Cloud Flink SQL editor
-- or run via Confluent CLI

-- ============================================================================
-- STEP 1: Create Source Tables
-- ============================================================================

-- Vital Signs source table (from Kafka topic)
CREATE TABLE vital_signs (
    `timestamp` DOUBLE,
    heart_rate_bpm DOUBLE,
    heart_rate_confidence DOUBLE,
    breathing_rate_bpm DOUBLE,
    breathing_rate_confidence DOUBLE,
    device_id STRING,
    patient_id STRING,
    event_time AS TO_TIMESTAMP(FROM_UNIXTIME(CAST(`timestamp` AS BIGINT))),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'vitalflow-vital-signs',
    'properties.bootstrap.servers' = '${CONFLUENT_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = 'SASL_SSL',
    'properties.sasl.mechanism' = 'PLAIN',
    'properties.sasl.jaas.config' = 'org.apache.kafka.common.security.plain.PlainLoginModule required username="${CONFLUENT_API_KEY}" password="${CONFLUENT_API_SECRET}";',
    'format' = 'json',
    'scan.startup.mode' = 'latest-offset'
);

-- Anomalies source table
CREATE TABLE anomalies (
    `timestamp` DOUBLE,
    anomaly_type STRING,
    severity STRING,
    current_value DOUBLE,
    normal_range_min DOUBLE,
    normal_range_max DOUBLE,
    confidence DOUBLE,
    device_id STRING,
    patient_id STRING,
    description STRING,
    recommended_action STRING,
    event_time AS TO_TIMESTAMP(FROM_UNIXTIME(CAST(`timestamp` AS BIGINT))),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'vitalflow-anomalies',
    'properties.bootstrap.servers' = '${CONFLUENT_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = 'SASL_SSL',
    'properties.sasl.mechanism' = 'PLAIN',
    'properties.sasl.jaas.config' = 'org.apache.kafka.common.security.plain.PlainLoginModule required username="${CONFLUENT_API_KEY}" password="${CONFLUENT_API_SECRET}";',
    'format' = 'json',
    'scan.startup.mode' = 'latest-offset'
);

-- ============================================================================
-- STEP 2: Create Sink Tables (for processed results)
-- ============================================================================

-- Rolling averages sink
CREATE TABLE vital_signs_averages (
    patient_id STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    avg_heart_rate DOUBLE,
    max_heart_rate DOUBLE,
    min_heart_rate DOUBLE,
    avg_breathing_rate DOUBLE,
    avg_confidence DOUBLE,
    measurement_count BIGINT,
    PRIMARY KEY (patient_id, window_start) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'vitalflow-vital-signs-aggregated',
    'properties.bootstrap.servers' = '${CONFLUENT_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = 'SASL_SSL',
    'properties.sasl.mechanism' = 'PLAIN',
    'properties.sasl.jaas.config' = 'org.apache.kafka.common.security.plain.PlainLoginModule required username="${CONFLUENT_API_KEY}" password="${CONFLUENT_API_SECRET}";',
    'format' = 'json',
    'key.format' = 'json'
);

-- Critical alerts sink (escalated anomalies)
CREATE TABLE critical_alerts (
    patient_id STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    alert_type STRING,
    alert_count BIGINT,
    max_severity STRING,
    requires_immediate_action BOOLEAN,
    latest_value DOUBLE,
    PRIMARY KEY (patient_id, window_start, alert_type) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'vitalflow-critical-alerts',
    'properties.bootstrap.servers' = '${CONFLUENT_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = 'SASL_SSL',
    'properties.sasl.mechanism' = 'PLAIN',
    'properties.sasl.jaas.config' = 'org.apache.kafka.common.security.plain.PlainLoginModule required username="${CONFLUENT_API_KEY}" password="${CONFLUENT_API_SECRET}";',
    'format' = 'json',
    'key.format' = 'json'
);

-- Patient trend analysis sink
CREATE TABLE patient_trends (
    patient_id STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    hr_trend STRING,  -- 'increasing', 'decreasing', 'stable'
    br_trend STRING,
    hr_variability DOUBLE,
    br_variability DOUBLE,
    health_score DOUBLE,
    PRIMARY KEY (patient_id, window_start) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'vitalflow-patient-trends',
    'properties.bootstrap.servers' = '${CONFLUENT_BOOTSTRAP_SERVERS}',
    'properties.security.protocol' = 'SASL_SSL',
    'properties.sasl.mechanism' = 'PLAIN',
    'properties.sasl.jaas.config' = 'org.apache.kafka.common.security.plain.PlainLoginModule required username="${CONFLUENT_API_KEY}" password="${CONFLUENT_API_SECRET}";',
    'format' = 'json',
    'key.format' = 'json'
);

-- ============================================================================
-- STEP 3: Streaming Analytics Queries
-- ============================================================================

-- Query 1: 1-Minute Rolling Averages
-- Computes running statistics for vital signs
INSERT INTO vital_signs_averages
SELECT 
    patient_id,
    TUMBLE_START(event_time, INTERVAL '1' MINUTE) AS window_start,
    TUMBLE_END(event_time, INTERVAL '1' MINUTE) AS window_end,
    AVG(heart_rate_bpm) AS avg_heart_rate,
    MAX(heart_rate_bpm) AS max_heart_rate,
    MIN(heart_rate_bpm) AS min_heart_rate,
    AVG(breathing_rate_bpm) AS avg_breathing_rate,
    AVG(heart_rate_confidence) AS avg_confidence,
    COUNT(*) AS measurement_count
FROM vital_signs
GROUP BY 
    patient_id,
    TUMBLE(event_time, INTERVAL '1' MINUTE);

-- Query 2: Critical Alert Escalation
-- Detects sustained anomalies requiring immediate attention
INSERT INTO critical_alerts
SELECT
    patient_id,
    TUMBLE_START(event_time, INTERVAL '30' SECOND) AS window_start,
    TUMBLE_END(event_time, INTERVAL '30' SECOND) AS window_end,
    anomaly_type AS alert_type,
    COUNT(*) AS alert_count,
    MAX(severity) AS max_severity,
    CASE 
        WHEN MAX(severity) = 'critical' AND COUNT(*) >= 2 THEN TRUE
        WHEN MAX(severity) = 'high' AND COUNT(*) >= 3 THEN TRUE
        ELSE FALSE 
    END AS requires_immediate_action,
    LAST_VALUE(current_value) AS latest_value
FROM anomalies
WHERE severity IN ('critical', 'high', 'medium')
GROUP BY 
    patient_id,
    anomaly_type,
    TUMBLE(event_time, INTERVAL '30' SECOND)
HAVING COUNT(*) >= 2;

-- Query 3: Patient Trend Analysis
-- Analyzes trends over 5-minute windows
INSERT INTO patient_trends
SELECT
    patient_id,
    TUMBLE_START(event_time, INTERVAL '5' MINUTE) AS window_start,
    TUMBLE_END(event_time, INTERVAL '5' MINUTE) AS window_end,
    -- Heart rate trend
    CASE
        WHEN LAST_VALUE(heart_rate_bpm) - FIRST_VALUE(heart_rate_bpm) > 10 THEN 'increasing'
        WHEN FIRST_VALUE(heart_rate_bpm) - LAST_VALUE(heart_rate_bpm) > 10 THEN 'decreasing'
        ELSE 'stable'
    END AS hr_trend,
    -- Breathing rate trend
    CASE
        WHEN LAST_VALUE(breathing_rate_bpm) - FIRST_VALUE(breathing_rate_bpm) > 3 THEN 'increasing'
        WHEN FIRST_VALUE(breathing_rate_bpm) - LAST_VALUE(breathing_rate_bpm) > 3 THEN 'decreasing'
        ELSE 'stable'
    END AS br_trend,
    -- Variability (standard deviation approximation)
    MAX(heart_rate_bpm) - MIN(heart_rate_bpm) AS hr_variability,
    MAX(breathing_rate_bpm) - MIN(breathing_rate_bpm) AS br_variability,
    -- Health score (0-100, higher is better)
    CAST(
        100 - 
        ABS(AVG(heart_rate_bpm) - 72) * 0.5 -  -- Deviation from ideal HR
        ABS(AVG(breathing_rate_bpm) - 14) * 1.0 -  -- Deviation from ideal BR
        (MAX(heart_rate_bpm) - MIN(heart_rate_bpm)) * 0.3  -- Penalize high variability
        AS DOUBLE
    ) AS health_score
FROM vital_signs
GROUP BY 
    patient_id,
    TUMBLE(event_time, INTERVAL '5' MINUTE);

-- ============================================================================
-- STEP 4: Real-Time Views (for dashboard queries)
-- ============================================================================

-- View: Current Patient Status
-- Real-time patient vital signs with classification
CREATE VIEW current_patient_status AS
SELECT 
    patient_id,
    heart_rate_bpm,
    breathing_rate_bpm,
    heart_rate_confidence,
    breathing_rate_confidence,
    CASE
        WHEN heart_rate_bpm < 40 OR heart_rate_bpm > 150 THEN 'CRITICAL'
        WHEN heart_rate_bpm < 50 OR heart_rate_bpm > 120 THEN 'WARNING'
        WHEN breathing_rate_bpm < 8 OR breathing_rate_bpm > 25 THEN 'WARNING'
        ELSE 'NORMAL'
    END AS status,
    event_time
FROM vital_signs;

-- View: Multi-Patient Overview
-- Aggregated view of all patients being monitored
CREATE VIEW ward_overview AS
SELECT 
    COUNT(DISTINCT patient_id) AS total_patients,
    SUM(CASE WHEN heart_rate_bpm < 40 OR heart_rate_bpm > 150 THEN 1 ELSE 0 END) AS critical_count,
    SUM(CASE WHEN heart_rate_bpm < 50 OR heart_rate_bpm > 120 THEN 1 ELSE 0 END) AS warning_count,
    AVG(heart_rate_bpm) AS ward_avg_hr,
    AVG(breathing_rate_bpm) AS ward_avg_br,
    TUMBLE_START(event_time, INTERVAL '1' MINUTE) AS window_start
FROM vital_signs
GROUP BY TUMBLE(event_time, INTERVAL '1' MINUTE);

-- ============================================================================
-- STEP 5: Advanced Pattern Detection
-- ============================================================================

-- Detect repeated anomalies pattern (potential deterioration)
SELECT
    a1.patient_id,
    a1.anomaly_type,
    COUNT(*) AS occurrence_count,
    'DETERIORATION_PATTERN' AS alert_type,
    'Patient showing repeated ' || a1.anomaly_type || ' episodes. Consider escalation.' AS message
FROM anomalies a1
JOIN anomalies a2 
    ON a1.patient_id = a2.patient_id 
    AND a1.anomaly_type = a2.anomaly_type
    AND a1.event_time BETWEEN a2.event_time - INTERVAL '10' MINUTE AND a2.event_time
WHERE a1.severity IN ('high', 'critical')
GROUP BY a1.patient_id, a1.anomaly_type
HAVING COUNT(*) >= 3;

-- Detect sudden changes (spike detection)
SELECT
    patient_id,
    heart_rate_bpm,
    LAG(heart_rate_bpm, 1) OVER (PARTITION BY patient_id ORDER BY event_time) AS prev_hr,
    ABS(heart_rate_bpm - LAG(heart_rate_bpm, 1) OVER (PARTITION BY patient_id ORDER BY event_time)) AS hr_change,
    CASE 
        WHEN ABS(heart_rate_bpm - LAG(heart_rate_bpm, 1) OVER (PARTITION BY patient_id ORDER BY event_time)) > 30 
        THEN 'SUDDEN_SPIKE'
        ELSE 'NORMAL'
    END AS spike_detected,
    event_time
FROM vital_signs
WHERE ABS(heart_rate_bpm - LAG(heart_rate_bpm, 1) OVER (PARTITION BY patient_id ORDER BY event_time)) > 30;
