import { useState, useEffect } from 'react'
import { useVitals } from '../contexts/VitalsContext'
import { Heart, Wind, Activity, AlertTriangle, TrendingUp, Clock, Radio, Brain, Sparkles, Zap, ArrowUpRight } from 'lucide-react'
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'
import clsx from 'clsx'
import { format } from 'date-fns'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || ''

function VitalCard({
  title,
  value,
  unit,
  confidence,
  icon: Icon,
  status,
  color,
}: {
  title: string
  value: number
  unit: string
  confidence: number
  icon: React.ElementType
  status: string
  color: 'red' | 'green' | 'blue'
}) {
  const colorClasses = {
    red: {
      bg: 'bg-red-500/10',
      text: 'text-red-500',
      border: 'border-red-500/20',
      gradient: 'from-red-500 to-rose-500',
    },
    green: {
      bg: 'bg-green-500/10',
      text: 'text-green-500',
      border: 'border-green-500/20',
      gradient: 'from-green-500 to-emerald-500',
    },
    blue: {
      bg: 'bg-blue-500/10',
      text: 'text-blue-500',
      border: 'border-blue-500/20',
      gradient: 'from-blue-500 to-cyan-500',
    },
  }

  const colors = colorClasses[color]
  const isAbnormal = status === 'warning' || status === 'critical'
  const isCalibrating = status === 'calibrating'

  return (
    <div
      className={clsx(
        'glass rounded-2xl p-6 relative overflow-hidden transition-all',
        isAbnormal && 'ring-2 ring-red-500/50',
        isCalibrating && 'ring-2 ring-blue-500/50'
      )}
    >
      {/* Status indicator */}
      <div
        className={clsx(
          'absolute top-0 left-0 right-0 h-1',
          status === 'critical' && 'bg-red-500',
          status === 'warning' && 'bg-yellow-500',
          status === 'calibrating' && 'bg-blue-500 animate-pulse',
          status === 'normal' && `bg-gradient-to-r ${colors.gradient}`
        )}
      />

      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-gray-400 text-sm font-medium">{title}</p>
        </div>
        <div className={clsx('p-3 rounded-xl', isCalibrating ? 'bg-blue-500/10' : colors.bg)}>
          <Icon className={clsx(
            'w-6 h-6', 
            isCalibrating ? 'text-blue-400' : colors.text, 
            color === 'red' && !isCalibrating && 'animate-heartbeat'
          )} />
        </div>
      </div>

      <div className="flex items-baseline gap-2 mb-4">
        <span
          className={clsx(
            'text-5xl font-bold',
            isCalibrating ? 'text-blue-400' : isAbnormal ? 'text-red-500' : 'text-white'
          )}
        >
          {isCalibrating ? '--' : value.toFixed(0)}
        </span>
        <span className="text-gray-400 text-lg">{unit}</span>
      </div>

      {/* Confidence bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Confidence</span>
          <span className="text-gray-400">{isCalibrating ? '--' : `${(confidence * 100).toFixed(0)}%`}</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={clsx(
              'h-full rounded-full bg-gradient-to-r transition-all duration-500',
              isCalibrating ? 'from-blue-500 to-cyan-500 animate-pulse' : colors.gradient
            )}
            style={{ width: isCalibrating ? '30%' : `${confidence * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}

function StatusCard({ status, alerts }: { status: string; alerts: any[] }) {
  const statusConfig = {
    normal: {
      color: 'bg-green-500',
      text: 'Normal',
      icon: Activity,
      bg: 'bg-green-500/10',
      textColor: 'text-green-500',
    },
    warning: {
      color: 'bg-yellow-500',
      text: 'Warning',
      icon: AlertTriangle,
      bg: 'bg-yellow-500/10',
      textColor: 'text-yellow-500',
    },
    critical: {
      color: 'bg-red-500',
      text: 'Critical',
      icon: AlertTriangle,
      bg: 'bg-red-500/10',
      textColor: 'text-red-500',
    },
    calibrating: {
      color: 'bg-blue-500',
      text: 'Calibrating',
      icon: Radio,
      bg: 'bg-blue-500/10',
      textColor: 'text-blue-500',
    },
  }

  const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.normal
  const Icon = config.icon
  const nonInfoAlerts = alerts.filter(a => a.severity !== 'info')

  return (
    <div className="glass rounded-2xl p-6">
      <h3 className="text-gray-400 text-sm font-medium mb-4">Patient Status</h3>
      
      <div className={clsx('flex items-center gap-3 p-4 rounded-xl mb-4', config.bg)}>
        <div className={clsx('p-2 rounded-lg', config.color, status === 'calibrating' && 'animate-pulse')}>
          <Icon className="w-6 h-6 text-white" />
        </div>
        <div>
          <p className={clsx('text-xl font-bold', config.textColor)}>{config.text}</p>
          <p className="text-gray-500 text-sm">
            {status === 'calibrating' 
              ? 'Sensor initializing...'
              : nonInfoAlerts.length > 0 
                ? `${nonInfoAlerts.length} active alert(s)` 
                : 'All vitals normal'}
          </p>
        </div>
      </div>

      {nonInfoAlerts.length > 0 && (
        <div className="space-y-2">
          {nonInfoAlerts.map((alert, i) => (
            <div
              key={i}
              className={clsx(
                'p-3 rounded-lg text-sm',
                alert.severity === 'critical'
                  ? 'bg-red-500/10 text-red-400'
                  : 'bg-yellow-500/10 text-yellow-400'
              )}
            >
              {alert.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Dashboard() {
  const { currentVitals, vitalsHistory } = useVitals()
  const [monitoringMode, setMonitoringMode] = useState<string>('unknown')
  const [healthSummary, setHealthSummary] = useState<any>(null)
  const [integrations, setIntegrations] = useState<any>({})
  const [streamingStats, setStreamingStats] = useState<any>(null)
  const [alertStats, setAlertStats] = useState<any>(null)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/health`)
        setMonitoringMode(response.data.monitoring_mode || 'simulation')
        setIntegrations(response.data.integrations || {})
        
        // Extract streaming stats from kafka integration
        if (response.data.integrations?.kafka?.stats) {
          setStreamingStats(response.data.integrations.kafka.stats)
        }
        
        // Extract alert stats
        if (response.data.alerts) {
          setAlertStats(response.data.alerts)
        }
      } catch (error) {
        console.error('Failed to fetch mode:', error)
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000) // More frequent updates for streaming
    return () => clearInterval(interval)
  }, [])

  // Listen for health summaries from WebSocket (sent every 30 seconds)
  useEffect(() => {
    // The summary comes through websocket broadcast, we can capture it from vitals context
    // For now, we'll fetch it on demand
    const fetchSummary = async () => {
      if (!currentVitals) return
      try {
        const response = await axios.post(`${API_URL}/api/vertex-ai/analyze?patient_id=default`)
        setHealthSummary(response.data.health_summary)
      } catch (error) {
        // AI not configured, that's okay
      }
    }
    
    // Fetch summary every 30 seconds
    fetchSummary()
    const interval = setInterval(fetchSummary, 30000)
    return () => clearInterval(interval)
  }, [currentVitals?.timestamp])

  const chartData = vitalsHistory.slice(-60).map((v) => ({
    time: format(new Date(v.timestamp * 1000), 'HH:mm:ss'),
    hr: v.heart_rate,
    br: v.breathing_rate,
  }))

  if (!currentVitals) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Connecting to radar...</p>
        </div>
      </div>
    )
  }

  // Handle calibrating status
  const isCalibrating = currentVitals.status === 'calibrating'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold">Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time vital signs monitoring</p>
        </div>
        <div className="flex items-center gap-4">
          {/* Monitoring Mode Badge */}
          <div className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium',
            monitoringMode === 'radar' 
              ? 'bg-green-500/20 text-green-400' 
              : 'bg-yellow-500/20 text-yellow-400'
          )}>
            <Radio className="w-4 h-4" />
            {monitoringMode === 'radar' ? 'Live Radar' : 'Simulation'}
          </div>
          
          {/* Kafka Badge */}
          {integrations.kafka?.connected && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-blue-500/20 text-blue-400">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"></span>
              Kafka
            </div>
          )}
          
          {/* Vertex AI Badge */}
          {integrations.vertex_ai?.initialized && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400">
              <Sparkles className="w-3 h-3" />
              AI
            </div>
          )}
          
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Clock className="w-4 h-4" />
            {format(new Date(currentVitals.timestamp * 1000), 'HH:mm:ss')}
          </div>
        </div>
      </div>

      {/* Calibrating Banner */}
      {isCalibrating && (
        <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-blue-500"></div>
            <div>
              <p className="text-blue-400 font-medium">Sensor Calibrating</p>
              <p className="text-blue-400/70 text-sm">
                {currentVitals.alerts[0]?.message || 'Please wait while the radar calibrates...'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Vital Signs Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
        <VitalCard
          title="Heart Rate"
          value={currentVitals.heart_rate}
          unit="BPM"
          confidence={currentVitals.heart_rate_confidence}
          icon={Heart}
          status={isCalibrating ? 'calibrating' : currentVitals.status}
          color="red"
        />
        <VitalCard
          title="Breathing Rate"
          value={currentVitals.breathing_rate}
          unit="bpm"
          confidence={currentVitals.breathing_rate_confidence}
          icon={Wind}
          status={isCalibrating ? 'calibrating' : currentVitals.status}
          color="green"
        />
        <div className="sm:col-span-2 lg:col-span-1">
          <StatusCard status={currentVitals.status} alerts={currentVitals.alerts} />
        </div>
      </div>

      {/* Chart */}
      <div className="glass rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp className="w-5 h-5 text-primary-500" />
          <h2 className="text-lg font-semibold">Vital Signs Trend</h2>
        </div>

        <div className="h-64 sm:h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="hrGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="brGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="time"
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem',
                }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Area
                type="monotone"
                dataKey="hr"
                stroke="#ef4444"
                strokeWidth={2}
                fill="url(#hrGradient)"
                name="Heart Rate"
              />
              <Area
                type="monotone"
                dataKey="br"
                stroke="#22c55e"
                strokeWidth={2}
                fill="url(#brGradient)"
                name="Breathing Rate"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="flex justify-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-sm text-gray-400">Heart Rate (BPM)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-sm text-gray-400">Breathing Rate (bpm)</span>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Avg HR (1h)', value: vitalsHistory.filter(v => v.heart_rate > 0).length > 0 
            ? (vitalsHistory.filter(v => v.heart_rate > 0).reduce((a, b) => a + b.heart_rate, 0) / vitalsHistory.filter(v => v.heart_rate > 0).length).toFixed(0)
            : '--', unit: 'BPM' },
          { label: 'Avg BR (1h)', value: vitalsHistory.filter(v => v.breathing_rate > 0).length > 0
            ? (vitalsHistory.filter(v => v.breathing_rate > 0).reduce((a, b) => a + b.breathing_rate, 0) / vitalsHistory.filter(v => v.breathing_rate > 0).length).toFixed(0)
            : '--', unit: 'bpm' },
          { label: 'Min HR', value: vitalsHistory.filter(v => v.heart_rate > 0).length > 0
            ? Math.min(...vitalsHistory.filter(v => v.heart_rate > 0).map(v => v.heart_rate)).toFixed(0)
            : '--', unit: 'BPM' },
          { label: 'Max HR', value: vitalsHistory.filter(v => v.heart_rate > 0).length > 0
            ? Math.max(...vitalsHistory.filter(v => v.heart_rate > 0).map(v => v.heart_rate)).toFixed(0)
            : '--', unit: 'BPM' },
        ].map((stat, i) => (
          <div key={i} className="glass rounded-xl p-4 text-center">
            <p className="text-gray-500 text-xs sm:text-sm">{stat.label}</p>
            <p className="text-xl sm:text-2xl font-bold mt-1">
              {stat.value}
              <span className="text-gray-500 text-sm ml-1">{stat.unit}</span>
            </p>
          </div>
        ))}
      </div>

      {/* Confluent Cloud Streaming Status */}
      {integrations.kafka?.connected && (
        <div className="glass rounded-2xl p-6 border border-blue-500/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-blue-500/20">
                <Zap className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Confluent Cloud Streaming</h2>
                <p className="text-xs text-gray-500">Real-time data pipeline</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
              <span className="text-xs text-green-400">Live</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-3">
              <p className="text-gray-500 text-xs">Messages Streamed</p>
              <p className="text-2xl font-bold text-blue-400">
                {streamingStats?.sent || 0}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-3">
              <p className="text-gray-500 text-xs">Throughput</p>
              <p className="text-2xl font-bold text-cyan-400">
                {(streamingStats?.messages_per_second || 0).toFixed(1)}
                <span className="text-xs text-gray-500 ml-1">/sec</span>
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-3">
              <p className="text-gray-500 text-xs">Active Alerts</p>
              <p className="text-2xl font-bold text-yellow-400">
                {alertStats?.unacknowledged || 0}
              </p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-3">
              <p className="text-gray-500 text-xs">Critical</p>
              <p className="text-2xl font-bold text-red-400">
                {alertStats?.by_severity?.critical || 0}
              </p>
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
                vitalflow-vital-signs
              </span>
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span>
                vitalflow-alerts
              </span>
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-yellow-400"></span>
                vitalflow-anomalies
              </span>
            </div>
            <a 
              href="https://confluent.cloud" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-blue-400 hover:text-blue-300"
            >
              Confluent Cloud <ArrowUpRight className="w-3 h-3" />
            </a>
          </div>
        </div>
      )}

      {/* AI Health Summary */}
      {healthSummary && (
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-5 h-5 text-purple-400" />
            <h2 className="text-lg font-semibold">AI Health Summary</h2>
            <span className={clsx(
              'ml-auto px-2 py-0.5 rounded text-xs font-medium',
              healthSummary.risk_level === 'CRITICAL' && 'bg-red-500/20 text-red-400',
              healthSummary.risk_level === 'HIGH' && 'bg-orange-500/20 text-orange-400',
              healthSummary.risk_level === 'MODERATE' && 'bg-yellow-500/20 text-yellow-400',
              healthSummary.risk_level === 'LOW' && 'bg-blue-500/20 text-blue-400',
              healthSummary.risk_level === 'NORMAL' && 'bg-green-500/20 text-green-400'
            )}>
              {healthSummary.risk_level}
            </span>
          </div>
          
          <p className="text-gray-300 mb-4">{healthSummary.summary_text}</p>
          
          {healthSummary.recommendations?.length > 0 && (
            <div className="space-y-2">
              <p className="text-gray-500 text-sm font-medium">Recommendations:</p>
              {healthSummary.recommendations.map((rec: string, i: number) => (
                <div key={i} className="flex items-start gap-2 text-sm text-gray-400">
                  <span className="text-purple-400">•</span>
                  {rec}
                </div>
              ))}
            </div>
          )}
          
          <div className="mt-4 pt-4 border-t border-slate-700 flex items-center justify-between text-xs text-gray-500">
            <span>Powered by {healthSummary.model_used === 'gemini-3-flash-preview' ? 'Google Gemini' : 'Rule-based Analysis'}</span>
            <span>{format(new Date(healthSummary.timestamp * 1000), 'HH:mm:ss')}</span>
          </div>
        </div>
      )}
    </div>
  )
}
