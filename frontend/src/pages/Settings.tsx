import { useState, useEffect } from 'react'
import { Wifi, Bell, Shield, Server, Radio, RefreshCw, Cloud, Brain, Zap, AlertTriangle, Play } from 'lucide-react'
import axios from 'axios'
import { useAuth } from '../contexts/AuthContext'
import { useVitals } from '../contexts/VitalsContext'
import clsx from 'clsx'

const API_URL = import.meta.env.VITE_API_URL || ''

export default function Settings() {
  const { user } = useAuth()
  const { isConnected, patientId, setPatientId } = useVitals()
  const [health, setHealth] = useState<any>(null)
  const [connecting, setConnecting] = useState(false)
  const [connectingKafka, setConnectingKafka] = useState(false)
  const [initializingAI, setInitializingAI] = useState(false)
  const [triggeringAnomaly, setTriggeringAnomaly] = useState<string | null>(null)
  const [alertStats, setAlertStats] = useState<any>(null)

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`)
      setHealth(response.data)
    } catch (error) {
      console.error('Failed to fetch health:', error)
    }
  }

  const fetchAlertStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/alerts/stats`)
      setAlertStats(response.data)
    } catch (error) {
      console.error('Failed to fetch alert stats:', error)
    }
  }

  useEffect(() => {
    fetchHealth()
    fetchAlertStats()
    const interval = setInterval(() => {
      fetchHealth()
      fetchAlertStats()
    }, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleConnectRadar = async () => {
    setConnecting(true)
    try {
      await axios.post(`${API_URL}/api/radar/connect`)
      await fetchHealth()
    } catch (error) {
      console.error('Failed to connect radar:', error)
    }
    setConnecting(false)
  }

  const handleDisconnectRadar = async () => {
    setConnecting(true)
    try {
      await axios.post(`${API_URL}/api/radar/disconnect`)
      await fetchHealth()
    } catch (error) {
      console.error('Failed to disconnect radar:', error)
    }
    setConnecting(false)
  }

  const handleConnectKafka = async () => {
    setConnectingKafka(true)
    try {
      await axios.post(`${API_URL}/api/kafka/connect`)
      await fetchHealth()
    } catch (error) {
      console.error('Failed to connect Kafka:', error)
    }
    setConnectingKafka(false)
  }

  const handleInitializeAI = async () => {
    setInitializingAI(true)
    try {
      await axios.post(`${API_URL}/api/vertex-ai/initialize`)
      await fetchHealth()
    } catch (error) {
      console.error('Failed to initialize Vertex AI:', error)
    }
    setInitializingAI(false)
  }

  const handleTriggerAnomaly = async (anomalyType: string) => {
    setTriggeringAnomaly(anomalyType)
    try {
      await axios.post(`${API_URL}/api/alerts/trigger/${anomalyType}`)
      // Refresh stats after triggering
      await fetchAlertStats()
      await fetchHealth()
    } catch (error) {
      console.error('Failed to trigger anomaly:', error)
    }
    setTriggeringAnomaly(null)
  }

  const radarStatus = health?.integrations?.radar || health?.radar || {}
  const kafkaStatus = health?.integrations?.kafka || {}
  const vertexStatus = health?.integrations?.vertex_ai || {}

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold">Settings</h1>
        <p className="text-gray-400 mt-1">System configuration and status</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Radar Sensor Status */}
        <div className="glass rounded-2xl p-6 lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className={clsx(
                'p-3 rounded-xl',
                radarStatus.is_connected ? 'bg-green-500/20' : 'bg-yellow-500/20'
              )}>
                <Radio className={clsx(
                  'w-6 h-6',
                  radarStatus.is_connected ? 'text-green-400' : 'text-yellow-400'
                )} />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Radar Sensor</h2>
                <p className="text-sm text-gray-400">AWR1642 mmWave Radar</p>
              </div>
            </div>
            {user?.role === 'admin' && (
              <button
                onClick={radarStatus.is_connected ? handleDisconnectRadar : handleConnectRadar}
                disabled={connecting}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                  radarStatus.is_connected
                    ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                    : 'bg-green-500/20 text-green-400 hover:bg-green-500/30',
                  connecting && 'opacity-50 cursor-not-allowed'
                )}
              >
                <RefreshCw className={clsx('w-4 h-4', connecting && 'animate-spin')} />
                {connecting ? 'Processing...' : radarStatus.is_connected ? 'Disconnect' : 'Connect Sensor'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Mode</p>
              <p className={clsx(
                'font-semibold',
                health?.monitoring_mode === 'radar' ? 'text-green-400' : 'text-yellow-400'
              )}>
                {health?.monitoring_mode === 'radar' ? '🟢 Live Radar' : '🟡 Simulation'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Sensor Detected</p>
              <p className={clsx(
                'font-semibold',
                radarStatus.sensor_detected ? 'text-green-400' : 'text-red-400'
              )}>
                {radarStatus.sensor_detected ? '✓ Yes' : '✗ No'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">CLI Port</p>
              <p className="font-mono text-sm">{radarStatus.cli_port || '/dev/ttyACM0'}</p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Data Port</p>
              <p className="font-mono text-sm">{radarStatus.data_port || '/dev/ttyACM1'}</p>
            </div>
          </div>

          {radarStatus.last_error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-red-400 text-sm">
                <strong>Error:</strong> {radarStatus.last_error}
              </p>
            </div>
          )}

          {!radarStatus.sensor_detected && (
            <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <p className="text-blue-400 text-sm">
                <strong>Tip:</strong> Connect your AWR1642 radar via USB. The system will auto-detect 
                serial ports at <code className="bg-slate-700 px-1 rounded">/dev/ttyACM0</code> and 
                <code className="bg-slate-700 px-1 rounded">/dev/ttyACM1</code>.
              </p>
            </div>
          )}
        </div>

        {/* Confluent Kafka Status */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className={clsx(
                'p-3 rounded-xl',
                kafkaStatus.connected ? 'bg-blue-500/20' : 'bg-gray-500/20'
              )}>
                <Cloud className={clsx(
                  'w-6 h-6',
                  kafkaStatus.connected ? 'text-blue-400' : 'text-gray-400'
                )} />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Confluent Kafka</h2>
                <p className="text-sm text-gray-400">Real-time Data Streaming</p>
              </div>
            </div>
            {user?.role === 'admin' && !kafkaStatus.connected && (
              <button
                onClick={handleConnectKafka}
                disabled={connectingKafka}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                  'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30',
                  connectingKafka && 'opacity-50 cursor-not-allowed'
                )}
              >
                <RefreshCw className={clsx('w-4 h-4', connectingKafka && 'animate-spin')} />
                {connectingKafka ? 'Connecting...' : 'Connect'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Status</p>
              <p className={clsx(
                'font-semibold',
                kafkaStatus.connected ? 'text-green-400' : 'text-gray-400'
              )}>
                {kafkaStatus.connected ? '🟢 Connected' : '⚪ Offline'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Environment</p>
              <p className={clsx(
                'font-semibold text-sm',
                kafkaStatus.is_confluent_cloud ? 'text-blue-400' : 'text-yellow-400'
              )}>
                {kafkaStatus.is_confluent_cloud ? '☁️ Confluent Cloud' : '🖥️ Local'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Messages Streamed</p>
              <p className="font-mono text-lg text-blue-400">{kafkaStatus.stats?.sent || 0}</p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Msg/sec</p>
              <p className="font-mono text-lg text-green-400">{kafkaStatus.stats?.messages_per_second || 0}</p>
            </div>
          </div>

          {kafkaStatus.last_error && (
            <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <p className="text-yellow-400 text-sm">
                <strong>Note:</strong> {kafkaStatus.last_error}
              </p>
            </div>
          )}
        </div>

        {/* Vertex AI / Gemini Status */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <div className={clsx(
                'p-3 rounded-xl',
                vertexStatus.initialized ? 'bg-purple-500/20' : 'bg-gray-500/20'
              )}>
                <Brain className={clsx(
                  'w-6 h-6',
                  vertexStatus.initialized ? 'text-purple-400' : 'text-gray-400'
                )} />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Google Vertex AI</h2>
                <p className="text-sm text-gray-400">Anomaly Detection & Gemini</p>
              </div>
            </div>
            {user?.role === 'admin' && !vertexStatus.initialized && (
              <button
                onClick={handleInitializeAI}
                disabled={initializingAI}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all',
                  'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30',
                  initializingAI && 'opacity-50 cursor-not-allowed'
                )}
              >
                <Zap className={clsx('w-4 h-4', initializingAI && 'animate-pulse')} />
                {initializingAI ? 'Initializing...' : 'Initialize'}
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Status</p>
              <p className={clsx(
                'font-semibold',
                vertexStatus.initialized ? 'text-green-400' : 'text-gray-400'
              )}>
                {vertexStatus.initialized ? '🟢 Active' : '⚪ Offline'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Gemini API</p>
              <p className={clsx(
                'font-semibold',
                vertexStatus.gemini_available ? 'text-purple-400' : 'text-gray-400'
              )}>
                {vertexStatus.gemini_available ? '✨ Available' : '— Not Configured'}
              </p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">Anomalies Detected</p>
              <p className="font-mono text-lg text-yellow-400">{vertexStatus.stats?.anomalies_detected || 0}</p>
            </div>
            
            <div className="bg-slate-800/50 rounded-xl p-4">
              <p className="text-gray-400 text-sm mb-1">AI Summaries</p>
              <p className="font-mono text-lg text-purple-400">{vertexStatus.stats?.summaries_generated || 0}</p>
            </div>
          </div>

          {vertexStatus.last_error && (
            <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
              <p className="text-yellow-400 text-sm">
                <strong>Note:</strong> {vertexStatus.last_error}
              </p>
            </div>
          )}

          {!vertexStatus.initialized && (
            <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-lg">
              <p className="text-purple-400 text-sm">
                <strong>AI Features:</strong> Set <code className="bg-slate-700 px-1 rounded">GOOGLE_CLOUD_PROJECT</code> environment 
                variable to enable Vertex AI anomaly detection and Gemini health summaries.
              </p>
            </div>
          )}
        </div>

        {/* System Status */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl bg-primary-500/20">
              <Server className="w-6 h-6 text-primary-400" />
            </div>
            <h2 className="text-lg font-semibold">System Status</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-700">
              <span className="text-gray-400">API Status</span>
              <span
                className={clsx(
                  'px-3 py-1 rounded-full text-xs font-medium',
                  health?.status === 'healthy'
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-red-500/20 text-red-400'
                )}
              >
                {health?.status || 'Unknown'}
              </span>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-slate-700">
              <span className="text-gray-400">WebSocket Connection</span>
              <span
                className={clsx(
                  'px-3 py-1 rounded-full text-xs font-medium',
                  isConnected
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-red-500/20 text-red-400'
                )}
              >
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-slate-700">
              <span className="text-gray-400">Monitoring Active</span>
              <span
                className={clsx(
                  'px-3 py-1 rounded-full text-xs font-medium',
                  health?.monitoring_active
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-red-500/20 text-red-400'
                )}
              >
                {health?.monitoring_active ? 'Active' : 'Inactive'}
              </span>
            </div>

            <div className="flex items-center justify-between py-3">
              <span className="text-gray-400">API Version</span>
              <span className="text-white">{health?.version || '--'}</span>
            </div>
          </div>
        </div>

        {/* User Info */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl bg-blue-500/20">
              <Shield className="w-6 h-6 text-blue-400" />
            </div>
            <h2 className="text-lg font-semibold">Account</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-700">
              <span className="text-gray-400">Role</span>
              <span
                className={clsx(
                  'px-3 py-1 rounded-full text-xs font-medium capitalize',
                  user?.role === 'admin'
                    ? 'bg-purple-500/20 text-purple-400'
                    : 'bg-blue-500/20 text-blue-400'
                )}
              >
                {user?.role || 'Unknown'}
              </span>
            </div>

            <div className="flex items-center justify-between py-3 border-b border-slate-700">
              <span className="text-gray-400">User ID</span>
              <span className="text-white text-sm font-mono">
                {user?.id?.slice(0, 8)}...
              </span>
            </div>
          </div>
        </div>

        {/* Monitoring Settings */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl bg-green-500/20">
              <Wifi className="w-6 h-6 text-green-400" />
            </div>
            <h2 className="text-lg font-semibold">Monitoring</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Active Patient
              </label>
              <select
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="default">Default Patient</option>
              </select>
            </div>
          </div>
        </div>

        {/* Notification Settings */}
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-3 rounded-xl bg-yellow-500/20">
              <Bell className="w-6 h-6 text-yellow-400" />
            </div>
            <h2 className="text-lg font-semibold">Notifications</h2>
          </div>

          <div className="space-y-4">
            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-gray-300">Critical Alerts</span>
              <input
                type="checkbox"
                defaultChecked
                className="sr-only peer"
              />
              <div className="relative w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>

            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-gray-300">Warning Alerts</span>
              <input
                type="checkbox"
                defaultChecked
                className="sr-only peer"
              />
              <div className="relative w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>

            <label className="flex items-center justify-between cursor-pointer">
              <span className="text-gray-300">Sound Alerts</span>
              <input
                type="checkbox"
                className="sr-only peer"
              />
              <div className="relative w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-500"></div>
            </label>
          </div>
        </div>
      </div>

      {/* Demo Controls Section */}
      <div className="glass rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 rounded-xl bg-orange-500/20">
            <AlertTriangle className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Demo Controls</h2>
            <p className="text-sm text-gray-400">Trigger simulated anomalies for testing the alert system</p>
          </div>
        </div>

        {/* Alert Stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-gray-400 text-xs">Total Alerts</p>
            <p className="text-2xl font-bold text-white">{alertStats?.total_alerts || 0}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-gray-400 text-xs">Critical</p>
            <p className="text-2xl font-bold text-red-400">{alertStats?.by_severity?.critical || 0}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-gray-400 text-xs">Warnings</p>
            <p className="text-2xl font-bold text-yellow-400">{alertStats?.by_severity?.warning || 0}</p>
          </div>
          <div className="bg-slate-800/50 rounded-xl p-4">
            <p className="text-gray-400 text-xs">Acknowledged</p>
            <p className="text-2xl font-bold text-green-400">{alertStats?.acknowledged || 0}</p>
          </div>
        </div>

        {/* Anomaly Trigger Buttons */}
        <div className="space-y-3">
          <p className="text-sm text-gray-400 mb-2">Click to simulate an anomaly (alerts auto-generated):</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <button
              onClick={() => handleTriggerAnomaly('tachycardia')}
              disabled={triggeringAnomaly !== null}
              className={clsx(
                'flex flex-col items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all',
                'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/30',
                triggeringAnomaly === 'tachycardia' && 'animate-pulse'
              )}
            >
              <Play className={clsx('w-5 h-5', triggeringAnomaly === 'tachycardia' && 'animate-spin')} />
              <span className="text-xs">Tachycardia</span>
              <span className="text-[10px] text-gray-400">HR &gt; 100</span>
            </button>
            <button
              onClick={() => handleTriggerAnomaly('bradycardia')}
              disabled={triggeringAnomaly !== null}
              className={clsx(
                'flex flex-col items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all',
                'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 border border-blue-500/30',
                triggeringAnomaly === 'bradycardia' && 'animate-pulse'
              )}
            >
              <Play className={clsx('w-5 h-5', triggeringAnomaly === 'bradycardia' && 'animate-spin')} />
              <span className="text-xs">Bradycardia</span>
              <span className="text-[10px] text-gray-400">HR &lt; 60</span>
            </button>
            <button
              onClick={() => handleTriggerAnomaly('apnea')}
              disabled={triggeringAnomaly !== null}
              className={clsx(
                'flex flex-col items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all',
                'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 border border-purple-500/30',
                triggeringAnomaly === 'apnea' && 'animate-pulse'
              )}
            >
              <Play className={clsx('w-5 h-5', triggeringAnomaly === 'apnea' && 'animate-spin')} />
              <span className="text-xs">Apnea</span>
              <span className="text-[10px] text-gray-400">BR &lt; 8</span>
            </button>
            <button
              onClick={() => handleTriggerAnomaly('tachypnea')}
              disabled={triggeringAnomaly !== null}
              className={clsx(
                'flex flex-col items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all',
                'bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 border border-orange-500/30',
                triggeringAnomaly === 'tachypnea' && 'animate-pulse'
              )}
            >
              <Play className={clsx('w-5 h-5', triggeringAnomaly === 'tachypnea' && 'animate-spin')} />
              <span className="text-xs">Tachypnea</span>
              <span className="text-[10px] text-gray-400">BR &gt; 20</span>
            </button>
            <button
              onClick={() => handleTriggerAnomaly('stress')}
              disabled={triggeringAnomaly !== null}
              className={clsx(
                'flex flex-col items-center gap-2 px-4 py-3 rounded-xl font-medium transition-all',
                'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30',
                triggeringAnomaly === 'stress' && 'animate-pulse'
              )}
            >
              <Play className={clsx('w-5 h-5', triggeringAnomaly === 'stress' && 'animate-spin')} />
              <span className="text-xs">Stress</span>
              <span className="text-[10px] text-gray-400">High HRV</span>
            </button>
          </div>
        </div>

        <p className="text-xs text-gray-500 mt-4">
          Anomalies trigger real alerts that are streamed through Confluent Cloud Kafka and processed by Vertex AI.
        </p>
      </div>

      {/* About Section */}
      <div className="glass rounded-2xl p-6">
        <h2 className="text-lg font-semibold mb-4">About VitalFlow Radar</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-sm">
          <div>
            <p className="text-gray-500">Version</p>
            <p className="font-medium">1.0.0</p>
          </div>
          <div>
            <p className="text-gray-500">Technology</p>
            <p className="font-medium">mmWave Radar (77GHz)</p>
          </div>
          <div>
            <p className="text-gray-500">Platform</p>
            <p className="font-medium">Raspberry Pi Edge</p>
          </div>
        </div>
        <p className="text-gray-500 text-sm mt-6">
          Contactless vital signs monitoring using Texas Instruments AWR1642 radar sensor.
          For wellness monitoring purposes only - not a medical device.
        </p>
      </div>
    </div>
  )
}
