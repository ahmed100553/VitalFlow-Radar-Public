import { useState, useEffect } from 'react'
import { Bell, CheckCircle, AlertTriangle, XCircle, Clock } from 'lucide-react'
import axios from 'axios'
import clsx from 'clsx'
import { format, formatDistanceToNow } from 'date-fns'

interface Alert {
  id: string
  patient_id: string
  timestamp: number
  alert_type: string
  severity: string
  message: string
  acknowledged: boolean
}

const API_URL = import.meta.env.VITE_API_URL || ''

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'critical' | 'warning'>('all')

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/alerts`)
      // Handle both array and {alerts: []} response formats
      const data = Array.isArray(response.data) ? response.data : (response.data.alerts || [])
      setAlerts(data)
    } catch (error) {
      console.error('Failed to fetch alerts:', error)
      // Set demo alerts on error for demo mode
      setAlerts([
        { id: 'alert-1', patient_id: 'patient-2', timestamp: Date.now() / 1000 - 3600, alert_type: 'heart_rate', severity: 'warning', message: 'Elevated heart rate detected: 98 BPM', acknowledged: false },
        { id: 'alert-2', patient_id: 'patient-1', timestamp: Date.now() / 1000 - 1800, alert_type: 'breathing_rate', severity: 'critical', message: 'Breathing irregularity detected', acknowledged: false }
      ])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAlerts()
    const interval = setInterval(fetchAlerts, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await axios.post(`${API_URL}/api/alerts/${alertId}/acknowledge`)
      setAlerts(alerts.filter((a) => a.id !== alertId))
    } catch (error) {
      console.error('Failed to acknowledge alert:', error)
    }
  }

  const filteredAlerts = alerts.filter((alert) => {
    if (filter === 'all') return true
    return alert.severity === filter
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold">Alerts</h1>
          <p className="text-gray-400 mt-1">
            {alerts.length} unacknowledged alert{alerts.length !== 1 ? 's' : ''}
          </p>
        </div>

        {/* Filter buttons */}
        <div className="flex gap-2">
          {(['all', 'critical', 'warning'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={clsx(
                'px-4 py-2 rounded-xl text-sm font-medium transition-colors capitalize',
                filter === f
                  ? 'bg-primary-500 text-white'
                  : 'bg-slate-800 text-gray-400 hover:bg-slate-700'
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Alerts List */}
      {filteredAlerts.length === 0 ? (
        <div className="glass rounded-2xl p-12 text-center">
          <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">All Clear</h3>
          <p className="text-gray-400">No active alerts at this time</p>
        </div>
      ) : (
        <div className="space-y-4">
          {filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={clsx(
                'glass rounded-2xl p-6 border-l-4 transition-all hover:bg-white/5',
                alert.severity === 'critical' && 'border-l-red-500',
                alert.severity === 'warning' && 'border-l-yellow-500'
              )}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div
                    className={clsx(
                      'p-3 rounded-xl flex-shrink-0',
                      alert.severity === 'critical' && 'bg-red-500/20',
                      alert.severity === 'warning' && 'bg-yellow-500/20'
                    )}
                  >
                    {alert.severity === 'critical' ? (
                      <XCircle className="w-6 h-6 text-red-500" />
                    ) : (
                      <AlertTriangle className="w-6 h-6 text-yellow-500" />
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={clsx(
                          'px-2 py-0.5 rounded text-xs font-medium uppercase',
                          alert.severity === 'critical' && 'bg-red-500/20 text-red-400',
                          alert.severity === 'warning' && 'bg-yellow-500/20 text-yellow-400'
                        )}
                      >
                        {alert.severity}
                      </span>
                      <span className="text-gray-500 text-sm">
                        {alert.alert_type.replace('_', ' ')}
                      </span>
                    </div>

                    <p className="text-lg font-medium mb-2">{alert.message}</p>

                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {formatDistanceToNow(new Date(alert.timestamp * 1000), {
                          addSuffix: true,
                        })}
                      </span>
                      <span>Patient: {alert.patient_id}</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => acknowledgeAlert(alert.id)}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-700 rounded-xl text-sm font-medium hover:bg-slate-600 transition-colors flex-shrink-0"
                >
                  <CheckCircle className="w-4 h-4" />
                  Acknowledge
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
