import { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react'
import axios from 'axios'

export interface VitalSigns {
  patient_id: string
  heart_rate: number
  heart_rate_confidence: number
  breathing_rate: number
  breathing_rate_confidence: number
  status: 'normal' | 'warning' | 'critical' | 'calibrating'
  timestamp: number
  alerts: Array<{
    type: string
    severity: string
    message: string
  }>
}

interface VitalsContextType {
  currentVitals: VitalSigns | null
  vitalsHistory: VitalSigns[]
  isConnected: boolean
  patientId: string
  setPatientId: (id: string) => void
  isDemoMode: boolean
}

const VitalsContext = createContext<VitalsContextType | null>(null)

const API_URL = import.meta.env.VITE_API_URL || ''

// Construct WebSocket URL based on current protocol (ws for http, wss for https)
const getWsUrl = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}`
}
const WS_URL = getWsUrl()

// Check if we're in demo mode (serverless deployment without WebSocket support)
const isDemoMode = () => {
  // If explicitly set, use that
  if (import.meta.env.VITE_DEMO_MODE === 'true') return true
  if (import.meta.env.VITE_DEMO_MODE === 'false') return false
  
  // Cloud Run supports WebSocket, so we can use it
  const hostname = window.location.hostname
  // Only use polling for platforms that don't support WebSocket
  return hostname.includes('vercel.app') || hostname.includes('netlify.app')
}

export function VitalsProvider({ children }: { children: ReactNode }) {
  const [currentVitals, setCurrentVitals] = useState<VitalSigns | null>(null)
  const [vitalsHistory, setVitalsHistory] = useState<VitalSigns[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [patientId, setPatientId] = useState('default')
  const [demoMode] = useState(isDemoMode())
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval>>()

  // HTTP Polling mode for demo/fallback
  const pollVitals = async () => {
    try {
      // Use demo endpoint (no auth required) for polling mode
      const response = await axios.get(`${API_URL}/api/demo/vitals/current`, {
        params: { patient_id: patientId }
      })
      const vitals: VitalSigns = response.data
      setCurrentVitals(vitals)
      setIsConnected(true)
      setVitalsHistory((prev) => {
        const updated = [...prev, vitals]
        return updated.slice(-300)
      })
    } catch (err) {
      console.error('Failed to poll vitals:', err)
      setIsConnected(false)
    }
  }

  // WebSocket mode for local/production with WS support
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close()
    }

    const ws = new WebSocket(`${WS_URL}/ws/${patientId}`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }
    
    ws.onmessage = (event) => {
      try {
        const data = event.data
        if (data === 'pong' || data === 'heartbeat') {
          return
        }
        const vitals: VitalSigns = JSON.parse(data)
        setCurrentVitals(vitals)
        setVitalsHistory((prev) => {
          const updated = [...prev, vitals]
          return updated.slice(-300)
        })
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000)
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
    
    wsRef.current = ws
  }

  useEffect(() => {
    if (demoMode) {
      // Use HTTP polling in demo mode
      console.log('VitalFlow: Using HTTP polling (demo mode)')
      pollVitals() // Initial fetch
      pollingIntervalRef.current = setInterval(pollVitals, 2000) // Poll every 2 seconds
    } else {
      // Use WebSocket for real-time data
      console.log('VitalFlow: Using WebSocket connection')
      connectWebSocket()
    }
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [patientId, demoMode])

  // Keep WebSocket connection alive with ping
  useEffect(() => {
    if (demoMode) return
    
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping')
      }
    }, 25000)
    
    return () => clearInterval(pingInterval)
  }, [demoMode])

  return (
    <VitalsContext.Provider
      value={{
        currentVitals,
        vitalsHistory,
        isConnected,
        patientId,
        setPatientId,
        isDemoMode: demoMode,
      }}
    >
      {children}
    </VitalsContext.Provider>
  )
}

export function useVitals() {
  const context = useContext(VitalsContext)
  if (!context) {
    throw new Error('useVitals must be used within VitalsProvider')
  }
  return context
}
