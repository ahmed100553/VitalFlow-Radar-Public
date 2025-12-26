import { useState, useEffect } from 'react'
import { Users, Plus, X, User, MapPin, Phone, AlertCircle } from 'lucide-react'
import axios from 'axios'
import clsx from 'clsx'
import { useAuth } from '../contexts/AuthContext'

interface Patient {
  id: string
  name: string
  room?: string
  age?: number
  conditions?: string[]
  emergency_contact?: string
  emergency_phone?: string
  current_vitals?: {
    heart_rate: number
    breathing_rate: number
    status: string
  }
}

const API_URL = import.meta.env.VITE_API_URL || ''

export default function Patients() {
  const { user } = useAuth()
  const [patients, setPatients] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    room: '',
    age: '',
    emergency_contact: '',
    emergency_phone: '',
  })

  const fetchPatients = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/patients`)
      // Handle both array and {patients: []} response formats
      const data = Array.isArray(response.data) ? response.data : (response.data.patients || [])
      setPatients(data)
    } catch (error) {
      console.error('Failed to fetch patients:', error)
      // Set demo patients on error for demo mode
      setPatients([
        { id: 'patient-1', name: 'Eleanor Thompson', room: '101A', age: 78, conditions: ['Hypertension', 'Diabetes Type 2'], current_vitals: { heart_rate: 72, breathing_rate: 16, status: 'normal' } },
        { id: 'patient-2', name: 'James Morrison', room: '102B', age: 82, conditions: ['COPD', 'Heart Arrhythmia'], current_vitals: { heart_rate: 78, breathing_rate: 18, status: 'normal' } },
        { id: 'patient-3', name: 'Margaret Chen', room: '103A', age: 71, conditions: ['Post-Surgery Recovery'], current_vitals: { heart_rate: 68, breathing_rate: 15, status: 'normal' } }
      ])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPatients()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await axios.post(`${API_URL}/api/patients`, {
        ...formData,
        age: formData.age ? parseInt(formData.age) : null,
      })
      setShowModal(false)
      setFormData({ name: '', room: '', age: '', emergency_contact: '', emergency_phone: '' })
      fetchPatients()
    } catch (error) {
      console.error('Failed to create patient:', error)
    }
  }

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
          <h1 className="text-2xl sm:text-3xl font-bold">Patients</h1>
          <p className="text-gray-400 mt-1">Manage monitored patients</p>
        </div>
        {user?.role === 'admin' && (
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-4 py-2 gradient-primary rounded-xl font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-5 h-5" />
            Add Patient
          </button>
        )}
      </div>

      {/* Patients Grid */}
      {patients.length === 0 ? (
        <div className="glass rounded-2xl p-12 text-center">
          <Users className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">No Patients</h3>
          <p className="text-gray-400 mb-6">Add a patient to start monitoring</p>
          {user?.role === 'admin' && (
            <button
              onClick={() => setShowModal(true)}
              className="px-6 py-3 gradient-primary rounded-xl font-medium"
            >
              Add First Patient
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6">
          {patients.map((patient) => (
            <div key={patient.id} className="glass rounded-2xl p-6 hover:bg-white/10 transition-colors">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-primary-500/20 flex items-center justify-center">
                    <User className="w-6 h-6 text-primary-400" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">{patient.name}</h3>
                    {patient.room && (
                      <p className="text-gray-400 text-sm flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        Room {patient.room}
                      </p>
                    )}
                  </div>
                </div>
                {patient.current_vitals && (
                  <span
                    className={clsx(
                      'px-3 py-1 rounded-full text-xs font-medium',
                      patient.current_vitals.status === 'normal' && 'bg-green-500/20 text-green-400',
                      patient.current_vitals.status === 'warning' && 'bg-yellow-500/20 text-yellow-400',
                      patient.current_vitals.status === 'critical' && 'bg-red-500/20 text-red-400'
                    )}
                  >
                    {patient.current_vitals.status}
                  </span>
                )}
              </div>

              {patient.current_vitals && (
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-gray-500 text-xs">Heart Rate</p>
                    <p className="text-xl font-bold text-red-400">
                      {patient.current_vitals.heart_rate.toFixed(0)}
                      <span className="text-sm font-normal text-gray-500 ml-1">BPM</span>
                    </p>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <p className="text-gray-500 text-xs">Breathing Rate</p>
                    <p className="text-xl font-bold text-green-400">
                      {patient.current_vitals.breathing_rate.toFixed(0)}
                      <span className="text-sm font-normal text-gray-500 ml-1">bpm</span>
                    </p>
                  </div>
                </div>
              )}

              {patient.emergency_contact && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <Phone className="w-4 h-4" />
                  {patient.emergency_contact}
                  {patient.emergency_phone && ` • ${patient.emergency_phone}`}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Add Patient Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="glass rounded-2xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold">Add Patient</h2>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 rounded-lg hover:bg-slate-700"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Patient Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Room
                  </label>
                  <input
                    type="text"
                    value={formData.room}
                    onChange={(e) => setFormData({ ...formData, room: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Age
                  </label>
                  <input
                    type="number"
                    value={formData.age}
                    onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Emergency Contact
                </label>
                <input
                  type="text"
                  value={formData.emergency_contact}
                  onChange={(e) => setFormData({ ...formData, emergency_contact: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Emergency Phone
                </label>
                <input
                  type="tel"
                  value={formData.emergency_phone}
                  onChange={(e) => setFormData({ ...formData, emergency_phone: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="flex-1 px-4 py-3 bg-slate-700 rounded-xl font-medium hover:bg-slate-600 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-3 gradient-primary rounded-xl font-medium hover:opacity-90 transition-opacity"
                >
                  Add Patient
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
