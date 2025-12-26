import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useVitals } from '../contexts/VitalsContext'
import {
  Heart,
  Users,
  Bell,
  Settings,
  LogOut,
  Activity,
  Menu,
  X,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

const navItems = [
  { path: '/dashboard', icon: Activity, label: 'Dashboard' },
  { path: '/dashboard/patients', icon: Users, label: 'Patients' },
  { path: '/dashboard/alerts', icon: Bell, label: 'Alerts' },
  { path: '/dashboard/settings', icon: Settings, label: 'Settings' },
]

export default function Layout() {
  const { logout, user } = useAuth()
  const { isConnected } = useVitals()
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">
      {/* Mobile header */}
      <header className="lg:hidden flex items-center justify-between p-4 bg-slate-800 border-b border-slate-700">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-2 rounded-lg hover:bg-slate-700"
        >
          <Menu className="w-6 h-6" />
        </button>
        <div className="flex items-center gap-2">
          <Heart className="w-6 h-6 text-primary-500 animate-heartbeat" />
          <span className="font-bold text-lg">VitalFlow</span>
        </div>
        <div className="flex items-center gap-2">
          {isConnected ? (
            <Wifi className="w-5 h-5 text-green-500" />
          ) : (
            <WifiOff className="w-5 h-5 text-red-500" />
          )}
        </div>
      </header>

      {/* Sidebar overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed lg:static inset-y-0 left-0 z-50 w-64 bg-slate-800 border-r border-slate-700 transform transition-transform duration-300 lg:transform-none',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-between p-6">
            <div className="flex items-center gap-3">
              <Heart className="w-8 h-8 text-primary-500 animate-heartbeat" />
              <span className="font-bold text-xl bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
                VitalFlow
              </span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-2 rounded-lg hover:bg-slate-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Connection status */}
          <div className="px-6 mb-4">
            <div
              className={clsx(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-sm',
                isConnected ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'
              )}
            >
              {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4">
            {navItems.map(({ path, icon: Icon, label }) => (
              <NavLink
                key={path}
                to={path}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-4 py-3 rounded-xl mb-2 transition-all',
                    isActive
                      ? 'bg-primary-500/20 text-primary-400'
                      : 'text-gray-400 hover:bg-slate-700 hover:text-white'
                  )
                }
              >
                <Icon className="w-5 h-5" />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* User section */}
          <div className="p-4 border-t border-slate-700">
            <div className="flex items-center gap-3 mb-4 px-4">
              <div className="w-10 h-10 rounded-full bg-primary-500/20 flex items-center justify-center">
                <span className="text-primary-400 font-medium">
                  {user?.role === 'admin' ? 'A' : 'U'}
                </span>
              </div>
              <div>
                <p className="text-sm font-medium">{user?.role === 'admin' ? 'Admin' : 'User'}</p>
                <p className="text-xs text-gray-500">{user?.role}</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-gray-400 hover:bg-red-500/10 hover:text-red-400 transition-all"
            >
              <LogOut className="w-5 h-5" />
              Sign Out
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-4 lg:p-6 max-w-7xl">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
