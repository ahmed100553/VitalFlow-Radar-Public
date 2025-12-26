import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import axios from 'axios'

interface User {
  id: string
  role: string
  isDemo?: boolean
}

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  loading: boolean
  isDemo: boolean
  login: (username: string, password: string) => Promise<void>
  demoLogin: () => void
  logout: () => void
  register: (username: string, password: string, email?: string) => Promise<void>
}

const AuthContext = createContext<AuthContextType | null>(null)

const API_URL = import.meta.env.VITE_API_URL || ''

// Demo credentials for hackathon judges
const DEMO_USER = {
  id: 'demo-user',
  role: 'admin',
  isDemo: true
}
const DEMO_TOKEN = 'demo-token-for-hackathon-judges'

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    const storedUser = localStorage.getItem('user')
    
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
      axios.defaults.headers.common['Authorization'] = `Bearer ${storedToken}`
    }
    setLoading(false)
  }, [])

  // Demo login - works without backend for hackathon
  const demoLogin = () => {
    setToken(DEMO_TOKEN)
    setUser(DEMO_USER)
    
    localStorage.setItem('token', DEMO_TOKEN)
    localStorage.setItem('user', JSON.stringify(DEMO_USER))
    
    axios.defaults.headers.common['Authorization'] = `Bearer ${DEMO_TOKEN}`
  }

  const login = async (username: string, password: string) => {
    // For demo credentials, use demo login (works without backend)
    if (username === 'admin' && password === 'admin123') {
      demoLogin()
      return
    }
    
    // Try real API login
    const response = await axios.post(`${API_URL}/api/auth/login`, {
      username,
      password,
    })
    
    const { access_token, user_id, role } = response.data
    
    setToken(access_token)
    setUser({ id: user_id, role })
    
    localStorage.setItem('token', access_token)
    localStorage.setItem('user', JSON.stringify({ id: user_id, role }))
    
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
  }

  const register = async (username: string, password: string, email?: string) => {
    const response = await axios.post(`${API_URL}/api/auth/register`, {
      username,
      password,
      email,
    })
    
    const { access_token, user_id, role } = response.data
    
    setToken(access_token)
    setUser({ id: user_id, role })
    
    localStorage.setItem('token', access_token)
    localStorage.setItem('user', JSON.stringify({ id: user_id, role }))
    
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
  }

  const logout = () => {
    setToken(null)
    setUser(null)
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    delete axios.defaults.headers.common['Authorization']
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token,
        loading,
        isDemo: user?.isDemo || false,
        login,
        demoLogin,
        logout,
        register,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}
