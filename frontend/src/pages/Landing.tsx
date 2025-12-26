import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { 
  Activity, 
  Heart, 
  Home, 
  Building2, 
  Users, 
  CheckCircle, 
  ArrowRight,
  Database,
  Brain,
  Radio,
  ChevronRight,
  Monitor,
  AlertTriangle,
  Lock,
  Play
} from 'lucide-react'

function Navigation() {
  const { demoLogin } = useAuth()
  const navigate = useNavigate()

  const handleDemoLogin = () => {
    demoLogin()
    navigate('/dashboard')
  }

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-md shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">VitalFlow</span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <button 
              onClick={() => scrollToSection('features')} 
              className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
            >
              Features
            </button>
            <button 
              onClick={() => scrollToSection('use-cases')} 
              className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
            >
              Use Cases
            </button>
            <button 
              onClick={() => scrollToSection('how-it-works')} 
              className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
            >
              How It Works
            </button>
            <button 
              onClick={() => scrollToSection('technology')} 
              className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
            >
              Technology
            </button>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleDemoLogin}
              className="hidden sm:inline-flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg font-medium transition-colors shadow-md hover:shadow-lg"
            >
              <Play className="w-4 h-4" />
              Try Demo
            </button>
            <Link 
              to="/login" 
              className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg font-medium transition-colors shadow-md hover:shadow-lg"
            >
              Get Started
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

function Hero() {
  return (
    <section className="pt-24 pb-16 bg-gradient-to-b from-blue-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-700 px-4 py-2 rounded-full text-sm font-medium">
              <Activity className="w-4 h-4" />
              Next-Generation Vital Signs Monitoring
            </div>
            
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
              Contactless Health Monitoring{' '}
              <span className="text-blue-600">for a Safer Future</span>
            </h1>
            
            <p className="text-lg text-gray-600 leading-relaxed max-w-xl">
              Revolutionary millimeter-wave radar technology monitors heart rate, breathing, 
              and vital signs without any physical contact. Ideal for hospitals, eldercare 
              facilities, and smart homes.
            </p>

            <div className="flex flex-wrap gap-4">
              <Link 
                to="/login"
                className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
              >
                Get Started Free
                <ArrowRight className="w-5 h-5" />
              </Link>
              <a 
                href="#how-it-works"
                className="inline-flex items-center gap-2 bg-white hover:bg-gray-50 text-gray-700 px-6 py-3 rounded-lg font-semibold transition-all border border-gray-200 shadow-sm"
              >
                Watch Demo
              </a>
            </div>

            <div className="flex flex-col gap-3 pt-4">
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span>Non-invasive, comfortable monitoring 24/7</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span>AI-powered health insights and anomaly detection</span>
              </div>
              <div className="flex items-center gap-2 text-gray-600">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span>Real-time alerts for caregivers and family</span>
              </div>
            </div>
          </div>

          <div className="relative">
            <div className="relative rounded-2xl overflow-hidden shadow-2xl">
              <img 
                src="https://images.unsplash.com/photo-1576765608535-5f04d1e3f289?w=800&auto=format&fit=crop&q=80" 
                alt="Elderly couple being monitored" 
                className="w-full h-auto object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-blue-900/50 to-transparent" />
              
              {/* Floating stat cards */}
              <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-lg">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                    <Heart className="w-6 h-6 text-red-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">72</p>
                    <p className="text-sm text-gray-500">Heart Rate BPM</p>
                  </div>
                </div>
              </div>
              
              <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-lg">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
                    <Activity className="w-6 h-6 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">16</p>
                    <p className="text-sm text-gray-500">Breaths/min</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Features() {
  const features = [
    {
      icon: Radio,
      title: 'Non-Contact Monitoring',
      description: 'Uses 77GHz mmWave radar technology to detect vital signs from a distance without any physical contact or wearables.',
      color: 'blue'
    },
    {
      icon: Heart,
      title: 'Heart Rate Detection',
      description: 'Accurate heart rate monitoring using advanced signal processing algorithms for continuous health tracking.',
      color: 'red'
    },
    {
      icon: Activity,
      title: 'Breathing Analysis',
      description: 'Real-time respiratory rate monitoring with breath-by-breath analysis for comprehensive health insights.',
      color: 'green'
    },
    {
      icon: Brain,
      title: 'AI Health Insights',
      description: 'Google Vertex AI-powered analytics detect anomalies and provide predictive health alerts.',
      color: 'purple'
    },
    {
      icon: AlertTriangle,
      title: 'Smart Alerts',
      description: 'Instant notifications to caregivers and family members when vital signs fall outside normal ranges.',
      color: 'orange'
    },
    {
      icon: Lock,
      title: 'Privacy First',
      description: 'No cameras or microphones - radar-only sensing ensures complete privacy while monitoring.',
      color: 'slate'
    }
  ]

  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    red: 'bg-red-100 text-red-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    orange: 'bg-orange-100 text-orange-600',
    slate: 'bg-slate-100 text-slate-600'
  }

  return (
    <section id="features" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <p className="text-blue-600 font-semibold mb-3">POWERFUL FEATURES</p>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Advanced Healthcare Monitoring
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Our cutting-edge technology combines mmWave radar sensing with AI-powered analytics 
            to deliver accurate, non-invasive vital signs monitoring for healthcare and home settings.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index} 
              className="p-6 rounded-2xl bg-gray-50 hover:bg-white hover:shadow-xl transition-all duration-300 border border-gray-100 group"
            >
              <div className={`w-14 h-14 rounded-xl ${colorClasses[feature.color as keyof typeof colorClasses]} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function UseCases() {
  const useCases = [
    {
      icon: Home,
      title: 'Smart Home',
      description: 'Enable peace of mind for families with elderly members living independently. Monitor vital signs discreetly from any room.',
      image: 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=600&auto=format&fit=crop&q=80',
      benefits: ['Sleep quality monitoring', 'Fall detection alerts', 'Activity tracking']
    },
    {
      icon: Building2,
      title: 'Healthcare Facilities',
      description: 'Continuous patient monitoring in hospitals, nursing homes, and rehabilitation centers without disrupting rest.',
      image: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=600&auto=format&fit=crop&q=80',
      benefits: ['ICU patient monitoring', 'Post-surgery recovery', 'Elderly care facilities']
    },
    {
      icon: Users,
      title: 'Remote Patient Monitoring',
      description: 'Enable telehealth providers to monitor patients at home, reducing hospital readmissions and improving outcomes.',
      image: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=600&auto=format&fit=crop&q=80',
      benefits: ['Chronic disease management', 'Post-discharge care', 'Preventive health alerts']
    }
  ]

  return (
    <section id="use-cases" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <p className="text-blue-600 font-semibold mb-3">USE CASES</p>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Designed for Every Healthcare Setting
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            From smart homes to professional healthcare facilities, VitalFlow adapts 
            to your monitoring needs with flexibility and precision.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {useCases.map((useCase, index) => (
            <div 
              key={index} 
              className="bg-white rounded-2xl overflow-hidden shadow-lg hover:shadow-2xl transition-all duration-300 group"
            >
              <div className="relative h-48 overflow-hidden">
                <img 
                  src={useCase.image} 
                  alt={useCase.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <div className="absolute bottom-4 left-4 flex items-center gap-2">
                  <div className="w-10 h-10 rounded-lg bg-white/90 flex items-center justify-center">
                    <useCase.icon className="w-5 h-5 text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-white">{useCase.title}</h3>
                </div>
              </div>
              <div className="p-6">
                <p className="text-gray-600 mb-4">{useCase.description}</p>
                <ul className="space-y-2">
                  {useCase.benefits.map((benefit, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm text-gray-500">
                      <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                      {benefit}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function HowItWorks() {
  const steps = [
    {
      step: '01',
      icon: Radio,
      title: 'Radar Sensing',
      description: 'The AWR1642 mmWave radar sensor transmits and receives signals that bounce off the human body, detecting micro-movements from breathing and heartbeat.'
    },
    {
      step: '02',
      icon: Database,
      title: 'Data Streaming',
      description: 'Raw sensor data is processed at the edge and streamed through Confluent Cloud for real-time, reliable data pipelines with sub-second latency.'
    },
    {
      step: '03',
      icon: Brain,
      title: 'AI Processing',
      description: 'Google Vertex AI analyzes the vital signs data, applies machine learning models for anomaly detection, and generates actionable health insights.'
    },
    {
      step: '04',
      icon: Monitor,
      title: 'Dashboard Display',
      description: 'Real-time vital signs, trends, and alerts are displayed on an intuitive dashboard accessible from any device, anywhere.'
    }
  ]

  return (
    <section id="how-it-works" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <p className="text-blue-600 font-semibold mb-3">HOW IT WORKS</p>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            From Sensor to Insight in Seconds
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Our end-to-end pipeline transforms raw radar signals into actionable health insights 
            using cutting-edge technology at every step.
          </p>
        </div>

        <div className="relative">
          {/* Connection line */}
          <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-1 bg-gradient-to-r from-blue-200 via-blue-400 to-blue-600 -translate-y-1/2 z-0" />
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 relative z-10">
            {steps.map((step, index) => (
              <div key={index} className="relative">
                <div className="bg-white rounded-2xl p-6 border-2 border-gray-100 hover:border-blue-200 transition-all hover:shadow-lg">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-4xl font-bold text-blue-100">{step.step}</span>
                    <div className="w-12 h-12 rounded-xl bg-blue-600 text-white flex items-center justify-center">
                      <step.icon className="w-6 h-6" />
                    </div>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">{step.title}</h3>
                  <p className="text-gray-600 text-sm leading-relaxed">{step.description}</p>
                </div>
                {index < steps.length - 1 && (
                  <ChevronRight className="hidden lg:block absolute -right-4 top-1/2 -translate-y-1/2 w-8 h-8 text-blue-400 z-20" />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function Technology() {
  const technologies = [
    {
      name: 'Texas Instruments AWR1642',
      category: 'mmWave Radar Sensor',
      description: '77GHz FMCW radar with integrated DSP for vital signs detection',
      logo: '📡'
    },
    {
      name: 'Confluent Cloud',
      category: 'Data Streaming',
      description: 'Enterprise Apache Kafka for real-time data pipelines',
      logo: '🌊'
    },
    {
      name: 'Google Vertex AI',
      category: 'AI/ML Platform',
      description: 'Machine learning for anomaly detection and health insights',
      logo: '🧠'
    },
    {
      name: 'React + Vite',
      category: 'Frontend Framework',
      description: 'Modern, responsive dashboard with real-time updates',
      logo: '⚛️'
    }
  ]

  return (
    <section id="technology" className="py-20 bg-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <p className="text-blue-400 font-semibold mb-3">TECHNOLOGY STACK</p>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Built with Industry-Leading Technology
          </h2>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Our solution combines best-in-class hardware and cloud services 
            to deliver reliable, scalable health monitoring.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {technologies.map((tech, index) => (
            <div 
              key={index} 
              className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700 hover:border-blue-500/50 transition-all hover:bg-slate-800"
            >
              <div className="text-4xl mb-4">{tech.logo}</div>
              <p className="text-sm text-blue-400 font-medium mb-1">{tech.category}</p>
              <h3 className="text-lg font-semibold mb-2">{tech.name}</h3>
              <p className="text-sm text-gray-400">{tech.description}</p>
            </div>
          ))}
        </div>

        <div className="mt-16 p-8 rounded-2xl bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div>
              <p className="text-4xl font-bold text-white mb-2">&lt;100ms</p>
              <p className="text-gray-400">End-to-end Latency</p>
            </div>
            <div>
              <p className="text-4xl font-bold text-white mb-2">99.9%</p>
              <p className="text-gray-400">System Uptime</p>
            </div>
            <div>
              <p className="text-4xl font-bold text-white mb-2">±2 BPM</p>
              <p className="text-gray-400">Heart Rate Accuracy</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function CallToAction() {
  return (
    <section className="py-20 bg-gradient-to-r from-blue-600 to-blue-700">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
          Ready to Transform Health Monitoring?
        </h2>
        <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
          Join healthcare providers and families who are already using VitalFlow 
          for safer, more comfortable vital signs monitoring.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Link 
            to="/login"
            className="inline-flex items-center gap-2 bg-white hover:bg-gray-100 text-blue-600 px-8 py-4 rounded-xl font-semibold transition-all shadow-lg hover:shadow-xl"
          >
            Start Free Trial
            <ArrowRight className="w-5 h-5" />
          </Link>
          <a 
            href="https://github.com/ahmed100553/VitalFlow-Radar-Public"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-blue-500/20 hover:bg-blue-500/30 text-white px-8 py-4 rounded-xl font-semibold transition-all border border-white/20"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="bg-slate-900 text-gray-400 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          <div className="col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">VitalFlow</span>
            </div>
            <p className="text-gray-400 max-w-md mb-4">
              Next-generation contactless vital signs monitoring powered by mmWave radar 
              and AI. Making healthcare safer and more comfortable.
            </p>
            <div className="flex gap-4">
              <a href="https://github.com/ahmed100553/VitalFlow-Radar-Public" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">
                GitHub
              </a>
            </div>
          </div>
          
          <div>
            <h4 className="text-white font-semibold mb-4">Product</h4>
            <ul className="space-y-2">
              <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#use-cases" className="hover:text-white transition-colors">Use Cases</a></li>
              <li><a href="#technology" className="hover:text-white transition-colors">Technology</a></li>
              <li><Link to="/login" className="hover:text-white transition-colors">Dashboard</Link></li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-white font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              <li><a href="https://github.com/ahmed100553/VitalFlow-Radar-Public" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="https://github.com/ahmed100553/VitalFlow-Radar-Public" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="mailto:support@vitalflow.com" className="hover:text-white transition-colors">Support</a></li>
              <li><a href="mailto:contact@vitalflow.com" className="hover:text-white transition-colors">Contact</a></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm">
            © 2024 VitalFlow Radar. Built for Google Cloud x Confluent Hackathon.
          </p>
          <p className="text-sm">
            Made with ❤️ by Ahmed
          </p>
        </div>
      </div>
    </footer>
  )
}

export default function Landing() {
  return (
    <div className="min-h-screen bg-white text-gray-900">
      <Navigation />
      <Hero />
      <Features />
      <UseCases />
      <HowItWorks />
      <Technology />
      <CallToAction />
      <Footer />
    </div>
  )
}
