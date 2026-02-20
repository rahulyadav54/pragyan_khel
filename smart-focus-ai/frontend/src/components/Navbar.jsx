import { motion } from 'framer-motion'
import { Cpu, Sparkles } from 'lucide-react'

export default function Navbar({ fps }) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative z-20 border-b border-white/10 bg-slate-950/40 backdrop-blur-xl"
    >
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-cyan-300/30 bg-gradient-to-br from-cyan-400/30 to-blue-500/30 shadow-neon">
            <Cpu className="h-5 w-5 text-cyan-200" />
          </div>
          <div>
            <p className="font-display text-lg font-semibold text-slate-50">Smart Focus AI</p>
            <p className="text-xs text-slate-400">Premium Object Tracking Studio</p>
          </div>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-cyan-300/25 bg-cyan-400/10 px-3 py-1.5">
          <Sparkles className="h-4 w-4 text-cyan-200" />
          <span className="text-xs font-medium text-cyan-100">{fps > 0 ? `${fps} FPS` : 'Idle'}</span>
        </div>
      </div>
    </motion.header>
  )
}
