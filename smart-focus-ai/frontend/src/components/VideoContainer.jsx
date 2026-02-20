import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

export default function VideoContainer({ children, fps, selectedObject, loading, title }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="relative overflow-hidden rounded-3xl border border-cyan-400/20 bg-white/5 p-3 shadow-premium backdrop-blur-xl sm:p-4"
    >
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-cyan-500/10 via-transparent to-blue-600/10" />
      <div className={`relative overflow-hidden rounded-2xl border bg-slate-950/80 p-2 sm:p-3 ${selectedObject ? 'border-cyan-300/60 glow-focus' : 'border-white/10'}`}>
        <div className="pointer-events-none absolute left-3 top-3 z-20 rounded-full border border-white/15 bg-slate-900/70 px-3 py-1 text-xs font-semibold text-slate-100 backdrop-blur-md sm:left-4 sm:top-4">
          {title}
        </div>

        <div className="pointer-events-none absolute right-3 top-3 z-20 rounded-full border border-cyan-300/40 bg-cyan-400/15 px-3 py-1 text-xs font-semibold text-cyan-100 backdrop-blur-md sm:right-4 sm:top-4">
          {fps > 0 ? `${fps} FPS` : 'Waiting'}
        </div>

        <motion.div
          animate={{ y: [0, -4, 0] }}
          transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
          className="relative"
        >
          {children}
        </motion.div>

        {loading && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/55 backdrop-blur-sm">
            <div className="rounded-2xl border border-cyan-300/25 bg-slate-900/80 px-5 py-4 text-center shadow-premium">
              <Loader2 className="mx-auto h-6 w-6 animate-spin text-cyan-300" />
              <p className="mt-2 text-sm font-medium text-slate-100">AI processing in progress</p>
            </div>
          </div>
        )}
      </div>
    </motion.section>
  )
}
