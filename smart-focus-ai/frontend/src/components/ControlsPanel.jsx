import { motion } from 'framer-motion'
import { Camera, RotateCcw, Square, Upload, Wand2 } from 'lucide-react'
import BlurSlider from './BlurSlider'

const buttonBase =
  'group flex w-full items-center justify-center gap-2 rounded-full border px-4 py-3 text-sm font-semibold transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/60'

export default function ControlsPanel({
  blurIntensity,
  onBlurChange,
  onStartWebcam,
  onUploadClick,
  onStop,
  onReset,
  selectedObject,
  isRunning,
  sourceMode
}) {
  return (
    <motion.aside
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.08 }}
      className="space-y-5 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-premium backdrop-blur-xl sm:p-5"
    >
      <div className="rounded-2xl border border-white/10 bg-slate-950/35 p-4">
        <p className="font-display text-lg font-semibold text-slate-100">Controls</p>
        <p className="mt-1 text-xs text-slate-400">Choose source, track object, and tune background blur.</p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
        <button
          type="button"
          onClick={onStartWebcam}
          className={`${buttonBase} border-cyan-300/30 bg-gradient-to-r from-cyan-500/30 to-blue-500/30 text-cyan-100 shadow-neon hover:-translate-y-0.5 hover:scale-[1.02] hover:shadow-cyan-400/30`}
        >
          <Camera className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" />
          Start Webcam
        </button>

        <button
          type="button"
          onClick={onUploadClick}
          className={`${buttonBase} border-blue-300/30 bg-gradient-to-r from-blue-500/30 to-indigo-500/30 text-blue-100 shadow-neon hover:-translate-y-0.5 hover:scale-[1.02] hover:shadow-blue-400/30`}
        >
          <Upload className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" />
          Upload Video
        </button>

        <button
          type="button"
          onClick={onStop}
          className={`${buttonBase} border-rose-300/30 bg-gradient-to-r from-rose-500/30 to-fuchsia-500/30 text-rose-100 shadow-neon hover:-translate-y-0.5 hover:scale-[1.02] hover:shadow-rose-400/30`}
        >
          <Square className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" />
          Stop
        </button>

        <button
          type="button"
          onClick={onReset}
          className={`${buttonBase} border-amber-300/30 bg-gradient-to-r from-amber-500/30 to-orange-500/30 text-amber-100 shadow-neon hover:-translate-y-0.5 hover:scale-[1.02] hover:shadow-amber-300/30`}
        >
          <RotateCcw className="h-4 w-4 transition-transform duration-300 group-hover:rotate-45" />
          Reset
        </button>
      </div>

      <BlurSlider value={blurIntensity} onChange={onBlurChange} />

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-2xl border border-white/10 bg-slate-950/35 p-3">
          <p className="text-[11px] uppercase tracking-wide text-slate-400">Source</p>
          <p className="mt-1 text-sm font-semibold text-slate-100">
            {sourceMode === 'webcam' ? 'Webcam' : sourceMode === 'upload' ? 'Upload' : 'None'}
          </p>
        </div>
        <div className="rounded-2xl border border-white/10 bg-slate-950/35 p-3">
          <p className="text-[11px] uppercase tracking-wide text-slate-400">Status</p>
          <p className={`mt-1 text-sm font-semibold ${isRunning ? 'text-emerald-300' : 'text-slate-300'}`}>
            {isRunning ? 'Live' : 'Idle'}
          </p>
        </div>
      </div>

      <motion.div
        initial={false}
        animate={{ opacity: 1, y: 0 }}
        className={`rounded-2xl border p-3 ${selectedObject ? 'border-cyan-300/40 bg-cyan-400/10' : 'border-white/10 bg-slate-950/35'}`}
      >
        <div className="flex items-center gap-2">
          <Wand2 className={`h-4 w-4 ${selectedObject ? 'text-cyan-200' : 'text-slate-300'}`} />
          <p className={`text-sm font-medium ${selectedObject ? 'text-cyan-100' : 'text-slate-300'}`}>
            {selectedObject ? 'Object focused, surroundings blurred' : 'Select an object to activate focus mode'}
          </p>
        </div>
      </motion.div>
    </motion.aside>
  )
}
