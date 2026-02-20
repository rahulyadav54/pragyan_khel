import { motion } from 'framer-motion'
import { SlidersHorizontal } from 'lucide-react'

export default function BlurSlider({ value, onChange }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between">
        <p className="flex items-center gap-2 text-sm font-medium text-slate-200">
          <SlidersHorizontal className="h-4 w-4 text-cyan-300" />
          Blur Intensity
        </p>
        <motion.span
          key={value}
          initial={{ scale: 0.9, opacity: 0.6 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.18 }}
          className="rounded-full border border-cyan-300/35 bg-cyan-400/15 px-3 py-1 text-xs font-semibold text-cyan-100"
        >
          {value}
        </motion.span>
      </div>
      <input
        type="range"
        min="5"
        max="99"
        step="2"
        value={value}
        onChange={(event) => onChange(parseInt(event.target.value, 10))}
        className="range-premium h-2 w-full cursor-pointer appearance-none rounded-full"
      />
    </div>
  )
}
