import { motion } from 'framer-motion'
import { Sliders, RotateCcw, Target } from 'lucide-react'

export default function Controls({ blurIntensity, onBlurChange, onReset, selectedObject }) {
  return (
    <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 space-y-6">
      <h3 className="text-white font-semibold text-lg flex items-center">
        <Sliders className="w-5 h-5 mr-2" />
        Controls
      </h3>

      {/* Blur Intensity Slider */}
      <div>
        <label className="text-gray-300 text-sm mb-2 block">
          Blur Intensity: {blurIntensity}
        </label>
        <input
          type="range"
          min="5"
          max="51"
          step="2"
          value={blurIntensity}
          onChange={(e) => onBlurChange(parseInt(e.target.value))}
          className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer slider"
        />
      </div>

      {/* Selected Object Indicator */}
      {selectedObject && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-green-500/20 border border-green-500/30 rounded-lg p-4"
        >
          <div className="flex items-center text-green-400">
            <Target className="w-5 h-5 mr-2" />
            <span className="font-medium">Object Tracked</span>
          </div>
        </motion.div>
      )}

      {/* Reset Button */}
      <button
        onClick={onReset}
        className="w-full py-3 px-4 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white font-medium rounded-lg transition-all shadow-lg flex items-center justify-center"
      >
        <RotateCcw className="w-5 h-5 mr-2" />
        Reset Tracking
      </button>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-gray-400 text-xs mb-1">Mode</div>
          <div className="text-white font-semibold">Real-time</div>
        </div>
        <div className="bg-white/5 rounded-lg p-3">
          <div className="text-gray-400 text-xs mb-1">Status</div>
          <div className="text-green-400 font-semibold">Active</div>
        </div>
      </div>
    </div>
  )
}
