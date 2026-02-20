import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Video, Camera, Sliders, RotateCcw, Zap } from 'lucide-react'
import VideoPlayer from './components/VideoPlayer'
import WebcamCapture from './components/WebcamCapture'
import Controls from './components/Controls'
import './index.css'

function App() {
  const [mode, setMode] = useState('upload') // 'upload' or 'webcam'
  const [videoFile, setVideoFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [blurIntensity, setBlurIntensity] = useState(25)
  const [selectedObject, setSelectedObject] = useState(null)
  const [fps, setFps] = useState(0)

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setVideoFile(URL.createObjectURL(file))
      setIsProcessing(false)
    }
  }

  const handleReset = () => {
    setSelectedObject(null)
    setIsProcessing(false)
    fetch('http://localhost:8000/api/reset-tracking', { method: 'POST' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute w-96 h-96 bg-purple-500/20 rounded-full blur-3xl -top-48 -left-48 animate-pulse-slow" />
        <div className="absolute w-96 h-96 bg-blue-500/20 rounded-full blur-3xl -bottom-48 -right-48 animate-pulse-slow" />
      </div>

      {/* Header */}
      <motion.header 
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="relative z-10 backdrop-blur-xl bg-white/5 border-b border-white/10"
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Smart Focus AI</h1>
                <p className="text-sm text-gray-400">AI-Powered Object Tracking</p>
              </div>
            </div>
            
            {fps > 0 && (
              <div className="px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-lg">
                <span className="text-green-400 font-mono">{fps} FPS</span>
              </div>
            )}
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Video Display */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="lg:col-span-2"
          >
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6 shadow-2xl">
              <div className="mb-4 flex space-x-2">
                <button
                  onClick={() => setMode('upload')}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                    mode === 'upload'
                      ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-lg'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <Upload className="w-5 h-5 inline mr-2" />
                  Upload Video
                </button>
                <button
                  onClick={() => setMode('webcam')}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
                    mode === 'webcam'
                      ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white shadow-lg'
                      : 'bg-white/5 text-gray-400 hover:bg-white/10'
                  }`}
                >
                  <Camera className="w-5 h-5 inline mr-2" />
                  Webcam
                </button>
              </div>

              <AnimatePresence mode="wait">
                {mode === 'upload' ? (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    {!videoFile ? (
                      <label className="block cursor-pointer">
                        <div className="border-2 border-dashed border-white/20 rounded-xl p-12 text-center hover:border-purple-500/50 transition-all">
                          <Video className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                          <p className="text-white font-medium mb-2">Click to upload video</p>
                          <p className="text-gray-400 text-sm">MP4, AVI, MOV up to 100MB</p>
                        </div>
                        <input
                          type="file"
                          accept="video/*"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                      </label>
                    ) : (
                      <VideoPlayer
                        videoSrc={videoFile}
                        blurIntensity={blurIntensity}
                        onFpsUpdate={setFps}
                      />
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="webcam"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <WebcamCapture
                      blurIntensity={blurIntensity}
                      onFpsUpdate={setFps}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

          {/* Controls Panel */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <Controls
              blurIntensity={blurIntensity}
              onBlurChange={setBlurIntensity}
              onReset={handleReset}
              selectedObject={selectedObject}
            />

            {/* Instructions */}
            <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-6">
              <h3 className="text-white font-semibold mb-4">How to Use</h3>
              <ol className="space-y-3 text-gray-300 text-sm">
                <li className="flex items-start">
                  <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs mr-3 flex-shrink-0">1</span>
                  <span>Upload a video or enable webcam</span>
                </li>
                <li className="flex items-start">
                  <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs mr-3 flex-shrink-0">2</span>
                  <span>Click on any object to track</span>
                </li>
                <li className="flex items-start">
                  <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs mr-3 flex-shrink-0">3</span>
                  <span>Adjust blur intensity with slider</span>
                </li>
                <li className="flex items-start">
                  <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs mr-3 flex-shrink-0">4</span>
                  <span>Click another object to switch focus</span>
                </li>
              </ol>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}

export default App
