import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Film } from 'lucide-react'
import Navbar from './components/Navbar'
import VideoContainer from './components/VideoContainer'
import ControlsPanel from './components/ControlsPanel'
import VideoPlayer from './components/VideoPlayer'
import WebcamCapture from './components/WebcamCapture'
import './index.css'

function App() {
  const [mode, setMode] = useState('idle')
  const [videoFile, setVideoFile] = useState(null)
  const [blurIntensity, setBlurIntensity] = useState(45)
  const [selectedObject, setSelectedObject] = useState(false)
  const [fps, setFps] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [sessionKey, setSessionKey] = useState(0)
  const fileInputRef = useRef(null)

  useEffect(() => {
    return () => {
      if (videoFile) {
        URL.revokeObjectURL(videoFile)
      }
    }
  }, [videoFile])

  const handleUploadTrigger = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  const handleFileUpload = (event) => {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    if (videoFile) {
      URL.revokeObjectURL(videoFile)
    }

    setVideoFile(URL.createObjectURL(file))
    setMode('upload')
    setSelectedObject(false)
    setFps(0)
    setIsLoading(true)
    setSessionKey((prev) => prev + 1)
  }

  const handleStartWebcam = () => {
    setMode('webcam')
    setSelectedObject(false)
    setFps(0)
    setIsLoading(true)
    setSessionKey((prev) => prev + 1)
  }

  const handleStop = () => {
    setMode('idle')
    setSelectedObject(false)
    setFps(0)
    setIsLoading(false)
    setSessionKey((prev) => prev + 1)
  }

  const handleReset = async () => {
    setSelectedObject(false)
    try {
      await fetch('/api/reset-tracking', { method: 'POST' })
    } catch (error) {
      console.error('Reset tracking failed:', error)
    }
  }

  const handleObjectSelect = ({ selected }) => {
    setSelectedObject(Boolean(selected))
  }

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-app text-slate-100">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="aurora aurora-a" />
        <div className="aurora aurora-b" />
        <div className="particle particle-a" />
        <div className="particle particle-b" />
        <div className="particle particle-c" />
      </div>

      <Navbar fps={fps} />

      <main className="relative z-10 mx-auto w-full max-w-7xl px-4 pb-8 pt-6 sm:px-6 md:pt-8 lg:px-8">
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(320px,1fr)] lg:gap-6"
        >
          <VideoContainer
            fps={fps}
            selectedObject={selectedObject}
            loading={isLoading}
            title="Smart Focus AI"
          >
            <AnimatePresence mode="wait">
              {mode === 'upload' && videoFile ? (
                <motion.div
                  key={`upload-${sessionKey}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.28 }}
                >
                  <VideoPlayer
                    videoSrc={videoFile}
                    blurIntensity={blurIntensity}
                    onFpsUpdate={setFps}
                    onObjectSelect={handleObjectSelect}
                    onLoadingChange={setIsLoading}
                  />
                </motion.div>
              ) : mode === 'webcam' ? (
                <motion.div
                  key={`webcam-${sessionKey}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.28 }}
                >
                  <WebcamCapture
                    blurIntensity={blurIntensity}
                    onFpsUpdate={setFps}
                    onObjectSelect={handleObjectSelect}
                    onLoadingChange={setIsLoading}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex h-[300px] items-center justify-center rounded-2xl border border-cyan-400/15 bg-slate-950/60 sm:h-[420px]"
                >
                  <div className="text-center">
                    <Film className="mx-auto mb-3 h-12 w-12 text-cyan-300/80" />
                    <p className="text-lg font-semibold text-slate-100">Ready to Process</p>
                    <p className="mt-1 text-sm text-slate-400">Start webcam or upload a video to begin focus tracking.</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </VideoContainer>

          <ControlsPanel
            blurIntensity={blurIntensity}
            onBlurChange={setBlurIntensity}
            onStartWebcam={handleStartWebcam}
            onUploadClick={handleUploadTrigger}
            onStop={handleStop}
            onReset={handleReset}
            selectedObject={selectedObject}
            isRunning={mode !== 'idle'}
            sourceMode={mode}
          />
        </motion.section>
      </main>

      <footer className="relative z-10 border-t border-white/10 px-4 py-4 text-center text-xs text-slate-400 sm:px-6 lg:px-8">
        Copyright {new Date().getFullYear()} Smart Focus AI. All rights reserved.
      </footer>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileUpload}
      />
    </div>
  )
}

export default App
