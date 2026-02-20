import { useRef, useEffect, useState } from 'react'

export default function VideoPlayer({ videoSrc, blurIntensity, onFpsUpdate }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [ws, setWs] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const fpsRef = useRef({ frames: 0, lastTime: Date.now() })

  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws/video')
    
    websocket.onopen = () => {
      setIsConnected(true)
      console.log('WebSocket connected')
    }
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'frame') {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const img = new Image()
        
        img.onload = () => {
          canvas.width = img.width
          canvas.height = img.height
          ctx.drawImage(img, 0, 0)
          
          // Update FPS
          fpsRef.current.frames++
          const now = Date.now()
          if (now - fpsRef.current.lastTime >= 1000) {
            onFpsUpdate(fpsRef.current.frames)
            fpsRef.current.frames = 0
            fpsRef.current.lastTime = now
          }
        }
        
        img.src = data.data
      }
    }
    
    setWs(websocket)
    
    return () => {
      websocket.close()
    }
  }, [])

  useEffect(() => {
    if (!videoRef.current || !ws || !isConnected) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    const sendFrame = () => {
      if (video.paused || video.ended) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)

      canvas.toBlob((blob) => {
        const reader = new FileReader()
        reader.onloadend = () => {
          ws.send(JSON.stringify({
            type: 'frame',
            data: reader.result,
            blur_intensity: blurIntensity
          }))
        }
        reader.readAsDataURL(blob)
      }, 'image/jpeg', 0.8)
    }

    const interval = setInterval(sendFrame, 33) // ~30 FPS

    return () => clearInterval(interval)
  }, [ws, isConnected, blurIntensity])

  const handleCanvasClick = (e) => {
    if (!ws || !isConnected) return

    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width))
    const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height))

    ws.send(JSON.stringify({
      type: 'select',
      x,
      y
    }))
  }

  return (
    <div className="relative">
      <video
        ref={videoRef}
        src={videoSrc}
        autoPlay
        loop
        muted
        className="hidden"
      />
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full rounded-lg cursor-crosshair shadow-2xl"
      />
      {!isConnected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
          <div className="text-white">Connecting to AI server...</div>
        </div>
      )}
    </div>
  )
}
