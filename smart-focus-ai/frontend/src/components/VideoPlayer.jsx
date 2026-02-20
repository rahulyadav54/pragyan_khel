import { useRef, useEffect, useState } from 'react'

const getWebSocketUrl = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}/ws/video`
}

export default function VideoPlayer({ videoSrc, blurIntensity, onFpsUpdate, onObjectSelect, onLoadingChange }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const sendingRef = useRef(false)
  const qualityRef = useRef(0.62)
  const sendEveryMs = 55
  const maxProcessingWidth = 960
  const [ws, setWs] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [hasFrame, setHasFrame] = useState(false)
  const fpsRef = useRef({ frames: 0, lastTime: Date.now() })

  useEffect(() => {
    if (onLoadingChange) {
      onLoadingChange(true)
    }

    const websocket = new WebSocket(getWebSocketUrl())
    
    websocket.onopen = () => {
      setIsConnected(true)
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
      if (onLoadingChange) {
        onLoadingChange(false)
      }
    }
    
    websocket.onmessage = (event) => {
      let data
      try {
        data = JSON.parse(event.data)
      } catch (error) {
        console.error('Invalid websocket payload:', error)
        return
      }
      
      if (data.type === 'frame') {
        sendingRef.current = false
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        const img = new Image()
        
        img.onload = () => {
          canvas.width = img.width
          canvas.height = img.height
          ctx.drawImage(img, 0, 0)
          setHasFrame(true)
          
          fpsRef.current.frames++
          const now = Date.now()
          if (now - fpsRef.current.lastTime >= 1000) {
            onFpsUpdate(fpsRef.current.frames)
            fpsRef.current.frames = 0
            fpsRef.current.lastTime = now
          }
        }
        
        img.src = data.data
      } else if (data.type === 'selected') {
        if (onObjectSelect) {
          onObjectSelect({ selected: data.track_id !== null && data.track_id !== undefined, trackId: data.track_id })
        }
      } else if (data.type === 'reset') {
        if (onObjectSelect) {
          onObjectSelect({ selected: false, trackId: null })
        }
      }
    }
    
    websocket.onclose = () => {
      setIsConnected(false)
      if (onLoadingChange) {
        onLoadingChange(false)
      }
    }

    setWs(websocket)
    
    return () => {
      websocket.close()
      setIsConnected(false)
      setIsVideoReady(false)
      setHasFrame(false)
      sendingRef.current = false
      if (onLoadingChange) {
        onLoadingChange(false)
      }
      onFpsUpdate(0)
    }
  }, [onFpsUpdate, onLoadingChange])

  useEffect(() => {
    if (onLoadingChange) {
      onLoadingChange(!(isConnected && isVideoReady && hasFrame))
    }
  }, [isConnected, isVideoReady, hasFrame, onLoadingChange])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    setIsVideoReady(false)
    setHasFrame(false)
    video.pause()
    video.load()

    const handleReady = async () => {
      try {
        await video.play()
        setIsVideoReady(true)
      } catch (error) {
        console.error('Video playback error:', error)
        setIsVideoReady(false)
      }
    }

    video.addEventListener('loadedmetadata', handleReady)
    video.addEventListener('canplay', handleReady)

    return () => {
      video.removeEventListener('loadedmetadata', handleReady)
      video.removeEventListener('canplay', handleReady)
    }
  }, [videoSrc])

  useEffect(() => {
    if (!videoRef.current || !ws || !isConnected || !isVideoReady) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    const sendFrame = () => {
      if (video.paused || video.ended) return
      if (!video.videoWidth || !video.videoHeight) return
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
      if (ws.readyState !== WebSocket.OPEN) return
      if (sendingRef.current) return

      const scale = video.videoWidth > maxProcessingWidth ? maxProcessingWidth / video.videoWidth : 1
      canvas.width = Math.max(1, Math.floor(video.videoWidth * scale))
      canvas.height = Math.max(1, Math.floor(video.videoHeight * scale))
      ctx.drawImage(video, 0, 0)

      canvas.toBlob((blob) => {
        if (!blob || ws.readyState !== WebSocket.OPEN) {
          sendingRef.current = false
          return
        }
        sendingRef.current = true
        const reader = new FileReader()
        reader.onloadend = () => {
          if (ws.readyState !== WebSocket.OPEN) {
            sendingRef.current = false
            return
          }
          ws.send(JSON.stringify({
            type: 'frame',
            data: reader.result,
            blur_intensity: blurIntensity
          }))
        }
        reader.readAsDataURL(blob)
      }, 'image/jpeg', qualityRef.current)
    }

    const interval = setInterval(sendFrame, sendEveryMs)

    return () => {
      clearInterval(interval)
      sendingRef.current = false
    }
  }, [ws, isConnected, isVideoReady, blurIntensity])

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
        playsInline
        className="hidden"
      />
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="aspect-video w-full rounded-xl bg-slate-900/80 object-contain shadow-2xl cursor-crosshair"
      />
      {!hasFrame && (
        <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-slate-950/60">
          <div className="text-sm text-slate-100">
            {!isConnected ? 'Connecting to AI server...' : 'Preparing video stream...'}
          </div>
        </div>
      )}
    </div>
  )
}
