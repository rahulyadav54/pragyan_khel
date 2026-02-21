import { useRef, useEffect, useState } from 'react'

const getWebSocketCandidates = () => {
  const candidates = []
  if (import.meta.env.VITE_WS_URL) {
    candidates.push(import.meta.env.VITE_WS_URL)
  }
  const secure = window.location.protocol === 'https:'
  const protocol = secure ? 'wss' : 'ws'
  const host = window.location.host
  const hostname = window.location.hostname

  candidates.push(`${protocol}://${host}/ws/video`)

  if (!secure && (hostname === 'localhost' || hostname === '127.0.0.1')) {
    candidates.push('ws://localhost:8000/ws/video')
    candidates.push('ws://127.0.0.1:8000/ws/video')
  }

  return [...new Set(candidates)]
}

export default function WebcamCapture({ blurIntensity, onFpsUpdate, onObjectSelect, onLoadingChange }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const sendingRef = useRef(false)
  const qualityRef = useRef(0.48)
  const sendEveryMs = 34
  const maxProcessingWidth = 640
  const [ws, setWs] = useState(null)
  const [isActive, setIsActive] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [hasFrame, setHasFrame] = useState(false)
  const [connectionError, setConnectionError] = useState('')
  const fpsRef = useRef({ frames: 0, lastTime: Date.now() })

  useEffect(() => {
    let activeSocket = null
    let cancelled = false

    if (onLoadingChange) {
      onLoadingChange(true)
    }

    const setupSocketHandlers = (socket) => {
      socket.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      socket.onmessage = (event) => {
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
      
      socket.onclose = () => {
        if (cancelled) return
        setIsConnected(false)
        setIsActive(false)
        if (onLoadingChange) {
          onLoadingChange(false)
        }
      }
    }

    const connectSocket = async () => {
      const candidates = getWebSocketCandidates()
      for (const url of candidates) {
        if (cancelled) return
        const socket = new WebSocket(url)
        activeSocket = socket
        setupSocketHandlers(socket)

        const opened = await new Promise((resolve) => {
          const timeout = setTimeout(() => resolve(false), 2200)
          const handleOpen = () => {
            clearTimeout(timeout)
            resolve(true)
          }
          const handleError = () => {
            clearTimeout(timeout)
            resolve(false)
          }
          socket.addEventListener('open', handleOpen, { once: true })
          socket.addEventListener('error', handleError, { once: true })
        })

        if (opened) {
          if (cancelled) {
            socket.close()
            return
          }
          setConnectionError('')
          setIsConnected(true)
          setWs(socket)
          startWebcam()
          return
        }
        socket.close()
      }

      setConnectionError('Unable to connect to backend websocket. Start backend on port 8000.')
      setIsConnected(false)
      if (onLoadingChange) {
        onLoadingChange(false)
      }
    }

    connectSocket()
    
    return () => {
      cancelled = true
      if (activeSocket) {
        activeSocket.close()
      }
      setIsConnected(false)
      stopWebcam()
      setIsActive(false)
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
      onLoadingChange(!(isConnected && isActive && hasFrame))
    }
  }, [isConnected, isActive, hasFrame, onLoadingChange])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 30, max: 30 }
        }
      })
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setIsActive(true)
    } catch (err) {
      console.error('Webcam error:', err)
      setIsActive(false)
      if (onLoadingChange) {
        onLoadingChange(false)
      }
    }
  }

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop())
    }
  }

  useEffect(() => {
    if (!videoRef.current || !ws || !isConnected || !isActive) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')

    const sendFrame = () => {
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
  }, [ws, isConnected, isActive, blurIntensity])

  const handleCanvasClick = (e) => {
    if (!ws) return

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
        autoPlay
        playsInline
        muted
        className="hidden"
      />
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="aspect-video w-full rounded-xl bg-slate-900/80 object-contain shadow-2xl cursor-crosshair"
      />
      {!isConnected || !isActive || !hasFrame ? (
        <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-slate-950/60">
          <div className="text-sm text-slate-100">
            {!isConnected ? (connectionError || 'Connecting to AI server...') : 'Starting webcam...'}
          </div>
        </div>
      ) : null}
    </div>
  )
}
