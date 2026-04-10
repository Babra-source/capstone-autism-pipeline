'use client'
import { useRef, useState, useCallback } from 'react'
import Image from 'next/image'
import type { ResultData } from './emotion'

type Props = {
  onResult: (data: ResultData) => void
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'

function dataUrlToBlob(dataUrl: string) {
  const [header, base64] = dataUrl.split(',')
  const mimeMatch = header.match(/:(.*?);/)
  const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg'
  const binary = atob(base64)
  const buffer = new Uint8Array(binary.length)

  for (let i = 0; i < binary.length; i += 1) {
    buffer[i] = binary.charCodeAt(i)
  }

  return new Blob([buffer], { type: mime })
}

export default function Webcam({ onResult }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [active, setActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [captured, setCaptured] = useState<string | null>(null)

  const startCam = async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setActive(true)
        setCaptured(null)
      }
    } catch {
      setError('Camera access denied. Please allow camera permissions.')
    }
  }

  const stopCam = () => {
    const video = videoRef.current
    if (video?.srcObject) {
      ;(video.srcObject as MediaStream).getTracks().forEach((t) => t.stop())
      video.srcObject = null
    }
    setActive(false)
  }

  const captureAndPredict = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')?.drawImage(video, 0, 0)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
    setCaptured(dataUrl)
    setLoading(true)
    setError(null)

    try {
      const blob = dataUrlToBlob(dataUrl)
      const file = new File([blob], 'capture.jpg', { type: blob.type })
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = (await res.json()) as ResultData
      onResult(data)
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message || 'Prediction failed.')
      } else {
        setError('Prediction failed.')
      }
    } finally {
      setLoading(false)
    }
  }, [onResult])

  return (
    <div className="flex flex-col gap-4 w-full max-w-sm">
      <p className="text-xs uppercase tracking-widest font-mono text-gray-400">Webcam</p>

      <div className="relative bg-gray-900 rounded-xl overflow-hidden aspect-video flex items-center justify-center">
        <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
            <div className="w-10 h-10 rounded-full border border-gray-600 flex items-center justify-center">
              <div className="w-4 h-4 rounded-full bg-gray-600" />
            </div>
            <p className="text-gray-500 text-xs">Camera off</p>
          </div>
        )}
        {captured && !active && (
          <Image
            src={captured}
            alt="captured"
            fill
            unoptimized
            className="object-cover opacity-60"
          />
        )}
      </div>
      <canvas ref={canvasRef} className="hidden" />

      <div className="bg-gray-50 border border-gray-100 rounded-lg px-4 py-3 text-xs text-gray-500 leading-relaxed">
        Start camera → capture a frame → send the image to the backend model.
      </div>

      <div className="flex gap-2">
        {!active ? (
          <button
            onClick={startCam}
            className="flex-1 py-2.5 bg-blue-500 text-white rounded-xl text-sm font-semibold hover:bg-blue-600 transition-colors"
          >
            Start camera
          </button>
        ) : (
          <>
            <button
              onClick={captureAndPredict}
              disabled={loading}
              className="flex-1 py-2.5 bg-blue-500 text-white rounded-xl text-sm font-semibold hover:bg-blue-600 disabled:opacity-60 transition-colors"
            >
              {loading ? 'Analysing…' : '📸 Capture frame'}
            </button>
            <button
              onClick={stopCam}
              className="px-4 py-2.5 border border-red-200 text-red-500 rounded-xl text-sm font-medium hover:bg-red-50 transition-colors"
            >
              Stop
            </button>
          </>
        )}
      </div>

      {error && (
        <p className="text-xs text-red-500 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
    </div>
  )
}
