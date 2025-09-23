import { useEffect, useRef, useState } from 'react'
import { PatchAttentionData } from '@/types/search'

interface AttentionHeatmapOverlayProps {
  imageUrl: string
  attentionData: PatchAttentionData | null
  isQuery?: boolean
  onPatchHover?: (patchIdx: number | null) => void
  highlightedPatch?: number | null
  className?: string
}

export default function AttentionHeatmapOverlay({
  imageUrl,
  attentionData,
  isQuery = false,
  onPatchHover,
  highlightedPatch,
  className = ''
}: AttentionHeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [imageError, setImageError] = useState(false)
  const [crossOriginFailed, setCrossOriginFailed] = useState(false)

  // Update canvas when image loads or attention data changes
  useEffect(() => {
    if (imageLoaded && attentionData) {
      drawAttentionOverlay()
    }
  }, [imageLoaded, attentionData, highlightedPatch])

  // Handle image load
  const handleImageLoad = () => {
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect()
      setCanvasSize({ width: rect.width, height: rect.height })
      setImageLoaded(true)
      setImageError(false)
    }
  }

  // Handle image error
  const handleImageError = () => {
    if (!crossOriginFailed) {
      // First try without crossOrigin
      setCrossOriginFailed(true)
      setImageError(false)
    } else {
      setImageError(true)
    }
  }

  // Draw attention heatmap overlay
  const drawAttentionOverlay = () => {
    if (!canvasRef.current || !attentionData || !imageRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = imageRef.current.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const gridSize = isQuery ? attentionData.query_grid_size : attentionData.candidate_grid_size
    const patchWidth = canvas.width / gridSize
    const patchHeight = canvas.height / gridSize

    // Draw patch grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1

    // Vertical lines
    for (let i = 1; i < gridSize; i++) {
      const x = i * patchWidth
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }

    // Horizontal lines
    for (let i = 1; i < gridSize; i++) {
      const y = i * patchHeight
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Highlight specific patch if provided
    if (highlightedPatch !== null && highlightedPatch !== undefined) {
      const row = Math.floor(highlightedPatch / gridSize)
      const col = highlightedPatch % gridSize
      const x = col * patchWidth
      const y = row * patchHeight

      // Draw highlight
      ctx.fillStyle = 'rgba(255, 215, 0, 0.4)' // Golden highlight
      ctx.fillRect(x, y, patchWidth, patchHeight)

      // Draw border
      ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)'
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, patchWidth, patchHeight)
    }
  }

  // Handle mouse move for patch detection
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!attentionData || !onPatchHover) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    const gridSize = isQuery ? attentionData.query_grid_size : attentionData.candidate_grid_size
    const patchWidth = canvas.width / gridSize
    const patchHeight = canvas.height / gridSize

    const col = Math.floor(x / patchWidth)
    const row = Math.floor(y / patchHeight)

    if (col >= 0 && col < gridSize && row >= 0 && row < gridSize) {
      const patchIdx = row * gridSize + col
      onPatchHover(patchIdx)
    }
  }

  const handleMouseLeave = () => {
    if (onPatchHover) {
      onPatchHover(null)
    }
  }

  return (
    <div className={`relative ${className}`}>
      {!imageError ? (
        <img
          ref={imageRef}
          src={imageUrl}
          alt={isQuery ? 'Query image' : 'Candidate image'}
          onLoad={handleImageLoad}
          onError={handleImageError}
          className="w-full h-full object-cover rounded-lg"
          crossOrigin={crossOriginFailed ? undefined : "anonymous"}
          key={`${imageUrl}-${crossOriginFailed}`} // Force re-render on crossOrigin change
        />
      ) : (
        <div className="w-full h-full bg-gray-200 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <svg className="w-16 h-16 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-sm">Failed to load image</p>
            <p className="text-xs mt-1">{isQuery ? 'Query' : 'Candidate'} image unavailable</p>
          </div>
        </div>
      )}

      {imageLoaded && attentionData && (
        <>
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none rounded-lg"
            style={{ width: canvasSize.width, height: canvasSize.height }}
          />

          {onPatchHover && (
            <div
              className="absolute inset-0 w-full h-full cursor-crosshair rounded-lg"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            />
          )}
        </>
      )}

      {/* Attention info overlay */}
      {attentionData && (
        <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
          <div className="text-center">
            <div className="font-medium">
              {isQuery ? 'Query' : 'Candidate'}
            </div>
            <div className="text-xs opacity-75">
              {attentionData.query_grid_size}×{attentionData.candidate_grid_size} patches
            </div>
            <div className="text-xs opacity-75">
              Similarity: {(attentionData.overall_similarity * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Patch coordinates tooltip */}
      {highlightedPatch !== null && highlightedPatch !== undefined && attentionData && (
        <div className="absolute bottom-2 left-2 bg-yellow-500 text-black text-xs px-2 py-1 rounded font-medium">
          Patch {highlightedPatch} ({Math.floor(highlightedPatch / (isQuery ? attentionData.query_grid_size : attentionData.candidate_grid_size))}, {highlightedPatch % (isQuery ? attentionData.query_grid_size : attentionData.candidate_grid_size)})
        </div>
      )}
    </div>
  )
}