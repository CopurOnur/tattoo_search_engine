import { useState, useRef } from 'react'

interface RobustImageProps {
  src: string
  alt: string
  className?: string
  onLoad?: () => void
  onError?: () => void
}

export default function RobustImage({ src, alt, className = '', onLoad, onError }: RobustImageProps) {
  const [attempts, setAttempts] = useState(0)
  const [imageError, setImageError] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)
  const imgRef = useRef<HTMLImageElement>(null)

  const handleImageError = () => {
    if (attempts === 0) {
      // First attempt failed, try without CORS
      setAttempts(1)
      if (imgRef.current) {
        imgRef.current.crossOrigin = ''
        imgRef.current.src = src // Force reload
      }
    } else {
      // All attempts failed
      setImageError(true)
      onError?.()
    }
  }

  const handleImageLoad = () => {
    setImageError(false)
    setImageLoaded(true)
    onLoad?.()
  }

  if (imageError) {
    return (
      <div className={`bg-gray-200 rounded-lg flex items-center justify-center ${className}`}>
        <div className="text-center text-gray-500 p-4">
          <svg className="w-16 h-16 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          <p className="text-sm font-medium">Image unavailable</p>
          <p className="text-xs mt-1">Unable to load from source</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`relative ${className}`}>
      {!imageLoaded && !imageError && (
        <div className="absolute inset-0 bg-gray-100 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="animate-spin w-8 h-8 border-2 border-gray-300 border-t-blue-500 rounded-full mx-auto mb-2"></div>
            <p className="text-sm">Loading image...</p>
          </div>
        </div>
      )}
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        className={`${className} ${imageLoaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
        onLoad={handleImageLoad}
        onError={handleImageError}
        crossOrigin={attempts === 0 ? "anonymous" : undefined}
      />
    </div>
  )
}