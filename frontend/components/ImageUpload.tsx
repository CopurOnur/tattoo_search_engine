import { useState, useRef, DragEvent, ChangeEvent } from 'react'
import Image from 'next/image'

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  selectedImage: File | null
  isLoading: boolean
}

export default function ImageUpload({
  onImageSelect,
  selectedImage,
  isLoading
}: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      onImageSelect(file)

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      alert('Please select an image file')
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full max-w-md mx-auto">
      <div
        className={`
          relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-colors duration-200
          ${dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400'
          }
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
          disabled={isLoading}
        />

        {previewUrl ? (
          <div className="space-y-4">
            <div className="relative w-48 h-48 mx-auto">
              <Image
                src={previewUrl}
                alt="Selected tattoo"
                fill
                className="object-cover rounded-lg"
              />
            </div>
            <p className="text-sm text-gray-600">
              {selectedImage?.name}
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                setPreviewUrl(null)
                onImageSelect(null as any)
              }}
              className="text-sm text-red-500 hover:text-red-700"
              disabled={isLoading}
            >
              Remove image
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="mx-auto w-12 h-12 text-gray-400">
              <svg
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            <div>
              <p className="text-lg font-medium text-gray-700">
                Upload a tattoo image
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Drag and drop or click to select
              </p>
            </div>
            <p className="text-xs text-gray-400">
              PNG, JPG, GIF up to 10MB
            </p>
          </div>
        )}
      </div>
    </div>
  )
}