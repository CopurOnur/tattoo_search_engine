import { useState } from 'react'
import Head from 'next/head'
import ImageUpload from '@/components/ImageUpload'
import SearchResults from '@/components/SearchResults'
import ModelSelector from '@/components/ModelSelector'

interface SearchResult {
  score: number
  url: string
}

interface SearchResponse {
  caption: string
  results: SearchResult[]
  embedding_model: string
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function TattooSearch() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [results, setResults] = useState<SearchResult[]>([])
  const [caption, setCaption] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [hasSearched, setHasSearched] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string>('clip')
  const [usedModel, setUsedModel] = useState<string>('')

  const handleImageSelect = (file: File | null) => {
    setSelectedImage(file)
    if (!file) {
      setResults([])
      setCaption('')
      setError('')
      setHasSearched(false)
      setUsedModel('')
    }
  }

  const handleSearch = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    setIsLoading(true)
    setError('')
    setResults([])
    setCaption('')

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const searchUrl = `${BACKEND_URL}/search?embedding_model=${encodeURIComponent(selectedModel)}`
      const response = await fetch(searchUrl, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: SearchResponse = await response.json()
      setResults(data.results)
      setCaption(data.caption)
      setUsedModel(data.embedding_model)
      setHasSearched(true)

      if (data.results.length === 0) {
        setError('No similar tattoos found. Try a different image.')
      }
    } catch (err) {
      console.error('Search error:', err)
      setError(
        err instanceof Error
          ? `Search failed: ${err.message}`
          : 'An error occurred while searching. Please try again.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <>
      <Head>
        <title>Tattoo Search Engine | Find Similar Tattoo Designs</title>
        <meta
          name="description"
          content="Upload a tattoo image and find similar designs using AI-powered visual search"
        />
      </Head>

      <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">
              Tattoo Search Engine
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Upload a tattoo image and discover similar designs from across the web.
              Our AI analyzes your image and finds visually similar tattoos.
            </p>
          </div>

          {/* Upload Section */}
          <div className="mb-8">
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              isLoading={isLoading}
            />
          </div>

          {/* Model Selection */}
          {selectedImage && (
            <div className="mb-8 max-w-md mx-auto">
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
                disabled={isLoading}
              />
            </div>
          )}

          {/* Search Button */}
          {selectedImage && (
            <div className="text-center mb-8">
              <button
                onClick={handleSearch}
                disabled={isLoading || !selectedImage}
                className={`
                  px-8 py-3 rounded-lg font-medium transition-colors duration-200
                  ${isLoading || !selectedImage
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-primary-600 text-white hover:bg-primary-700'
                  }
                `}
              >
                {isLoading ? (
                  <span className="flex items-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Searching for similar tattoos...
                  </span>
                ) : (
                  'Search for Similar Tattoos'
                )}
              </button>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-8">
              <div className="max-w-md mx-auto bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg
                      className="h-5 w-5 text-red-400"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-red-800">
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          <SearchResults
            results={results}
            caption={caption}
            isLoading={isLoading}
            embeddingModel={usedModel}
          />

          {/* Instructions */}
          {!hasSearched && !selectedImage && (
            <div className="mt-16 max-w-3xl mx-auto">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6 text-center">
                How it works
              </h2>
              <div className="grid md:grid-cols-3 gap-8">
                <div className="text-center">
                  <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg
                      className="w-6 h-6 text-primary-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    1. Upload Image
                  </h3>
                  <p className="text-gray-600">
                    Upload a clear photo of the tattoo you want to find similar designs for
                  </p>
                </div>

                <div className="text-center">
                  <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg
                      className="w-6 h-6 text-primary-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    2. AI Analysis
                  </h3>
                  <p className="text-gray-600">
                    Our AI generates a description and searches for visually similar tattoos
                  </p>
                </div>

                <div className="text-center">
                  <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg
                      className="w-6 h-6 text-primary-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                      />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    3. Discover Results
                  </h3>
                  <p className="text-gray-600">
                    Browse through similar tattoo designs ranked by similarity score
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}