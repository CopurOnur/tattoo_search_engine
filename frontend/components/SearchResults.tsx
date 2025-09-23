import Image from 'next/image'
import { useState } from 'react'
import { SearchResult } from '@/types/search'

interface SearchResultsProps {
  results: SearchResult[]
  caption: string
  isLoading: boolean
  embeddingModel?: string
  patchAttentionEnabled?: boolean
  onAnalyzeAttention?: (result: SearchResult) => void
}

export default function SearchResults({
  results,
  caption,
  isLoading,
  embeddingModel,
  patchAttentionEnabled = false,
  onAnalyzeAttention
}: SearchResultsProps) {
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set())

  const handleImageError = (url: string) => {
    setFailedImages(prev => new Set(prev).add(url))
  }

  if (isLoading) {
    return (
      <div className="w-full max-w-4xl mx-auto">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="space-y-3">
                <div className="aspect-square bg-gray-200 rounded-lg"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!results.length) {
    return null
  }

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Caption Display */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">
          Search Results
        </h2>
        <div className="space-y-2">
          <p className="text-gray-600 bg-gray-50 px-4 py-2 rounded-lg">
            <span className="font-medium">AI Caption:</span> "{caption}"
          </p>
          {embeddingModel && (
            <p className="text-sm text-gray-500 bg-blue-50 px-4 py-2 rounded-lg">
              <span className="font-medium">Embedding Model:</span> {embeddingModel}
            </p>
          )}
        </div>
      </div>

      {/* Results Grid */}
      <div className={`grid gap-6 ${patchAttentionEnabled ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
        {results.map((result, index) => (
          <div
            key={`${result.url}-${index}`}
            className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300"
          >
            <div className="aspect-square relative bg-gray-100">
              {!failedImages.has(result.url) ? (
                <Image
                  src={result.url}
                  alt={`Similar tattoo ${index + 1}`}
                  fill
                  className="object-cover"
                  onError={() => handleImageError(result.url)}
                  unoptimized
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-2 text-gray-400">
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
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                    </div>
                    <p className="text-sm text-gray-500">
                      Image unavailable
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-700">
                    Similarity:
                  </span>
                  <span className="text-sm font-bold text-primary-600">
                    {(result.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <div
                    className="w-16 bg-gray-200 rounded-full h-2"
                  >
                    <div
                      className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${result.score * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Patch Attention Info */}
              {result.patch_attention && (
                <div className="mb-3 p-2 bg-purple-50 rounded border border-purple-200">
                  <div className="text-xs font-medium text-purple-700 mb-1">
                    Patch Analysis Available
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs text-purple-600">
                    <div>
                      Overall: {(result.patch_attention.overall_similarity * 100).toFixed(1)}%
                    </div>
                    <div>
                      High attention: {result.patch_attention.attention_summary.high_attention_patches}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between">
                <a
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-sm text-primary-600 hover:text-primary-700"
                >
                  View original
                  <svg
                    className="w-3 h-3 ml-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                    />
                  </svg>
                </a>

                {/* Detailed Analysis Button */}
                {onAnalyzeAttention && (
                  <button
                    onClick={() => onAnalyzeAttention(result)}
                    className="inline-flex items-center text-sm text-purple-600 hover:text-purple-700 font-medium"
                  >
                    <svg
                      className="w-3 h-3 mr-1"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                    Analyze
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Results Summary */}
      <div className="mt-8 text-center text-sm text-gray-500">
        Showing {results.length} most similar tattoos
      </div>
    </div>
  )
}