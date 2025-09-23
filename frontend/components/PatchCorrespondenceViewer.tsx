import { useState } from 'react'
import { PatchCorrespondence } from '@/types/search'
import AttentionHeatmapOverlay from './AttentionHeatmapOverlay'
import RobustImage from './RobustImage'

interface PatchCorrespondenceViewerProps {
  queryImageUrl: string
  candidateImageUrl: string
  correspondences: PatchCorrespondence[]
  queryGridSize: number
  candidateGridSize: number
  className?: string
}

export default function PatchCorrespondenceViewer({
  queryImageUrl,
  candidateImageUrl,
  correspondences,
  queryGridSize,
  candidateGridSize,
  className = ''
}: PatchCorrespondenceViewerProps) {
  const [selectedQueryPatches, setSelectedQueryPatches] = useState<Set<number>>(new Set())
  const [hoveredPatch, setHoveredPatch] = useState<{ type: 'query' | 'candidate', idx: number } | null>(null)
  const [showTop10, setShowTop10] = useState(true)

  // Get top 10 similar patches for all selected query patches
  const getTopSimilarPatches = () => {
    if (selectedQueryPatches.size === 0) return []

    const allSimilarPatches: Array<{ candidateIdx: number, queryIdx: number, similarity: number }> = []

    selectedQueryPatches.forEach(queryIdx => {
      const correspondence = correspondences.find(c => c.query_patch_idx === queryIdx)
      if (correspondence) {
        correspondence.top_candidate_indices.slice(0, 10).forEach((candidateIdx, rank) => {
          allSimilarPatches.push({
            candidateIdx,
            queryIdx,
            similarity: correspondence.similarity_scores[rank]
          })
        })
      }
    })

    // Sort by similarity and take top 10
    return allSimilarPatches
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 10)
  }

  const topSimilarPatches = getTopSimilarPatches()

  const handlePatchHover = (type: 'query' | 'candidate', patchIdx: number | null) => {
    if (patchIdx === null) {
      setHoveredPatch(null)
    } else {
      setHoveredPatch({ type, idx: patchIdx })
    }
  }

  const handleQueryPatchClick = (patchIdx: number) => {
    const newSelected = new Set(selectedQueryPatches)
    if (newSelected.has(patchIdx)) {
      newSelected.delete(patchIdx)
    } else {
      newSelected.add(patchIdx)
    }
    setSelectedQueryPatches(newSelected)
  }

  const clearSelection = () => {
    setSelectedQueryPatches(new Set())
  }

  const getQueryPatchHighlights = () => {
    const patches: Array<{ idx: number, type: 'selected' | 'hovered' }> = []

    // Add selected patches
    selectedQueryPatches.forEach(idx => {
      patches.push({ idx, type: 'selected' })
    })

    // Add hovered patch if it's not already selected
    if (hoveredPatch?.type === 'query' && !selectedQueryPatches.has(hoveredPatch.idx)) {
      patches.push({ idx: hoveredPatch.idx, type: 'hovered' })
    }

    return patches
  }

  const getCandidatePatchHighlights = () => {
    const patches: Array<{ idx: number, similarity: number, querySource: number, rank: number }> = []

    if (showTop10) {
      topSimilarPatches.forEach((patch, rank) => {
        patches.push({
          idx: patch.candidateIdx,
          similarity: patch.similarity,
          querySource: patch.queryIdx,
          rank: rank + 1
        })
      })
    }

    return patches
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          Interactive Patch Correspondence Analysis
        </h3>
        <p className="text-sm text-gray-600">
          Click patches on the query image to see the top 10 most similar patches in the candidate image
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <div className="text-sm">
            <span className="font-medium text-gray-700">Selected patches:</span>
            <span className="ml-2 bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">
              {selectedQueryPatches.size}
            </span>
          </div>
          {topSimilarPatches.length > 0 && (
            <div className="text-sm">
              <span className="font-medium text-gray-700">Top matches:</span>
              <span className="ml-2 bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                {topSimilarPatches.length}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={clearSelection}
            disabled={selectedQueryPatches.size === 0}
            className="px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Clear Selection
          </button>
        </div>
      </div>

      {/* Image Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Query Image */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 text-center">
            Query Image - Click patches to select
          </h4>
          <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden relative">
            <img
              src={queryImageUrl}
              alt="Query tattoo image"
              className="w-full h-full object-cover rounded-lg"
            />

            {/* Interactive grid overlay */}
            <div className="absolute inset-0">
              <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                {/* Grid lines */}
                {Array.from({ length: queryGridSize - 1 }, (_, i) => {
                  const pos = ((i + 1) * 100) / queryGridSize;
                  return (
                    <g key={i}>
                      <line x1={pos} y1="0" x2={pos} y2="100" stroke="rgba(255,255,255,0.4)" strokeWidth="0.3" />
                      <line x1="0" y1={pos} x2="100" y2={pos} stroke="rgba(255,255,255,0.4)" strokeWidth="0.3" />
                    </g>
                  );
                })}

                {/* Patch highlights */}
                {getQueryPatchHighlights().map((patch, idx) => {
                  const row = Math.floor(patch.idx / queryGridSize);
                  const col = patch.idx % queryGridSize;
                  const x = (col * 100) / queryGridSize;
                  const y = (row * 100) / queryGridSize;
                  const width = 100 / queryGridSize;
                  const height = 100 / queryGridSize;

                  return (
                    <rect
                      key={`query-${patch.idx}`}
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill={patch.type === 'selected' ? "rgba(59, 130, 246, 0.4)" : "rgba(255, 255, 255, 0.2)"}
                      stroke={patch.type === 'selected' ? "rgba(59, 130, 246, 0.8)" : "rgba(255, 255, 255, 0.6)"}
                      strokeWidth="1"
                      className="cursor-pointer"
                    />
                  );
                })}

                {/* Interactive click areas */}
                {Array.from({ length: queryGridSize * queryGridSize }, (_, i) => {
                  const row = Math.floor(i / queryGridSize);
                  const col = i % queryGridSize;
                  const x = (col * 100) / queryGridSize;
                  const y = (row * 100) / queryGridSize;
                  const width = 100 / queryGridSize;
                  const height = 100 / queryGridSize;

                  return (
                    <rect
                      key={`click-${i}`}
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill="transparent"
                      className="cursor-pointer hover:fill-blue-100 hover:fill-opacity-30"
                      onClick={() => handleQueryPatchClick(i)}
                      onMouseEnter={() => handlePatchHover('query', i)}
                      onMouseLeave={() => handlePatchHover('query', null)}
                    />
                  );
                })}
              </svg>
            </div>
          </div>
        </div>

        {/* Candidate Image */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 text-center">
            Candidate Image - Top 10 similar patches
          </h4>
          <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden relative">
            <RobustImage
              src={candidateImageUrl}
              alt="Candidate tattoo image"
              className="w-full h-full object-cover rounded-lg"
            />

            {/* Grid overlay with highlights */}
            <div className="absolute inset-0">
              <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                {/* Grid lines */}
                {Array.from({ length: candidateGridSize - 1 }, (_, i) => {
                  const pos = ((i + 1) * 100) / candidateGridSize;
                  return (
                    <g key={i}>
                      <line x1={pos} y1="0" x2={pos} y2="100" stroke="rgba(255,255,255,0.4)" strokeWidth="0.3" />
                      <line x1="0" y1={pos} x2="100" y2={pos} stroke="rgba(255,255,255,0.4)" strokeWidth="0.3" />
                    </g>
                  );
                })}

                {/* Top 10 similar patches */}
                {getCandidatePatchHighlights().map((patch) => {
                  const row = Math.floor(patch.idx / candidateGridSize);
                  const col = patch.idx % candidateGridSize;
                  const x = (col * 100) / candidateGridSize;
                  const y = (row * 100) / candidateGridSize;
                  const width = 100 / candidateGridSize;
                  const height = 100 / candidateGridSize;

                  // Color gradient from green (best) to yellow (worst)
                  const intensity = Math.max(0.3, 1 - (patch.rank - 1) / 9);
                  const hue = patch.rank <= 3 ? 120 : patch.rank <= 6 ? 60 : 30; // Green -> Yellow -> Orange

                  return (
                    <g key={`candidate-${patch.idx}`}>
                      <rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={`hsla(${hue}, 70%, 50%, ${intensity * 0.4})`}
                        stroke={`hsla(${hue}, 70%, 40%, 0.8)`}
                        strokeWidth="1"
                      />
                      {/* Rank number */}
                      <text
                        x={x + width/2}
                        y={y + height/2}
                        fill="white"
                        fontSize="2"
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontWeight="bold"
                        stroke="rgba(0,0,0,0.5)"
                        strokeWidth="0.3"
                      >
                        {patch.rank}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Analysis */}
      {topSimilarPatches.length > 0 && (
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-900 mb-4">
            Top 10 Similar Patches ({selectedQueryPatches.size} query patches selected)
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-3">
            {topSimilarPatches.slice(0, 10).map((patch, idx) => {
              const queryCoord = correspondences.find(c => c.query_patch_idx === patch.queryIdx)?.query_patch_coord || [0, 0];
              const candidateCoord = correspondences.find(c => c.query_patch_idx === patch.queryIdx)?.top_candidate_coords.find((_, i) =>
                correspondences.find(c => c.query_patch_idx === patch.queryIdx)?.top_candidate_indices[i] === patch.candidateIdx
              ) || [0, 0];

              return (
                <div key={`${patch.queryIdx}-${patch.candidateIdx}`} className="bg-gray-50 rounded-lg p-3 text-center border">
                  <div className="text-xs font-medium text-gray-700 mb-1">
                    Rank #{idx + 1}
                  </div>
                  <div className="text-xs text-gray-600 mb-1">
                    Query ({queryCoord[0]}, {queryCoord[1]}) → ({candidateCoord[0]}, {candidateCoord[1]})
                  </div>
                  <div className="text-xs font-semibold mb-2" style={{
                    color: idx < 3 ? '#10b981' : idx < 6 ? '#f59e0b' : '#f97316'
                  }}>
                    {(patch.similarity * 100).toFixed(1)}% similarity
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5">
                    <div
                      className="h-1.5 rounded-full transition-all duration-300"
                      style={{
                        width: `${patch.similarity * 100}%`,
                        backgroundColor: idx < 3 ? '#10b981' : idx < 6 ? '#f59e0b' : '#f97316'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h5 className="text-xs font-semibold text-gray-700 mb-2">How to use:</h5>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• <strong>Click patches</strong> on the query image to select/deselect them</li>
          <li>• <strong>Selected patches</strong> appear in blue on the query image</li>
          <li>• <strong>Top 10 matches</strong> automatically appear on the candidate image</li>
          <li>• <strong>Color coding:</strong> Green = best matches, Yellow = good, Orange = moderate</li>
          <li>• <strong>Numbers</strong> show ranking (1 = most similar)</li>
          <li>• <strong>Hover effects</strong> provide visual feedback</li>
        </ul>
      </div>
    </div>
  )
}