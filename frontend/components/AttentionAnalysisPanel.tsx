import { useState } from 'react'
import { DetailedAttentionAnalysis } from '@/types/search'

interface AttentionAnalysisPanelProps {
  analysis: DetailedAttentionAnalysis
  className?: string
}

export default function AttentionAnalysisPanel({
  analysis,
  className = ''
}: AttentionAnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'statistics' | 'visualizations'>('overview')
  const [showRawData, setShowRawData] = useState(false)

  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'statistics', name: 'Statistics', icon: '📈' },
    { id: 'visualizations', name: 'Visualizations', icon: '🎨' }
  ] as const

  const renderOverview = () => (
    <div className="space-y-4">
      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
          <div className="text-xs font-medium text-blue-700 mb-1">Overall Similarity</div>
          <div className="text-lg font-bold text-blue-900">
            {(analysis.similarity_analysis.overall_similarity * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
          <div className="text-xs font-medium text-green-700 mb-1">Max Similarity</div>
          <div className="text-lg font-bold text-green-900">
            {(analysis.similarity_analysis.max_similarity * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
          <div className="text-xs font-medium text-purple-700 mb-1">High Attention</div>
          <div className="text-lg font-bold text-purple-900">
            {analysis.similarity_analysis.high_attention_patches}
          </div>
          <div className="text-xs text-purple-600">patches</div>
        </div>

        <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg border border-orange-200">
          <div className="text-xs font-medium text-orange-700 mb-1">Model Used</div>
          <div className="text-sm font-semibold text-orange-900">
            {analysis.embedding_model}
          </div>
        </div>
      </div>

      {/* Image Information */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Image Details</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-xs font-medium text-gray-700">Query Image</div>
            <div className="text-sm text-gray-600">
              Size: {analysis.query_image_size[0]} × {analysis.query_image_size[1]}
            </div>
            <div className="text-sm text-gray-600">
              Patches: {analysis.similarity_analysis.query_patches_count}
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-xs font-medium text-gray-700">Candidate Image</div>
            <div className="text-sm text-gray-600">
              Size: {analysis.candidate_image_size[0]} × {analysis.candidate_image_size[1]}
            </div>
            <div className="text-sm text-gray-600">
              Patches: {analysis.similarity_analysis.candidate_patches_count}
            </div>
          </div>
        </div>
      </div>

      {/* Top Correspondences Preview */}
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Top Patch Correspondences</h4>
        <div className="space-y-2">
          {analysis.top_correspondences.slice(0, 3).map((correspondence, idx) => (
            <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div className="text-sm">
                <span className="font-medium">#{idx + 1}</span>
                {' '}Query ({correspondence.query_patch_coord[0]}, {correspondence.query_patch_coord[1]})
                {' '}→ Candidate ({correspondence.top_candidate_coords[0][0]}, {correspondence.top_candidate_coords[0][1]})
              </div>
              <div className="text-sm font-semibold text-blue-600">
                {(correspondence.similarity_scores[0] * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderStatistics = () => (
    <div className="space-y-4">
      {/* Statistical Distribution */}
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Similarity Distribution</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-xs font-medium text-blue-700 mb-1">Mean</div>
            <div className="text-lg font-bold text-blue-900">
              {(analysis.similarity_analysis.overall_similarity * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-xs font-medium text-green-700 mb-1">Maximum</div>
            <div className="text-lg font-bold text-green-900">
              {(analysis.similarity_analysis.max_similarity * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center p-3 bg-red-50 rounded-lg">
            <div className="text-xs font-medium text-red-700 mb-1">Minimum</div>
            <div className="text-lg font-bold text-red-900">
              {(analysis.similarity_analysis.min_similarity * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>

      {/* Attention Matrix Info */}
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">Attention Matrix Properties</h4>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs font-medium text-gray-700 mb-1">Matrix Shape</div>
            <div className="text-sm text-gray-900">
              {analysis.attention_matrix_shape[0]} × {analysis.attention_matrix_shape[1]}
            </div>
          </div>
          <div>
            <div className="text-xs font-medium text-gray-700 mb-1">Standard Deviation</div>
            <div className="text-sm text-gray-900">
              {analysis.similarity_analysis.std_similarity.toFixed(4)}
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Correspondences */}
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-gray-900">All Correspondences</h4>
          <button
            onClick={() => setShowRawData(!showRawData)}
            className="text-xs text-blue-600 hover:text-blue-700"
          >
            {showRawData ? 'Hide' : 'Show'} Raw Data
          </button>
        </div>

        <div className="max-h-64 overflow-y-auto">
          {analysis.top_correspondences.map((correspondence, idx) => (
            <div key={idx} className="py-2 border-b border-gray-100 last:border-b-0">
              <div className="flex items-center justify-between">
                <div className="text-sm">
                  <span className="font-medium text-gray-900">#{idx + 1}</span>
                  <span className="text-gray-600 ml-2">
                    Query ({correspondence.query_patch_coord[0]}, {correspondence.query_patch_coord[1]})
                  </span>
                </div>
                <div className="text-sm font-semibold text-blue-600">
                  {(correspondence.similarity_scores[0] * 100).toFixed(1)}%
                </div>
              </div>

              {showRawData && (
                <div className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded">
                  <div>Top matches: {correspondence.top_candidate_coords.slice(0, 3).map(coord => `(${coord[0]},${coord[1]})`).join(', ')}</div>
                  <div>Scores: {correspondence.similarity_scores.slice(0, 3).map(s => (s * 100).toFixed(1)).join(', ')}%</div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const renderVisualizations = () => (
    <div className="space-y-4">
      {analysis.visualizations ? (
        <>
          {/* Attention Heatmap */}
          {analysis.visualizations.attention_heatmap && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3">Attention Heatmap</h4>
              <div className="text-center">
                <img
                  src={analysis.visualizations.attention_heatmap}
                  alt="Attention heatmap visualization"
                  className="max-w-full h-auto rounded-lg shadow-sm"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2 text-center">
                Shows overall attention patterns between query and candidate images
              </p>
            </div>
          )}

          {/* Top Correspondences Visualization */}
          {analysis.visualizations.top_correspondences && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <h4 className="text-sm font-semibold text-gray-900 mb-3">Top Patch Correspondences</h4>
              <div className="text-center">
                <img
                  src={analysis.visualizations.top_correspondences}
                  alt="Top patch correspondences visualization"
                  className="max-w-full h-auto rounded-lg shadow-sm"
                />
              </div>
              <p className="text-xs text-gray-500 mt-2 text-center">
                Visual comparison of the most similar patch pairs
              </p>
            </div>
          )}

          {/* Export Options */}
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <h4 className="text-sm font-semibold text-gray-900 mb-3">Export Visualizations</h4>
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  if (analysis.visualizations?.attention_heatmap) {
                    const link = document.createElement('a')
                    link.href = analysis.visualizations.attention_heatmap
                    link.download = 'attention_heatmap.png'
                    link.click()
                  }
                }}
                className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Download Heatmap
              </button>
              <button
                onClick={() => {
                  if (analysis.visualizations?.top_correspondences) {
                    const link = document.createElement('a')
                    link.href = analysis.visualizations.top_correspondences
                    link.download = 'patch_correspondences.png'
                    link.click()
                  }
                }}
                className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
              >
                Download Correspondences
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <div className="text-sm">No visualizations available</div>
          <div className="text-xs mt-1">Visualizations may have been disabled or failed to generate</div>
        </div>
      )}
    </div>
  )

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Detailed Attention Analysis</h3>
        <p className="text-sm text-gray-600 mt-1">
          In-depth analysis of patch-level attention patterns
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex space-x-0">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex-1 py-3 px-4 text-sm font-medium text-center border-b-2 transition-colors
                ${activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <span className="mr-1">{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="p-4">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'statistics' && renderStatistics()}
        {activeTab === 'visualizations' && renderVisualizations()}
      </div>
    </div>
  )
}