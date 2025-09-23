import { useState } from 'react'

interface PatchAttentionToggleProps {
  enabled: boolean
  onChange: (enabled: boolean) => void
  disabled?: boolean
  modelSupported?: boolean
}

export default function PatchAttentionToggle({
  enabled,
  onChange,
  disabled = false,
  modelSupported = true
}: PatchAttentionToggleProps) {
  const [showTooltip, setShowTooltip] = useState(false)

  const isDisabled = disabled || !modelSupported

  const handleToggle = () => {
    if (!isDisabled) {
      onChange(!enabled)
    }
  }

  return (
    <div className="relative">
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-200">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <svg
                className="w-5 h-5 text-purple-600"
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
            </div>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-semibold text-gray-900">
                Patch Attention Analysis
              </h3>
              <button
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
            <p className="text-xs text-gray-600 mt-1">
              {!modelSupported
                ? 'Not supported by current model'
                : enabled
                ? 'Analyze which image patches are most similar'
                : 'Enable to see detailed patch-level comparisons'
              }
            </p>
          </div>
        </div>

        <div className="flex-shrink-0">
          <button
            onClick={handleToggle}
            disabled={isDisabled}
            className={`
              relative inline-flex h-6 w-11 items-center rounded-full transition-colors
              ${isDisabled
                ? 'bg-gray-200 cursor-not-allowed'
                : enabled
                ? 'bg-purple-600 hover:bg-purple-700'
                : 'bg-gray-200 hover:bg-gray-300'
              }
            `}
          >
            <span
              className={`
                inline-block h-4 w-4 transform rounded-full bg-white transition-transform
                ${enabled ? 'translate-x-6' : 'translate-x-1'}
              `}
            />
          </button>
        </div>
      </div>

      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute top-full left-0 right-0 mt-2 z-10">
          <div className="bg-gray-900 text-white text-xs rounded-lg p-3 shadow-lg">
            <p className="font-medium mb-1">Patch Attention Analysis</p>
            <p>
              This advanced feature breaks down images into patches and shows which parts
              are most similar between your query and result images. It provides:
            </p>
            <ul className="mt-2 space-y-1 text-xs">
              <li>• Attention heatmaps showing similarity patterns</li>
              <li>• Top matching patch pairs</li>
              <li>• Detailed similarity statistics</li>
            </ul>
            <p className="mt-2 text-xs text-gray-300">
              Note: Processing time may be longer with this feature enabled.
            </p>
          </div>
        </div>
      )}

      {/* Warning for unsupported models */}
      {!modelSupported && (
        <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-800">
          <div className="flex items-center space-x-1">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <span>Patch attention is not supported by the currently selected embedding model.</span>
          </div>
        </div>
      )}
    </div>
  )
}