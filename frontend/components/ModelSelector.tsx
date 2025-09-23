import { useEffect, useState } from 'react'

interface ModelConfig {
  model_name: string
  description: string
}

interface ModelConfigs {
  [key: string]: ModelConfig
}

interface ModelSelectorProps {
  selectedModel: string
  onModelChange: (model: string) => void
  disabled?: boolean
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function ModelSelector({
  selectedModel,
  onModelChange,
  disabled = false
}: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [modelConfigs, setModelConfigs] = useState<ModelConfigs>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/models`)
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        setAvailableModels(data.available_models || [])
        setModelConfigs(data.model_configs || {})
      } catch (err) {
        console.error('Failed to fetch models:', err)
        setError('Failed to load models')
        // Fallback to default models
        setAvailableModels(['clip', 'dinov2', 'siglip'])
        setModelConfigs({
          clip: {
            model_name: 'ViT-B-32',
            description: 'OpenAI CLIP model - good general purpose vision-language model'
          },
          dinov2: {
            model_name: 'dinov2_vitb14',
            description: 'Meta DINOv2 - self-supervised vision transformer, good for visual features'
          },
          siglip: {
            model_name: 'google/siglip-base-patch16-224',
            description: 'Google SigLIP - improved CLIP-like model with better training'
          }
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchModels()
  }, [])

  if (isLoading) {
    return (
      <div className="space-y-3">
        <div className="h-4 bg-gray-200 rounded animate-pulse"></div>
        <div className="h-10 bg-gray-200 rounded animate-pulse"></div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <label htmlFor="model-select" className="block text-sm font-medium text-gray-700">
        Embedding Model
        {error && (
          <span className="text-red-500 text-xs ml-2">({error})</span>
        )}
      </label>

      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={disabled}
        className={`
          block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm
          focus:outline-none focus:ring-primary-500 focus:border-primary-500
          ${disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
        `}
      >
        {availableModels.map((model) => (
          <option key={model} value={model}>
            {model.toUpperCase()} - {modelConfigs[model]?.model_name || model}
          </option>
        ))}
      </select>

      {modelConfigs[selectedModel] && (
        <p className="text-xs text-gray-500 mt-1">
          {modelConfigs[selectedModel].description}
        </p>
      )}
    </div>
  )
}