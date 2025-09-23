export interface SearchResult {
  score: number
  url: string
  patch_attention?: PatchAttentionData
}

export interface PatchAttentionData {
  overall_similarity: number
  query_grid_size: number
  candidate_grid_size: number
  attention_summary: AttentionSummary
}

export interface AttentionSummary {
  overall_similarity: number
  max_similarity: number
  min_similarity: number
  std_similarity: number
  query_patches_count: number
  candidate_patches_count: number
  high_attention_patches: number
  model_name: string
}

export interface PatchCorrespondence {
  query_patch_idx: number
  query_patch_coord: [number, number]
  top_candidate_indices: number[]
  top_candidate_coords: Array<[number, number]>
  similarity_scores: number[]
}

export interface SearchResponse {
  caption: string
  results: SearchResult[]
  embedding_model: string
  patch_attention_enabled?: boolean
}

export interface DetailedAttentionAnalysis {
  query_image_size: [number, number]
  candidate_image_size: [number, number]
  candidate_url: string
  embedding_model: string
  similarity_analysis: AttentionSummary
  attention_matrix_shape: [number, number]
  top_correspondences: PatchCorrespondence[]
  visualizations?: {
    attention_heatmap: string
    top_correspondences: string
  }
}