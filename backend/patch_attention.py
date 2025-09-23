import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict, Any
import io
import base64
import math


class PatchAttentionAnalyzer:
    """Utility class for computing and visualizing patch-level attention between images."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def compute_patch_similarities(self, query_image: Image.Image, candidate_image: Image.Image) -> Dict[str, Any]:
        """
        Compute patch-level similarities between query and candidate images.

        Returns:
            Dictionary containing attention matrix, top correspondences, and metadata
        """
        try:
            # Get patch features for both images
            query_patches = self.embedding_model.encode_image_patches(query_image)
            candidate_patches = self.embedding_model.encode_image_patches(candidate_image)

            # Compute attention matrix
            attention_matrix = self.embedding_model.compute_patch_attention(query_patches, candidate_patches)

            # Get grid dimensions (assuming square patches for ViT models)
            query_grid_size = int(math.sqrt(query_patches.shape[0]))
            candidate_grid_size = int(math.sqrt(candidate_patches.shape[0]))

            # Find top correspondences for each query patch
            top_correspondences = []
            for i in range(attention_matrix.shape[0]):
                patch_similarities = attention_matrix[i]
                top_indices = torch.topk(patch_similarities, k=min(5, patch_similarities.shape[0]))

                top_correspondences.append({
                    'query_patch_idx': i,
                    'query_patch_coord': self._patch_idx_to_coord(i, query_grid_size),
                    'top_candidate_indices': top_indices.indices.tolist(),
                    'top_candidate_coords': [self._patch_idx_to_coord(idx.item(), candidate_grid_size)
                                           for idx in top_indices.indices],
                    'similarity_scores': top_indices.values.tolist()
                })

            return {
                'attention_matrix': attention_matrix.cpu().numpy(),
                'query_grid_size': query_grid_size,
                'candidate_grid_size': candidate_grid_size,
                'top_correspondences': top_correspondences,
                'query_patches_shape': query_patches.shape,
                'candidate_patches_shape': candidate_patches.shape,
                'overall_similarity': torch.mean(attention_matrix).item()
            }

        except NotImplementedError:
            raise ValueError(f"Patch-level encoding not supported for {self.embedding_model.get_model_name()}")
        except Exception as e:
            raise RuntimeError(f"Error computing patch similarities: {e}")

    def _patch_idx_to_coord(self, patch_idx: int, grid_size: int) -> Tuple[int, int]:
        """Convert flat patch index to (row, col) coordinate."""
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        return (row, col)

    def visualize_attention_heatmap(self, query_image: Image.Image, candidate_image: Image.Image,
                                  similarity_data: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Create a visualization showing attention heatmap between patches.
        Returns base64 encoded PNG image.
        """
        attention_matrix = similarity_data['attention_matrix']
        query_grid_size = similarity_data['query_grid_size']
        candidate_grid_size = similarity_data['candidate_grid_size']

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Patch Attention Analysis - Overall Similarity: {similarity_data["overall_similarity"]:.3f}',
                     fontsize=14, fontweight='bold')

        # Plot original images
        axes[0, 0].imshow(query_image)
        axes[0, 0].set_title('Query Image')
        axes[0, 0].axis('off')
        self._overlay_patch_grid(axes[0, 0], query_image.size, query_grid_size)

        axes[0, 1].imshow(candidate_image)
        axes[0, 1].set_title('Candidate Image')
        axes[0, 1].axis('off')
        self._overlay_patch_grid(axes[0, 1], candidate_image.size, candidate_grid_size)

        # Plot attention matrix
        im = axes[1, 0].imshow(attention_matrix, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Attention Matrix')
        axes[1, 0].set_xlabel('Candidate Patches')
        axes[1, 0].set_ylabel('Query Patches')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # Plot attention summary (max attention per query patch)
        max_attention_per_query = np.max(attention_matrix, axis=1)
        attention_grid = max_attention_per_query.reshape(query_grid_size, query_grid_size)

        im2 = axes[1, 1].imshow(attention_grid, cmap='hot', interpolation='nearest')
        axes[1, 1].set_title('Max Attention per Query Patch')
        axes[1, 1].set_xlabel('Patch Column')
        axes[1, 1].set_ylabel('Patch Row')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    def visualize_top_correspondences(self, query_image: Image.Image, candidate_image: Image.Image,
                                    similarity_data: Dict[str, Any], num_top_patches: int = 6) -> str:
        """
        Visualize the top corresponding patches between query and candidate images.
        Returns base64 encoded PNG image.
        """
        top_correspondences = similarity_data['top_correspondences']
        query_grid_size = similarity_data['query_grid_size']
        candidate_grid_size = similarity_data['candidate_grid_size']

        # Sort by best similarity score
        sorted_correspondences = sorted(
            top_correspondences,
            key=lambda x: max(x['similarity_scores']),
            reverse=True
        )[:num_top_patches]

        fig, axes = plt.subplots(2, num_top_patches, figsize=(3*num_top_patches, 6))
        fig.suptitle('Top Patch Correspondences', fontsize=14, fontweight='bold')

        for i, correspondence in enumerate(sorted_correspondences):
            query_coord = correspondence['query_patch_coord']
            best_candidate_coord = correspondence['top_candidate_coords'][0]
            best_score = correspondence['similarity_scores'][0]

            # Extract and show query patch
            query_patch = self._extract_patch_from_image(query_image, query_coord, query_grid_size)
            axes[0, i].imshow(query_patch)
            axes[0, i].set_title(f'Q-Patch {query_coord}\nScore: {best_score:.3f}')
            axes[0, i].axis('off')

            # Extract and show best matching candidate patch
            candidate_patch = self._extract_patch_from_image(candidate_image, best_candidate_coord, candidate_grid_size)
            axes[1, i].imshow(candidate_patch)
            axes[1, i].set_title(f'C-Patch {best_candidate_coord}')
            axes[1, i].axis('off')

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()

        return base64.b64encode(plot_data).decode()

    def _overlay_patch_grid(self, ax, image_size: Tuple[int, int], grid_size: int):
        """Overlay patch grid lines on image."""
        width, height = image_size
        patch_width = width / grid_size
        patch_height = height / grid_size

        # Draw vertical lines
        for i in range(1, grid_size):
            x = i * patch_width
            ax.axvline(x=x, color='white', alpha=0.5, linewidth=1)

        # Draw horizontal lines
        for i in range(1, grid_size):
            y = i * patch_height
            ax.axhline(y=y, color='white', alpha=0.5, linewidth=1)

    def _extract_patch_from_image(self, image: Image.Image, patch_coord: Tuple[int, int], grid_size: int) -> Image.Image:
        """Extract a specific patch from an image based on grid coordinates."""
        row, col = patch_coord
        width, height = image.size

        patch_width = width // grid_size
        patch_height = height // grid_size

        left = col * patch_width
        top = row * patch_height
        right = min((col + 1) * patch_width, width)
        bottom = min((row + 1) * patch_height, height)

        return image.crop((left, top, right, bottom))

    def get_similarity_summary(self, similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of similarity statistics."""
        attention_matrix = similarity_data['attention_matrix']

        return {
            'overall_similarity': similarity_data['overall_similarity'],
            'max_similarity': float(np.max(attention_matrix)),
            'min_similarity': float(np.min(attention_matrix)),
            'std_similarity': float(np.std(attention_matrix)),
            'query_patches_count': similarity_data['query_patches_shape'][0],
            'candidate_patches_count': similarity_data['candidate_patches_shape'][0],
            'high_attention_patches': int(np.sum(attention_matrix > (np.mean(attention_matrix) + np.std(attention_matrix)))),
            'model_name': self.embedding_model.get_model_name()
        }