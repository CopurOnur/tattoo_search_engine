from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.preprocess = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model and preprocessing."""
        pass

    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode an image into feature vector."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass

    def compute_similarity(self, query_features: torch.Tensor, candidate_features: torch.Tensor) -> float:
        """Compute similarity between query and candidate features."""
        return torch.mm(query_features, candidate_features.T).item()


class CLIPEmbedding(EmbeddingModel):
    """CLIP-based embedding model."""

    def __init__(self, device: torch.device, model_name: str = "ViT-B-32"):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        """Load CLIP model and preprocessing."""
        try:
            import open_clip
            logger.info(f"Loading CLIP model: {self.model_name}")

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai"
            )
            self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            logger.info(f"CLIP model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using CLIP."""
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_input)
                features = F.normalize(features, p=2, dim=1)

            return features
        except Exception as e:
            logger.error(f"Failed to encode image with CLIP: {e}")
            raise

    def get_model_name(self) -> str:
        return f"CLIP-{self.model_name}"


class DINOv2Embedding(EmbeddingModel):
    """DINOv2-based embedding model."""

    def __init__(self, device: torch.device, model_name: str = "dinov2_vitb14"):
        super().__init__(device)
        self.model_name = model_name
        self.load_model()

    def load_model(self) -> None:
        """Load DINOv2 model and preprocessing."""
        try:
            import torch.hub
            from torchvision import transforms

            logger.info(f"Loading DINOv2 model: {self.model_name}")

            # Load DINOv2 model from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # DINOv2 preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info(f"DINOv2 model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using DINOv2."""
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(image_input)
                features = F.normalize(features, p=2, dim=1)

            return features
        except Exception as e:
            logger.error(f"Failed to encode image with DINOv2: {e}")
            raise

    def get_model_name(self) -> str:
        return f"DINOv2-{self.model_name}"


class SigLIPEmbedding(EmbeddingModel):
    """SigLIP-based embedding model."""

    def __init__(self, device: torch.device, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__(device)
        self.model_name = model_name
        self.processor = None
        self.load_model()

    def load_model(self) -> None:
        """Load SigLIP model and preprocessing."""
        try:
            # Check for required dependencies
            try:
                import sentencepiece
            except ImportError:
                raise ImportError(
                    "SentencePiece is required for SigLIP. Install with: pip install sentencepiece"
                )

            from transformers import SiglipVisionModel, SiglipProcessor

            logger.info(f"Loading SigLIP model: {self.model_name}")

            self.model = SiglipVisionModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self.processor = SiglipProcessor.from_pretrained(self.model_name)

            logger.info(f"SigLIP model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            raise

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image using SigLIP."""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                features = F.normalize(features, p=2, dim=1)

            return features
        except Exception as e:
            logger.error(f"Failed to encode image with SigLIP: {e}")
            raise

    def get_model_name(self) -> str:
        return f"SigLIP-{self.model_name.split('/')[-1]}"


class EmbeddingModelFactory:
    """Factory class for creating embedding models."""

    AVAILABLE_MODELS = {
        "clip": CLIPEmbedding,
        "dinov2": DINOv2Embedding,
        "siglip": SigLIPEmbedding,
    }

    @classmethod
    def create_model(cls, model_type: str, device: torch.device, **kwargs) -> EmbeddingModel:
        """Create an embedding model instance.

        Args:
            model_type: Type of model ('clip', 'dinov2', 'siglip')
            device: PyTorch device
            **kwargs: Additional arguments for specific models

        Returns:
            EmbeddingModel instance
        """
        if model_type.lower() not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls.AVAILABLE_MODELS.keys())}")

        model_class = cls.AVAILABLE_MODELS[model_type.lower()]

        try:
            return model_class(device, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            # Fallback to CLIP if the requested model fails
            if model_type.lower() != 'clip':
                logger.info("Falling back to CLIP model")
                return cls.AVAILABLE_MODELS['clip'](device, **kwargs)
            else:
                raise

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls.AVAILABLE_MODELS.keys())


def get_default_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for each model type."""
    return {
        "clip": {
            "model_name": "ViT-B-32",
            "description": "OpenAI CLIP model - good general purpose vision-language model"
        },
        "dinov2": {
            "model_name": "dinov2_vitb14",
            "description": "Meta DINOv2 - self-supervised vision transformer, good for visual features"
        },
        "siglip": {
            "model_name": "google/siglip-base-patch16-224",
            "description": "Google SigLIP - improved CLIP-like model with better training"
        }
    }