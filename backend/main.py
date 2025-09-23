import io
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from PIL import Image
from search_engines import SearchEngineManager
from utils import SearchCache, URLValidator
from embeddings import EmbeddingModelFactory, EmbeddingModel, get_default_model_configs
from patch_attention import PatchAttentionAnalyzer

# Load environment variables from .env file
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tattoo Search Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TattooSearchEngine:
    def __init__(self, embedding_model_type: str = "clip"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize HuggingFace InferenceClient for VLM captioning
        logger.info("Initializing HuggingFace InferenceClient...")
        self.client = InferenceClient(
            provider="novita",
            api_key=HF_TOKEN,
        )
        self.vlm_model = "zai-org/GLM-4.5V"
        logger.info(f"Using VLM model: {self.vlm_model}")

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model_type}")
        self.embedding_model = EmbeddingModelFactory.create_model(
            embedding_model_type, self.device
        )
        logger.info(f"Using embedding model: {self.embedding_model.get_model_name()}")

        # Initialize new search system
        logger.info("Initializing search system...")
        self.search_manager = SearchEngineManager(max_workers=5)
        self.url_validator = URLValidator(max_workers=10, timeout=10)
        self.search_cache = SearchCache(default_ttl=3600, max_size=1000)

        # Setup enhanced web scraping
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        ]

        logger.info("Search system initialized successfully!")

    def generate_caption(self, image: Image.Image) -> str:
        """Generate tattoo caption using HuggingFace InferenceClient."""
        try:
            # Convert PIL image to base64 URL format
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG", quality=95)
            img_buffer.seek(0)

            # Create image URL for the API
            import base64

            image_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{image_b64}"

            # completion = self.client.chat.completions.create(
            #     model=self.vlm_model,
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": [
            #                 {
            #                     "type": "text",
            #                     "text": "Generate a one search engine query to find the most similar tattoos to this image. Response in json format",
            #                 },
            #                 {
            #                     "type": "image_url",
            #                     "image_url": {"url": image_url},
            #                 },
            #             ],
            #         }
            #     ],
            # )
            caption = '<|begin_of_box|>{"search_query": "hand tattoo geometric human figure abstract blackwork"}<|end_of_box|>'
            # caption = completion.choices[0].message.content
            if caption:
                match = re.search(r"\{.*\}", caption)
                if match:
                    data = json.loads(match.group())
                    search_query = data["search_query"]
                    return search_query

            else:
                logger.warning("No caption generated from VLM")
                return "tattoo artwork"

        except Exception as e:
            logger.error(f"Failed to generate caption: {e}")
            return "tattoo artwork"

    def search_images(self, query: str, max_results: int = 50) -> List[str]:
        """Search for tattoo images across multiple platforms with caching and validation."""
        # Check cache first
        cache_key = SearchCache.create_cache_key(query, max_results)
        cached_result = self.search_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            return cached_result

        logger.info(f"Searching for images: {query}")

        # Use new search system with fallback
        search_result = self.search_manager.search_with_fallback(
            query=query, max_results=max_results, min_results_threshold=10
        )

        # Extract URLs from search results
        urls = [image.url for image in search_result.images]

        if not urls:
            logger.warning(f"No URLs found for query: {query}")
            return []

        # Validate URLs
        logger.info(f"Validating {len(urls)} URLs...")
        valid_urls = self.url_validator.validate_urls(urls)

        if not valid_urls:
            logger.warning(f"No valid URLs found for query: {query}")
            return []

        # Cache the result
        self.search_cache.set(cache_key, valid_urls, ttl=3600)

        logger.info(
            f"Search completed: {len(valid_urls)} valid URLs from "
            f"{len(search_result.platforms_used)} platforms in "
            f"{search_result.search_duration:.2f}s"
        )

        return valid_urls[:max_results]

    def download_image(self, url: str, max_retries: int = 3) -> Image.Image:
        for attempt in range(max_retries):
            try:
                # Instagram-optimized headers
                headers = {
                    "User-Agent": random.choice(self.user_agents),
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "image",
                    "Sec-Fetch-Mode": "no-cors",
                    "Sec-Fetch-Site": "cross-site",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                }

                # Pinterest-specific headers
                if "pinterest" in url.lower() or "pinimg" in url.lower():
                    headers.update(
                        {
                            "Referer": "https://www.pinterest.com/",
                            "Origin": "https://www.pinterest.com",
                            "X-Requested-With": "XMLHttpRequest",
                            "Sec-Fetch-User": "?1",
                            "X-Pinterest-Source": "web",
                            "X-APP-VERSION": "web",
                        }
                    )
                else:
                    headers["Referer"] = "https://www.google.com/"

                response = requests.get(
                    url, headers=headers, timeout=15, allow_redirects=True, stream=True
                )
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get("content-type", "").lower()
                if not content_type.startswith("image/"):
                    logger.warning(f"Invalid content type for {url}: {content_type}")
                    return None

                # Check file size (avoid downloading huge files)
                content_length = response.headers.get("content-length")
                if (
                    content_length and int(content_length) > 10 * 1024 * 1024
                ):  # 10MB limit
                    logger.warning(f"Image too large: {url} ({content_length} bytes)")
                    return None

                # Download and process image
                image_data = response.content
                if len(image_data) < 1024:  # Skip very small images (likely broken)
                    logger.warning(f"Image too small: {url} ({len(image_data)} bytes)")
                    return None

                image = Image.open(io.BytesIO(image_data)).convert("RGB")

                # Validate image dimensions
                if image.size[0] < 50 or image.size[1] < 50:
                    logger.warning(f"Image dimensions too small: {url} {image.size}")
                    return None

                return image

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    logger.info(f"Retry {attempt + 1} for {url} in {wait_time:.1f}s")
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"Failed to download image {url} after {max_retries} attempts: {e}"
                    )
            except Exception as e:
                logger.warning(f"Failed to process image {url}: {e}")
                break

        return None

    def download_and_process_image(
        self, url: str, query_features: torch.Tensor, query_image: Image.Image = None,
        include_patch_attention: bool = False
    ) -> Dict[str, Any]:
        """Download and compute similarity for a single image"""
        candidate_image = self.download_image(url)
        if candidate_image is None:
            return None

        try:
            candidate_features = self.embedding_model.encode_image(candidate_image)
            similarity = self.embedding_model.compute_similarity(query_features, candidate_features)

            result = {"score": float(similarity), "url": url}

            # Add patch attention analysis if requested
            if include_patch_attention and query_image is not None:
                try:
                    analyzer = PatchAttentionAnalyzer(self.embedding_model)
                    patch_data = analyzer.compute_patch_similarities(query_image, candidate_image)
                    result["patch_attention"] = {
                        "overall_similarity": patch_data["overall_similarity"],
                        "query_grid_size": patch_data["query_grid_size"],
                        "candidate_grid_size": patch_data["candidate_grid_size"],
                        "attention_summary": analyzer.get_similarity_summary(patch_data)
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute patch attention for {url}: {e}")
                    result["patch_attention"] = None

            return result

        except Exception as e:
            logger.warning(f"Error processing candidate image {url}: {e}")
            return None

    def compute_similarity(
        self, query_image: Image.Image, candidate_urls: List[str], include_patch_attention: bool = False
    ) -> List[Dict[str, Any]]:
        # Encode query image using the selected embedding model
        query_features = self.embedding_model.encode_image(query_image)

        results = []

        # Use ThreadPoolExecutor for concurrent downloading and processing
        max_workers = min(10, len(candidate_urls))  # Limit concurrent downloads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(
                    self.download_and_process_image, url, query_features, query_image, include_patch_attention
                ): url
                for url in candidate_urls
            }

            # Process completed downloads with rate limiting
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)

                        # Stop early if we have enough good results (unless patch attention is needed)
                        target_count = 5 if include_patch_attention else 20
                        if len(results) >= target_count:
                            # Cancel remaining futures
                            for remaining_future in future_to_url:
                                remaining_future.cancel()
                            break

                except Exception as e:
                    logger.warning(f"Error in concurrent processing for {url}: {e}")

                # Small delay to be respectful to servers
                time.sleep(0.1)

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        final_count = 3 if include_patch_attention else 15
        return results[:final_count]


# Global variable to store search engine instance
search_engine = None

def get_search_engine(embedding_model: str = "clip") -> TattooSearchEngine:
    """Get or create search engine instance with specified embedding model."""
    global search_engine
    if search_engine is None or search_engine.embedding_model.get_model_name().lower() != embedding_model:
        search_engine = TattooSearchEngine(embedding_model)
    return search_engine


@app.post("/search")
async def search_tattoos(
    file: UploadFile = File(...),
    embedding_model: str = Query(default="clip", description="Embedding model to use (clip, dinov2, siglip)"),
    include_patch_attention: bool = Query(default=False, description="Include patch-level attention analysis")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Validate embedding model
        available_models = EmbeddingModelFactory.get_available_models()
        if embedding_model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedding model. Available: {available_models}"
            )

        # Get search engine with specified embedding model
        engine = get_search_engine(embedding_model)

        # Read and process the uploaded image
        image_data = await file.read()
        query_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate caption
        logger.info("Generating caption...")
        caption = engine.generate_caption(query_image)
        logger.info(f"Generated caption: {caption}")

        # Search for candidate images
        logger.info("Searching for candidate images...")
        candidate_urls = engine.search_images(caption, max_results=100)

        if not candidate_urls:
            return {"caption": caption, "results": [], "embedding_model": engine.embedding_model.get_model_name()}

        # Compute similarities and rank
        logger.info("Computing similarities...")
        results = engine.compute_similarity(query_image, candidate_urls, include_patch_attention)

        return {
            "caption": caption,
            "results": results,
            "embedding_model": engine.embedding_model.get_model_name(),
            "patch_attention_enabled": include_patch_attention
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-attention")
async def analyze_patch_attention(
    query_file: UploadFile = File(...),
    candidate_url: str = Query(..., description="URL of the candidate image to compare"),
    embedding_model: str = Query(default="clip", description="Embedding model to use (clip, dinov2, siglip)"),
    include_visualizations: bool = Query(default=True, description="Include attention visualizations")
):
    """Analyze patch-level attention between query image and a specific candidate image."""
    if not query_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Query file must be an image")

    try:
        # Validate embedding model
        available_models = EmbeddingModelFactory.get_available_models()
        if embedding_model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid embedding model. Available: {available_models}"
            )

        # Get search engine with specified embedding model
        engine = get_search_engine(embedding_model)

        # Read query image
        query_image_data = await query_file.read()
        query_image = Image.open(io.BytesIO(query_image_data)).convert("RGB")

        # Download candidate image
        candidate_image = engine.download_image(candidate_url)
        if candidate_image is None:
            raise HTTPException(status_code=400, detail="Failed to download candidate image")

        # Analyze patch attention
        analyzer = PatchAttentionAnalyzer(engine.embedding_model)
        similarity_data = analyzer.compute_patch_similarities(query_image, candidate_image)

        result = {
            "query_image_size": query_image.size,
            "candidate_image_size": candidate_image.size,
            "candidate_url": candidate_url,
            "embedding_model": engine.embedding_model.get_model_name(),
            "similarity_analysis": analyzer.get_similarity_summary(similarity_data),
            "attention_matrix_shape": similarity_data['attention_matrix'].shape,
            "top_correspondences": similarity_data['top_correspondences'][:10]  # Top 10
        }

        # Add visualizations if requested
        if include_visualizations:
            try:
                attention_heatmap = analyzer.visualize_attention_heatmap(
                    query_image, candidate_image, similarity_data
                )
                top_correspondences_viz = analyzer.visualize_top_correspondences(
                    query_image, candidate_image, similarity_data
                )

                result["visualizations"] = {
                    "attention_heatmap": f"data:image/png;base64,{attention_heatmap}",
                    "top_correspondences": f"data:image/png;base64,{top_correspondences_viz}"
                }
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
                result["visualizations"] = None

        return result

    except Exception as e:
        logger.error(f"Error analyzing patch attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_available_models():
    """Get list of available embedding models and their configurations."""
    models = EmbeddingModelFactory.get_available_models()
    configs = get_default_model_configs()
    return {
        "available_models": models,
        "model_configs": configs
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
