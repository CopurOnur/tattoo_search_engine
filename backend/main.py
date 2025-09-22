import io
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import open_clip
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from PIL import Image
from search_engines import SearchEngineManager
from utils import SearchCache, URLValidator

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
    def __init__(self):
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

        # Load CLIP for similarity
        logger.info("Loading CLIP model...")
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        )
        self.clip_model.to(self.device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

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
        self, url: str, query_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Download and compute similarity for a single image"""
        candidate_image = self.download_image(url)
        if candidate_image is None:
            return None

        try:
            candidate_input = (
                self.clip_preprocess(candidate_image).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                candidate_features = self.clip_model.encode_image(candidate_input)
                candidate_features = F.normalize(candidate_features, p=2, dim=1)

                similarity = torch.mm(query_features, candidate_features.T).item()

            return {"score": float(similarity), "url": url}

        except Exception as e:
            logger.warning(f"Error processing candidate image {url}: {e}")
            return None

    def compute_clip_similarity(
        self, query_image: Image.Image, candidate_urls: List[str]
    ) -> List[Dict[str, Any]]:
        # Encode query image
        query_input = self.clip_preprocess(query_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_features = self.clip_model.encode_image(query_input)
            query_features = F.normalize(query_features, p=2, dim=1)

        results = []

        # Use ThreadPoolExecutor for concurrent downloading and processing
        max_workers = min(10, len(candidate_urls))  # Limit concurrent downloads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_url = {
                executor.submit(
                    self.download_and_process_image, url, query_features
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

                        # Stop early if we have enough good results
                        if (
                            len(results) >= 20
                        ):  # Process more than needed for better ranking
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

        return results[:15]  # Return top 5


# Initialize the search engine
search_engine = TattooSearchEngine()


@app.post("/search")
async def search_tattoos(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process the uploaded image
        image_data = await file.read()
        query_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Generate caption
        logger.info("Generating caption...")
        caption = search_engine.generate_caption(query_image)
        logger.info(f"Generated caption: {caption}")

        # Search for candidate images
        logger.info("Searching for candidate images...")
        candidate_urls = search_engine.search_images(caption, max_results=100)

        if not candidate_urls:
            return {"caption": caption, "results": []}

        # Compute similarities and rank
        logger.info("Computing similarities...")
        results = search_engine.compute_clip_similarity(query_image, candidate_urls)

        return {"caption": caption, "results": results}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
