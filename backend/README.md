# Tattoo Search Engine Backend

A FastAPI service that uses computer vision models to search for similar tattoo images based on an uploaded image.

## Pipeline

1. **Image Upload** - User uploads a tattoo image
2. **Caption Generation** - BLIP-base generates a text description
3. **Image Search** - DuckDuckGo searches for candidate images
4. **Similarity Ranking** - CLIP computes embeddings and ranks by cosine similarity
5. **Results** - Returns top-5 most similar tattoos with scores

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /search
Upload an image to search for similar tattoos.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "caption": "black ink dragon tattoo on forearm",
  "results": [
    {
      "score": 0.87,
      "url": "https://example.com/tattoo1.jpg"
    },
    {
      "score": 0.82,
      "url": "https://example.com/tattoo2.jpg"
    }
  ]
}
```

### GET /health
Health check endpoint.

## Models Used

- **BLIP-base**: Image captioning (`Salesforce/blip-image-captioning-base`)
- **CLIP ViT-B/32**: Image similarity (`openai` pretrained)
- **DuckDuckGo**: Image search API

## Requirements

- Python 3.8+
- GPU recommended for faster inference
- Internet connection for image search and model downloads