# Tattoo Search Engine

A full-stack application for finding similar tattoo designs using AI-powered image analysis and multi-platform search capabilities.

## 🔍 Overview

This project combines computer vision and web scraping to help users find tattoo inspiration by uploading an image and receiving similar tattoo designs from various platforms. The system uses CLIP (Contrastive Language-Image Pre-training) for image similarity matching and advanced search engines to gather tattoo images from multiple sources.

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **FastAPI** server with CORS support
- **CLIP model** (ViT-B-32) for image similarity computation
- **HuggingFace InferenceClient** for image captioning using GLM-4.5V
- **Multi-platform search engines** (Instagram, Pinterest, Reddit)
- **URL validation and caching system**
- **Concurrent image processing** with ThreadPoolExecutor

### Frontend (Next.js/React)
- **Next.js 14** with TypeScript
- **Tailwind CSS** for styling
- **React** components for UI
- **Image upload and results display**

## 🚀 Features

- **AI-Powered Image Analysis**: Upload a tattoo image and get AI-generated search queries
- **Multi-Platform Search**: Searches across Instagram, Pinterest, and Reddit
- **Similarity Ranking**: Uses CLIP model to rank results by visual similarity
- **Smart Caching**: Caches search results and validated URLs for performance
- **Concurrent Processing**: Parallel image downloading and processing
- **Robust Error Handling**: Retry mechanisms and validation for reliable results

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (optional, for faster processing)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your HuggingFace token to .env
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🔧 Configuration

Create a `.env` file in the backend directory:
```
HF_TOKEN=your_huggingface_token_here
```

## 📚 API Endpoints

- `POST /search` - Upload an image and get similar tattoo results
- `GET /health` - Health check endpoint

## 🛠️ Technology Stack

**Backend:**
- FastAPI
- PyTorch + OpenCLIP
- HuggingFace Hub
- Pillow (PIL)
- Requests
- DuckDuckGo Search

**Frontend:**
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS

## 🔄 Current Status

- ✅ Backend API with image search functionality
- ✅ Multi-platform search engine integration
- ✅ CLIP-based similarity ranking
- ✅ Frontend UI framework setup
- 🔄 Frontend-backend integration (in progress)
- 🔄 UI components for search and results display

## 📝 Development

The project is structured as a monorepo with separate backend and frontend directories. The backend provides a REST API that the frontend consumes for tattoo search functionality.
