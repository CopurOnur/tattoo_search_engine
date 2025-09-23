# 🎨 AI-Powered Tattoo Search Engine

A sophisticated full-stack application for finding similar tattoo designs using advanced AI-powered image analysis, multi-platform search capabilities, and interactive patch-level attention visualization.

## 🔍 Overview

This project combines state-of-the-art computer vision, web scraping, and interactive visualizations to help users find tattoo inspiration. Upload an image and receive similar tattoo designs from various platforms, with deep insights into which parts of images are most similar through our revolutionary patch attention analysis system.

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **FastAPI** server with advanced CORS support
- **Multiple embedding models**: CLIP (ViT-B-32), DINOv2, SigLIP for diverse similarity computation
- **Patch-level attention analysis** with interactive visualization generation
- **HuggingFace InferenceClient** for image captioning using GLM-4.5V
- **Multi-platform search engines** (Instagram, Pinterest, Reddit, Google Images)
- **Advanced URL validation and intelligent caching system**
- **Concurrent image processing** with ThreadPoolExecutor and rate limiting
- **Mathematical visualization** with matplotlib and seaborn for attention heatmaps

### Frontend (Next.js/React/TypeScript)
- **Next.js 14** with full TypeScript support
- **Interactive patch visualization** with SVG-based 16x16 grids
- **Real-time patch selection** and similarity analysis
- **Responsive design** with Tailwind CSS
- **Advanced error handling** with CORS fallback mechanisms
- **Progressive enhancement** from basic search to detailed analysis

## 🚀 Features

### 🔬 **Advanced AI Analysis**
- **Multi-Model Support**: Choose between CLIP, DINOv2, and SigLIP embedding models
- **Patch-Level Attention**: Revolutionary 16x16 grid analysis showing which image parts are most similar
- **Interactive Visualization**: Click patches on query images to see top 10 matching patches in results
- **Real-time Analysis**: Dynamic patch selection with immediate visual feedback

### 🌐 **Intelligent Search**
- **Multi-Platform Discovery**: Searches across Instagram, Pinterest, Reddit, and Google Images
- **AI-Generated Queries**: Uses GLM-4.5V for intelligent search term generation
- **Smart Ranking**: Advanced similarity scoring with multiple embedding models
- **Cached Results**: Intelligent caching system for improved performance

### 🎯 **User Experience**
- **Progressive Interface**: Basic search → Enable patch attention → Detailed analysis
- **Visual Feedback**: Color-coded similarity rankings (Green/Yellow/Orange)
- **Interactive Grids**: Hover and click effects for intuitive exploration
- **Export Capabilities**: Download attention heatmaps and correspondence visualizations
- **Responsive Design**: Works seamlessly on desktop and mobile devices

### ⚡ **Performance & Reliability**
- **Concurrent Processing**: Parallel image downloading and analysis
- **CORS Handling**: Robust image loading with automatic fallback strategies
- **Rate Limiting**: Respectful server interaction with intelligent delays
- **Error Recovery**: Comprehensive error handling and retry mechanisms

## 📦 Installation

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **Node.js 18+** with npm
- **CUDA-compatible GPU** (optional, for faster processing)
- **HuggingFace Account** for API access

### Backend Setup
```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your HuggingFace token to .env

# Start the server
python main.py
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### Quick Start
```bash
# Terminal 1 - Backend
cd backend && python main.py

# Terminal 2 - Frontend
cd frontend && npm run dev

# Open http://localhost:3000 in your browser
```

## 🔧 Configuration

Create a `.env` file in the backend directory:
```
HF_TOKEN=your_huggingface_token_here
```

## 📚 API Endpoints

### Core Search
- `POST /search` - Upload an image and get similar tattoo results
  - Query params: `embedding_model` (clip/dinov2/siglip), `include_patch_attention` (boolean)
  - Returns: Search results with optional patch attention data

### Advanced Analysis
- `POST /analyze-attention` - Detailed patch-level attention analysis
  - Query params: `candidate_url`, `embedding_model`, `include_visualizations` (boolean)
  - Returns: Comprehensive attention analysis with visualizations

### Utility
- `GET /models` - Get available embedding models and configurations
- `GET /health` - Health check endpoint

### Example Usage
```bash
# Basic search
curl -X POST "http://localhost:8000/search?embedding_model=clip" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tattoo.jpg"

# Search with patch attention
curl -X POST "http://localhost:8000/search?embedding_model=clip&include_patch_attention=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tattoo.jpg"

# Detailed analysis
curl -X POST "http://localhost:8000/analyze-attention" \
  -H "Content-Type: multipart/form-data" \
  -F "query_file=@tattoo.jpg" \
  -F "candidate_url=https://example.com/tattoo2.jpg" \
  -F "embedding_model=clip"
```

## 🛠️ Technology Stack

### **Backend Technologies**
| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | High-performance async API with auto-documentation |
| **AI Models** | PyTorch + OpenCLIP + DINOv2 + SigLIP | Multiple embedding models for similarity computation |
| **Image Processing** | Pillow (PIL) + NumPy | Image manipulation and preprocessing |
| **Visualization** | Matplotlib + Seaborn | Attention heatmap generation |
| **ML Hub** | HuggingFace Transformers | Model loading and inference |
| **Search Engines** | Custom multi-platform scrapers | Instagram, Pinterest, Reddit, Google Images |
| **Caching** | In-memory + TTL | Smart result caching system |
| **Concurrency** | ThreadPoolExecutor | Parallel image processing |

### **Frontend Technologies**
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | Next.js 14 + React 18 | Modern full-stack React framework |
| **Language** | TypeScript | Type-safe JavaScript development |
| **Styling** | Tailwind CSS | Utility-first CSS framework |
| **Visualization** | SVG + Canvas | Interactive patch grid overlays |
| **State Management** | React Hooks | Component state and effects |
| **HTTP Client** | Fetch API | API communication |
| **Image Handling** | Next.js Image + Custom components | Optimized image loading with CORS handling |

## ✅ Project Status

### **Completed Features**
- ✅ **Multi-Model Backend API** - Full FastAPI implementation with CLIP, DINOv2, SigLIP
- ✅ **Patch Attention Analysis** - Revolutionary 16x16 grid similarity analysis
- ✅ **Interactive Frontend** - Complete Next.js/React interface with TypeScript
- ✅ **Multi-Platform Search** - Instagram, Pinterest, Reddit, Google Images integration
- ✅ **Advanced Visualizations** - Attention heatmaps, correspondence analysis, exports
- ✅ **Progressive UI** - Basic search → Patch attention → Detailed analysis workflow
- ✅ **CORS Handling** - Robust image loading with multiple fallback strategies
- ✅ **Performance Optimization** - Caching, concurrent processing, rate limiting
- ✅ **Error Recovery** - Comprehensive error handling and retry mechanisms

### **Key Achievements**
- 🎯 **World-class UX** - Intuitive patch selection with real-time feedback
- 🔬 **Research-grade Analysis** - Attention matrix computation and visualization
- ⚡ **Production Ready** - Scalable architecture with proper error handling
- 📱 **Mobile Responsive** - Works seamlessly across all device sizes

## 🔮 Future Enhancements

- **Advanced Models**: Integration with newer vision transformers (EVA, SAM)
- **Video Analysis**: Extend patch attention to video frames
- **3D Visualization**: WebGL-based 3D attention landscapes
- **Social Features**: User accounts, favorites, sharing
- **API Rate Limiting**: Redis-based rate limiting for production
- **Deployment**: Docker containerization and cloud deployment guides

## 🤝 Contributing

This project represents cutting-edge computer vision research applied to real-world tattoo discovery. Contributions welcome in:

- **Model Integration**: Adding new embedding models
- **Visualization Improvements**: Enhanced interactive components
- **Performance Optimization**: Caching strategies and speed improvements
- **Search Engine Expansion**: New platform integrations
- **Mobile App**: React Native implementation

## 📄 License

MIT License - Feel free to use this project for research, education, or commercial purposes.

---

### 🌟 **Built with ❤️ for the tattoo community and computer vision researchers**
