# Tattoo Search Engine - Migration to HuggingFace Inference

## Tasks for migrating to HuggingFace Inference APIs (Vercel-compatible)

### Pending Tasks

- [ ] Replace local CLIP embedding model with HuggingFace Inference API
- [ ] Replace local DINOv2 embedding model with HuggingFace Inference API
- [ ] Replace local SigLIP embedding model with HuggingFace Inference API
- [ ] Update EmbeddingModel classes to use API calls instead of local inference
- [ ] Remove PyTorch dependencies from requirements.txt
- [ ] Test HuggingFace Inference endpoints for all embedding models
- [ ] Update backend to be Vercel-compatible without GPU requirements
- [ ] Uncomment and test VLM captioning with HuggingFace API

### Benefits of Migration

- No GPU requirements locally
- Serverless scaling
- Can deploy to Vercel with API-only backend
- Lower infrastructure costs
- Faster cold starts

### Notes

- Currently using HuggingFace InferenceClient with Novita provider for VLM
- Local models: CLIP, DINOv2, SigLIP for embeddings
- Backend is FastAPI with heavy ML dependencies
- Frontend is Next.js (already Vercel-compatible)