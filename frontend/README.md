# Tattoo Search Engine Frontend

A Next.js frontend for the tattoo image search engine that integrates with your existing portfolio site.

## Features

- **Image Upload**: Drag & drop or click to upload tattoo images
- **Visual Search**: Find similar tattoos using AI-powered image analysis
- **Results Display**: View similar tattoos with similarity scores
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: User-friendly error messages and loading states

## Setup

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Update the backend URL in .env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Run development server
npm run dev
```

The page will be available at `http://localhost:3000/projects/tattoo-search`

## Integration with Portfolio

To integrate this into your existing Next.js portfolio:

1. **Copy the page**:
   ```bash
   cp pages/projects/tattoo-search.tsx /path/to/your/portfolio/pages/projects/
   ```

2. **Copy the components**:
   ```bash
   cp -r components/* /path/to/your/portfolio/components/
   ```

3. **Add the styles** to your existing `globals.css`:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```

4. **Install dependencies** in your portfolio:
   ```bash
   npm install tailwindcss autoprefixer postcss
   ```

5. **Configure Tailwind** by copying `tailwind.config.js` and `postcss.config.js`

6. **Update Next.js config** to allow remote images by copying the `next.config.js` settings

## Environment Variables

- `NEXT_PUBLIC_BACKEND_URL`: URL of your FastAPI backend (default: `http://localhost:8000`)

## Page Structure

- **URL**: `/projects/tattoo-search`
- **Components**:
  - `ImageUpload`: Handles image upload with drag & drop
  - `SearchResults`: Displays search results with similarity scores

## API Integration

The frontend communicates with the FastAPI backend through:

- **Endpoint**: `POST /search`
- **Payload**: FormData with image file
- **Response**: JSON with caption and results array

## Styling

Built with:
- **Tailwind CSS**: Utility-first CSS framework
- **Responsive Design**: Mobile-first approach
- **Custom Colors**: Primary blue theme that can be customized

## Development

```bash
# Development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint
```

## Deployment

This frontend is designed to be deployed as part of your existing Next.js portfolio on Vercel. Make sure to:

1. Set the `NEXT_PUBLIC_BACKEND_URL` environment variable in Vercel
2. Ensure your backend is deployed and accessible
3. Update CORS settings in your backend to allow requests from your domain