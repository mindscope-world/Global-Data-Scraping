# Deployment Update Guide

## Changes Made to Fix API Issues

1. **Fixed `/people/stats` Endpoint**:
   - Increased search results from 10 to 20 for better coverage
   - Enhanced the instruction prompt to better extract disaster type, title, and summary
   - Improved error handling for missing metadata

2. **Performance Improvements**:
   - Increased concurrent fetches from 3 to 5
   - Fixed type annotations for better code quality
   - Updated httpx client initialization to correctly handle proxies

3. **Added Render Deployment Configuration**:
   - Created render.yaml for easy deployment
   - Added health check endpoint integration

## How to Deploy

### Option 1: Deploy to Render with Blueprint

1. Push your changes to GitHub
2. Connect your GitHub repository to Render
3. Render will automatically detect render.yaml and set up a Web Service
4. Add your OPENAI_API_KEY in the Render environment variables

### Option 2: Manual Deployment

1. Set up a new Web Service in Render
2. Connect to GitHub repository
3. Select Python runtime
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

## Testing After Deployment

1. Access the deployed API at your Render URL
2. Visit `/docs` to access the Swagger UI
3. Test the `/people/stats` endpoint
4. Verify that disaster types, titles, and summaries are now showing correctly

## Troubleshooting

If you're still not seeing the disaster types and titles:
1. Check the API logs in Render dashboard
2. Ensure your OpenAI API key has sufficient credits
3. Check that the duckduckgo-search package is installed correctly 