#!/bin/bash
set -e

# Configuration - Replace these values with your own
PROJECT_ID="your-gcp-project-id"
IMAGE_NAME="disaster-api"
REGION="us-central1"
SERVICE_NAME="disaster-api-service"

# Build the Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Test the image locally (optional)
echo "Testing image locally..."
docker run -p 8080:8080 --rm gcr.io/$PROJECT_ID/$IMAGE_NAME &
PID=$!
sleep 5
curl http://localhost:8080/docs
kill $PID

# Push the image to Google Container Registry
echo "Pushing image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars "OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)"

echo "Deployment complete! Your service is available at:"
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)' 