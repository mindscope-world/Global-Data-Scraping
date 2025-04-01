# Configuration - Update these values as needed
$PROJECT_ID = "wdcdatawarehouse"  # Your GCP project ID
$IMAGE_NAME = "disaster-api"
$REGION = "us-central1"
$SERVICE_NAME = "disaster-api-service"
$REPOSITORY = "disaster-images" # Artifact Registry repository

# Create Artifact Repository if it doesn't exist (one-time setup)
Write-Host "Creating Artifact Registry repository (if it doesn't exist)..." -ForegroundColor Green
gcloud artifacts repositories create $REPOSITORY --repository-format=docker --location=$REGION --description="Docker repository for Disaster API" --async

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME .

# Test the image locally (optional)
Write-Host "Testing image locally..." -ForegroundColor Green
$job = Start-Job -ScriptBlock { docker run -p 8080:8080 --rm $using:REGION-docker.pkg.dev/$using:PROJECT_ID/$using:REPOSITORY/$using:IMAGE_NAME }
Start-Sleep -Seconds 5
try {
    Invoke-WebRequest -Uri "http://localhost:8080/docs" -Method GET
} catch {
    Write-Host "Error accessing the local server: $_" -ForegroundColor Red
}
Stop-Job -Job $job
Remove-Job -Job $job

# Configure Docker to use gcloud as credential helper for Artifact Registry
Write-Host "Configuring Docker authentication for Artifact Registry..." -ForegroundColor Green
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# Push the image to Artifact Registry
Write-Host "Pushing image to Artifact Registry..." -ForegroundColor Green
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME

# Get OpenAI API Key from .env file
$OPENAI_API_KEY = (Get-Content .env | Where-Object { $_ -match "OPENAI_API_KEY=" }) -replace "OPENAI_API_KEY=",""

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..." -ForegroundColor Green
gcloud run deploy $SERVICE_NAME `
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --memory 1Gi `
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY"

Write-Host "Deployment complete! Your service is available at:" -ForegroundColor Green
gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)' 