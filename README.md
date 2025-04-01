# Global Data Scraping - World Disaster Center API

A FastAPI application that provides information about global disasters, funding appeals, and affected populations.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) installed locally
2. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
3. A Google Cloud Platform project with billing enabled
4. OpenAI API key
5. Permissions to create resources in your Google Cloud project

## Deployment to Google Cloud Run

### Important: Enable Billing

Before deploying, make sure billing is enabled for your project:
- Visit https://console.cloud.google.com/billing/projects
- Select your project and enable billing if not already enabled

### Manual Deployment

1. Clone this repository and navigate to the project directory

2. Update the configuration variables in `deploy.ps1` (Windows) or `deploy.sh` (Linux/MacOS):
   ```
   PROJECT_ID="your-gcp-project-id"
   IMAGE_NAME="disaster-api"
   REGION="us-central1"
   SERVICE_NAME="disaster-api-service"
   REPOSITORY="disaster-images"
   ```

3. Make sure your OpenAI API key is in the `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

4. Run the deployment script:
   - Windows: `.\deploy.ps1`
   - Linux/MacOS: `bash deploy.sh`

### GitHub Actions Deployment

To set up automatic deployments with GitHub Actions:

1. Add the following secrets to your GitHub repository:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: Service account key JSON (base64 encoded)
   - `OPENAI_API_KEY`: Your OpenAI API key

2. Create a service account in GCP with the following roles:
   - Cloud Run Admin
   - Storage Admin
   - Artifact Registry Administrator
   - Service Account User

3. Create and download a JSON key for the service account

4. Base64 encode the JSON key file:
   - Windows: `[System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes((Get-Content -Raw -Path key.json)))`
   - Linux/macOS: `cat key.json | base64`

5. Add the encoded string as the `GCP_SA_KEY` secret in GitHub

6. Push your code to the `main` branch, and GitHub Actions will handle the deployment

## Troubleshooting

If you encounter permission issues:
1. Ensure you're logged in to gcloud with sufficient permissions
2. Verify billing is enabled for your project
3. Check IAM permissions for your user account or service account

If the container fails to start on Cloud Run:
1. Check logs in Google Cloud Console
2. Verify the OPENAI_API_KEY environment variable is correctly set

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI application:
   ```
   uvicorn app:app --reload
   ```

3. Access the API documentation at http://localhost:8000/docs

## Building and Running with Docker Locally

```bash
# Build the Docker image
docker build -t disaster-api .

# Run the container
docker run -p 8080:8080 -e OPENAI_API_KEY=your-openai-api-key disaster-api

# Access the API at http://localhost:8080/docs
```
