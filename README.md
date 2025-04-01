# World Disaster Center API

A FastAPI application that provides real-time information about global disasters, funding appeals, and affected populations using AI-powered web crawling and structured data extraction.

## Features

- **Real-time Disaster Information**: Crawls and analyzes recent disaster reports from around the world
- **Structured Data Extraction**: Uses AI to extract key information about:
  - Disaster types and locations
  - Affected populations and casualties
  - Funding appeals and humanitarian response plans
- **Multi-Region Coverage**: Implements 20 specialized search queries targeting diverse disaster types and regions
- **Priority Source Handling**: Prioritizes authoritative humanitarian sources (UNOCHA, UNHCR, WHO, etc.)
- **Comprehensive API Endpoints**:
  - `/disasters/by_country/{country_name}`: Get recent disasters for a specific country
  - `/funding/stats`: Get aggregated funding statistics from recent reports
  - `/people/stats`: Get aggregated affected population statistics
  - `/chat`: Conversational AI assistant for disaster information

## Prerequisites

1. Python 3.8 or higher
2. OpenAI API key
3. (Optional) Docker for local containerization

## Local Development

1. Clone this repository and navigate to the project directory

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

5. Run the FastAPI application:
   ```bash
   uvicorn app:app --reload
   ```

6. Access the API documentation at http://localhost:8000/docs

## Docker Deployment

```bash
# Build the Docker image
docker build -t disaster-api .

# Run the container
docker run -p 8080:8080 -e OPENAI_API_KEY=your-openai-api-key disaster-api

# Access the API at http://localhost:8080/docs
```

## Deployment to Render

The application is configured for deployment on Render using the provided `render.yaml`:

1. Push your code to a Git repository (GitHub, GitLab, etc.)

2. Create a new Web Service on Render:
   - Connect your repository
   - Select the Docker environment
   - Add your environment variables:
     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

3. Deploy! Render will automatically build and deploy your application

## API Usage

### Get Recent Disasters by Country
```http
GET /disasters/by_country/{country_name}
```

### Get Funding Statistics
```http
GET /funding/stats
```

### Get Affected People Statistics
```http
GET /people/stats
```

### Chat with Disaster Assistant
```http
POST /chat
Content-Type: application/json

{
    "message": "What are the recent disasters in Japan?"
}
```

## Data Quality and Limitations

- Data is sampled from crawled web pages and may be incomplete
- Statistics are not comprehensive global figures
- Information accuracy depends on source availability and AI interpretation
- Always verify critical information through official channels

## Troubleshooting

If you encounter issues:

1. Check the application logs for error messages
2. Verify your OpenAI API key is correctly set
3. Ensure all required dependencies are installed
4. Check for rate limiting issues from search providers
5. Verify network connectivity for web crawling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
