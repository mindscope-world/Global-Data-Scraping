# Disaster Information Collection Update

## Changes Made to Improve Global Disaster Data Collection

### 1. Multi-Region Search Strategy
- **Implemented 20 specialized search queries** targeting diverse disaster types and regions
- Each query addresses specific disaster types: earthquakes, floods, conflicts, wildfires, etc.
- Geographically diverse targeting ensures coverage across all continents

### 2. Enhanced Content Processing
- Increased timeout for disaster-related content fetching (30 seconds vs 20 seconds)
- Added priority domain sorting to favor reliable disaster information sources
- Added fallback disaster type detection based on search query content
- Improved extraction prompts to focus specifically on disaster information

### 3. Quality Improvements
- Added date recency indicators to search queries
- Prioritized authoritative humanitarian sources like UNOCHA, UNHCR, WHO, etc.
- Improved handling of locations with better specificity in extraction

## Expected Outcomes

This update should significantly increase:
1. **Number of disaster locations** reported in the API response
2. **Diversity of disaster types** captured from around the world
3. **Quality of metadata** including disaster titles and descriptions
4. **Geographical coverage** across continents and disaster-prone regions

## Deployment Instructions

1. Push the updated code to your GitHub repository
2. Ensure your Render deployment has the OpenAI API key set
3. The service should automatically rebuild with the new changes
4. Monitor logs to make sure all queries are running successfully

## Testing the Changes

After deployment, test `/people/stats` endpoint and verify:
- Response includes multiple locations with disaster information
- Each location has disaster_type and title properly populated
- Response includes a diverse range of disaster types
- Data is sourced from at least 10-15 different websites

## Troubleshooting

If you still see limited results:
1. Check the OpenAI API key to ensure it has not expired or hit rate limits
2. Verify the duckduckgo-search package is installed and working
3. Look for any rate limiting issues in the logs from search providers
4. Ensure all search queries are being processed (check for timeout errors) 