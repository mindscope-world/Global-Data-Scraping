import asyncio
import json
import logging
import os
from collections import defaultdict
from typing import List, Optional, Dict, Any

import httpx # Still needed for OpenAI client, potentially health checks
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError

# +++ Import crawl4ai +++
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.errors import CrawlingError, ExtractionError
from crawl4ai.cache_modes import CacheMode # Optional: for caching control

# --- Configuration & Initialization ---

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="World Disaster Center Assistant API",
    description="AI-powered service using web crawling and LLM extraction for recent disaster information, funding appeals, and affected population statistics. WARNING: Data is sampled, potentially incomplete, and accuracy depends on source availability and LLM interpretation.",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AsyncOpenAI client
# Ensure OPENAI_API_KEY is set in your .env file or environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    # You might want to exit or raise a configuration error here
    # For now, we'll let it potentially fail later if used.

# Configure httpx client for OpenAI, handling proxy if needed
openai_http_client = None
https_proxy = os.getenv("HTTPS_PROXY")
if https_proxy:
    logger.info(f"Using HTTPS proxy for OpenAI: {https_proxy}")
    openai_http_client = httpx.AsyncClient(proxies=https_proxy, timeout=httpx.Timeout(45.0)) # Increased timeout for LLM calls
else:
    openai_http_client = httpx.AsyncClient(timeout=httpx.Timeout(45.0))

# Use try-except block for client initialization as API key might be missing/invalid
try:
    client = AsyncOpenAI(
        api_key=openai_api_key,
        http_client=openai_http_client
    )
    # Optional: Test connection or list models if needed
    # asyncio.run(client.models.list())
    logger.info("AsyncOpenAI client initialized successfully.")
except OpenAIError as e:
    logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
    client = None # Set client to None to indicate failure
except Exception as e:
    logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}", exc_info=True)
    client = None

# --- Pydantic Models ---

# --- Models for Structured Extraction ---

class BaseExtractionModel(BaseModel):
    source_url: Optional[HttpUrl] = Field(None, description="The source URL where the data was found")

    @field_validator('*', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "":
            return None
        return v

    @field_validator('*', mode='before')
    @classmethod
    def zero_to_none_or_int(cls, v, info):
        # Convert numeric fields from potential strings, handle "N/A" etc.
        numeric_fields = {'funding_requested', 'funding_received', 'affected_count', 'casualty_count', 'displaced_count'}
        if info.field_name in numeric_fields:
            if isinstance(v, str):
                try:
                    # Remove commas, currency symbols, etc.
                    cleaned_v = ''.join(filter(str.isdigit, v.split('.')[0])) # Basic cleaning, might need refinement
                    if cleaned_v:
                        return int(cleaned_v)
                    else: return 0 # Treat non-numeric strings or empty strings as 0
                except ValueError:
                    return 0 # Cannot convert, treat as 0
            elif isinstance(v, (int, float)):
                return int(v) # Ensure integer for counts
            return 0 # Default to 0 if type is unexpected or value is None/falsy
        return v


class DisasterInfo(BaseExtractionModel):
    disaster_type: Optional[str] = Field(None, description="Type of disaster (e.g., earthquake, flood)")
    location: Optional[str] = Field(None, description="Specific city, region, or area affected")
    date_reported: Optional[str] = Field(None, description="Approximate date or time frame mentioned")
    severity: Optional[str] = Field(None, description="Indication of severity (e.g., magnitude, category, impact description)")
    summary: Optional[str] = Field(None, description="Brief summary of the situation")

class FundingInfo(BaseExtractionModel):
    funding_requested: Optional[float] = Field(0.0, description="Amount of funding requested (USD or unspecified currency)")
    funding_received: Optional[float] = Field(0.0, description="Amount of funding received/pledged (USD or unspecified currency)")
    funding_gap: Optional[float] = Field(0.0, description="Calculated funding gap (requested - received)")
    percentage_funded: Optional[float] = Field(0.0, description="Funding received as a percentage of requested")
    location: Optional[str] = Field("Unknown Location", description="Country or region the funding pertains to")
    organizations_involved: Optional[List[str]] = Field(default_factory=list, description="Organizations mentioned in the appeal/report")
    appeal_name: Optional[str] = Field(None, description="Name of the funding appeal or plan, if mentioned")
    report_date: Optional[str] = Field(None, description="Date of the funding report or appeal")

class PeopleStatsInfo(BaseExtractionModel):
    location: Optional[str] = Field("Unknown Location", description="Country or region the stats pertain to")
    affected_count: Optional[int] = Field(0, description="Number of people affected")
    casualty_count: Optional[int] = Field(0, description="Number of deaths or casualties reported")
    injured_count: Optional[int] = Field(0, description="Number of people injured")
    displaced_count: Optional[int] = Field(0, description="Number of people displaced (IDPs, refugees)")
    report_date: Optional[str] = Field(None, description="Date the statistics were reported")

# --- Models for API Responses ---

class BaseStatsResponse(BaseModel):
    warning: str = Field("WARNING: Data is sampled from crawled web pages and may be incomplete or inaccurate. Not a comprehensive global statistic. Verify critical information.", description="Standard warning about data limitations")
    source_urls_processed: List[HttpUrl] = Field(default_factory=list, description="List of URLs successfully crawled and processed")
    errors: Dict[str, str] = Field(default_factory=dict, description="URLs that failed to crawl or extract data")

class LocationFundingStats(BaseModel):
    requested: float = 0.0
    received: float = 0.0

class FundingStatsResponse(BaseStatsResponse):
    total_requested: float = 0.0
    total_received: float = 0.0
    total_funding_gap: float = 0.0
    overall_percentage_funded: float = 0.0
    by_location: Dict[str, LocationFundingStats] = Field(default_factory=dict)
    by_organization: Dict[str, int] = Field(default_factory=dict, description="Count of times an organization was mentioned in funding contexts")

class LocationPeopleStats(BaseModel):
    affected: int = 0
    casualties: int = 0
    injured: int = 0
    displaced: int = 0

class PeopleStatsResponse(BaseStatsResponse):
    total_affected: int = 0
    total_casualties: int = 0
    total_injured: int = 0
    total_displaced: int = 0
    by_location: Dict[str, LocationPeopleStats] = Field(default_factory=dict)

class DisasterListResponse(BaseStatsResponse):
    country: str
    disasters_found: List[DisasterInfo] = Field(default_factory=list)

# Model for the existing /chat endpoint
class ChatRequest(BaseModel):
    message: str

class ChatSource(BaseModel):
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    summary: Optional[str] = None
    # detailed_analysis: Optional[str] = None # Removed, replaced by structured data where applicable

class ChatResponse(BaseModel):
    analysis: str
    sources: List[ChatSource] = []
    safety_notice: str = "Always follow official emergency instructions first. Verify information with local authorities."


# --- Helper Functions ---

def enhanced_search(query: str, max_results: int = 10) -> list[dict]:
    """Performs a web search using DuckDuckGoSearchAPIWrapper."""
    logger.info(f"Performing search for: '{query}' (max_results={max_results})")
    try:
        # Use 'd' for day, 'w' for week, 'm' for month, or None for any time
        search = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="w", max_results=max_results)
        results = search.results(query, max_results=max_results)
        logger.info(f"Search returned {len(results)} results.")
        # Ensure results have 'link' key
        valid_results = [r for r in results if isinstance(r, dict) and 'link' in r and 'title' in r and 'snippet' in r]
        if len(valid_results) != len(results):
             logger.warning(f"Filtered out {len(results) - len(valid_results)} search results missing 'link', 'title', or 'snippet'.")
        return valid_results
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
        return []

async def fetch_url_content(url: str, timeout: int = 20) -> Optional[str]:
    """Fetches text content of a URL, handling redirects and common errors."""
    logger.debug(f"Attempting to fetch content from: {url}")
    try:
        # Use a temporary client for each fetch for isolation
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=False) as fetch_client: # Added verify=False for potential SSL issues, use with caution
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} # Common user agent
            response = await fetch_client.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4xx or 5xx status codes
            content_type = response.headers.get('content-type', '').lower()

            # Basic check if it's likely HTML or text, avoid PDFs, images etc directly
            if 'text/html' in content_type or 'text/plain' in content_type or 'application/json' in content_type or not content_type:
                 # Attempt decoding, falling back to latin-1 if utf-8 fails
                try:
                    text_content = response.content.decode('utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 decoding failed for {url}, trying latin-1.")
                    text_content = response.content.decode('latin-1', errors='ignore')

                logger.info(f"Successfully fetched content from: {url}")
                # Basic cleaning (optional): remove excessive whitespace
                return ' '.join(text_content.split())
            else:
                logger.warning(f"Skipping non-text content type '{content_type}' for URL: {url}")
                return None

    except httpx.RequestError as e:
        logger.error(f"HTTP Request error fetching URL {url}: {type(e).__name__} - {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Status error {e.response.status_code} for URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching URL {url}: {type(e).__name__} - {e}", exc_info=True)
        return None

async def extract_structured_data_llm(
    content: str,
    url: HttpUrl,
    output_model: type[BaseModel],
    instruction_prompt: str,
    max_content_chars: int = 10000 # Limit content sent to LLM
) -> Optional[BaseModel]:
    """
    Uses OpenAI's function calling/JSON mode to extract structured data.
    """
    if not client:
        logger.error("OpenAI client is not available. Cannot perform extraction.")
        return None
    if not content:
        logger.warning(f"No content provided for extraction from {url}.")
        return None

    model_to_use = "gpt-4-1106-preview" # Or "gpt-3.5-turbo-1106" or newer models supporting JSON mode
    logger.info(f"Attempting structured extraction from {url} using model {model_to_use}")
    logger.debug(f"Extraction instruction: {instruction_prompt}")
    # logger.debug(f"Output schema: {output_model.model_json_schema()}") # Can be verbose

    try:
        response = await client.chat.completions.create(
            model=model_to_use,
            response_format={"type": "json_object"}, # Enforce JSON output
            messages=[
                {"role": "system", "content": f"{instruction_prompt}\n\nExtract information conforming precisely to this JSON Schema:\n{output_model.model_json_schema(by_alias=True)}. If no relevant information is found in the text, return a JSON object with null values or empty arrays for all fields, matching the schema structure."},
                {"role": "user", "content": f"Extract data from the following text (from URL: {url}):\n\n{content[:max_content_chars]}"}
            ],
            temperature=0.1, # Lower temperature for more deterministic extraction
            # max_tokens=1000 # Adjust as needed based on expected output size
        )

        response_content = response.choices[0].message.content
        logger.debug(f"Raw LLM JSON response from {url}: {response_content}")

        if not response_content:
            logger.warning(f"LLM returned empty content for {url}.")
            return None

        # Validate the LLM's JSON output against the Pydantic model
        try:
            # Use model_validate_json for Pydantic v2
            extracted_data = output_model.model_validate_json(response_content)
            # Add the source URL back to the validated object
            if hasattr(extracted_data, 'source_url'):
                 extracted_data.source_url = url
            logger.info(f"Successfully extracted and validated data for {url}")
            return extracted_data
        except ValidationError as e:
            logger.error(f"LLM output validation failed for {url}. Error: {e}. Raw response: {response_content}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"LLM output was not valid JSON for {url}. Error: {e}. Raw response: {response_content}", exc_info=True)
            return None

    except OpenAIError as e:
        logger.error(f"OpenAI API error during extraction for {url}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during LLM extraction for {url}: {type(e).__name__} - {e}", exc_info=True)
        return None


async def search_crawl_extract(
    search_query: str,
    output_model: type[BaseModel],
    instruction_prompt: str,
    max_search_results: int = 5, # Limit search results to manage cost/time
    max_concurrent_fetches: int = 3 # Limit concurrent fetches
) -> tuple[List[BaseModel], Dict[str, str]]:
    """
    Core workflow: Search -> Fetch URL Content -> Extract Structured Data.
    Returns a list of successfully extracted data objects and a dict of errors.
    """
    search_results = enhanced_search(search_query, max_results=max_search_results)
    extracted_data_list = []
    errors = {}
    urls_to_process = [r['link'] for r in search_results if r.get('link')]

    if not urls_to_process:
        logger.warning(f"No valid URLs found in search results for query: '{search_query}'")
        return [], errors

    semaphore = asyncio.Semaphore(max_concurrent_fetches)

    async def process_url(url_str: str):
        async with semaphore:
            try:
                # Validate URL before processing
                url = HttpUrl(url_str) # Validate/parse using Pydantic
            except ValidationError:
                logger.warning(f"Invalid URL format skipped: {url_str}")
                errors[url_str] = "Invalid URL format"
                return None

            content = await fetch_url_content(str(url)) # Fetch needs string
            if content:
                extracted = await extract_structured_data_llm(content, url, output_model, instruction_prompt)
                if extracted:
                    return extracted
                else:
                    errors[str(url)] = "Failed to extract structured data (LLM error or no data found)"
                    return None
            else:
                errors[str(url)] = "Failed to fetch content or content type unsuitable"
                return None

    # Process URLs concurrently
    tasks = [process_url(url) for url in urls_to_process]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results and handle potential exceptions from gather
    for i, result in enumerate(results):
        url = urls_to_process[i]
        if isinstance(result, Exception):
            logger.error(f"Exception during processing task for {url}: {result}", exc_info=result)
            if url not in errors: # Avoid overwriting specific errors
                 errors[url] = f"Unhandled exception during processing: {type(result).__name__}"
        elif result: # If result is not None and not an Exception
            extracted_data_list.append(result)
        # No need for else, errors are logged and added within process_url

    logger.info(f"Extraction summary for '{search_query}': {len(extracted_data_list)} successes, {len(errors)} failures out of {len(urls_to_process)} URLs.")

    return extracted_data_list, errors


# --- API Endpoints ---

@app.get("/", tags=["Utility"])
async def read_root():
    """Root endpoint providing basic API information and endpoint overview."""
    return {
        "message": "Welcome to the World Disaster Center Assistant API (v1.1 - Structured Extraction).",
        "version": app.version,
        "description": app.description,
        "docs_url": app.docs_url,
        "redoc_url": app.redoc_url,
        "health_url": "/health",
        "endpoints": {
            "/disasters/by_country/{country_name}": "Get recent disaster info for a specific country.",
            "/funding/stats": "Get aggregated sample funding statistics from recent reports.",
            "/people/stats": "Get aggregated sample affected people statistics from recent reports.",
            "/chat": "(POST) General conversational AI assistant for disaster info (less structured)."
        },
        "warning": "This API uses dynamic web crawling and LLM extraction. Data accuracy, completeness, and availability vary significantly. Statistics are sampled, not comprehensive global figures. ALWAYS verify critical information through official channels."
    }

@app.get("/health", tags=["Utility"])
async def health_check():
    """Check the health of the API and its dependencies (basic)."""
    # Basic check if OpenAI client was initialized
    openai_status = "healthy" if client else "unhealthy (initialization failed)"
    # You could add a simple OpenAI API call here if needed, e.g., list models
    # but be mindful of cost and rate limits.
    return {"status": "healthy", "openai_client_status": openai_status}


@app.get(
    "/disasters/by_country/{country_name}",
    response_model=DisasterListResponse,
    tags=["Disasters"],
    summary="Get Recent Disasters by Country",
)
async def get_disasters_by_country(
    country_name: str = Path(..., description="Name of the country to search for disasters.", min_length=3, example="Philippines")
):
    """
    Searches for recent disaster reports or emergency situation updates
    in the specified country. Attempts to crawl top search results and
    extract structured information about each reported disaster using an LLM.

    Returns a list of structured disaster information objects for successfully
    processed sources. Includes standard warnings about data limitations.
    """
    search_query = f"recent disaster OR emergency situation report in {country_name}"
    instruction_prompt = "You are an expert data extractor. Analyze the provided text from a webpage about potential disasters. Extract key details for any *specific* disaster event mentioned (like earthquake, flood, storm). Focus on the disaster type, specific location (city/province/area), reported date/timeframe, and severity indicators. If multiple distinct disasters are mentioned, create a separate entry for each if possible within the JSON structure constraints."

    disaster_data_list, errors = await search_crawl_extract(
        search_query=search_query,
        output_model=DisasterInfo,
        instruction_prompt=instruction_prompt,
        max_search_results=7 # Fetch a few more for country specifics
    )

    response = DisasterListResponse(
        country=country_name,
        disasters_found=disaster_data_list,
        source_urls_processed=[d.source_url for d in disaster_data_list if d.source_url],
        errors=errors,
    )
    return response


@app.get(
    "/funding/stats",
    response_model=FundingStatsResponse,
    tags=["Funding"],
    summary="Get Sampled Funding Statistics",
)
async def get_funding_statistics():
    """
    Attempts to crawl & extract funding data (requested/received amounts, locations,
    organizations) from web pages found via a general search for recent major
    disaster funding appeals or humanitarian response plan updates.

    Aggregates data ONLY from successfully crawled and processed pages.
    Provides totals, location breakdown, and organization mentions.

    **WARNING:** This provides sampled data based on recent web searches and LLM
    extraction, not a comprehensive global statistic. Amounts are treated as USD
    unless specified otherwise by the LLM (currency conversion is not implemented).
    """
    search_query = "recent major disaster funding appeal OR humanitarian response plan funding update OR OCHA situation report funding OR CERF allocation"
    instruction_prompt = "You are an expert financial data extractor. Analyze the provided text from a webpage about disaster funding appeals or reports. Extract details like the total funding requested, funding received/pledged, the specific location (country/region) the funding is for, and any organizations involved (requesting, receiving, or implementing). Provide numeric values for funding amounts (assume USD if not specified, omit currency symbols). If specific amounts aren't mentioned, use 0."

    funding_data_list, errors = await search_crawl_extract(
        search_query=search_query,
        output_model=FundingInfo,
        instruction_prompt=instruction_prompt,
        max_search_results=10 # More results for broader stats
    )

    # --- Aggregation Logic ---
    stats = FundingStatsResponse(
        source_urls_processed=[f.source_url for f in funding_data_list if f.source_url],
        errors=errors,
    )
    location_stats = defaultdict(lambda: defaultdict(float)) # Nested defaultdict for easier summing
    org_counts = defaultdict(int)

    for item in funding_data_list:
        # Use getattr with default 0.0 to safely access potentially None fields
        req = getattr(item, 'funding_requested', 0.0) or 0.0
        rec = getattr(item, 'funding_received', 0.0) or 0.0

        stats.total_requested += req
        stats.total_received += rec

        loc = getattr(item, 'location', "Unknown Location") or "Unknown Location"
        location_stats[loc]["requested"] += req
        location_stats[loc]["received"] += rec

        orgs = getattr(item, 'organizations_involved', []) or []
        if orgs:
            for org in orgs:
                if org and isinstance(org, str): # Ensure org is a non-empty string
                     org_counts[org.strip()] += 1 # Count valid org names, strip whitespace

    stats.total_funding_gap = max(0.0, stats.total_requested - stats.total_received) # Ensure gap isn't negative
    if stats.total_requested > 0:
         stats.overall_percentage_funded = round((stats.total_received / stats.total_requested) * 100, 2)
    else:
         stats.overall_percentage_funded = 0.0 if stats.total_received == 0 else 100.0 # Handle division by zero / case where only received funds reported

    # Convert defaultdicts back to regular dicts for the response model
    stats.by_location = {k: LocationFundingStats(requested=v["requested"], received=v["received"]) for k, v in location_stats.items()}
    stats.by_organization = dict(org_counts)

    return stats


@app.get(
    "/people/stats",
    response_model=PeopleStatsResponse,
    tags=["People Affected"],
    summary="Get Sampled Affected People Statistics",
)
async def get_people_statistics():
    """
    Attempts to crawl & extract affected people data (affected counts,
    casualties, displacement figures) from web pages found via a general search
    for recent disaster situation reports.

    Aggregates data ONLY from successfully crawled and processed pages.
    Provides totals and a location breakdown.

    **WARNING:** This provides sampled data based on recent web searches and LLM
    extraction, not a comprehensive global statistic. Figures can vary wildly
    between reports and may be estimates.
    """
    search_query = "recent disaster situation report casualty figures OR affected population numbers OR displacement statistics OR OCHA situation report population"
    instruction_prompt = "You are an expert data extractor focused on humanitarian impact. Analyze the provided text from a webpage, typically a disaster situation report. Extract statistics about affected populations: specifically, the number of people affected, number of casualties (deaths), number of injured, and number of displaced people (IDPs or refugees). Also, identify the primary location (country/region) these figures refer to. Provide integer numbers for counts. If a figure is not mentioned, use 0."

    people_data_list, errors = await search_crawl_extract(
        search_query=search_query,
        output_model=PeopleStatsInfo,
        instruction_prompt=instruction_prompt,
        max_search_results=10 # More results for broader stats
    )

    # --- Aggregation Logic ---
    stats = PeopleStatsResponse(
        source_urls_processed=[p.source_url for p in people_data_list if p.source_url],
        errors=errors,
    )
    location_stats = defaultdict(lambda: defaultdict(int)) # Nested defaultdict for easier summing

    for item in people_data_list:
        # Use getattr with default 0 to safely access potentially None fields
        aff = getattr(item, 'affected_count', 0) or 0
        cas = getattr(item, 'casualty_count', 0) or 0
        inj = getattr(item, 'injured_count', 0) or 0
        dis = getattr(item, 'displaced_count', 0) or 0

        stats.total_affected += aff
        stats.total_casualties += cas
        stats.total_injured += inj
        stats.total_displaced += dis

        loc = getattr(item, 'location', "Unknown Location") or "Unknown Location"
        location_stats[loc]["affected"] += aff
        location_stats[loc]["casualties"] += cas
        location_stats[loc]["injured"] += inj
        location_stats[loc]["displaced"] += dis

    # Convert defaultdicts back to regular dicts for the response model
    stats.by_location = {
        k: LocationPeopleStats(
            affected=v["affected"],
            casualties=v["casualties"],
            injured=v["injured"],
            displaced=v["displaced"]
        ) for k, v in location_stats.items()
    }

    return stats


# --- Existing /chat Endpoint (Revised) ---
# Keep this simpler, focusing on summarization/analysis rather than forced structure
# as the other endpoints now handle structured data.

async def summarize_content_for_chat(content: str, url: str) -> str:
    """Generates a brief summary for the chat response sources."""
    if not client: return "OpenAI client unavailable."
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", # Cheaper model for simple summary
            messages=[
                {"role": "system", "content": "Briefly summarize the key disaster-related information (what, where, when, impact) from this text in 1-2 sentences. Mention the source URL."},
                {"role": "user", "content": f"Source URL: {url}\n\nContent: {content[:3000]}"} # Limit context
            ],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content or "Summary could not be generated."
    except Exception as e:
        logger.error(f"Chat summarization failed for {url}: {e}", exc_info=True)
        return "Unable to summarize content."


async def analyze_results_for_chat(results: list, query: str, summaries: dict) -> str:
    """Provides a high-level analysis for the /chat endpoint."""
    if not client: return "OpenAI client unavailable. Cannot perform analysis."
    if not results: return "No search results to analyze."

    system_prompt = """You are an AI assistant summarizing disaster information from web search results. Briefly synthesize the key findings based on the user's query and the provided snippets/summaries. Focus on the most recent and significant events mentioned. Highlight key locations, disaster types, and reported impacts. Mention the limitations (based on search results). Keep the analysis concise (3-5 sentences)."""

    combined_info = []
    for r in results:
        source_info = f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}"
        if r.get('link') in summaries:
            source_info += f"\nSummary: {summaries[r['link']]}"
        combined_info.append(source_info)

    user_prompt = f"User Query: {query}\n\nAnalyze these findings:\n\n" + "\n---\n".join(combined_info)

    try:
        response = await client.chat.completions.create(
            model="gpt-4", # Use GPT-4 for better synthesis here
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt[:15000]} # Limit overall prompt length
            ],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content or "Analysis could not be generated."
    except Exception as e:
        logger.error(f"Chat analysis failed: {e}", exc_info=True)
        # Fallback to just listing snippets if analysis fails
        return "Unable to perform full analysis. Key points from search results:\n" + \
               "\n".join([f"- {r.get('snippet', 'N/A')}" for r in results[:5]]) # Limit fallback output



# (Keep enhanced_search)
def enhanced_search(query: str, max_results: int = 10) -> list[dict]:
    # ... (keep existing implementation) ...
    logger.info(f"Performing search for: '{query}' (max_results={max_results})")
    try:
        search = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="w", max_results=max_results)
        results = search.results(query, max_results=max_results)
        logger.info(f"Search returned {len(results)} results.")
        valid_results = [r for r in results if isinstance(r, dict) and 'link' in r and 'title' in r and 'snippet' in r]
        if len(valid_results) != len(results):
             logger.warning(f"Filtered out {len(results) - len(valid_results)} search results missing 'link', 'title', or 'snippet'.")
        return valid_results
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
        return []

# (Remove fetch_url_content - crawl4ai will handle fetching)
# (Remove summarize_content_for_chat - crawl4ai's LLM strategy will handle summarization)

# --- API Endpoints ---

# (Keep read_root, health_check, get_disasters_by_country, get_funding_statistics, get_people_statistics)
# ... (keep existing endpoint implementations) ...


# --- Existing /chat Endpoint (Revised with crawl4ai) ---

# (Keep analyze_results_for_chat - it now receives crawl4ai summaries)
async def analyze_results_for_chat(results: list, query: str, summaries: dict) -> str:
    # ... (keep existing implementation) ...
    if not client: return "OpenAI client unavailable. Cannot perform analysis."
    if not results: return "No search results to analyze."

    system_prompt = """You are an AI assistant summarizing disaster information from web search results. Briefly synthesize the key findings based on the user's query and the provided snippets/summaries. Focus on the most recent and significant events mentioned. Highlight key locations, disaster types, and reported impacts. Mention the limitations (based on search results). Keep the analysis concise (3-5 sentences)."""

    combined_info = []
    for r in results:
        source_info = f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}"
        link = r.get('link')
        if link and link in summaries: # Check if link exists before accessing summaries
            summary_text = summaries[link]
            # Add summary only if it's not an error message
            if not summary_text.startswith("Failed") and not summary_text.startswith("Error"):
                 source_info += f"\nSummary: {summary_text}"
            else:
                 source_info += f"\nSummary: (Could not be generated - {summary_text})" # Indicate failure
        combined_info.append(source_info)


    user_prompt = f"User Query: {query}\n\nAnalyze these findings:\n\n" + "\n---\n".join(combined_info)

    try:
        response = await client.chat.completions.create(
            model="gpt-4", # Use GPT-4 for better synthesis here
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt[:15000]} # Limit overall prompt length
            ],
            temperature=0.5,
            max_tokens=300
        )
        analysis = response.choices[0].message.content or "Analysis could not be generated."
        # Basic check if analysis seems generic/uninformative
        if len(analysis.split()) < 15 and ("no specific" in analysis.lower() or "do not provide" in analysis.lower()):
             analysis += "\n(Note: The underlying sources might lack specific details or couldn't be fully processed.)"
        return analysis

    except Exception as e:
        logger.error(f"Chat analysis failed: {e}", exc_info=True)
        # Fallback to just listing snippets if analysis fails
        return "Unable to perform full analysis. Key points from search results:\n" + \
               "\n".join([f"- {r.get('snippet', 'N/A')}" for r in results[:5]]) # Limit fallback output


@app.post("/chat", response_model=ChatResponse, tags=["Chat Assistant"])
async def disaster_assistant_chat(request: ChatRequest):
    """
    General conversational assistant using robust crawling (crawl4ai) for summaries.
    Provides analysis based on web search results and AI-generated content summaries.
    """
    if not client: # Check if primary OpenAI client is ok for analysis later
        raise HTTPException(status_code=503, detail="AI service initialization failed. Service unavailable.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
         raise HTTPException(status_code=500, detail="OpenAI API key not configured for crawling/summarization.")

    try:
        # 1. Search
        raw_results = enhanced_search(request.message, max_results=5) # Limit results for chat
        valid_results = [r for r in raw_results if r.get('link')]
        urls_to_summarize = [r['link'] for r in valid_results]

        # 2. Crawl and Summarize using crawl4ai (Concurrently)
        summaries = {}
        if urls_to_summarize:
            semaphore = asyncio.Semaphore(3) # Limit concurrent crawls

            # Define the summarization strategy using crawl4ai's LLM feature
            summarization_strategy = LLMExtractionStrategy(
                provider="openai", # Or "openai/gpt-4o-mini" directly
                # model_name="gpt-4o-mini", # Specify model if not in provider
                api_token=openai_api_key,
                instruction="""You are an expert summarizer. Read the main content of the webpage.
                Extract and provide a concise summary (2-4 sentences) focusing *only* on specific,
                recent disaster events, emergency situations, or related warnings mentioned.
                Include location, type of disaster, date/timeframe, and key impacts if available.
                If no specific disaster information is found, state 'No specific recent disaster information found in the content.'
                Do not include general news headlines or unrelated information.""",
                # extraction_type="text", # Often implicit for summarization based on instruction
                max_tokens=150,
                temperature=0.2,
            )

            async def crawl_and_summarize_url(url: str):
                async with semaphore:
                    logger.info(f"Initiating crawl and summarization for: {url}")
                    summary_text = f"Failed to process URL: {url}" # Default error message
                    try:
                        # Use crawl4ai for fetching and summarizing in one step
                        # CacheMode.ENABLED can speed up repeated requests during testing
                        async with AsyncWebCrawler(cache_mode=CacheMode.ENABLED, verbose=False) as crawler:
                            result = await crawler.arun(
                                url=url,
                                extraction_strategy=summarization_strategy,
                                word_count_threshold=50 # Ignore very short pages
                            )
                            if result and result.extracted_content:
                                # The LLM should return the summary text directly
                                summary_text = result.extracted_content.strip()
                                logger.info(f"Successfully summarized: {url}")
                            elif result and not result.extracted_content:
                                logger.warning(f"No content extracted by LLM for: {url}")
                                summary_text = "Summary could not be generated (no content extracted)."
                            else:
                                logger.warning(f"Crawling returned no result object for: {url}")
                                summary_text = "Failed to crawl or process the page."

                    except (CrawlingError, ExtractionError) as e:
                        logger.error(f"crawl4ai error for {url}: {type(e).__name__} - {e}", exc_info=True)
                        summary_text = f"Failed ({type(e).__name__})"
                    except Exception as e:
                        logger.error(f"Unexpected error during crawl/summary for {url}: {type(e).__name__} - {e}", exc_info=True)
                        summary_text = f"Failed (Unexpected Error)"
                    return url, summary_text

            # Run crawl/summary tasks concurrently
            summary_tasks = [crawl_and_summarize_url(url) for url in urls_to_summarize]
            summary_results = await asyncio.gather(*summary_tasks, return_exceptions=False) # Don't let one failure stop others

            # Collect summaries, handling potential exceptions caught within the task function
            for res in summary_results:
                if isinstance(res, tuple):
                    url, summary = res
                    summaries[url] = summary
                # else: This case should ideally not happen if return_exceptions=False and tasks handle errors

        # 3. Analyze Results with Summaries (using the existing function)
        final_analysis = await analyze_results_for_chat(valid_results, request.message, summaries)

        # 4. Format Response
        response_sources = []
        for r in valid_results:
            link = r.get('link')
            if link: # Ensure link exists
                 response_sources.append(ChatSource(
                     title=r.get('title'),
                     url=link,
                     summary=summaries.get(link, "Summary not available") # Use generated summary or default
                 ))

        return ChatResponse(
            analysis=final_analysis,
            sources=response_sources
            # Safety notice is now a default in the model
        )

    except HTTPException as he:
        raise he # Re-raise known HTTP exceptions
    except Exception as e:
        logger.error(f"Unhandled error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred processing the chat request.")


# --- Run Application ---

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Use reload=True for development only
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
    # For production, use: uvicorn main:app --host 0.0.0.0 --port 8080