import os
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Union
from dotenv import load_dotenv

import httpx
import pycountry
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient
from openai import OpenAI
from aiocache import cached, Cache

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the global logging level to INFO
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress INFO logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("tavily").setLevel(logging.WARNING)  # Add other libraries if needed

# Validate environment variables
required_env_vars = ['TAVILY_API_KEY', 'OPENAI_API_KEY', 'OPENWEATHERMAP_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    missing = ', '.join(missing_vars)
    logger.error(f"Missing required environment variables: {missing}")
    raise EnvironmentError(f"Missing required environment variables: {missing}")

# Initialize Tavily client
tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define dependencies required by the agent
@dataclass
class ForecastingDependencies:
    product_name: str
    location: str

# Define the structure of the agent's output
class ForecastResult(BaseModel):
    forecast: str = Field(description='Predicted demand trend')
    percentage_change: float = Field(description='Predicted percentage change in demand')

# Initialize the agent
forecasting_agent = Agent(
    model='openai:gpt-4',
    deps_type=ForecastingDependencies,
    result_type=ForecastResult,
    system_prompt=(
        'You are a demand forecasting assistant. '
        'Analyze various factors such as economic indicators, environmental conditions, '
        'market trends, and expert opinions to predict the demand for a given product in a specific location.'
    ),
)

# Initialize HTTP client
http_client = httpx.AsyncClient()

# Define a tool to fetch market trend data using Tavily API
@forecasting_agent.tool
@cached(ttl=3600, cache=Cache.MEMORY)  # Cache for 1 hour
async def get_market_trends(ctx: RunContext[ForecastingDependencies]) -> Dict[str, str]:
    """
    Fetch market trend data for the specified product and location using Tavily API.
    """
    query = f"Market trends for {ctx.deps.product_name} in {ctx.deps.location}"
    try:
        # Run the synchronous search method in a separate thread
        response = await asyncio.to_thread(tavily_client.search, query)

        # Adjust the following lines based on the actual response structure
        # Assuming 'response' is a dict with a 'results' key
        trends = {result['title']: result['content'] for result in response.get('results', [])}
        logger.info("Fetched market trends successfully.")
        return trends
    except Exception as e:
        logger.error(f"Error fetching market trends: {e}")
        return {}

# Define a tool to fetch economic indicators using the World Bank's API
@forecasting_agent.tool
@cached(ttl=86400, cache=Cache.MEMORY)  # Cache for 1 day
async def get_economic_indicators(ctx: RunContext[ForecastingDependencies]) -> Optional[Dict[str, Dict[str, Optional[Union[float, str]]]]]:
    """
    Fetch economic indicators relevant to the specified location using the World Bank's API.
    """
    location = ctx.deps.location
    country = pycountry.countries.get(name=location)
    if not country:
        logger.error(f"Could not find country code for location: {location}")
        return None
    country_code = country.alpha_2
    indicators = {
        'GDP': 'NY.GDP.MKTP.CD',
        'Inflation': 'FP.CPI.TOTL.ZG',
        'Unemployment': 'SL.UEM.TOTL.ZS'
    }
    base_url = 'http://api.worldbank.org/v2/country/{}/indicator/{}'
    economic_data = {}
    for indicator_name, indicator_code in indicators.items():
        url = base_url.format(country_code, indicator_code)
        params = {'format': 'json', 'per_page': 100, 'date': '2020:2023'}  # Adjust date range as needed
        try:
            resp = await http_client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 1 and data[1]:
                # Extract the most recent value
                latest_record = next((item for item in data[1] if item['value'] is not None), None)
                if latest_record:
                    economic_data[indicator_name] = {
                        'value': latest_record['value'],
                        'date': latest_record['date']
                    }
            logger.info(f"Fetched economic indicator: {indicator_name}")
        except httpx.HTTPError as e:
            logger.error(f"An error occurred while fetching {indicator_name}: {e}")
            economic_data[indicator_name] = None
    return economic_data

# Define a tool to fetch environmental conditions using OpenWeatherMap API
@forecasting_agent.tool
@cached(ttl=1800, cache=Cache.MEMORY)  # Cache for 30 minutes
async def get_environmental_conditions(ctx: RunContext[ForecastingDependencies]) -> Dict[str, Optional[Union[float, str]]]:
    """
    Fetch environmental conditions relevant to the specified location using OpenWeatherMap API.
    """
    api_key = os.getenv('OPENWEATHERMAP_API_KEY')
    city = ctx.deps.location
    url = 'http://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        conditions = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'weather_description': data['weather'][0]['description']
        }
        logger.info("Fetched environmental conditions successfully.")
        return conditions
    except httpx.HTTPError as e:
        logger.error(f"An error occurred while fetching environmental conditions: {e}")
        return {}

# Define a tool to fetch expert opinions using OpenAI's GPT model
@forecasting_agent.tool
@cached(ttl=3600, cache=Cache.MEMORY)  # Cache for 1 hour
async def get_expert_opinions(ctx: RunContext[ForecastingDependencies]) -> Optional[str]:
    """
    Generate expert opinions on the demand for the specified product in the specified location.
    """
    prompt = (
        f"As an expert in market analysis, provide your insights on the demand for {ctx.deps.product_name} "
        f"in {ctx.deps.location} for the upcoming month."
    )
    try:
        # Run the synchronous OpenAI API call in a separate thread
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1000
        )
        # Correctly access the response attributes
        # Depending on the OpenAI client version, adjust the access accordingly
        # Here, assuming response.choices[0].message.content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            # Depending on the OpenAI library version, access the content appropriately
            # For example, it might be response.choices[0].message['content'] or response.choices[0].message.content
            # Adjust accordingly based on your library's response structure
            try:
                expert_opinion = response.choices[0].message.content.strip()
            except AttributeError:
                # Fallback if 'message' is a dict
                expert_opinion = response.choices[0].message['content'].strip()
            logger.info("Fetched expert opinions successfully.")
            return expert_opinion
        else:
            logger.error("Unexpected response structure from OpenAI.")
            return None
    except Exception as e:
        logger.error(f"An error occurred while fetching expert opinions: {e}")
        return None

# Define the agent's main execution logic
@forecasting_agent.run
async def forecasting_logic(ctx: RunContext[ForecastingDependencies]) -> ForecastResult:
    """
    Main logic to aggregate data from various tools and generate a demand forecast.
    """
    # Concurrently fetch all required data
    market_trends, economic_indicators, environmental_conditions, expert_opinions = await asyncio.gather(
        get_market_trends(ctx),
        get_economic_indicators(ctx),
        get_environmental_conditions(ctx),
        get_expert_opinions(ctx)
    )

    # Basic logic to generate forecast based on fetched data
    # In a real scenario, you would implement a more sophisticated model or analysis
    if not all([market_trends, economic_indicators, environmental_conditions]):
        logger.warning("Incomplete data received. Forecast may be inaccurate.")

    # Handle the possibility of missing expert opinions
    expert_opinion_text = f"and {expert_opinions.lower()}" if expert_opinions else ""

    # Placeholder logic for forecast generation
    # This should be replaced with actual analysis using the fetched data
    gdp_value = economic_indicators.get('GDP', {}).get('value', 0)
    forecast_trend = "increase" if gdp_value > 0 else "decrease"
    percentage_change = gdp_value * 0.1  # Simplistic calculation

    forecast = (
        f"Based on the current market trends, economic indicators, and environmental conditions, "
        f"the demand for {ctx.deps.product_name} in {ctx.deps.location} is expected to {forecast_trend} "
        f"by approximately {percentage_change:.2f}% in the upcoming month. "
        f"Expert opinions also suggest that {expert_opinion_text}."
    )

    return ForecastResult(forecast=forecast, percentage_change=percentage_change)

# Example usage
async def main():
    try:
        # Define dependencies
        deps = ForecastingDependencies(
            product_name='Milk',
            location='Estonia'
        )

        # Run the agent with a specific query
        result = await forecasting_agent.run(
            'Predict the demand for the next month.',
            deps=deps
        )

        # Access the forecast and percentage change
        print(f"Demand Forecast: {result.data.forecast}")
        print(f"Predicted Percentage Change: {result.data.percentage_change:.2f}%")
    except Exception as e:
        logger.error(f"An error occurred during forecasting: {e}")
    finally:
        # Close the HTTP client within the same event loop
        await http_client.aclose()

# Execute the main function
if __name__ == "__main__":
    asyncio.run(main())
