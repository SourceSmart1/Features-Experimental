from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
from openai import OpenAIError
from dotenv import load_dotenv
from typing import List
import pandas as pd
import logging
import asyncio
import json
import os
import io
import re
import sys

#disable logging
logging.disable(logging.CRITICAL)

# Suppress "Event loop is closed" warning on Windows
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define data models
class ResultItem(BaseModel):
    title: str = Field(description="Any key metric from the database")
    amount: str = Field(description="Numeric value, formatted as string")
    result: str = Field(description="Comparison from either last month or last year")
    type: str = Field(description="Type of the metric: 'number' or 'currency'")
    status: str = Field(description="Status of the metric: 'increase', 'stable', or 'decrease'")
    icon: str = Field(
        description=(
            "Icon for the metric. "
            "Possible values: 'dollar', 'shopping-cart', 'bar-chart', 'pie-chart', 'users', 'tools'."
        )
    )

class ForecastResults(BaseModel):
    results: List[ResultItem]

# Helper function to convert Timestamps and Timedeltas
def convert_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(v) for v in obj]
    else:
        return obj

# Function to read Excel file
def read_excel(file_path: str) -> pd.DataFrame:
    """
    Reads an Excel file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the Excel file is empty.
        Exception: For other unexpected errors.
    """
    absolute_path = os.path.abspath(file_path)
    logger.info(f"Attempting to read Excel file at: {absolute_path}")
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        if df.empty:
            logger.error("The Excel file is empty.")
            raise ValueError("The Excel file is empty.")
        logger.info(f"Excel file '{file_path}' read successfully with {len(df)} records.")
        return df
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("No data found in the Excel file.")
        raise
    except Exception as e:
        logger.error("Error reading Excel file: %s", e)
        raise

# Function to perform basic EDA
def perform_eda(df: pd.DataFrame):
    """
    Performs exploratory data analysis on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    logger.info("Starting Exploratory Data Analysis (EDA).")
    
    # Display basic information
    logger.info("DataFrame Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    logger.info(info_str)
    
    # Display summary statistics
    logger.info("DataFrame Description:")
    description = df.describe(include='all').to_string()
    logger.info(description)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    logger.info(f"Missing Values:\n{missing_values}")
    
    # Additional EDA steps can be added here
    # For example: plotting, correlation analysis, etc.

# Function to summarize the DataFrame
def summarize_dataframe(df: pd.DataFrame) -> str:
    """
    Summarizes the DataFrame to a JSON string suitable for GPT-4 analysis.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.

    Returns:
        str: A JSON-formatted string summarizing the DataFrame.
    """
    summary = {
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.apply(lambda x: x.name).to_dict(),
        "summary_statistics": df.describe(include='all').to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict(orient='records')  # First 5 rows as sample
    }
    
    # Convert any Timestamps or Timedeltas to strings
    summary = convert_timestamps(summary)
    
    return json.dumps(summary, indent=4)

# Updated generate_prompt function requesting the "icon" field
def generate_prompt(summary: str) -> str:
    """
    Generates a prompt for GPT-4 to extract a diverse set of eight business metrics from summarized data,
    including an icon for each metric.

    Args:
        summary (str): The JSON-formatted summary of the DataFrame.

    Returns:
        str: The generated prompt.
    """
    prompt = (
        "You are a business analyst. Analyze the following summarized data and identify exactly eight key business-related metrics that provide a comprehensive overview of the business performance.\n\n"
        "Summarized Data:\n"
        + summary + "\n\n"
        "For each of the eight identified metrics, provide the following details:\n"
        "- Title: Name of the metric (e.g., Total Revenue, Average Quantity Purchased)\n"
        "- Amount: Current value (formatted as a string with appropriate units)\n"
        "- Result: Comparison from either last month or last year (e.g., +20.1% since last month)\n"
        "- Type: Specify the type of metric (e.g., 'total', 'average', 'count', 'percentage', 'maximum', 'minimum')\n"
        "- Status: 'increase', 'stable', or 'decrease'\n"
        "- Icon: Chosen from the following list based on the metric's meaning:\n"
        "  'dollar', 'shopping-cart', 'bar-chart', 'pie-chart', 'users', 'tools'\n\n"
        "**Examples of desired metrics:**\n"
        "- Total Revenue\n"
        "- Average Quantity Purchased\n"
        "- Total Units Sold\n"
        "- Percentage Increase in Revenue\n"
        "- Maximum Daily Sales\n"
        "- Count of Returning Customers\n"
        "- Minimum Inventory Level\n"
        "- Average Production Cost\n\n"
        "Ensure that the metrics include a mix of different types, such as totals/sums, averages, counts, and other relevant statistics to provide a well-rounded view of the business performance.\n\n"
        "**The output should be a JSON array of exactly eight objects with the following structure only:**\n\n"
        "[\n"
        "    {\n"
        '        "title": "Metric Title",\n'
        '        "amount": "Value",\n'
        '        "result": "Comparison",\n'
        '        "type": "Metric Type",\n'
        '        "status": "increase, stable, or decrease",\n'
        '        "icon": "dollar, shopping-cart, bar-chart, pie-chart, users, or tools"\n'
        "    },\n"
        "    ...\n"
        "]\n\n"
        "**Important:** Do not include any additional text, explanations, or code fences. The response should be **only** the JSON array as specified."
    )
    return prompt

# Helper function to extract JSON from response
def extract_json(response: str) -> str:
    """
    Extracts the JSON array from the response string.

    Args:
        response (str): The response string from GPT-4.

    Returns:
        str: The extracted JSON string.
    """
    # Regex pattern to find JSON array
    pattern = r'\[\s*\{.*\}\s*\]'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ""

# Function to interact with GPT-4 and retrieve business metrics
async def get_business_metrics(summary: str, openai_api_key: str) -> List[dict]:
    """
    Sends the summarized data to GPT-4 and retrieves business metrics.

    Args:
        summary (str): The JSON-formatted summary of the DataFrame.
        openai_api_key (str): OpenAI API key.

    Returns:
        List[dict]: A list of dictionaries containing the business metrics.
    """
    # Instantiate the AsyncOpenAI client
    client = AsyncOpenAI(api_key=openai_api_key)
    
    prompt = generate_prompt(summary)
    
    try:
        # Make the API call using the client-based interface
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to match the correct GPT-4 model name if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts key business metrics from summarized data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=1500,
        )
        
        # Log the entire response for debugging
        logger.debug("OpenAI Response: %s", response)
        
        # Ensure that 'choices' is not empty
        if not response.choices:
            logger.error("No choices returned in the OpenAI response.")
            return []
        
        # Access the message object
        message = response.choices[0].message
        
        # Check if 'content' attribute exists
        if not hasattr(message, 'content'):
            logger.error("The message object does not have a 'content' attribute.")
            logger.debug("Message Object: %s", message)
            return []
        
        # Extract the assistant's reply using attribute access
        reply = message.content
        
        # Attempt to extract JSON from the reply
        json_str = extract_json(reply)
        
        if not json_str:
            logger.error("No JSON array found in GPT-4 response.")
            logger.error("GPT-4 Response: %s", reply)
            return []
        
        # Attempt to parse the extracted JSON string
        metrics = json.loads(json_str)
        
        # Verify that exactly eight metrics are returned
        if len(metrics) != 8:
            logger.error("Expected 8 metrics, but received %d.", len(metrics))
            return []
        
        logger.info("Business metrics extracted successfully.")
        return metrics
    except json.JSONDecodeError:
        logger.error("Failed to parse GPT-4 response as JSON.")
        logger.error("Extracted JSON: %s", json_str)
        return []
    except OpenAIError as oe:
        logger.error("OpenAI API error: %s", oe)
        return []
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching business metrics.")
        return []

# Function to validate and structure the metrics
def validate_metrics(metrics: List[dict]) -> ForecastResults:
    """
    Validates and structures the metrics using pydantic models.

    Args:
        metrics (List[dict]): A list of dictionaries containing the business metrics.

    Returns:
        ForecastResults: A pydantic model containing the structured results.
    """
    try:
        forecast_results = ForecastResults(results=[ResultItem(**metric) for metric in metrics])
        logger.info("Metrics validated successfully.")
        return forecast_results
    except ValidationError as ve:
        logger.error("Validation error: %s", ve)
        raise
    except Exception as e:
        logger.error("An error occurred during metric validation: %s", e)
        raise

# Main execution logic
async def fetch_metrics(summary):
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OpenAI API key not found in environment variables.")
            return "API key not found."

        #perform_eda(df)  # Optional EDA step
        
        # Summarize the DataFrame
        #summary = summarize_dataframe(df)
        
        # Get business metrics from GPT-4
        metrics = await get_business_metrics(summary, openai_api_key)
        
        if not metrics:
            logger.error("No metrics were extracted. Exiting.")
            return "No metrics extracted."
        
        # Validate and structure the metrics
        forecast_results = validate_metrics(metrics)
        
        # Output the results using model_dump_json
        return forecast_results.model_dump_json(indent=4)
    
    except FileNotFoundError:
        logger.error("Please ensure the Excel file exists in the specified path.")
    except ValidationError as ve:
        logger.error("Validation error: %s", ve)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
    finally:
        logger.info("Processing completed.")
