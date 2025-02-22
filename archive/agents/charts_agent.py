# chart_agent.py

from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv
from typing import List, Dict, Any
import pandas as pd
import asyncio
import json
import os
import io
import re
from prophet import Prophet

# Load environment variables from .env file
load_dotenv()

# Define Pydantic models
class ForecastInstruction(BaseModel):
    forecast: Dict[str, List[str]]  # e.g., {'demand': ['x', 'y'], 'price': ['a', 'b']}

class ForecastDataPoint(BaseModel):
    month: str
    demand: int = None
    predictedDemand: int = None
    price: float = None
    predictedPrice: float = None

class ChartItem(BaseModel):
    title: str
    data: List[Dict[str, Any]]
    xAxisDataKey: str
    yAxisDataKeys: List[str]
    xAxisLabel: str
    yAxisLabel: str

class ForecastCharts(BaseModel):
    charts: List[ChartItem]

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
    """
    absolute_path = os.path.abspath(file_path)
    print(f"Attempting to read Excel file at: {absolute_path}")
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        if df.empty:
            print("Error: The Excel file is empty.")
            raise ValueError("The Excel file is empty.")
        print(f"Excel file '{file_path}' read successfully with {len(df)} records.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: No data found in the Excel file.")
        raise
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise

# Function to perform basic EDA (optional)
def perform_eda(df: pd.DataFrame):
    """
    Performs exploratory data analysis on the DataFrame.
    """
    print("Starting Exploratory Data Analysis (EDA).")
    
    # Display basic information
    print("DataFrame Info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)
    
    # Display summary statistics
    print("DataFrame Description:")
    description = df.describe(include='all').to_string()
    print(description)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values}")
    
    # Additional EDA steps can be added here

# Function to summarize the DataFrame
def summarize_dataframe(df: pd.DataFrame) -> str:
    """
    Summarizes the DataFrame to a JSON string suitable for GPT-4 analysis.
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
    
    summary_json = json.dumps(summary, indent=4)
    print("DataFrame Summary:")
    print(summary_json)
    return summary_json

# Function to generate prompt for Agent #1
def generate_prompt_agent1(summary: str) -> str:
    """
    Generates a prompt for GPT-4 to determine the types of forecast charts to create.
    """
    prompt = (
        "You are a data analyst. Analyze the following summarized data and identify the types of forecast charts that would be most beneficial for understanding future trends.\n\n"
        "Summarized Data:\n"
        + summary + "\n\n"
        "For each forecast chart, specify the following details:\n"
        "- Forecast Type: The type of forecast (e.g., demand, price)\n"
        "- Columns: The columns from the dataset to use for forecasting\n\n"
        "**Examples of forecast charts:**\n"
        "- {'forecast':{'demand':['x','y'],'price':['a','b']}}\n\n"
        "Ensure that you only provide 2 forecast charts that cover different aspects of the business performance and provide actionable insights.\n\n"
        "**The output should be a JSON object with the following structure only:**\n\n"
        "{\n"
        '    "forecast": {\n'
        '        "demand": ["column_x", "column_y"],\n'
        '        "price": ["column_a", "column_b"]\n'
        "    }\n"
        "}\n\n"
        "**Important:** Do not include any additional text, explanations, or code fences. The response should be **only** the JSON object as specified."
    )
    return prompt

# Function to generate prompt for Agent #2
def generate_prompt_agent2(forecast_type: str, columns: List[str]) -> str:
    """
    Generates a prompt for GPT-4 to assist in writing forecasting code using Prophet.
    """
    prompt = (
        f"You are a Python developer tasked with writing clean and functional code to forecast '{forecast_type}' using the Prophet library.\n\n"
        f"Given the dataset with columns {columns}, write a Python function named 'generate_forecast' that accepts a file path as an argument and performs the following steps:\n"
        "1. Preprocesses the data.\n"
        "2. Fits a Prophet model.\n"
        "3. Makes future predictions for the next 6 months.\n"
        "4. Returns the forecasted data in JSON format with the following structure:\n\n"
        "{\n"
        f'    "title": "{forecast_type.capitalize()} Forecast",\n'  # Dynamically set the title
        '    "data": [\n'
        '        {\n'
        '            "month": "Jan",\n'
        '            "demand": 400,\n'
        '            "predictedDemand": 350\n'
        '        },\n'
        '        // More data points...\n'
        '    ],\n'
        '    "xAxisDataKey": "month",\n'
        '    "yAxisDataKeys": ["demand", "predictedDemand"],\n'
        '    "xAxisLabel": "Month",\n'
        '    "yAxisLabel": "Demand (Units)"\n'
        "}\n\n"
        "Ensure the code is well-structured, includes necessary error handling, and follows best practices.\n\n"
        "**Important:** Do not include any additional explanations, text, or code fences. The output should be **only** the Python code as specified."
    )
    return prompt

# Helper function to extract JSON from response
def extract_json(response: str) -> str:
    """
    Extracts the JSON object or array from the response string.
    """
    # Regex pattern to find JSON object or array
    pattern = r'(\{.*\}|\[.*\])'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ""

# Helper function to extract code from response (remove code fences)
def extract_code(response: str) -> str:
    """
    Extracts Python code from a response string, removing any code fences.
    """
    # Regex to extract content between ```python and ```
    pattern = r'```python\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1)
        return code.strip()
    else:
        # If no code fences, assume the entire response is code
        return response.strip()

# Function to interact with GPT-4 and retrieve forecast instructions (Agent #1)
async def agent1_get_forecast_instructions(summary: str, openai_api_key: str) -> ForecastInstruction:
    """
    Agent #1: Sends the summary to GPT-4 and retrieves forecast chart instructions.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    prompt = generate_prompt_agent1(summary)
    
    try:
        print("Agent #1: Sending prompt to GPT-4.")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use the appropriate GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines forecast chart types based on summarized data."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=500,
        )
        
        print("Agent #1: Received response from GPT-4.")
        # Debug: Print the raw response
        print(f"Agent #1 GPT-4 Response: {response}")
        
        if not response.choices:
            print("Agent #1 Error: No choices returned in the OpenAI response.")
            return None
        
        message = response.choices[0].message
        if not hasattr(message, 'content'):
            print("Agent #1 Error: The message object does not have a 'content' attribute.")
            print(f"Agent #1 Message Object: {message}")
            return None
        
        reply = message.content
        json_str = extract_json(reply)
        
        if not json_str:
            print("Agent #1 Error: No JSON object found in GPT-4 response.")
            print(f"Agent #1 GPT-4 Response Content: {reply}")
            return None
        
        print(f"Agent #1 Extracted JSON: {json_str}")
        instructions = ForecastInstruction(**json.loads(json_str))
        
        print("Agent #1: Forecast instructions extracted successfully.")
        print(f"Agent #1 Instructions: {instructions.json()}")
        return instructions
    except json.JSONDecodeError:
        print("Agent #1 Error: Failed to parse GPT-4 response as JSON.")
        return None
    except OpenAIError as oe:
        print(f"Agent #1 OpenAI API Error: {oe}")
        return None
    except ValidationError as ve:
        print(f"Agent #1 Validation Error: {ve}")
        return None
    except Exception as e:
        print(f"Agent #1 Unexpected Error: {e}")
        return None

# Function to interact with GPT-4 and generate forecasting code (Agent #2)
async def agent2_generate_forecast_code(forecast_type: str, columns: List[str], openai_api_key: str) -> str:
    """
    Agent #2: Generates forecasting code using GPT-4 based on forecast type and columns.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    prompt = generate_prompt_agent2(forecast_type, columns)
    
    try:
        print(f"Agent #2: Sending prompt to GPT-4 for forecast type '{forecast_type}'.")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a skilled Python developer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # Lower temperature for more deterministic output
            max_tokens=1000,
        )
        
        print(f"Agent #2: Received response from GPT-4 for '{forecast_type}'.")
        # Debug: Print the raw response
        print(f"Agent #2 GPT-4 Response for '{forecast_type}': {response}")
        
        if not response.choices:
            print(f"Agent #2 Error: No choices returned in the OpenAI response for '{forecast_type}'.")
            return None
        
        message = response.choices[0].message
        if not hasattr(message, 'content'):
            print(f"Agent #2 Error: The message object does not have a 'content' attribute for '{forecast_type}'.")
            print(f"Agent #2 Message Object for '{forecast_type}': {message}")
            return None
        
        code_response = message.content.strip()
        code = extract_code(code_response)
        
        if not code:
            print(f"Agent #2 Error: No Python code found in the GPT-4 response for '{forecast_type}'.")
            print(f"Agent #2 GPT-4 Response Content for '{forecast_type}': {code_response}")
            return None
        
        print(f"Agent #2: Forecasting code for '{forecast_type}' generated successfully.")
        print(f"Agent #2: Generated Code for '{forecast_type}':\n{code}")
        return code
    except OpenAIError as oe:
        print(f"Agent #2 OpenAI API Error for '{forecast_type}': {oe}")
        return None
    except Exception as e:
        print(f"Agent #2 Unexpected Error for '{forecast_type}': {e}")
        return None

# Function to execute generated code and capture JSON output
def execute_generated_code(code: str, file_path: str) -> Dict[str, Any]:
    """
    Executes the generated Python code safely and captures the JSON output.
    """
    try:
        print("Executing generated forecasting code.")
        # Define a local namespace to execute the code
        local_namespace = {}
        print(f"Executing code:\n{code}")
        exec(code, local_namespace, local_namespace)  # Modified exec call
        
        # Assuming the generated code defines a function 'generate_forecast' that accepts 'file_path' and returns JSON
        if 'generate_forecast' in local_namespace:
            forecast_json = local_namespace['generate_forecast'](file_path)
            print("Executed generated forecasting code successfully.")
            print(f"Forecast JSON:\n{forecast_json}")
            # Attempt to parse JSON if it's a string
            if isinstance(forecast_json, str):
                try:
                    forecast_json = json.loads(forecast_json)
                except json.JSONDecodeError:
                    print("Error: The generated forecast is not valid JSON.")
                    return {}
            return forecast_json
        else:
            print("Error: The generated code does not define a 'generate_forecast' function.")
            return {}
    except Exception as e:
        print(f"Error executing generated code: {e}")
        return {}

# Function to perform forecasting using Agent #2
async def agent2_perform_forecast(forecast_type: str, columns: List[str], file_path: str, openai_api_key: str) -> Dict[str, Any]:
    """
    Agent #2: Generates and executes forecasting code, returning the forecast data.
    """
    # Generate forecasting code using GPT-4
    code = await agent2_generate_forecast_code(forecast_type, columns, openai_api_key)
    if not code:
        print(f"Agent #2: Failed to generate code for '{forecast_type}'.")
        return {}
    
    # Execute the generated code and capture the forecast JSON
    forecast_json = execute_generated_code(code, file_path)
    
    # Validate the forecast JSON structure
    try:
        chart_item = ChartItem(**forecast_json)
        print(f"Agent #2: Forecast data for '{forecast_type}' validated successfully.")
        print(f"Agent #2: Chart Item for '{forecast_type}': {chart_item.json()}")
        return chart_item.dict()
    except ValidationError as ve:
        print(f"Agent #2 Validation Error for '{forecast_type}': {ve}")
        return {}
    except Exception as e:
        print(f"Agent #2 Unexpected Error while validating forecast data for '{forecast_type}': {e}")
        return {}

# Function to validate and structure the forecast charts
def validate_forecast_charts(charts: List[Dict[str, Any]]) -> ForecastCharts:
    """
    Validates and structures the forecast charts using Pydantic models.
    """
    try:
        forecast_charts = ForecastCharts(charts=[ChartItem(**chart) for chart in charts])
        print("Forecast charts validated successfully.")
        print(f"Forecast Charts:\n{forecast_charts.json(indent=4)}")
        return forecast_charts
    except ValidationError as ve:
        print(f"Forecast Charts Validation Error: {ve}")
        raise
    except Exception as e:
        print(f"Error during forecast charts validation: {e}")
        raise

# Main Coordinator Function
async def generate_forecast_charts(file_path: str, summary: str) -> str:
    """
    Main function to generate forecast charts from the Excel dataset using an agentic model.
    """
    try:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("Coordinator Error: OpenAI API key not found in environment variables.")
            return json.dumps({"error": "API key not found."})
        
        # Agent #1: Get forecast instructions
        instructions = await agent1_get_forecast_instructions(summary, openai_api_key)
        
        if not instructions or not instructions.forecast:
            print("Coordinator Error: No forecast instructions were extracted. Exiting.")
            return json.dumps({"error": "No forecast instructions extracted."})
        
        print(f"Coordinator: Forecast instructions: {instructions.json()}")
        
        # Read the Excel file
        df = read_excel(file_path)
        
        # Verify that specified columns exist in the DataFrame
        missing_columns = []
        for forecast_type, columns in instructions.forecast.items():
            for column in columns:
                if column not in df.columns:
                    missing_columns.append((forecast_type, column))
        
        if missing_columns:
            for forecast_type, column in missing_columns:
                print(f"Coordinator Error: Column '{column}' specified for '{forecast_type}' does not exist in the DataFrame.")
            return json.dumps({"error": "Specified columns not found in the dataset.", "details": missing_columns})
        
        # Agent #2: Perform forecasts based on instructions
        forecast_charts = []
        for forecast_type, columns in instructions.forecast.items():
            print(f"Coordinator: Initiating forecast for '{forecast_type}' using columns {columns}.")
            chart = await agent2_perform_forecast(forecast_type, columns, file_path, openai_api_key)
            if chart:
                forecast_charts.append(chart)
            else:
                print(f"Coordinator Warning: Forecast chart for '{forecast_type}' was not generated.")
        
        if not forecast_charts:
            print("Coordinator Error: No forecast charts were generated.")
            return json.dumps({"error": "No forecast charts were generated."})
        
        # Validate and structure the forecast charts
        forecast_results = validate_forecast_charts(forecast_charts)
        
        # Output the results as JSON
        return forecast_results.model_dump_json(indent=4)
    
    except ValidationError as ve:
        print(f"Coordinator Validation Error: {ve}")
        return json.dumps({"error": "Validation error.", "details": ve.errors()})
    except Exception as e:
        print(f"Coordinator Unexpected Error: {e}")
        return json.dumps({"error": "An unexpected error occurred.", "details": str(e)})
    finally:
        print("Forecast chart generation process completed.")
