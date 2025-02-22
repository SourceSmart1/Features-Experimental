from flask import Flask, request, jsonify, redirect,Response
from supabase import create_client, Client
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
import pandas as pd
import json
import logging
import os
import asyncio
import numpy as np
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from agents.metrics_agent import fetch_metrics,summarize_dataframe
from agents.charts_agent import generate_forecast_charts

# Load environment variables
load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
#logging.basicConfig(level=#logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    ##logging.error("Supabase URL or Key is not set in environment variables.")
    raise EnvironmentError("Missing Supabase configuration.")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    ##logging.debug("Supabase client created successfully.")
except Exception as e:
    ##logging.error(f"Failed to create Supabase client: {e}")
    raise

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    ##logging.error("OpenAI API key is not set in environment variables.")
    raise EnvironmentError("Missing OpenAI API key.")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    ##logging.debug("OpenAI client created successfully.")
except Exception as e:
    ##logging.error(f"Failed to create OpenAI client: {e}")
    raise

def file_exists(bucket_name, file_path):
    """
    Checks if a file exists in the specified Supabase storage bucket.
    Returns True if exists, False otherwise.
    """
    try:
        # Attempt to download the file; if it succeeds, the file exists
        supabase.storage.from_(bucket_name).download(file_path)
        ##logging.debug(f"File '{file_path}' exists in bucket '{bucket_name}'.")
        return True
    except Exception as e:
        # Check if the error message indicates the file was not found
        if 'not_found' in str(e).lower() or 'object not found' in str(e).lower():
            ##logging.debug(f"File '{file_path}' does not exist in bucket '{bucket_name}'.")
            return False
        else:
            # Re-raise unexpected exceptions
            ###logging.error(f"Error checking if file exists: {e}")
            raise
    except Exception as e:
        ##logging.error(f"Unexpected error when checking if file exists: {e}")
        raise

def user_login(user_id, password):
    try:
        # Query the User_details table for the provided email_id
        response = supabase.table('User_details').select('password').eq('email_id', user_id).single().execute()

        # Check if a matching record was found
        if response.data:
            # Compare the retrieved password with the provided password
            stored_password = response.data['password']
            if stored_password == password:
                return True
            else:
                print("Password mismatch")
                return False
        else:
            print("Email ID not found")
            return False
    except Exception as e:
        # Handle errors (e.g., network issues, Supabase errors)
        print(f"Error during login: {e}")
        return False

def upload_or_replace_file(user_id, file_stream, original_filename):
    """
    Uploads or replaces the Excel file for a user and updates the Supabase table,
    yielding multiple progress messages in the process.
    """
    converted_file = None
    try:
        # 1) Convert file to XLSX in memory
        converted_file = convert_to_xlsx(file_stream, original_filename)
        yield {"message": "File converted to XLSX successfully."}

        df=pd.read_excel(converted_file)
        # 2) Prepare paths and check for existence
        file_path = f"{user_id}_file.xlsx"
        bucket_name = "excel_file_store"

        yield {"message": "Checking if file exists..."}
        if file_exists(bucket_name, file_path):
            yield {"message": "File already exists. Deleting existing file..."}
            delete_response = supabase.storage.from_(bucket_name).remove([file_path])
            if (isinstance(delete_response, dict) 
                and "error" in delete_response 
                and delete_response["error"]):
                error_message = delete_response["error"].get("message", "Unknown delete error")
                raise Exception(error_message)

        # 3) Upload the new file
        yield {"message": "Uploading file..."}

        # Get raw bytes from the BytesIO object
        file_bytes = converted_file.getvalue()

        # Pass bytes to Supabase, not the BytesIO object
        response = supabase.storage.from_(bucket_name).upload(file_path, file_bytes)
        if (isinstance(response, dict) 
            and "error" in response 
            and response["error"]):
            error_message = response["error"].get("message", "Unknown upload error")
            raise Exception(error_message)

        yield {"message": "Preprocessing the data..."}
        summary=summarize_dataframe(df)
        # 4) Update DB with new file URL
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{file_path}"
        yield {"message": "Updating database with file URL."}
        update_response = supabase.table("User_file_storage").upsert({
            "user_id": user_id,
            "excel_file_url": file_url,
            "excel_file_summary":summary
        }).execute()
        if "error" in update_response:
            error_message = update_response["error"].get("message", "Unknown database error")
            raise Exception(error_message)

        # 5) Final success message
        yield {"message": "File uploaded and stored successfully."}
        
    except Exception as e:
        raise
    finally:
        # We used in-memory BytesIO; no disk file to remove
        pass

def get_file_by_user_id(user_id):
    """Retrieves the Excel file URL for a user from Supabase table."""
    try:
        # Execute the query
        response = supabase.table("User_file_storage") \
                           .select("excel_file_url") \
                           .eq("user_id", user_id) \
                           .single() \
                           .execute()
        
        # Debugging: Log the entire response object (optional)
        # logging.debug(f"Supabase select response: {response}")

        # Check if data exists in the response
        if response.data and "excel_file_url" in response.data:
            file_url = response.data["excel_file_url"]
            logging.debug(f"Retrieved file URL: {file_url}")
            return file_url
        else:
            logging.debug(f"No file URL found for User ID: {user_id}")
            return None

    except Exception as e:
        logging.error(f"Error retrieving file: {e}")
        raise

def get_summary_by_user_id(user_id):
    """Retrieves the Excel file summary for a user from the 'User_file_storage' table."""
    try:
        response = supabase.table("User_file_storage") \
                           .select("excel_file_summary") \
                           .eq("user_id", user_id) \
                           .single() \
                           .execute()

        if response.data and "excel_file_summary" in response.data:
            summary_text = response.data["excel_file_summary"]
            logging.debug(f"Summary Retrieved: {summary_text}")
            return summary_text
        else:
            logging.debug("No file summary found for User ID")
            return None

    except Exception as e:
        logging.error(f"Error retrieving summary: {e}")
        raise

def get_metrics_by_user_id(user_id):
    """Retrieves the metrics_data for a user from the 'User_file_storage' table."""
    try:
        response = supabase.table("User_file_storage") \
                           .select("metrics_data") \
                           .eq("user_id", user_id) \
                           .single() \
                           .execute()

        if response.data and "metrics_data" in response.data:
            summary_text = response.data["metrics_data"]
            logging.debug(f"Summary Retrieved: {summary_text}")
            return summary_text
        else:
            logging.debug("No metrics for User ID")
            return None

    except Exception as e:
        logging.error(f"Error retrieving summary: {e}")
        raise

def upload_metrics(user_id, metrics_data):
    """Updates only the metrics_data column for a user in the 'User_file_storage' table."""
    try:
        response = supabase.table("User_file_storage") \
                           .update({"metrics_data": metrics_data}) \
                           .eq("user_id", user_id) \
                           .execute()
        if "error" in response:
            error_message = response["error"].get("message", "Unknown database error")
            raise Exception(error_message)
        return response
    except Exception as e:
        logging.error(f"Error in upload_metrics: {e}")
        raise

def retrieve_forecasts(column_headers, sample_data, df, model="gpt-4o"):
    prompt = f"""
        You are a data forecasting assistant. The dataset has these columns:
        {column_headers}

        Sample Data:
        {sample_data}

        Your Task:
        1. Identify Key Metrics using the sample data and column headers.:
        - Analyze the columns and sample data.
        - Select 3-4 relevant numeric metrics to forecast.

        2. Forecasting Approach (this is the output):
        - Use the selected metrics to forecast the next 3 months for the actual data stored in df dataframe.
        - Use `scikit-learn` models such as:
            - LinearRegression
            - RandomForestRegressor
            - SVR
        - Choose the best model based on your intuition for that metric.
        - Forecast the next 3 months.

        Return Format:
        - Generate Python code that produces a dictionary like:
        {{"metric_1": "+20%", "metric_2": "-15%", "metric_3": "+10%"}}

        Important Notes:
        - You can access the dataset using the provided DataFrame `df` which is already loaded with the data.
        - Use only the provided DataFrame `df`.
        - Ensure the generated code returns the `result` dictionary.
        - Correct any errors if they occur during execution.
        - Log the best model and its MSE value.
        - Do not return the answer directly; only return code.
    """

    globals_dict = {
        "pd": pd,
        "np": np,
        "df": df,
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "SVR": SVR,
        "mean_squared_error": mean_squared_error,
        "train_test_split": train_test_split,
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Python forecasting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        generated_code = clean_code_block(response.choices[0].message.content.strip())
        execution_result = execute_code(generated_code, globals_dict)

        if "error" in execution_result:
            correction_prompt = f"""
            The following code returned an error:
            ```
            {generated_code}
            ```

            The error was:
            {execution_result['error']}

            Correct the code. Ensure the `result` dictionary is returned after execution.
            """

            corrected_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a Python expert."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            corrected_code = corrected_response.choices[0].message.content.strip()
            #print("openai corrected code:",corrected_code)
            execution_result = execute_code(corrected_code, globals_dict)

        return execution_result

    except Exception as e:
        return {"error": str(e)}

def convert_to_xlsx(file_stream, original_filename):
    """
    Converts the uploaded file to XLSX format in-memory and returns a BytesIO buffer.
    """
    try:
        if original_filename.lower().endswith(".csv"):
            df = pd.read_csv(file_stream)
        else:
            df = pd.read_excel(file_stream, engine="openpyxl")

        output_buffer = BytesIO()
        df.to_excel(output_buffer, index=False, engine="openpyxl")

        # Reset to start so future reads begin at correct position
        output_buffer.seek(0)
        return output_buffer

    except Exception as e:
        raise Exception(f"Error converting file to XLSX: {e}")

def clean_response_content(content):
    """Clean response content by stripping Markdown code blocks."""
    if content.startswith("```json") and content.endswith("```"):
        return content[7:-3].strip()  # Strip both markers and whitespace
    if content.startswith("```") and content.endswith("```"):
        return content[3:-3].strip()  # Strip generic code block
    return content.strip()

def retrieve_charts(column_headers, model="gpt-4o"):
    """
    Function to retrieve recommended charts with relevant features.
    Ensures minimum 2 and maximum 4 charts are suggested.
    """
    prompt = f"""
    You are a data visualization assistant. Based on the following dataset columns:

    {column_headers}

    Choose between 2 and 4 suitable chart types from this list: 
    **line, bar, pie, area, radial**.

    Select appropriate features from the dataset columns intuitively based on the type of visualization. 
    Respond strictly in this JSON format, keeping **original column names unchanged**:

    {{
        "chart_type_1": ["feature1", "feature2"],
        "chart_type_2": ["feature3", "feature4"],
        "chart_type_3": ["feature5", "feature6"],
        "chart_type_4": ["feature7", "feature8"]
    }}

    **Important**:
    - Use **exactly** the column names as provided.
    - Ensure the number of charts returned is **between 2 and 4**.
    - Do not include unsupported chart types or alter the dataset features.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data visualization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        if not response or not response.choices or not response.choices[0].message.content:
            return {"error": "Empty response from OpenAI."}

        response_content = clean_response_content(response.choices[0].message.content)

        try:
            charts = json.loads(response_content)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse response as JSON. Details: {str(e)}", "response": response_content}

        if not (2 <= len(charts) <= 4):
            return {"error": "Invalid number of charts returned. Ensure between 2 and 4 charts."}

        return charts

    except Exception as e:
        return {"error": str(e)}

def clean_code_block(code):
    return code.replace("```python", "").replace("```", "").strip()

def safe_convert_to_python_type(value):
    if isinstance(value, (np.int64, np.float64, int, float)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value

def execute_code(code, globals_dict):
    try:
        exec(code, globals_dict)
        result = globals_dict.get("result", {})
        result = {key: safe_convert_to_python_type(value) for key, value in result.items()}  
        return result
    except Exception as e:
        print(e)
        return {"error": str(e)}

@app.route('/')
def home():
    return redirect("https://sourcesmart.ai")

@app.route('/upload_dataset', methods=['POST'])
def upload_data():
    user_id = request.form.get('user_id')
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        # If not found in `request.files`, check `request.data`
        raw_data = request.data
        if not raw_data:
            return jsonify({"error": "Missing file"}), 400
        # Convert raw_data into a BytesIO
        input_buffer = BytesIO(raw_data)
        input_filename = "uploaded_file"
    else:
        # We have a file from `request.files`
        # Read it fully into a BytesIO so it won't be closed
        input_buffer = BytesIO(uploaded_file.read())
        input_filename = uploaded_file.filename or "uploaded_file"

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    def sse_stream():
        try:
            # Pass the in-memory buffer and original filename
            for item in upload_or_replace_file(user_id, input_buffer, input_filename):
                if isinstance(item, dict):
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    yield f"data: {json.dumps({'message': item})}\n\n"
        except Exception as e:
            error_data = {"error": "File upload failed", "details": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(sse_stream(), mimetype='text/event-stream')

@app.route('/download_dataset', methods=['GET'])
def get_files():
    """Downloads the file for a user and sends it directly."""
    user_id = request.args.get('user_id')

    if not user_id:
        logging.warning("Missing user_id in get_files request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        file_url = get_file_by_user_id(user_id)
        if not file_url:
            logging.error(f"No file found for User ID: {user_id}")
            return jsonify({"error": "No file found for this user"}), 404
        else:
            # Option 1: Return the file URL as JSON
            return jsonify({"file_url": file_url}), 200
    except Exception as e:
        logging.error(f"Error retrieving file: {e}")
        return jsonify({"error": "Failed to retrieve file"}), 500
    
@app.route('/get_charts', methods=['GET'])
def get_charts():
    """Retrieve charts for a user's uploaded Excel file without saving locally."""
    user_id = request.args.get('user_id')
    print(user_id)
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        summary = get_summary_by_user_id(user_id)
        file_path=get_file_by_user_id(user_id)
        print("summary")
        if not summary or not file_path:
            return jsonify({"error": "No file found for this user"}), 404
        metrics = asyncio.run(generate_forecast_charts(file_path,summary))
        return metrics, 200
    except Exception as e:
        return jsonify({"error": e}), 500
    
@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    user_id = request.args.get('user_id')
    print(user_id)
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        metrics= get_metrics_by_user_id(user_id)
        if metrics:
            return metrics, 200
        summary = get_summary_by_user_id(user_id)
        #print("summary")
        if not summary:
            return jsonify({"error": "No file found for this user"}), 404
        metrics = asyncio.run(fetch_metrics(summary))
        #print(metrics)
        threading.Thread(target=upload_metrics, args=(user_id, metrics), daemon=True).start()
        return metrics, 200
    except Exception as e:
        return jsonify({"error": e}), 500


@app.route('/get_forecasts', methods=['GET'])
def get_forecasts():
    user_id = request.args.get('user_id')

    if not user_id:
        #logging.warning("Missing user_id in get_forecasts request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        file_url = get_file_by_user_id(user_id)
        if not file_url:
            return jsonify({"error": "No file found for this user"}), 404

        if file_url.endswith(".csv"):
            df = pd.read_csv(file_url)
        else:
            df = pd.read_excel(file_url, engine="openpyxl")

        column_headers = df.columns.tolist()
        sample_data = df.head().to_string()
        forecasts = retrieve_forecasts(column_headers, sample_data, df)
        print(forecasts)
        return jsonify({"forecasts": forecasts}), 200
    except Exception as e:
        #logging.error(f"Error retrieving forecasts: {e}")
        return jsonify({"error": "Failed to retrieve forecasts"}), 500

@app.route('/login', methods=['GET'])
def login():
    """Handle user login and manage potential errors."""
    try:
        user_id = request.args.get('user_id')
        password = request.args.get('password')

        # Print for debugging purposes
        print(f"Request received: user_id={user_id}, password={password}")

        if not user_id or not password:
            print("Missing user_id or password in request")
            return jsonify({"error": "Missing user_id or password"}), 400

        if user_login(user_id, password):
            print("Login successful")
            return jsonify({"key": "true"}), 200
        else:
            print("Invalid credentials")
            return jsonify({"key": "false"}), 401
    except Exception as e:
        # Print the exception for debugging
        print(f"An error occurred during login: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
