import streamlit as st
import http.client
import json
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv
from openai import OpenAI  # Using new client instantiation

# Load environment variables (ensure you have a .env file with SERPER_API_KEY and OPENAI_API_KEY)
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Instantiate the OpenAI client (v1.0.0 style)
client = OpenAI(api_key=openai_api_key)

# --------------------- Streamlit Page Config & Styling ---------------------
st.set_page_config(page_title="PricePredictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1>PricePredictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p>Get current pricing info and AI-driven predictions for your material needs.</p>",
    unsafe_allow_html=True,
)

# --------------------- Sidebar Inputs ---------------------
# Define a mapping from country name to its Google GL code
countries = {
    "Estonia": "ee",
    "United States": "us",
    "United Kingdom": "uk",
    "Germany": "de",
    "France": "fr",
    "Spain": "es",
    "Italy": "it",
    "Canada": "ca",
    "Australia": "au",
    "Netherlands": "nl",
    "Sweden": "se",
    "Finland": "fi",
    "India": "in",
}

with st.sidebar:
    st.markdown("### Enter Search Details")
    material = st.text_input("Material Name", "Aluminum")
    country = st.selectbox("Country", list(countries.keys()), index=list(countries.keys()).index("Estonia"))
    search_clicked = st.button("Search")

# --------------------- SERPER API Search Function ---------------------
def perform_search(query: str, country_name: str, gl_code: str) -> dict:
    """
    Calls the SERPER API using the provided query, country (as location),
    and the country code (gl).
    """
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": query,
            "location": country_name,  # full country name (e.g., "India")
            "gl": gl_code,             # e.g., "in"
            "num": 30
        })
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        st.error(f"Error during search: {e}")
        return {}

# --------------------- Collecting Source Information ---------------------
def compile_sources(search_result: dict, source_type: str) -> str:
    """
    Build a string that lists the sources from the search result.
    source_type can be "organic" for regular search results or "news".
    """
    sources_info = ""
    if source_type in search_result:
        for item in search_result[source_type]:
            title = item.get("title", "No Title")
            snippet = item.get("snippet", "No Snippet")
            link = item.get("link", "No URL")
            sources_info += f"- **{title}**\n  - Snippet: {snippet}\n  - [Link]({link})\n\n"
    return sources_info.strip()

# --------------------- Main Section ---------------------
if search_clicked:
    with st.spinner("Processing your request..."):
        progress_bar = st.progress(0)
        gl_code = countries[country]

        # ----- Step 1: Get current price search results -----
        progress_bar.progress(10)
        price_query = f"{material} current price {country}"
        price_result = perform_search(price_query, country, gl_code)
        price_sources = compile_sources(price_result, "organic")
        
        # Use a default in case we don't extract any price directly
        current_price_fallback = "20$ per ton"
        extracted_price = current_price_fallback
        if "organic" in price_result and price_result["organic"]:
            snippet = price_result["organic"][0].get("snippet", "")
            match = re.search(r"(\$?\d+(?:\.\d+)?\s*\$?)", snippet)
            if match:
                extracted_price = match.group(0)
        
        progress_bar.progress(40)
        
        # ----- Step 2: Get recent news search results -----
        news_query = f"{material} price news {country}"
        news_result = perform_search(news_query, country, gl_code)
        news_sources = compile_sources(news_result, "news")
        progress_bar.progress(70)
        
        # ----- Step 3: Prepare all sources for GPT prompt -----
        all_sources = "### Price Sources\n" + (price_sources if price_sources else "No price sources found.") + "\n\n"
        all_sources += "### News Sources\n" + (news_sources if news_sources else "No news sources found.")

        # ----- Step 4: Use OpenAI to predict next month's price with reasoning -----
        prompt = (
            f"You are an expert market analyst. Based on the following search results, "
            f"please determine the current price for {material} in {country} and predict the price for next month. "
            f"Also provide a detailed explanation citing the sources provided.\n\n"
            f"**Extracted Current Price (fallback):** {extracted_price}\n\n"
            f"**Sources:**\n{all_sources}\n\n"
            f"Please output your answer as a JSON object with the following keys:\n"
            f"- `current_price`: Include the price, currency, and unit (e.g., '2639.42 USD per tonne')\n"
            f"- `predicted_price`\n"
            f"- `reason`\n"
            f"Ensure that the JSON is valid."
        )

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0
            )
            response_content = completion.choices[0].message.content.strip()
            
            # Remove markdown formatting if present (e.g., triple backticks and language tags)
            if response_content.startswith("```"):
                response_content = response_content.strip("`")
                if response_content.lower().startswith("json"):
                    response_content = response_content[4:].strip()
            
            # Parse the cleaned JSON response
            prediction = json.loads(response_content)
            current_price = prediction.get("current_price", extracted_price)
            predicted_price = prediction.get("predicted_price", "N/A")
            reason = prediction.get("reason", "N/A")
        except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
            current_price = extracted_price
            predicted_price = "N/A"
            reason = "Could not generate prediction due to an error."

    # Helper function to format a price dictionary or string
    def format_price(price):
        if isinstance(price, dict):
            price_val = price.get("price", "")
            currency = price.get("currency", "")
            unit = price.get("unit", "")
            return f"{price_val} {currency} {unit}"
        return price

    # ----- Display the Results -----
    st.markdown("## Results")
    st.markdown(f"**Current Price:** {format_price(current_price)}")
    st.markdown(f"**Predicted Price Next Month:** {format_price(predicted_price)}")
    st.markdown(f"**Reason:** {reason}")

    # Optionally, display the entire JSON nicely:
    with st.expander("View Raw JSON Response"):
        st.json(prediction)


        
        progress_bar.progress(100)
    