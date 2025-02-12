import streamlit as st
import http.client
import json
import pandas as pd
import os
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from dotenv import load_dotenv

# Load environment variables (ensure you have a .env file with SERPER_API_KEY)
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

# --------------------- Streamlit Page Config & Styling ---------------------
st.set_page_config(page_title="Supplier Search", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1>Supplier Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Find suppliers for your material needs quickly and easily.</p>", unsafe_allow_html=True)

# --------------------- Sidebar Inputs ---------------------
with st.sidebar:
    st.markdown("### Enter Search Details")
    material_name = st.text_input("Material Name", "Aluminum")
    countries = [
        "Estonia", "United States", "United Kingdom", "Germany", "France", 
        "Spain", "Italy", "Canada", "Australia", "Netherlands", "Sweden", "Finland"
    ]
    country = st.selectbox("Country", countries, index=countries.index("Estonia"))
    city = st.text_input("City (Optional)", "Tallin")
    query = f"{material_name} supplier in {city}, {country}"
    st.markdown("**Generated Query:**")
    st.markdown(f"`{query}`")
    search_clicked = st.button("Search")

# --------------------- Function to Call the Serper API ---------------------
def perform_search(query: str) -> dict:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query,
        "location": "Estonia",
        "gl": "ee",
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

# --------------------- Email Extraction Helpers ---------------------
def get_email(html_text: str):
    """Extracts emails from text using regex."""
    try:
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html_text)
        return list(set(emails))  # Remove duplicates
    except Exception:
        return []

def extract_contact_email(url: str):
    """
    Given a website URL, this function attempts to extract a contact email.
    It first checks the home page, then looks for a 'contact' link if no email is found.
    Returns the first found email or None.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            text = soup.get_text(separator=" ", strip=True)
            emails = get_email(text)
            if emails:
                return emails[0]
            # Look for a 'contact' link if no email is found on the home page
            contact_link = soup.find('a', string=re.compile('contact', re.IGNORECASE))
            if contact_link and 'href' in contact_link.attrs:
                contact_url = urljoin(url, contact_link['href'])
                response_contact = requests.get(contact_url, timeout=10)
                if response_contact.status_code == 200:
                    soup_contact = BeautifulSoup(response_contact.text, 'lxml')
                    text_contact = soup_contact.get_text(separator=" ", strip=True)
                    emails_contact = get_email(text_contact)
                    if emails_contact:
                        return emails_contact[0]
        return None
    except Exception:
        return None

# --------------------- Main Section ---------------------
if search_clicked:
    with st.spinner("Searching..."):
        result = perform_search(query)
    
    if "organic" in result:
        # Build initial list of supplier dictionaries (without displaying an initial table)
        suppliers = []
        for item in result["organic"]:
            name = item.get("title", "N/A")
            website = item.get("link", "N/A")
            suppliers.append({"Name": name, "Website": website})
        df = pd.DataFrame(suppliers)
        live_table_placeholder = st.empty()  # Placeholder for the live-updating table
        progress_bar = st.progress(0)
        final_rows = []
        total = len(df)
        
        # Loop through each supplier and update the table live.
        for idx, row in df.iterrows():
            website = row["Website"]
            contact_email = extract_contact_email(website)
            if contact_email:
                row["Contact Email"] = contact_email
                final_rows.append(row)
            progress_bar.progress((idx + 1) / total)
            # Update the live table with the current results
            live_table_placeholder.table(pd.DataFrame(final_rows))
        
        if not final_rows:
            st.error("No contact emails found on any of the supplier websites.")
    else:
        st.error("No search results found.")
