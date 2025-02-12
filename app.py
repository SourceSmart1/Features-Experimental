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

# Load environment variables (make sure you have a .env file with SERPER_API_KEY)
load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

# --------------------- Streamlit Page Config & Custom Styling ---------------------
st.set_page_config(page_title="Supplier Search", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1>Supplier Search</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Find suppliers for your material needs quickly and easily.</p>", unsafe_allow_html=True)

# --------------------- Sidebar Inputs ---------------------
with st.sidebar:
    st.markdown("### Enter Search Details")
    # 1. Material Name (text input)
    material_name = st.text_input("Material Name", "Aluminum")
    
    # 2. Country (dropdown)
    countries = [
        "Estonia", "United States", "United Kingdom", "Germany", "France", 
        "Spain", "Italy", "Canada", "Australia", "Netherlands", "Sweden", "Finland"
    ]
    country = st.selectbox("Country", countries, index=countries.index("Estonia"))
    
    # 3. City (text input, optional)
    city = st.text_input("City (Optional)", "Tallin")
    
    # Construct the query string
    query = f"{material_name} supplier in {city}, {country}"
    st.markdown("**Generated Query:**")
    st.markdown(f"`{query}`")
    
    # Trigger for the search
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
    """
    Extracts and returns a list of email addresses from a text string.
    """
    try:
        # Regex for emails; matches a variety of TLD lengths.
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html_text)
        return list(set(emails))  # Remove duplicates
    except Exception:
        return []

def extract_contact_email(url: str):
    """
    Given a website URL, this function tries to extract a contact email.
    First it checks the home page; if none is found, it looks for a "contact" link
    and attempts to extract an email from that page.
    Returns the first email found or None.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            text = soup.get_text(separator=" ", strip=True)
            emails = get_email(text)
            if emails:
                return emails[0]  # Return first email found
            # If no email on home page, look for a "contact" link (case-insensitive)
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
        # Build initial DataFrame with supplier names and websites
        rows = []
        for item in result["organic"]:
            name = item.get("title", "N/A")
            website = item.get("link", "N/A")
            rows.append({"Name": name, "Website": website})
        df = pd.DataFrame(rows)
        
        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
        st.success("Initial search complete!")
        st.table(df)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ---------------------
        # Now loop through each website to extract the contact email.
        # Only include rows with a found email.
        # ---------------------
        final_rows = []
        progress_bar = st.progress(0)
        total = len(df)
        for idx, row in df.iterrows():
            website = row["Website"]
            contact_email = extract_contact_email(website)
            if contact_email:  # Only keep row if email is found
                row["Contact Email"] = contact_email
                final_rows.append(row)
            progress_bar.progress((idx + 1) / total)
        
        if final_rows:
            final_df = pd.DataFrame(final_rows)
            st.markdown("<div class='result-section'>", unsafe_allow_html=True)
            st.markdown("<h2>Final Results with Contact Emails</h2>", unsafe_allow_html=True)
            st.table(final_df)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("No contact emails found on any of the supplier websites.")
    else:
        st.error("No search results found.")
