import re
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time

def search_suppliers(material, location):
    """Search DuckDuckGo for suppliers and return 10 URLs."""
    query = f"{material} supplier in {location}"
    results = []
    
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=10):
            results.append((result['href'], result['title']))
            time.sleep(0.5)  # Add delay to avoid rate limiting

    return results

def extract_footer_info(url):
    """Scrape the footer for contact information."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')

        footer = soup.find('footer')
        if not footer:
            footer = soup.find("div", {"class": "footer"})  # Fallback for some websites

        if footer:
            footer_text = footer.get_text(" ", strip=True)
        else:
            footer_text = ""

        # Extract email addresses
        emails = list(set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", footer_text)))

        # Extract phone numbers
        phones = list(set(re.findall(r"\+?\d{1,4}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,9}", footer_text)))

        # Extract "Contact Us" page link
        contact_link = None
        for a_tag in soup.find_all("a", href=True):
            if "contact" in a_tag.text.lower() or "support" in a_tag.text.lower():
                contact_link = a_tag["href"]
                if contact_link.startswith("/"):
                    contact_link = url.rstrip("/") + contact_link  # Convert relative URL to absolute
                break

        return {
            "emails": emails if emails else "Not found",
            "phones": phones if phones else "Not found",
            "contact_page": contact_link if contact_link else "Not found"
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    material = input("Enter the raw material name: ").strip()
    location = input("Enter the location: ").strip()

    if material and location:
        print("\nSearching for suppliers...\n")
        suppliers = search_suppliers(material, location)

        for i, (url, title) in enumerate(suppliers, start=1):
            print(f"\n[{i}] {title}\nURL: {url}")
            footer_info = extract_footer_info(url)
            print(f"📧 Emails: {footer_info['emails']}")
            print(f"📞 Phones: {footer_info['phones']}")
            print(f"🔗 Contact Page: {footer_info['contact_page']}\n")
    else:
        print("Both material and location must be provided.")