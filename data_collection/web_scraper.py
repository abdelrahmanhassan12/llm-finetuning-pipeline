import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from common elements like paragraphs, headings, etc.
        text_content = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        return text_content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

if __name__ == "__main__":
    # Example usage (will be replaced by actual usage in orchestration)
    example_url = "https://www.example.com"
    scraped_text = scrape_website(example_url)
    if scraped_text:
        print(f"Scraped text from {example_url}:\n{scraped_text[:500]}...")


