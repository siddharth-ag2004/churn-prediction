import os
import datetime
import requests # Import the requests library
import google.generativeai as genai
from dotenv import load_dotenv

def fetch_news(api_key, start_date, end_date, keywords):
    """
    Fetches news articles from MediaStack API for a given date range and keywords.

    Args:
        api_key (str): Your MediaStack API key.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        keywords (list): A list of keywords to search for.

    Returns:
        str: A single string containing the concatenated content of all found articles,
             or None if no articles are found.
    """
    print(f"Fetching news from {start_date} to {end_date} using MediaStack...")
    
    # MediaStack API endpoint
    # NOTE: The free plan uses HTTP. If you have a paid plan, change this to https.
    API_ENDPOINT = "http://api.mediastack.com/v1/news"
    all_articles_content = []

    for keyword in keywords:
        print(f"  - Searching for keyword: '{keyword}'")
        
        # Parameters for the MediaStack API request
        params = {
            'access_key': api_key,
            'keywords': keyword,
            'date': f'{start_date},{end_date}',
            'languages': 'en',
            'sort': 'published_desc',
            'limit': 30
        }

        try:
            # Make the GET request to the API
            response = requests.get(API_ENDPOINT, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            api_response = response.json()

            for article in api_response.get('data', []):
                title = article.get('title', '')
                content = article.get('description', '')
                source = article.get('source', 'Unknown Source')
                if content:
                    full_text = f"Source: {source}\nTitle: {title}\nContent: {content}\n"
                    all_articles_content.append(full_text)

        except requests.exceptions.RequestException as e:
            print(f"    An error occurred while fetching news for '{keyword}': {e}")
            print("    This could be due to network issues or an invalid API key/plan.")
            # Check for common MediaStack free plan errors
            if "historical_searches_not_supported" in str(e):
                 print("    Your MediaStack plan does not support historical searches.")

    if not all_articles_content:
        return None

    print(f"\nFound a total of {len(all_articles_content)} relevant article snippets.")
    return "\n--- ARTICLE SEPARATOR ---\n".join(all_articles_content)


def analyze_with_gemini(api_key, news_context, start_date, end_date):
    """
    Analyzes news context using the Gemini API to identify trends and churn hotspots.
    (This function remains unchanged as it is independent of the news source)
    """
    print("Sending news context to Gemini for analysis...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = f"""
        **Role:** You are an expert risk analyst for the US insurance industry.

        **Task:** I have provided a collection of news articles from the period of {start_date} to {end_date}.
        Your primary task is to analyze this information and predict where auto and home insurance churn is most likely to increase.

        **Instructions:**
        1.  First, briefly identify 2-3 overarching themes from the news content that are pressuring consumers (e.g., Rising Premiums, Climate Risk, Post-Pandemic Vehicle Costs).
        2.  Next, perform a "Churn Hotspot Analysis".
        3.  List the specific US states or large regions (e.g., "The Gulf Coast") that are at a higher risk for increased customer churn.
        4.  For each location identified, provide a brief, bullet-pointed explanation linking the news trends directly to why churn is likely to increase there.
        5.  If a trend is national and affects all customers, state this explicitly.
        6.  Base your analysis strictly on the provided news context. Do not use outside knowledge.

        **News Context:**
        ---
        {news_context}
        ---
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while communicating with the Gemini API: {e}"


def main():
    """Main function to run the churn analysis pipeline."""
    load_dotenv()
    # --- UPDATED TO USE MEDIASTACK_API_KEY ---
    news_api_key = os.getenv("MEDIASTACK_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not news_api_key or not gemini_api_key:
        print("ERROR: API keys not found. Please check your .env file for MEDIASTACK_API_KEY and GEMINI_API_KEY.")
        return

    # Configuration for the last 6 months (180 days)
    today = datetime.date.today()
    six_months_ago = today - datetime.timedelta(days=180)
    START_DATE = six_months_ago.strftime("%Y-%m-%d")
    END_DATE = today.strftime("%Y-%m-%d")

    KEYWORDS = [
        "insurance rates"
    ]

    print("="*70)
    print("Starting Geospatial Churn Analysis Pipeline (using MediaStack)")
    print("="*70)

    # Phase 1: Fetch News
    news_articles_text = fetch_news(news_api_key, START_DATE, END_DATE, KEYWORDS)

    if not news_articles_text:
        print("\nCould not retrieve news articles. This may be due to the limitations of your MediaStack plan (e.g., historical access). Exiting.")
        return

    # Phase 2: Analyze with Gemini
    analysis_report = analyze_with_gemini(gemini_api_key, news_articles_text, START_DATE, END_DATE)

    # Phase 3: Display Report
    print("\n\n" + "="*70)
    print("Churn Hotspot Analysis Report Generated by Gemini")
    print("="*70)
    print(analysis_report)
    print("="*70)

if __name__ == "__main__":
    main()