import os
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv


def fetch_news(api_key, start_date, end_date, keywords, max_articles_per_keyword=30):
    """
    Fetches news articles from MediaStack API for a given date range and keywords.
    Returns a list of dictionaries with title, content, date, source, and URL.
    """
    API_ENDPOINT = "http://api.mediastack.com/v1/news"
    all_articles = []

    for keyword in keywords:
        params = {
            'access_key': api_key,
            'keywords': keyword,
            'date': f'{start_date},{end_date}',
            'languages': 'en',
            'sort': 'published_desc',
            'limit': max_articles_per_keyword
        }

        try:
            response = requests.get(API_ENDPOINT, params=params)
            response.raise_for_status()
            api_response = response.json()

            for article in api_response.get('data', []):
                all_articles.append({
                    'title': article.get('title', ''),
                    'content': article.get('description', ''),
                    'source': article.get('source', 'Unknown'),
                    'url': article.get('url', 'N/A'),
                    'date': article.get('published_at', ''),
                    'region': 'USA'
                })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for '{keyword}': {e}")

    return all_articles


def analyze_with_gemini(api_key, articles, peak_dates):
    """
    Uses Gemini to correlate churn spike dates with nearby news.
    Returns a text summary of insights.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')

    news_text = ""
    for art in articles:
        news_text += (
            f"Date: {art['date']}\n"
            f"Title: {art['title']}\n"
            f"Content: {art['content']}\n"
            f"URL: {art['url']}\n---\n"
        )

    prompt = f"""
    You are an insurance churn analyst.

    I will give you:
    - A list of churn spike dates (dates when many customers left)
    - Recent insurance-related news articles

    Your task:
    🔹 For each churn spike date, check if any nearby news (within ~5–10 days) might explain the spike.
    🔹 Output concise, dashboard-friendly insights such as:
       "High churn around early October could likely be explained by news on 2025-10-06 about proposed FAIR Plan rate hikes in California — (https://example.com)"
    🔹 Always include the article's publication date and its URL in parentheses at the end.
    🔹 Keep explanations short (1–2 sentences per date).
    🔹 Only explain dates where you find a reasonable match.
    🔹 Maintain a professional tone suitable for executive reporting.

    Churn Spike Dates:
    {peak_dates}

    News Context:
    {news_text}
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


def generate_churn_insights(peak_dates):
    """
    Main callable function:
    Takes a list of churn spike dates and returns Gemini-generated churn insights.
    """
    load_dotenv()
    news_api_key = os.getenv("MEDIASTACK_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not news_api_key or not gemini_api_key:
        return "ERROR: Missing API keys. Check .env for MEDIASTACK_API_KEY and GEMINI_API_KEY."

    today = datetime.date.today()
    six_months_ago = today - datetime.timedelta(days=180)
    start_date = six_months_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    KEYWORDS = ["insurance rates"]

    articles = fetch_news(news_api_key, start_date, end_date, KEYWORDS)

    if not articles:
        return "No relevant news articles found. Try adjusting date range or API plan."

    insights = analyze_with_gemini(gemini_api_key, articles, peak_dates)

    return insights or "No insights generated."


# Run standalone for testing
if __name__ == "__main__":
    sample_dates = ["2025-07-08", "2025-07-28", "2025-10-06"]
    print("\n🔍 Dashboard-Ready Churn Insights Based on Peak Dates:\n")
    print(generate_churn_insights(sample_dates))
