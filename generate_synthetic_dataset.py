import os
import datetime
import requests
import pandas as pd
import numpy as np
import random
import google.generativeai as genai
import ast
from dotenv import load_dotenv
from tqdm import tqdm

# fetch_news_content and select_major_events_with_gemini functions are unchanged...
def fetch_news_content(api_key, start_date, end_date, keyword):
    """
    Fetches the full text content of news articles from MediaStack API.
    """
    print(f"Fetching raw news intelligence for keyword '{keyword}'...")
    API_ENDPOINT = "http://api.mediastack.com/v1/news"
    all_articles_text = ""
    params = {
        'access_key': api_key,
        'keywords': keyword,
        'date': f'{start_date},{end_date}',
        'languages': 'en',
        'sort': 'published_desc',
        'limit': 100
    }
    try:
        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()
        api_response = response.json()
        articles = api_response.get('data', [])
        print(f"Found {len(articles)} articles to analyze.")
        for article in articles:
            all_articles_text += f"Date: {article.get('published_at', '')}\nTitle: {article.get('title', '')}\nDescription: {article.get('description', '')}\n\n"
        return all_articles_text
    except requests.exceptions.RequestException as e:
        print(f"    An error occurred while fetching news: {e}")
        return None

def select_major_events_with_gemini(api_key, news_context):
    """
    Uses Gemini to analyze news and select the 3-4 most significant event dates.
    """
    print("Sending news intelligence to Gemini for major event selection...")
    if not news_context:
        print("No news context to analyze.")
        return []
    
    try:
        genai.configure(api_key=api_key)
        # Corrected model name
        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = f"""
        **Role:** You are an expert insurance industry analyst.

        **Task:** I have provided a collection of news articles. Analyze them to identify the 3 to 4 most significant and distinct news events that would cause widespread consumer concern about rising insurance rates. A significant event is one that signals a major market shift, a large-scale disaster, or a major economic announcement.

        **Output Format Constraint:** Your response MUST BE ONLY a valid Python list of strings, where each string is a date in 'YYYY-MM-DD' format corresponding to the most impactful event dates. For example: `['2025-09-15', '2025-10-01', '2025-10-10']`. Do not include any other text, explanation, or markdown formatting.

        **News Context:**
        ---
        {news_context}
        ---
        """

        response = model.generate_content(prompt)
        cleaned_text = response.text.replace("```python", "").replace("```", "").strip()
        major_event_date_strings = ast.literal_eval(cleaned_text)
        major_event_dates = [datetime.datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in major_event_date_strings]
        
        print(f"Gemini identified {len(major_event_dates)} major event dates.")
        return major_event_dates

    except (SyntaxError, ValueError) as e:
        print(f"Error parsing Gemini's response. The response may not be in the expected format. Error: {e}")
        print(f"Raw Gemini Response:\n---\n{response.text}\n---")
        return []
    except Exception as e:
        print(f"An error occurred while communicating with the Gemini API: {e}")
        return []

# =================================================================================
# === THIS IS THE MODIFIED FUNCTION WITH THE DECAYING PROBABILITY LOGIC ===
# =================================================================================
def generate_synthetic_dataset(original_df, config):
    """
    Generates a new synthetic dataset with a decaying churn probability around events.
    """
    print("Creating a copy of the dataset for simulation...")
    df_new = original_df.copy()
    sim_end_date, sim_start_date = config['SIMULATION_END_DATE'], config['SIMULATION_START_DATE']
    high_churn_event_dates = config['HIGH_CHURN_EVENT_DATES']
    base_churn_prob = config['BASE_CHURN_PROBABILITY']
    max_event_increase = config['EVENT_CHURN_PROBABILITY_INCREASE'] # This is now the MAX increase
    event_window_days = config['EVENT_WINDOW_DAYS']
    event_window_delta = datetime.timedelta(days=event_window_days)
    new_churn_status, new_susp_dates, new_orig_dates, new_tenures = [], [], [], []

    print(f"Simulating {config['SIMULATION_DURATION_DAYS']} days with decaying event-driven churn...")
    for _, row in tqdm(df_new.iterrows(), total=df_new.shape[0]):
        days_in_past = random.randint(365, 365*10)
        cust_orig_date = sim_end_date - datetime.timedelta(days=days_in_past)
        new_orig_dates.append(cust_orig_date)
        
        total_sim_days = (sim_end_date - sim_start_date).days
        random_day_in_sim = random.randint(0, total_sim_days)
        check_date = sim_start_date + datetime.timedelta(days=random_day_in_sim)
        
        current_churn_prob = base_churn_prob
        
        for event_date in high_churn_event_dates:
            # Check if the customer's date is within the event window
            if abs(check_date - event_date) <= event_window_delta:
                # --- NEW DECAY LOGIC ---
                # Calculate how many days away from the event this check_date is
                days_difference = abs((check_date - event_date).days)
                
                # Calculate the decay factor (1.0 on event day, 0.0 at the edge of the window)
                decay_factor = (event_window_days - days_difference) / event_window_days
                
                # Calculate the dynamic probability increase
                dynamic_increase = max_event_increase * decay_factor
                
                # Add the decayed probability to the current probability
                current_churn_prob += dynamic_increase
                
                # An event has been found and applied, no need to check others
                break
        
        if random.random() < current_churn_prob:
            new_churn_status.append(1)
            acct_suspd_date = check_date
            new_susp_dates.append(acct_suspd_date)
            new_tenures.append((acct_suspd_date - cust_orig_date).days)
        else:
            new_churn_status.append(0)
            new_susp_dates.append(pd.NaT)
            new_tenures.append((sim_end_date - cust_orig_date).days)
            
    print("Updating dataframe with new synthetic data...")
    # Using direct assignment which is safer and more efficient than df.update
    df_new['Churn'] = new_churn_status
    df_new['cust_orig_date'] = new_orig_dates
    df_new['acct_suspd_date'] = new_susp_dates
    df_new['days_tenure'] = new_tenures
    
    return df_new

def main():
    ORIGINAL_FILENAME = './dataset/archive/autoinsurance_churn.csv'
    OUTPUT_FILENAME = 'gemini_selected_event_data_90days.csv'

    load_dotenv()
    mediastack_api_key = os.getenv("MEDIASTACK_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not (mediastack_api_key and gemini_api_key):
        print("ERROR: API keys not found in .env file. Please check for MEDIASTACK_API_KEY and GEMINI_API_KEY.")
        return

    # --- CONFIGURATION ---
    SIMULATION_DURATION_DAYS = 90
    SIMULATION_END_DATE = datetime.date.today()
    SIMULATION_START_DATE = SIMULATION_END_DATE - datetime.timedelta(days=SIMULATION_DURATION_DAYS)
    EVENT_WINDOW_DAYS = 3
    # This is now the MAXIMUM increase on the event day itself
    EVENT_CHURN_PROBABILITY_INCREASE = 0.05 
    EVENT_DISCOVERY_KEYWORD = "insurance rates"

    print("="*70)
    print("Gemini-Selected Event-Driven Synthetic Data Generation Pipeline")
    print("="*70)

    # Phase 1 & 2: Fetch news and have Gemini select major events
    news_text = fetch_news_content(mediastack_api_key, SIMULATION_START_DATE.strftime("%Y-%m-%d"), SIMULATION_END_DATE.strftime("%Y-%m-%d"), EVENT_DISCOVERY_KEYWORD)
    major_event_dates = select_major_events_with_gemini(gemini_api_key, news_text)

    if not major_event_dates:
        print("\nGemini did not identify any major events. Cannot proceed. Exiting.")
        return
        
    print("\nGemini-Selected Major Event Dates for Simulation:")
    for date in major_event_dates:
        print(f"  - {date.strftime('%Y-%m-%d')}")

    # Phase 3: Generate the dataset using only these major events
    print(f"\nLoading original dataset from: {ORIGINAL_FILENAME}")
    try:
        df_original = pd.read_csv(ORIGINAL_FILENAME)
    except FileNotFoundError:
        print(f"ERROR: Original dataset not found at '{ORIGINAL_FILENAME}'.")
        return

    base_churn_rate = df_original['Churn'].value_counts(normalize=True)[1]
    config = {
        'SIMULATION_DURATION_DAYS': SIMULATION_DURATION_DAYS, 'SIMULATION_END_DATE': SIMULATION_END_DATE,
        'SIMULATION_START_DATE': SIMULATION_START_DATE, 'HIGH_CHURN_EVENT_DATES': major_event_dates,
        'BASE_CHURN_PROBABILITY': base_churn_rate, 'EVENT_CHURN_PROBABILITY_INCREASE': EVENT_CHURN_PROBABILITY_INCREASE,
        'EVENT_WINDOW_DAYS': EVENT_WINDOW_DAYS
    }
    df_synthetic = generate_synthetic_dataset(df_original, config)

    # Phase 4: Save and Report
    print(f"\nSaving new synthetic dataset to: {OUTPUT_FILENAME}")
    df_synthetic.to_csv(OUTPUT_FILENAME, index=False)
    print("\n--- Analysis Complete ---")
    print(f"Original churn rate: {base_churn_rate:.2%}")
    new_churn_rate = df_synthetic['Churn'].value_counts(normalize=True).get(1, 0)
    print(f"New synthetic churn rate: {new_churn_rate:.2%}")
    print("The new churn rate is higher due to concentrated churn clusters around Gemini-selected major event dates.")
    print("="*70)

if __name__ == "__main__":
    main()