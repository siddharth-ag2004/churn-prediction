import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
import json
import numpy as np
import folium
from folium.plugins import HeatMap
import random
from datetime import datetime, timedelta # New import
from churn_insights import generate_churn_insights
from urllib.parse import quote
from flask import redirect, request
from email_draft import generate_retention_email

import joblib
import shap
from groq import Groq
from flask import session
import os
import re
from agent_backend import chat_with_agent, MOCK_CUSTOMER_DATA, DISCOUNT_LOGS

# --- Agent Configuration ---
# Configure Groq API
load_dotenv() # Ensure env vars are loaded
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Global State for Demo ---
# Moved to agent_backend.py


# --- Add this section to load the model and explainers once when the app starts ---
# This is much more efficient than loading them on every request.
try:
    model = joblib.load('logistic_regression_model.pkl')
    marital_status_encoder = joblib.load('marital_status_encoder.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    print("Model, encoder, and SHAP explainer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or helper files not found. Please run the notebook to generate them.")
    model, marital_status_encoder, explainer = None, None, None

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secret_key_for_demo_session') # Load from env, fallback for dev

# --- Mock News Events for the Insights Page ---
# In a real application, this would come from a database or an external API
MOCK_NEWS_EVENTS = [
    {"date": (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d'), "headline": "Major Competitor Announces 20% Price Cut"},
    {"date": (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'), "headline": "New Federal Regulations Impact Insurance Premiums"},
]


def load_and_prepare_data():
    """
    Loads, merges, and preprocesses the auto insurance churn datasets.
    """
    try:
        # Load the datasets
        df_customer = pd.read_csv('./dataset/archive/customer.csv',engine="pyarrow")
        df_address = pd.read_csv('./dataset/archive/address.csv',engine="pyarrow")
        df_termination = pd.read_csv('./dataset/archive/termination.csv',engine="pyarrow")
        df_demographic = pd.read_csv('./dataset/archive/demographic.csv',engine="pyarrow")

        # Merge the dataframes
        df = pd.merge(df_customer, df_address, on='ADDRESS_ID')
        df = pd.merge(df, df_demographic, on='INDIVIDUAL_ID', how='left')
        df = pd.merge(df, df_termination, on='INDIVIDUAL_ID', how='left')

        # Create the 'CHURN' column
        df['CHURN'] = df['ACCT_SUSPD_DATE'].notna().astype(int)
        
        # --- Data Cleaning and Preparation ---
        df['AGE_IN_YEARS'] = pd.to_numeric(df['AGE_IN_YEARS'], errors='coerce')
        df['INCOME'] = pd.to_numeric(df['INCOME'], errors='coerce')
        df['CURR_ANN_AMT'] = pd.to_numeric(df['CURR_ANN_AMT'], errors='coerce')
        df['DAYS_TENURE'] = pd.to_numeric(df['DAYS_TENURE'], errors='coerce')
        
        df['MARITAL_STATUS'] = df['MARITAL_STATUS'].fillna('Unknown')
        df['GOOD_CREDIT'] = df['GOOD_CREDIT'].fillna(False).astype(bool)
        df['HOME_OWNER'] = df['HOME_OWNER'].fillna('Unknown')
        
        # Drop rows where key numeric data is missing
        df.dropna(subset=['AGE_IN_YEARS', 'INCOME', 'CURR_ANN_AMT', 'DAYS_TENURE'], inplace=True)
        
        return df
    except FileNotFoundError:
        print("Error: One or more dataset files not found.")
        return pd.DataFrame()


def generate_churn_heatmap(df, sample_size=5000):
    """
    Generates a folium map with layer toggles.
    """
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
 
    df_stayed = df[df['CHURN'] == 0]
    df_left = df[df['CHURN'] == 1]
    df_high_value = df[df['CURR_ANN_AMT'] >= df['CURR_ANN_AMT'].quantile(0.9)]
 
    m = folium.Map(location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()], zoom_start=10, tiles='OpenStreetMap')
 
    fg_stayed = folium.FeatureGroup(name='Retained Customers', show=True)
    fg_left = folium.FeatureGroup(name='Churned Customers', show=True)
    fg_high = folium.FeatureGroup(name='High-Value Customers', show=False)
 
    HeatMap(df_stayed[['LATITUDE', 'LONGITUDE']], radius=8, blur=12, gradient={0.2: 'green', 0.8: 'lime'}, min_opacity=0.5).add_to(fg_stayed)
    HeatMap(df_left[['LATITUDE', 'LONGITUDE']], radius=8, blur=10, gradient={0.2: 'yellow', 0.8: 'red'}, min_opacity=0.5).add_to(fg_left)
    HeatMap(df_high_value[['LATITUDE', 'LONGITUDE']], radius=3, blur=5, gradient={0.2: 'blue', 0.8: 'cyan'}, min_opacity=0.6).add_to(fg_high)
 
    fg_stayed.add_to(m)
    fg_left.add_to(m)
    fg_high.add_to(m)
 
    folium.LayerControl(collapsed=True, position='topright').add_to(m)
 
    return m._repr_html_()

def calculate_optimal_discount(customer_data, model, feature_names):
    """
    Calculates the optimal discount to reduce churn probability below 0.5.
    Returns: (suggested_discount, discount_analysis_list)
    """
    suggested_discount = None
    discount_analysis = []
    original_premium = customer_data['curr_ann_amt']
    
    # Try discounts from 1% to 15%
    for discount_pct in range(1, 16, 1):
        sim_customer_data = customer_data.copy()
        sim_customer_data['curr_ann_amt'] = original_premium * (1 - (discount_pct / 100.0))
        x_sim_df = pd.DataFrame([sim_customer_data], columns=feature_names)
        sim_probability = float(model.predict_proba(x_sim_df)[0][1])
        
        discount_analysis.append({
            'discount_pct': discount_pct,
            'new_premium': round(sim_customer_data['curr_ann_amt'], 2),
            'new_probability': sim_probability
        })
        
        if sim_probability < 0.5 and suggested_discount is None:
            suggested_discount = discount_pct
            # We don't break here because we want the full analysis for the chart
            
    if suggested_discount is None:
        suggested_discount = 20
        
    return suggested_discount, discount_analysis

# Load the data once when the app starts
df = load_and_prepare_data()

# ==============================================================================
# --- DASHBOARD PAGE ---
# ==============================================================================
# ==============================================================================
# --- DASHBOARD PAGE ---
# ==============================================================================
@app.route('/')
def landing():
    """Renders the role selection landing page."""
    return render_template('landing.html')

@app.route('/dashboard')
def index():
    """
    Renders the main dashboard page.
    """
    if df.empty:
        return "Error: Could not load dataset. Please check file paths."
    
    churned_df = df[df['CHURN'] == 1]
    total_customers = len(df)
    churned_customers = len(churned_df)
    churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
    retained_customers = total_customers - churned_customers
    churn_overview_data = {'labels': ['Retained', 'Churned'], 'data': [int(retained_customers), int(churned_customers)]}
    
    # Other data preparations... (code omitted for brevity, it's the same as your original)
    income_bins = [0, 30000, 60000, 100000, 150000, df['INCOME'].max() + 1]
    income_labels = ['< $30k', '$30k-$60k', '$60k-$100k', '$100k-$150k', '$150k+']
    df['INCOME_GROUP'] = pd.cut(df['INCOME'], bins=income_bins, labels=income_labels, right=False)
    churn_by_income = df.groupby('INCOME_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_income_data = {'labels': churn_by_income.index.tolist(), 'churn_rate': (churn_by_income[1] * 100).tolist()}
    
    credit_churn_counts = churned_df['GOOD_CREDIT'].value_counts()
    churn_by_credit_pie_data = {'labels': ['No Good Credit', 'Has Good Credit'], 'data': [int(credit_churn_counts.get(False, 0)), int(credit_churn_counts.get(True, 0))]}
    
    marital_churn_counts = churned_df.groupby('MARITAL_STATUS').size()
    churn_by_marital_pie_data = {'labels': marital_churn_counts.index.tolist(), 'data': marital_churn_counts.values.tolist()}
    
    premium_bins = [0, 500, 1000, 1500, 2000, df['CURR_ANN_AMT'].max() + 1]
    premium_labels = ['< $500', '$500-$1k', '$1k-$1.5k', '$1.5k-$2k', '$2k+']
    df['PREMIUM_GROUP'] = pd.cut(df['CURR_ANN_AMT'], bins=premium_bins, labels=premium_labels, right=False)
    churn_by_premium = df.groupby('PREMIUM_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_premium_data = {'labels': churn_by_premium.index.tolist(), 'churn_rate': (churn_by_premium[1] * 100).tolist()}
    
    tenure_bins = [0, 365, 365*3, 365*5, 365*10, df['DAYS_TENURE'].max() + 1]
    tenure_labels = ['0-1 Yr', '1-3 Yrs', '3-5 Yrs', '5-10 Yrs', '10+ Yrs']
    df['TENURE_GROUP'] = pd.cut(df['DAYS_TENURE'], bins=tenure_bins, labels=tenure_labels, right=False)
    churn_by_tenure = df.groupby('TENURE_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_tenure_data = {'labels': churn_by_tenure.index.tolist(), 'churn_rate': (churn_by_tenure[1] * 100).tolist()}
    
    owner_churn_counts = churned_df.groupby('HOME_OWNER').size()
    churn_by_owner_pie_data = {'labels': ['Renter', 'Owner'], 'data': owner_churn_counts.values.tolist()}
    
    churn_map_html = generate_churn_heatmap(df)
 
    return render_template(
        'index.html',
        total_customers=f"{total_customers:,}",
        churned_customers=f"{churned_customers:,}",
        churn_rate=f"{churn_rate:.2f}",
        churn_overview_data=json.dumps(churn_overview_data),
        churn_by_income_data=json.dumps(churn_by_income_data),
        churn_by_premium_data=json.dumps(churn_by_premium_data),
        churn_by_credit_pie_data=json.dumps(churn_by_credit_pie_data),
        churn_by_marital_pie_data=json.dumps(churn_by_marital_pie_data),
        churn_by_tenure_data=json.dumps(churn_by_tenure_data),
        churn_by_owner_pie_data=json.dumps(churn_by_owner_pie_data),
        churn_map_html=churn_map_html
    )

# ==============================================================================
# --- PREDICTION PAGE ---
# ==============================================================================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles the individual customer prediction page."""
    
    # Initialize form_data with default values for the initial GET request
    form_data = {
        'tenure': 500,
        'annual_premium': 1200.50,
        'income': 65000,
        'age': 45,
        'marital_status': 'Married',
        'home_owner': '0',
        'good_credit': 'on',
        'has_children': None # None means it will be unchecked by default
    }

    prediction_data = None
    if request.method == 'POST':
        if not all([model, marital_status_encoder, explainer]):
            return "Error: Model components not loaded.", 500

        try:
            # 1. Capture the raw form data to send back to the template
            form_data = request.form.to_dict()

            # 2. Process and predict as before
            marital_status_str = form_data['marital_status']
            marital_status_encoded = marital_status_encoder.transform([marital_status_str])[0]

            customer_data = {
                'curr_ann_amt': float(form_data['annual_premium']),
                'days_tenure': int(form_data['tenure']),
                'age_in_years': int(form_data['age']),
                'income': float(form_data['income']),
                'has_children': 1 if 'has_children' in form_data else 0,
                'marital_status': marital_status_encoded,
                'home_owner': int(form_data['home_owner']),
                'good_credit': 1 if 'good_credit' in form_data else 0,
            }

            # ... (rest of your prediction and SHAP logic is unchanged) ...

            # Create DataFrame
            feature_names = [
                'curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
                'has_children', 'marital_status', 'home_owner', 'good_credit'
            ]
            x_df = pd.DataFrame([customer_data], columns=feature_names)

            # Get probability and explanation
            probability = float(model.predict_proba(x_df)[0][1])
            shap_values = explainer(x_df)
            
            # Format SHAP explanation
            explanation_list = []
            for i, feature in enumerate(feature_names):
                explanation_list.append({
                    "feature": feature.replace('_', ' ').title(),
                    "value": float(shap_values.values[0][i])
                })
            explanation_json = {
                "base_value": float(shap_values.base_values[0]),
                "explanation": sorted(explanation_list, key=lambda item: abs(item['value']), reverse=True)
            }

            # ... (discount analysis logic is unchanged) ...
            discount_analysis = []
            # Calculate optimal discount
            suggested_discount, discount_analysis = calculate_optimal_discount(customer_data, model, feature_names)

            # Package all data for the template
            prediction_data = {
                "probability": probability,
                "explanation": explanation_json,
                "discount_analysis": discount_analysis,
                "suggested_discount": suggested_discount
            }

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return f"An error occurred: {e}", 400

    # Pass BOTH prediction_data AND form_data to the template
    return render_template('prediction.html', prediction_data=prediction_data, form_data=form_data)


# ==============================================================================
# --- NEW: INSIGHTS PAGE & API ---
# ==============================================================================

def generate_ai_analysis(dates, news_events):
    """
    Simulates an AI model generating insights for selected dates.
    In a real-world scenario, this would involve complex analysis or calls to a GenAI API.
    """
    if not dates:
        return "<p>No dates were provided for analysis.</p>"

    # Find relevant news for the selected dates
    relevant_news = [event for event in news_events if event['date'] in dates]

    # Build the HTML response string
    analysis_html = f"<p>Analysis for {len(dates)} selected date(s):</p>"
    analysis_html += "<ul>"
    
    for date in sorted(dates):
        news_item = next((item for item in relevant_news if item['date'] == date), None)
        if news_item:
            analysis_html += f"<li><strong>On {date}:</strong> The churn spike likely correlates with the news event: <em>'{news_item['headline']}'</em>. This event may have caused customer uncertainty, leading to cancellations.</li>"
        else:
            analysis_html += f"<li><strong>On {date}:</strong> A notable churn spike was observed. No direct external news event was found in our data, suggesting the cause could be internal, such as a recent price change, service outage, or a competitor's marketing campaign.</li>"

    analysis_html += "</ul>"
    
    if len(dates) > 1:
        analysis_html += "<p class='mt-4'><strong>Overall Trend:</strong> The multiple selected churn spikes suggest a period of market volatility or heightened customer sensitivity. Recommend reviewing internal communications and competitor activities around these dates.</p>"
        
    return analysis_html

@app.route('/insights')
def insights():
    """
    Renders the news impact analysis page.
    Generates mock data for the timeline chart.
    """
    # Generate 60 days of sample data for the chart
    base_date = datetime.now()
    dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(60)][::-1]
    
    # Simulate churn rates with some random spikes
    base_churn = 1.5
    churn_rates = [random.uniform(base_churn - 0.5, base_churn + 0.5) for _ in dates]
    # Add spikes on the days we have news events
    for event in MOCK_NEWS_EVENTS:
        if event['date'] in dates:
            idx = dates.index(event['date'])
            churn_rates[idx] = random.uniform(3.5, 4.5) # Make it a clear spike

    timeline_data = {
        "dates": dates,
        "churn_rates": [round(rate, 2) for rate in churn_rates]
    }

    return render_template(
        'insights.html',
        timeline_data=json.dumps(timeline_data),
        news_events=json.dumps(MOCK_NEWS_EVENTS)
    )

@app.route('/api/insights', methods=['POST'])
def get_insights():
    """
    API endpoint that receives dates and returns an AI-generated analysis.
    """
    data = request.get_json()
    selected_dates = data.get('dates', [])
    
    # Call the function to get the analysis for the selected dates
    # We pass the mock news events to the function so it can correlate them
    analysis_text = generate_churn_insights(selected_dates)
    
    print(f"Generated analysis for dates {selected_dates}: {analysis_text}")
    
    return jsonify({'analysis': analysis_text})

@app.route('/generate_email_draft')
def generate_email_draft():
    # Get customer name from query params
    customer_name = request.args.get("name", "Valued Customer")
    
    # Get discount from query params (sent from JS using the prediction_data)
    discount = int(request.args.get("discount", 10))
    
    # Generate email text
    email_text = generate_retention_email(customer_name, discount)
    
    # Optional: set a subject line
    subject_line = f"We're Sorry to See You Go, {customer_name} — A Special Offer from Us"
    
    # Encode for mailto
    mailto_link = (
        f"mailto:?subject={quote(subject_line)}"
        f"&body={quote(email_text)}"
    )
    
    # Redirect to mail client
    return redirect(mailto_link)

# ==============================================================================
# --- AGENT CHAT ROUTES ---
# ==============================================================================

# ==============================================================================
# --- AGENT CHAT ROUTES ---
# ==============================================================================

@app.route('/customer')
def customer_portal():
    """Renders the customer portal with chat interface."""
    # Clear history on new page load
    session['chat_history'] = []
    session['discount_applied'] = False # Reset discount flag
    
    # Mock customer data for the demo (Alex Johnson)
    # Use global data so updates persist
    customer_data = MOCK_CUSTOMER_DATA
    
    # Calculate optimal discount for this customer
    feature_names = [
        'curr_ann_amt', 'days_tenure', 'age_in_years', 'income',
        'has_children', 'marital_status', 'home_owner', 'good_credit'
    ]
    
    try:
        if all([model, explainer]):
             suggested_discount, _ = calculate_optimal_discount(customer_data, model, feature_names)
             session['target_discount'] = suggested_discount
             print(f"DEBUG: Calculated target discount for customer: {suggested_discount}%")
        else:
             session['target_discount'] = 15 # Fallback
             print("DEBUG: Model not loaded, using fallback discount 15%")
    except Exception as e:
        print(f"Error calculating discount for customer portal: {e}")
        session['target_discount'] = 15 # Fallback

    return render_template('customer_portal.html')

@app.route('/admin/discounts')
def discount_logs():
    """Renders the stakeholder discount logs."""
    return render_template('discount_logs.html', logs=DISCOUNT_LOGS)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Retrieve history from session
    history = session.get('chat_history', [])
    
    try:
        # Call the backend agent logic
        agent_text, new_premium = chat_with_agent(user_message, history, session)
        
        # Update session history
        history.append({'role': 'user', 'text': user_message})
        history.append({'role': 'model', 'text': agent_text})
        session['chat_history'] = history
        
        response_data = {'response': agent_text}
        if new_premium:
            response_data['new_premium'] = new_premium
            
        return jsonify(response_data)

    except Exception as e:
        print(f"Agent Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)