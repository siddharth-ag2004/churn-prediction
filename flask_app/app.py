import pandas as pd
from flask import Flask, render_template, jsonify, request
import json
import numpy as np
import folium
from folium.plugins import HeatMap
import random

# Initialize the Flask application
app = Flask(__name__)

def load_and_prepare_data():
    """
    Loads, merges, and preprocesses the auto insurance churn datasets.
    """
    try:
        # Load the datasets
        df_customer = pd.read_csv('../dataset/archive/customer.csv',engine="pyarrow")
        df_address = pd.read_csv('../dataset/archive/address.csv',engine="pyarrow")
        df_termination = pd.read_csv('../dataset/archive/termination.csv',engine="pyarrow")
        df_demographic = pd.read_csv('../dataset/archive/demographic.csv',engine="pyarrow")

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
    Generates a folium map with layer toggles:
    - Retained Customers (default visible)
    - Churned Customers (default visible)
    - High-Value Customers (default hidden)
    Returns HTML for embedding in Flask.
    """
    # Drop rows without lat/lon
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
 
    df_stayed = df[df['CHURN'] == 0]
    df_left = df[df['CHURN'] == 1]
    df_high_value = df[df['CURR_ANN_AMT'] >= df['CURR_ANN_AMT'].quantile(0.9)]  # top 10% premium
 
    # Base map
    m = folium.Map(
        location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
 
    # --- Feature Groups ---
    # Retained and Churned layers will be added directly to map so they show by default
    fg_stayed = folium.FeatureGroup(name='Retained Customers', show=True)
    fg_left = folium.FeatureGroup(name='Churned Customers', show=True)
    fg_high = folium.FeatureGroup(name='High-Value Customers', show=False)
 
    # HeatMaps
    HeatMap(
        df_stayed[['LATITUDE', 'LONGITUDE']],
        radius=8,
        blur=12,
        gradient={0.2: 'green', 0.8: 'lime'},
        min_opacity=0.5
    ).add_to(fg_stayed)
 
    HeatMap(
        df_left[['LATITUDE', 'LONGITUDE']],
        radius=8,
        blur=10,
        gradient={0.2: 'yellow', 0.8: 'red'},
        min_opacity=0.5
    ).add_to(fg_left)
 
    HeatMap(
        df_high_value[['LATITUDE', 'LONGITUDE']],
        radius=3,
        blur=8,
        gradient={0.2: 'blue', 0.8: 'cyan'},
        min_opacity=0.6
    ).add_to(fg_high)
 
    # Add layers to map
    fg_stayed.add_to(m)
    fg_left.add_to(m)
    fg_high.add_to(m)
 
    # Layer Control — collapsed for a cleaner UI
    folium.LayerControl(collapsed=True, position='topright').add_to(m)
 
    return m._repr_html_()
# Load the data once when the app starts
df = load_and_prepare_data()

def predict_churn_probability(customer_data):
    """Simulates a model prediction."""
    # A simple simulation: higher premium and lower tenure increase churn risk
    base_risk = 0.1
    risk_from_premium = (customer_data['CURR_ANN_AMT'] / 2000) * 0.3
    risk_from_tenure = ((3650 - customer_data['DAYS_TENURE']) / 3650) * 0.3
    risk_from_credit = 0 if customer_data['GOOD_CREDIT'] else 0.1
    
    total_risk = base_risk + risk_from_premium + risk_from_tenure + risk_from_credit
    return min(random.uniform(total_risk - 0.05, total_risk + 0.05), 0.95)

def get_shap_explanation(customer_data):
    """Simulates a SHAP explanation for a prediction."""
    # These values represent the positive (red) or negative (green) impact on churn risk
    explanation = [
        {"feature": "Tenure", "value": -random.uniform(0.1, 0.3)},
        {"feature": "Annual Premium", "value": random.uniform(0.05, 0.25)},
        {"feature": "Income", "value": -random.uniform(0.05, 0.15)},
        {"feature": "Good Credit", "value": -random.uniform(0.05, 0.1) if customer_data['GOOD_CREDIT'] else random.uniform(0.05, 0.1)},
        {"feature": "Age", "value": -random.uniform(0.01, 0.05)}
    ]
    random.shuffle(explanation)
    return {"base_value": 0.15, "explanation": explanation}

@app.route('/')
def index():
    """
    Renders the main dashboard page.
    Passes the aggregated data to the template.
    """
    if df.empty:
        return "Error: Could not load dataset. Please check file paths."

    # --- Prepare data for visualizations ---
    churned_df = df[df['CHURN'] == 1]

    # 1. KPI Data
    total_customers = len(df)
    churned_customers = len(churned_df)
    churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
    
    # 2. Donut Chart: Churn vs. Retained
    retained_customers = total_customers - churned_customers
    churn_overview_data = {
        'labels': ['Retained', 'Churned'],
        'data': [int(retained_customers), int(churned_customers)]
    }

    # 3. Bar Chart: Churn by Income Group
    income_bins = [0, 30000, 60000, 100000, 150000, df['INCOME'].max() + 1]
    income_labels = ['< $30k', '$30k-$60k', '$60k-$100k', '$100k-$150k', '$150k+']
    df['INCOME_GROUP'] = pd.cut(df['INCOME'], bins=income_bins, labels=income_labels, right=False)
    churn_by_income = df.groupby('INCOME_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_income_data = {
        'labels': churn_by_income.index.tolist(),
        'churn_rate': (churn_by_income[1] * 100).tolist()
    }
    
    # 4. Pie Chart: Distribution of Churned Customers by Good Credit
    credit_churn_counts = churned_df['GOOD_CREDIT'].value_counts()
    churn_by_credit_pie_data = {
        'labels': ['No Good Credit', 'Has Good Credit'],
        'data': [int(credit_churn_counts.get(False, 0)), int(credit_churn_counts.get(True, 0))]
    }
    
    # 5. Pie Chart: Distribution of Churned Customers by Marital Status
    marital_churn_counts = churned_df.groupby('MARITAL_STATUS').size()
    churn_by_marital_pie_data = {
        'labels': marital_churn_counts.index.tolist(),
        'data': marital_churn_counts.values.tolist()
    }

    # 6. Bar Chart: Churn by Annual Premium
    premium_bins = [0, 500, 1000, 1500, 2000, df['CURR_ANN_AMT'].max() + 1]
    premium_labels = ['< $500', '$500-$1k', '$1k-$1.5k', '$1.5k-$2k', '$2k+']
    df['PREMIUM_GROUP'] = pd.cut(df['CURR_ANN_AMT'], bins=premium_bins, labels=premium_labels, right=False)
    churn_by_premium = df.groupby('PREMIUM_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_premium_data = {
        'labels': churn_by_premium.index.tolist(),
        'churn_rate': (churn_by_premium[1] * 100).tolist()
    }
    
    # 7. Bar Chart: Churn Rate by Customer Tenure
    tenure_bins = [0, 365, 365*3, 365*5, 365*10, df['DAYS_TENURE'].max() + 1]
    tenure_labels = ['0-1 Yr', '1-3 Yrs', '3-5 Yrs', '5-10 Yrs', '10+ Yrs']
    df['TENURE_GROUP'] = pd.cut(df['DAYS_TENURE'], bins=tenure_bins, labels=tenure_labels, right=False)
    churn_by_tenure = df.groupby('TENURE_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_tenure_data = {
        'labels': churn_by_tenure.index.tolist(),
        'churn_rate': (churn_by_tenure[1] * 100).tolist()
    }

    # 8. Pie Chart for Churn Distribution by Home Ownership
    owner_churn_counts = churned_df.groupby('HOME_OWNER').size()
    churn_by_owner_pie_data = {
        'labels': owner_churn_counts.index.tolist(),
        'data': owner_churn_counts.values.tolist()
    }

    # 10. Geospatial Churn Analysis Heatmap
    churn_map_html = generate_churn_heatmap(df)
 
    # Pass all data to the template
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
        churn_map_html=churn_map_html  # <-- pass Folium map HTML
    )
    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles the individual customer prediction page."""
    prediction_data = None
    if request.method == 'POST':
        # 1. Get data from the form
        customer_data = {
            'CURR_ANN_AMT': float(request.form['annual_premium']),
            'DAYS_TENURE': int(request.form['tenure']),
            'INCOME': float(request.form['income']),
            'AGE_IN_YEARS': int(request.form['age']),
            'GOOD_CREDIT': request.form.get('good_credit') == 'on'
        }
        
        # 2. Get simulated prediction and explanation
        probability = predict_churn_probability(customer_data)
        explanation = get_shap_explanation(customer_data)
        
        prediction_data = {
            "probability": probability,
            "explanation": explanation
        }

    return render_template('prediction.html', prediction_data=prediction_data)

if __name__ == '__main__':
    app.run(debug=True)