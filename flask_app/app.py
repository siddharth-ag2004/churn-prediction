import pandas as pd
from flask import Flask, render_template, jsonify
import json
import numpy as np
import folium
from folium.plugins import HeatMap

# Initialize the Flask application
app = Flask(__name__)

def load_and_prepare_data():
    """
    Loads, merges, and preprocesses the auto insurance churn datasets.
    """
    try:
        # Load the datasets
        df_customer = pd.read_csv('../dataset/archive/customer.csv')
        df_address = pd.read_csv('../dataset/archive/address.csv')
        df_termination = pd.read_csv('../dataset/archive/termination.csv')
        df_demographic = pd.read_csv('../dataset/archive/demographic.csv')

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
    Generates a folium HeatMap showing churned vs retained customers.
    Returns the HTML representation of the map.
    """
    # Drop rows without lat/lon
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    df_stayed = df[df['CHURN'] == 0]
    df_left = df[df['CHURN'] == 1]

    # Base map — zoomed in more
    m = folium.Map(
        location=[df['LATITUDE'].mean(), df['LONGITUDE'].mean()],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # HeatMap for customers who stayed
    HeatMap(
        df_stayed[['LATITUDE', 'LONGITUDE']],
        radius=8,
        blur=12,
        gradient={0.2: 'green', 0.8: 'lime'},
        min_opacity=0.5
    ).add_to(m)

    # HeatMap for customers who left
    HeatMap(
        df_left[['LATITUDE', 'LONGITUDE']],
        radius=8,
        blur=10,
        gradient={0.2: 'yellow', 0.8: 'red'},
        min_opacity=0.5
    ).add_to(m)

    return m._repr_html_()


# Load the data once when the app starts
df = load_and_prepare_data()

@app.route('/')
def index():
    """
    Renders the main dashboard page.
    Passes the aggregated data to the template.
    """
    if df.empty:
        return "Error: Could not load dataset. Please check file paths."

    # --- Prepare data for visualizations ---

    # 1. KPI Data
    total_customers = len(df)
    churned_customers = df['CHURN'].sum()
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
    churned_df = df[df['CHURN'] == 1]
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

    # --- NEW PLOTS DATA ---
    
    # 7. NEW Bar Chart: Churn Rate by Customer Tenure
    tenure_bins = [0, 365, 365*3, 365*5, 365*10, df['DAYS_TENURE'].max() + 1]
    tenure_labels = ['0-1 Yr', '1-3 Yrs', '3-5 Yrs', '5-10 Yrs', '10+ Yrs']
    df['TENURE_GROUP'] = pd.cut(df['DAYS_TENURE'], bins=tenure_bins, labels=tenure_labels, right=False)
    churn_by_tenure = df.groupby('TENURE_GROUP', observed=True)['CHURN'].value_counts(normalize=True).unstack().fillna(0)
    churn_by_tenure_data = {
        'labels': churn_by_tenure.index.tolist(),
        'churn_rate': (churn_by_tenure[1] * 100).tolist()
    }

    # 8. NEW Bar Chart: Churn Rate by Home Ownership
    owner_churn_counts = churned_df.groupby('HOME_OWNER').size()
    churn_by_owner_pie_data = {
        'labels': ['Renter', 'Owner'],
        'data': owner_churn_counts.values.tolist()
    }

    # 9. NEW Heatmap: Correlation of Numeric Features
    numeric_cols = ['AGE_IN_YEARS', 'INCOME', 'DAYS_TENURE', 'CURR_ANN_AMT', 'CHURN']
    corr_matrix = df[numeric_cols].corr()
    heatmap_data = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            heatmap_data.append({'x': col1, 'y': col2, 'v': round(corr_matrix.iloc[i, j], 2)})
    heatmap_labels = corr_matrix.columns.tolist()

    # 10. Folium Heatmap
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
        churn_by_owner_data=json.dumps(churn_by_owner_pie_data),
        heatmap_data=json.dumps(heatmap_data),
        heatmap_labels=json.dumps(heatmap_labels),
        churn_map_html=churn_map_html  # <-- pass Folium map HTML
    )

if __name__ == '__main__':
    app.run(debug=True)
