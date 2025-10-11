import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Load your dataset
df = pd.read_csv('./dataset/archive/autoinsurance_churn.csv')

# Clean / prepare data
df = df.dropna(subset=['latitude', 'longitude'])
df['Churn_Status'] = df['Churn'].map({0: 'Stayed', 1: 'Left'})

# Take a random sample to reduce number of points (e.g., 500)
sample_size = 10000  # adjust as needed
df_sample = df.sample(n=sample_size, random_state=42)

# Create a base map centered on the sample
m = folium.Map(location=[df_sample['latitude'].mean(), df_sample['longitude'].mean()],
               zoom_start=6,
               tiles='OpenStreetMap')

marker_cluster = MarkerCluster().add_to(m)

def churn_color(churn):
    return 'red' if churn == 1 else 'green'

# Add sampled customers
for _, row in df_sample.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=churn_color(row['Churn']),
        fill=True,
        fill_opacity=0.6,
        popup=(
            f"<b>City:</b> {row['city']}<br>"
            f"<b>Age:</b> {row['age_in_years']}<br>"
            f"<b>Income:</b> {row['income']}<br>"
            f"<b>Tenure (days):</b> {row['days_tenure']}<br>"
            f"<b>Status:</b> {row['Churn_Status']}"
        )
    ).add_to(marker_cluster)

# Save map
m.save('churn_map_sample.html')
