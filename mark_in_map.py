import pandas as pd
import folium
from folium.plugins import HeatMap

# Load dataset
df = pd.read_csv('./dataset/archive/autoinsurance_churn.csv')

# Drop rows without lat/lon
df = df.dropna(subset=['latitude', 'longitude'])

# Add readable churn status
df['Churn_Status'] = df['Churn'].map({0: 'Stayed', 1: 'Left'})

# Take a random sample to reduce plotting load
sample_size = 2000
df_sample = df.sample(n=sample_size, random_state=42)

# Separate customers
df_stayed = df_sample[df_sample['Churn'] == 0]
df_left = df_sample[df_sample['Churn'] == 1]

# Base map
m = folium.Map(
    location=[df_sample['latitude'].mean(), df_sample['longitude'].mean()],
    zoom_start=6,
    tiles='OpenStreetMap'
)

# HeatMap for customers who stayed
HeatMap(
    df_stayed[['latitude', 'longitude']],
    radius=10,        # smaller radius so heat is visible at different zooms
    blur=15,          # slightly spread out
    gradient={0.2: 'green', 0.8: 'lime'},  # green shades
    min_opacity=0.3,
    max_val=1          # normalize intensity
).add_to(m)

# HeatMap for customers who left
HeatMap(
    df_left[['latitude', 'longitude']],
    radius=10,
    blur=15,
    gradient={0.2: 'yellow', 0.8: 'red'},  # red shades
    min_opacity=0.3,
    max_val=1
).add_to(m)

# Save the map
m.save('churn_heatmap_adjusted.html')
