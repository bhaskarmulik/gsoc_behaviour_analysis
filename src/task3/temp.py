import pandas as pd
import spacy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import HeatMap
from IPython.display import IFrame


# Load dataset
data_path = './../../data/final_reddit_data.csv'
df = pd.read_csv(data_path)

# Initialize spaCy model for NER (place recognition)
# en_core_web_trf is more accurate but larger; fallback to en_core_web_sm if needed
try:
    nlp = spacy.load('en_core_web_trf')
except OSError:
    nlp = spacy.load('en_core_web_sm')

# Function to extract GPE entities from text
def extract_locations(text):
    doc = nlp(str(text))
    return [ent.text for ent in doc.ents if ent.label_ == 'GPE']

# Apply extraction to posts
df['locations'] = df['selftext'].apply(extract_locations)

# Explode to one location per row
df_exploded = df.explode('locations').dropna(subset=['locations'])

# Count crisis-related posts per location
# Assuming all posts are crisis-related; if not, filter by a crisis keyword column
location_counts = df_exploded['locations'].value_counts().rename_axis('location').reset_index(name='count')

# Initialize geocoder with rate limiter
geolocator = Nominatim(user_agent='crisis_heatmap_app')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Geocode unique locations
location_counts['geocode'] = location_counts['location'].apply(lambda loc: geocode(loc))
# Drop failed geocodes
location_counts = location_counts.dropna(subset=['geocode'])
# Extract lat/lon
location_counts['latitude'] = location_counts['geocode'].apply(lambda x: x.latitude)
location_counts['longitude'] = location_counts['geocode'].apply(lambda x: x.longitude)

# Get top 5 locations
top5 = location_counts.nlargest(5, 'count')
print("Top 5 locations with highest crisis discussions:")
print(top5[['location', 'count']])

# Create base Folium map
# Center on the mean coordinates
map_center = [location_counts['latitude'].mean(), location_counts['longitude'].mean()]
folium_map = folium.Map(location=map_center, zoom_start=2)

# Prepare heat data
heat_data = list(zip(location_counts['latitude'], location_counts['longitude'], location_counts['count']))

# Add heatmap layer
HeatMap(heat_data, radius=15, max_zoom=10).add_to(folium_map)

# Save map to HTML
output_map = 'crisis_heatmap.html'
folium_map.save(output_map)
print(f"Heatmap saved to {output_map}")

# Optionally, display within a Jupyter environment
try:
    display(IFrame(output_map, width=800, height=600))
except ImportError:
    pass
