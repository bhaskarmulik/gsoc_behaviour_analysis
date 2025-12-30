import pandas as pd
import spacy
import geocoder
import folium
from folium.plugins import HeatMap
import plotly.express as px
from tqdm import tqdm
import time

######### Helper functions #########
####################################

#Get lat/long
def geocoding(loc_df):  
    unique_locations = loc_df['location'].unique()
    
    geo_dict = {}
    
    for loc in tqdm(unique_locations):
        try:
            g = geocoder.arcgis(loc)
            if g.ok:
                geo_dict[loc] = {
                    'lat': g.lat,
                    'lng': g.lng,
                    'address': g.address
                }
            else:
                print(f"Could not geocode: {loc}")
            
            # Rate limit stuff
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error geocoding {loc}: {e}")
    
    geocoded_df = loc_df.copy()
    geocoded_df['lat'] = geocoded_df['location'].map(lambda x: geo_dict.get(x, {}).get('lat'))
    geocoded_df['lng'] = geocoded_df['location'].map(lambda x: geo_dict.get(x, {}).get('lng'))
    geocoded_df['full_address'] = geocoded_df['location'].map(lambda x: geo_dict.get(x, {}).get('address'))
    
    # Remove rows with failed geocoding
    geocoded_df = geocoded_df.dropna(subset=['lat', 'lng'])
    
    print(f"Successfully geocoded {len(geocoded_df)} out of {len(loc_df)} location mentions")
    return geocoded_df

############## Visualization functions #########
################################################

def folium_heatmap(geocoded_df):
    
    center_lat = geocoded_df['lat'].mean()
    center_lng = geocoded_df['lng'].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=3)
    
    # Temp df
    temp_df = [[row['lat'], row['lng'], row['risk_score']] 
                 for _, row in geocoded_df.iterrows()]
    
    HeatMap(temp_df, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1: 'red'}).add_to(m)
    
    #Markers for top locs
    top_locations = geocoded_df.groupby(['location', 'lat', 'lng', 'full_address'])['risk_score'].agg(['count', 'mean']).reset_index()
    top_locations = top_locations.sort_values('count', ascending=False).head(5)
    
    for _, row in top_locations.iterrows():
        folium.Marker(
            location=[row['lat'], row['lng']],
            popup=f"<b>{row['location']}</b><br>Posts: {row['count']}<br>Avg Risk: {row['mean']:.2f}",
            icon=folium.Icon(color='darkred', icon='info-sign')
        ).add_to(m)
    
    # Save the map
    m.save('reddit_crisis_heatmap.html')
    return m, top_locations

def create_plotly_map(geocoded_df):
    
    #Using groupby - KISS (keep it simple, stupid)
    location_summary = geocoded_df.groupby(['location', 'lat', 'lng']).agg(
        count=pd.NamedAgg(column='post_id', aggfunc='nunique'),
        avg_risk=pd.NamedAgg(column='risk_score', aggfunc='mean')
    ).reset_index()
    
    # Color coding df
    location_summary['risk_category'] = pd.cut(
        location_summary['avg_risk'],
        bins=[0, 3, 5, 10],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    fig = px.scatter_geo(
        location_summary,
        lat='lat',
        lon='lng',
        color='risk_category',
        size='count',
        hover_name='location',
        hover_data={
            'lat': False,
            'lng': False,
            'count': True,
            'avg_risk': ':.2f',
            'risk_category': True
        },
        projection='natural earth',
        title='Crisis Mentions by Location in Reddit Posts',
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
    )
    
    fig.update_layout(
        height=600,
        geo=dict(
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)',
            coastlinecolor='rgb(255, 255, 255)',
            showocean=True,
            oceancolor='rgb(220, 230, 255)'
        )
    )
    
    # Save as HTML file
    fig.write_image('reddit_crisis_plotly_map.jpeg', format='jpeg')
    
    return fig, location_summary
    
################# Main function #####################
#####################################################

if __name__ == "__main__":

    # Step 1: Load the data
    data_file = './../../data/final_reddit_data.csv'
    df = pd.read_csv(data_file)
    df = df.dropna(subset=['title', 'selftext'])
    spacy.require_gpu()
    nlp = spacy.load("en_core_web_lg")
    
    locations = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        combined_text = f"{row['title']} {row['selftext']}"        
        doc = nlp(combined_text)
        
        loc_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        
        if loc_entities:
            # Associate locations with risk score
            for loc in loc_entities:
                locations.append({
                    'post_id': row['id'],
                    'location': loc,
                    'risk_score': row['final_risk_score'],
                    'risk_class': row['risk_classification']
                })
    
    loc_df = pd.DataFrame(locations)
    print(f"Found {len(loc_df)} location mentions in {len(set(loc_df['post_id']))} posts")

    
    geocoded_df = geocoding(loc_df)
    
    #Creating viz
    folium_heatmap(geocoded_df)
    create_plotly_map(geocoded_df)

    
    location_stats = geocoded_df.groupby('location').agg(
        post_count=pd.NamedAgg(column='post_id', aggfunc='nunique'),
        avg_risk_score=pd.NamedAgg(column='risk_score', aggfunc='mean'),
        high_risk_count=pd.NamedAgg(column='risk_class', aggfunc=lambda x: sum(x == 'High Risk'))
    ).reset_index()
    
    top_locations_by_count = location_stats.sort_values('post_count', ascending=False).head(10)
    top_locations_by_risk = location_stats[location_stats['post_count'] >= 2].sort_values('avg_risk_score', ascending=False).head(10)
    top_locations_by_high_risk = location_stats.sort_values('high_risk_count', ascending=False).head(10)
    
    print(f"\nTop 5 Locations by Post Count: {top_locations_by_count[['location', 'post_count', 'avg_risk_score']].head(5)}")
    
    print(f"\nTop 5 Locations by Average Risk Score (min 2 posts):{top_locations_by_risk[['location', 'post_count', 'avg_risk_score']].head(5)}")
    
    print(f"\nTop 5 Locations by High Risk Post Count:{top_locations_by_high_risk[['location', 'high_risk_count', 'post_count']].head(5)}")
    