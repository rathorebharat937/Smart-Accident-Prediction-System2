import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np

def create_basic_map(df, limit=1000):
    """Create a basic map with accident markers"""
    # Sample data if too large
    if len(df) > limit:
        df_sample = df.sample(n=limit, random_state=42)
    else:
        df_sample = df.copy()
    
    # Remove any rows with invalid coordinates
    df_sample = df_sample.dropna(subset=['Start_Lat', 'Start_Lng'])
    df_sample = df_sample[
        (df_sample['Start_Lat'].between(-90, 90)) &
        (df_sample['Start_Lng'].between(-180, 180))
    ]
    
    if len(df_sample) == 0:
        # Return empty map centered on US
        return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Calculate center
    center_lat = df_sample['Start_Lat'].mean()
    center_lng = df_sample['Start_Lng'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Use MarkerCluster for better performance
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers
    severity_colors = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red'}
    
    for idx, row in df_sample.iterrows():
        try:
            severity = row['Severity']
            color = severity_colors.get(severity, 'gray')
            
            popup_text = f"""
            <b>Severity:</b> {severity}<br>
            <b>City:</b> {row.get('City', 'Unknown')}<br>
            <b>Weather:</b> {row.get('Weather_Condition', 'Unknown')}
            """
            
            folium.Marker(
                location=[row['Start_Lat'], row['Start_Lng']],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(marker_cluster)
        except Exception as e:
            # Skip problematic markers
            continue
    
    return m

def create_heatmap(df, limit=5000):
    """Create a heatmap of accident locations"""
    # Sample data if too large
    if len(df) > limit:
        df_sample = df.sample(n=limit, random_state=42)
    else:
        df_sample = df.copy()
    
    # Remove invalid coordinates
    df_sample = df_sample.dropna(subset=['Start_Lat', 'Start_Lng'])
    df_sample = df_sample[
        (df_sample['Start_Lat'].between(-90, 90)) &
        (df_sample['Start_Lng'].between(-180, 180))
    ]
    
    if len(df_sample) == 0:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Calculate center
    center_lat = df_sample['Start_Lat'].mean()
    center_lng = df_sample['Start_Lng'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Prepare heatmap data
    heat_data = []
    for idx, row in df_sample.iterrows():
        try:
            # Weight by severity
            weight = row['Severity'] / 4.0  # Normalize to 0-1
            heat_data.append([row['Start_Lat'], row['Start_Lng'], weight])
        except Exception:
            continue
    
    # Add heatmap layer
    if heat_data:
        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            max_zoom=13,
            min_opacity=0.3,
            gradient={
                0.0: 'blue',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
    
    return m

def create_severity_map(df, limit=2000):
    """Create a map with color-coded severity markers"""
    # Sample data if too large
    if len(df) > limit:
        df_sample = df.sample(n=limit, random_state=42)
    else:
        df_sample = df.copy()
    
    # Remove invalid coordinates
    df_sample = df_sample.dropna(subset=['Start_Lat', 'Start_Lng'])
    df_sample = df_sample[
        (df_sample['Start_Lat'].between(-90, 90)) &
        (df_sample['Start_Lng'].between(-180, 180))
    ]
    
    if len(df_sample) == 0:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Calculate center
    center_lat = df_sample['Start_Lat'].mean()
    center_lng = df_sample['Start_Lng'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Severity color mapping
    severity_colors = {
        1: '#90EE90',  # Light green
        2: '#FFD700',  # Gold
        3: '#FFA500',  # Orange
        4: '#FF4500'   # Red-orange
    }
    
    # Add circle markers
    for idx, row in df_sample.iterrows():
        try:
            severity = row['Severity']
            color = severity_colors.get(severity, '#808080')
            
            popup_text = f"""
            <div style='font-family: Arial; font-size: 12px;'>
                <b>Severity Level:</b> {severity}<br>
                <b>Location:</b> {row.get('City', 'Unknown')}, {row.get('State', 'Unknown')}<br>
                <b>Weather:</b> {row.get('Weather_Condition', 'Unknown')}<br>
                <b>Time:</b> {row.get('Start_Time', 'Unknown')}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['Start_Lat'], row['Start_Lng']],
                radius=4 + severity,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)
        except Exception:
            continue
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 160px; height: 140px; 
                background-color: white; z-index:9999; font-size:12px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <p style="margin:0; font-weight:bold; font-size:14px;">Severity Level</p>
        <p style="margin:5px 0;"><span style="color:#90EE90; font-size:18px;">●</span> Level 1 (Minor)</p>
        <p style="margin:5px 0;"><span style="color:#FFD700; font-size:18px;">●</span> Level 2 (Moderate)</p>
        <p style="margin:5px 0;"><span style="color:#FFA500; font-size:18px;">●</span> Level 3 (Serious)</p>
        <p style="margin:5px 0;"><span style="color:#FF4500; font-size:18px;">●</span> Level 4 (Severe)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_cluster_map(df, limit=5000):
    """Create a map with marker clustering"""
    # Sample data if too large
    if len(df) > limit:
        df_sample = df.sample(n=limit, random_state=42)
    else:
        df_sample = df.copy()
    
    # Remove invalid coordinates
    df_sample = df_sample.dropna(subset=['Start_Lat', 'Start_Lng'])
    df_sample = df_sample[
        (df_sample['Start_Lat'].between(-90, 90)) &
        (df_sample['Start_Lng'].between(-180, 180))
    ]
    
    if len(df_sample) == 0:
        return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Calculate center
    center_lat = df_sample['Start_Lat'].mean()
    center_lng = df_sample['Start_Lng'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Create separate clusters for each severity
    clusters = {
        1: MarkerCluster(name='Severity 1', overlay=True, control=True),
        2: MarkerCluster(name='Severity 2', overlay=True, control=True),
        3: MarkerCluster(name='Severity 3', overlay=True, control=True),
        4: MarkerCluster(name='Severity 4', overlay=True, control=True)
    }
    
    for cluster in clusters.values():
        cluster.add_to(m)
    
    severity_colors = {1: 'green', 2: 'blue', 3: 'orange', 4: 'red'}
    
    # Add markers to appropriate clusters
    for idx, row in df_sample.iterrows():
        try:
            severity = row['Severity']
            color = severity_colors.get(severity, 'gray')
            
            popup_text = f"""
            <b>Severity:</b> {severity}<br>
            <b>City:</b> {row.get('City', 'Unknown')}<br>
            <b>State:</b> {row.get('State', 'Unknown')}<br>
            <b>Weather:</b> {row.get('Weather_Condition', 'Unknown')}
            """
            
            folium.Marker(
                location=[row['Start_Lat'], row['Start_Lng']],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(clusters[severity])
        except Exception:
            continue
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def validate_coordinates(df):
    """Validate and clean coordinate data"""
    df_clean = df.copy()
    
    # Check if required columns exist
    if 'Start_Lat' not in df_clean.columns or 'Start_Lng' not in df_clean.columns:
        raise ValueError("DataFrame must contain 'Start_Lat' and 'Start_Lng' columns")
    
    # Remove NaN values
    df_clean = df_clean.dropna(subset=['Start_Lat', 'Start_Lng'])
    
    # Remove invalid ranges
    df_clean = df_clean[
        (df_clean['Start_Lat'].between(-90, 90)) &
        (df_clean['Start_Lng'].between(-180, 180))
    ]
    
    # Convert to numeric if needed
    df_clean['Start_Lat'] = pd.to_numeric(df_clean['Start_Lat'], errors='coerce')
    df_clean['Start_Lng'] = pd.to_numeric(df_clean['Start_Lng'], errors='coerce')
    
    # Remove any remaining NaN
    df_clean = df_clean.dropna(subset=['Start_Lat', 'Start_Lng'])
    
    return df_clean