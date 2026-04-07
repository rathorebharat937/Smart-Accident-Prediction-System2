import pandas as pd
import numpy as np
from functools import lru_cache

def format_number(num):
    """Format large numbers with commas"""
    return f"{int(num):,}"

def get_summary_statistics(df):
    """Calculate summary statistics for the dashboard"""
    stats = {
        'total_accidents': len(df),
        'avg_severity': round(df['Severity'].mean(), 2),
        'high_severity_count': len(df[df['Severity'] >= 3]),
        'peak_hour': int(df['Hour'].mode()[0]) if 'Hour' in df.columns else 0,
        'date_range': (df['Start_Time'].min(), df['Start_Time'].max()) if 'Start_Time' in df.columns else (None, None)
    }
    return stats

@lru_cache(maxsize=1000)
def get_city_coordinates(city, df_hash):
    """Get coordinates for a city (cached for performance)"""
    # Note: df_hash should be a hashable representation of df
    # In practice, you'd pass the actual df, but caching requires hashable args
    pass

def get_city_coordinates(city, df):
    """Get average coordinates for a city"""
    city_data = df[df['City'] == city]
    if len(city_data) > 0:
        avg_lat = city_data['Start_Lat'].mean()
        avg_lng = city_data['Start_Lng'].mean()
        return (avg_lat, avg_lng)
    return None

def calculate_distance(coord1, coord2):
    """Calculate distance between two coordinates using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Earth's radius in miles
    radius = 3959
    distance = radius * c
    
    return distance

def get_time_period(hour):
    """Categorize hour into time period"""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def get_season(month):
    """Get season from month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def calculate_risk_level(severity, weather, visibility):
    """Calculate overall risk level"""
    risk_score = 0
    
    # Severity contribution
    risk_score += severity * 25
    
    # Weather contribution
    high_risk_weather = ['Rain', 'Snow', 'Fog', 'Heavy Rain', 'Thunderstorm']
    if weather in high_risk_weather:
        risk_score += 20
    
    # Visibility contribution
    if visibility < 2:
        risk_score += 30
    elif visibility < 5:
        risk_score += 15
    
    # Classify risk level
    if risk_score >= 70:
        return 'Very High'
    elif risk_score >= 50:
        return 'High'
    elif risk_score >= 30:
        return 'Medium'
    else:
        return 'Low'

def get_top_n_items(df, column, n=10):
    """Get top N items by count in a column"""
    return df[column].value_counts().head(n)

def get_weather_categories(weather_condition):
    """Categorize weather conditions into broader categories"""
    if pd.isna(weather_condition):
        return 'Unknown'
    
    weather_str = str(weather_condition).lower()
    
    if any(word in weather_str for word in ['clear', 'fair']):
        return 'Clear'
    elif any(word in weather_str for word in ['rain', 'drizzle', 'shower']):
        return 'Rainy'
    elif any(word in weather_str for word in ['snow', 'sleet', 'ice']):
        return 'Snowy'
    elif any(word in weather_str for word in ['fog', 'mist', 'haze']):
        return 'Foggy'
    elif any(word in weather_str for word in ['cloud', 'overcast']):
        return 'Cloudy'
    elif any(word in weather_str for word in ['thunder', 'storm']):
        return 'Stormy'
    elif any(word in weather_str for word in ['wind']):
        return 'Windy'
    else:
        return 'Other'

def add_derived_features(df):
    """Add derived features for analysis"""
    df_copy = df.copy()
    
    # Time-based features
    if 'Hour' in df_copy.columns:
        df_copy['Time_Period'] = df_copy['Hour'].apply(get_time_period)
    
    if 'Month' in df_copy.columns:
        df_copy['Season'] = df_copy['Month'].apply(get_season)
    
    # Weather category
    if 'Weather_Condition' in df_copy.columns:
        df_copy['Weather_Category'] = df_copy['Weather_Condition'].apply(get_weather_categories)
    
    # Risk level
    if all(col in df_copy.columns for col in ['Severity', 'Weather_Condition', 'Visibility(mi)']):
        df_copy['Risk_Level'] = df_copy.apply(
            lambda x: calculate_risk_level(x['Severity'], x['Weather_Condition'], x['Visibility(mi)']),
            axis=1
        )
    
    return df_copy

def filter_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range"""
    if 'Start_Time' in df.columns:
        mask = (df['Start_Time'] >= start_date) & (df['Start_Time'] <= end_date)
        return df[mask]
    return df

def get_state_statistics(df):
    """Get statistics grouped by state"""
    state_stats = df.groupby('State').agg({
        'ID': 'count',
        'Severity': ['mean', 'max'],
        'Start_Lat': 'mean',
        'Start_Lng': 'mean'
    }).round(2)
    
    state_stats.columns = ['Accident_Count', 'Avg_Severity', 'Max_Severity', 'Avg_Lat', 'Avg_Lng']
    state_stats = state_stats.sort_values('Accident_Count', ascending=False)
    
    return state_stats

def get_hourly_statistics(df):
    """Get statistics grouped by hour"""
    if 'Hour' in df.columns:
        hourly_stats = df.groupby('Hour').agg({
            'ID': 'count',
            'Severity': 'mean'
        }).round(2)
        
        hourly_stats.columns = ['Accident_Count', 'Avg_Severity']
        return hourly_stats
    return pd.DataFrame()

def export_to_csv(df, filename='export.csv'):
    """Export dataframe to CSV"""
    df.to_csv(filename, index=False)
    return filename

def get_data_quality_report(df):
    """Generate data quality report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return report

def sample_stratified(df, n=1000, strata_column='Severity'):
    """Create stratified sample maintaining distribution"""
    if strata_column not in df.columns:
        return df.sample(n=min(n, len(df)))
    
    # Calculate proportional samples for each stratum
    strata_counts = df[strata_column].value_counts()
    strata_props = strata_counts / len(df)
    
    samples = []
    for stratum, prop in strata_props.items():
        stratum_n = int(n * prop)
        stratum_data = df[df[strata_column] == stratum]
        stratum_sample = stratum_data.sample(n=min(stratum_n, len(stratum_data)), random_state=42)
        samples.append(stratum_sample)
    
    return pd.concat(samples, ignore_index=True)