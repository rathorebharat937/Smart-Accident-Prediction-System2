"""
Configuration file for optimizing performance with large datasets
"""

# Data Loading Configuration
DATA_CONFIG = {
    'chunk_size': 100000,  # Number of rows to read at once
    'use_chunks': True,     # Enable chunked reading for very large files
    'sample_size_default': 50000,  # Default sample size
    'max_memory_mb': 4096,  # Maximum memory usage in MB
}

# Feature Configuration
REQUIRED_COLUMNS = [
    'ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng',
    'City', 'State', 'Weather_Condition', 'Temperature(F)',
    'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
    'Wind_Speed(mph)', 'Precipitation(in)'
]

# Optimized data types for memory efficiency
# Note: Categorical types are set AFTER handling missing values
DTYPE_CONFIG = {
    'ID': 'str',
    'Severity': 'int8',
    'Temperature(F)': 'float32',
    'Humidity(%)': 'float32',
    'Pressure(in)': 'float32',
    'Visibility(mi)': 'float32',
    'Wind_Speed(mph)': 'float32',
    'Precipitation(in)': 'float32',
    'Start_Lat': 'float32',
    'Start_Lng': 'float32'
}

# Visualization Configuration
VIZ_CONFIG = {
    'max_map_points': 10000,  # Maximum points to display on map
    'default_map_points': 1000,
    'heatmap_radius': 15,
    'heatmap_blur': 20,
    'max_hotspots_display': 50,
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'route_search_radius_default': 10,  # miles
    'hotspot_eps_default': 0.5,  # degrees
    'hotspot_min_samples_default': 10,
    'spatial_index_threshold': 10000,  # Use spatial indexing above this count
}

# ML Configuration
ML_CONFIG = {
    'test_size': 0.25,
    'random_state': 42,
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    'gb_n_estimators': 100,
    'gb_max_depth': 5,
    'max_training_samples': 100000,  # Limit training data for speed
}

# Cache Configuration
CACHE_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # Time to live in seconds
    'max_cache_size_mb': 1024,
}

# Performance Optimization Flags
PERFORMANCE_FLAGS = {
    'use_parallel_processing': True,
    'n_jobs': -1,  # Use all available cores
    'optimize_memory': True,
    'enable_progress_bars': True,
}

# Display Configuration
DISPLAY_CONFIG = {
    'max_table_rows': 20,
    'max_plot_categories': 10,
    'figure_dpi': 100,
    'figure_size': (10, 6),
}

# Thresholds
THRESHOLDS = {
    'high_severity': 3,
    'low_visibility': 2.0,  # miles
    'high_risk_weather': ['Rain', 'Snow', 'Fog', 'Heavy Rain', 'Thunderstorm'],
    'dangerous_hours': [0, 1, 2, 3, 4, 5, 22, 23],
}

def get_sample_size_recommendation(total_rows):
    """
    Recommend appropriate sample size based on total data size
    """
    if total_rows < 10000:
        return total_rows
    elif total_rows < 100000:
        return 10000
    elif total_rows < 500000:
        return 50000
    elif total_rows < 1000000:
        return 100000
    else:
        return 200000

def estimate_memory_usage(num_rows):
    """
    Estimate memory usage for given number of rows
    Returns memory in MB
    """
    # Approximate bytes per row (with optimized dtypes)
    bytes_per_row = 150
    memory_mb = (num_rows * bytes_per_row) / (1024 ** 2)
    return memory_mb

def should_use_big_data_mode(total_rows, available_memory_mb=8192):
    """
    Determine if big data mode should be recommended
    """
    estimated_memory = estimate_memory_usage(total_rows)
    # Use big data mode if we have enough memory (with 2x safety margin)
    return estimated_memory * 2 < available_memory_mb

# Feature engineering configuration
FEATURE_ENGINEERING = {
    'extract_time_features': True,
    'create_weather_categories': True,
    'calculate_risk_scores': True,
    'add_spatial_features': False,  # Expensive for large datasets
}

# Export configuration
EXPORT_CONFIG = {
    'default_format': 'csv',
    'compression': 'gzip',
    'include_derived_features': True,
}