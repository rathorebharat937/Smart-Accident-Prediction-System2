import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath, limit=None, use_chunks=False):
    """
    Load accident data with optimizations for large datasets
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    limit : int, optional
        Number of rows to load. If None, loads entire dataset
    use_chunks : bool
        Whether to use chunked reading for very large files
    """
    print(f"Loading data from {filepath}...")
    
    # Define only the columns we need to reduce memory
    required_columns = [
        'ID', 'Severity', 'Start_Time', 'Start_Lat', 'Start_Lng',
        'City', 'State', 'Weather_Condition', 'Temperature(F)',
        'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
        'Wind_Speed(mph)', 'Precipitation(in)'
    ]
    
    # Optimize data types to reduce memory usage
    # Don't set categorical during loading - do it after filling nulls
    dtype_dict = {
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
    
    try:
        if limit is not None:
            # Load limited sample
            df = pd.read_csv(
                filepath,
                nrows=limit,
                usecols=required_columns,
                dtype=dtype_dict,
                parse_dates=['Start_Time'],
                low_memory=False
            )
        else:
            # Load full dataset with chunking for memory efficiency
            if use_chunks:
                chunks = []
                chunk_size = 100000
                for chunk in pd.read_csv(
                    filepath,
                    usecols=required_columns,
                    dtype=dtype_dict,
                    parse_dates=['Start_Time'],
                    chunksize=chunk_size,
                    low_memory=False
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                print(f"Loaded {len(df):,} rows in chunks")
            else:
                df = pd.read_csv(
                    filepath,
                    usecols=required_columns,
                    dtype=dtype_dict,
                    parse_dates=['Start_Time'],
                    low_memory=False
                )
        
        # Feature engineering
        df = preprocess_data(df)
        
        print(f"Successfully loaded {len(df):,} accident records")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess and add derived features"""
    
    # Extract time features
    df['Hour'] = df['Start_Time'].dt.hour
    df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    
    # Handle missing values BEFORE converting to category
    # Fill categorical missing values first
    if 'Weather_Condition' in df.columns:
        df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    if 'City' in df.columns:
        df['City'] = df['City'].fillna('Unknown')
    if 'State' in df.columns:
        df['State'] = df['State'].fillna('Unknown')
    
    # Now convert to category dtype for memory efficiency
    for col in ['Weather_Condition', 'City', 'State']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def get_data_info(df):
    """Get information about the loaded dataset"""
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'date_range': (df['Start_Time'].min(), df['Start_Time'].max()),
        'states_count': df['State'].nunique(),
        'cities_count': df['City'].nunique(),
        'missing_values': df.isnull().sum().to_dict()
    }
    return info

def sample_data(df, n=10000, method='random'):
    """
    Create a sample from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    n : int
        Number of samples
    method : str
        'random' or 'stratified' (by severity)
    """
    if method == 'random':
        return df.sample(n=min(n, len(df)), random_state=42)
    elif method == 'stratified':
        # Stratified sampling to maintain severity distribution
        return df.groupby('Severity', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n // df['Severity'].nunique()), random_state=42)
        )
    else:
        raise ValueError("method must be 'random' or 'stratified'")