import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_severity(df):
    """Plot severity distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    severity_counts = df['Severity'].value_counts().sort_index()
    
    colors = ['#90EE90', '#FFD700', '#FFA500', '#FF4500']
    ax.bar(severity_counts.index, severity_counts.values, color=colors)
    ax.set_xlabel('Severity Level', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title('Accident Severity Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(severity_counts.values):
        ax.text(severity_counts.index[i], v, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_daywise(df):
    """Plot accidents by day of week"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check which column exists
    day_col = 'Day_of_Week' if 'Day_of_Week' in df.columns else 'DayOfWeek'
    
    if day_col not in df.columns:
        ax.text(0.5, 0.5, 'Day of Week data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    day_counts = df[day_col].value_counts().sort_index()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    ax.bar(range(len(day_counts)), day_counts.values, color='skyblue')
    ax.set_xticks(range(len(day_counts)))
    ax.set_xticklabels([day_names[i] for i in day_counts.index], rotation=45, ha='right')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title('Accidents by Day of Week', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_hourly(df):
    """Plot hourly distribution of accidents"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'Hour' not in df.columns:
        ax.text(0.5, 0.5, 'Hour data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    hourly_counts = df['Hour'].value_counts().sort_index()
    
    ax.plot(hourly_counts.index, hourly_counts.values, marker='o', 
            linewidth=2, markersize=6, color='steelblue')
    ax.fill_between(hourly_counts.index, hourly_counts.values, alpha=0.3, color='steelblue')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title('Hourly Distribution of Accidents', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    # Highlight peak hours
    peak_hour = hourly_counts.idxmax()
    ax.axvline(peak_hour, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak_hour}:00')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_top_cities(df, top_n=10):
    """Plot top N cities by accident count"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'City' not in df.columns:
        ax.text(0.5, 0.5, 'City data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    city_counts = df['City'].value_counts().head(top_n)
    
    ax.barh(range(len(city_counts)), city_counts.values, color='coral')
    ax.set_yticks(range(len(city_counts)))
    ax.set_yticklabels(city_counts.index)
    ax.set_xlabel('Number of Accidents', fontsize=12)
    ax.set_ylabel('City', fontsize=12)
    ax.set_title(f'Top {top_n} Cities by Accident Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(city_counts.values):
        ax.text(v, i, f' {v:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_weather_vs_severity(df):
    """Plot weather conditions vs severity"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'Weather_Condition' not in df.columns:
        ax.text(0.5, 0.5, 'Weather data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    # Get top weather conditions
    top_weather = df['Weather_Condition'].value_counts().head(10).index
    df_filtered = df[df['Weather_Condition'].isin(top_weather)]
    
    # Create crosstab
    weather_severity = pd.crosstab(df_filtered['Weather_Condition'], 
                                   df_filtered['Severity'], 
                                   normalize='index') * 100
    
    weather_severity.plot(kind='bar', stacked=True, ax=ax, 
                         color=['#90EE90', '#FFD700', '#FFA500', '#FF4500'])
    
    ax.set_xlabel('Weather Condition', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Severity Distribution by Weather Condition (Top 10)', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Severity', labels=['1', '2', '3', '4'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_state_distribution(df, top_n=15):
    """Plot accident distribution by state"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'State' not in df.columns:
        ax.text(0.5, 0.5, 'State data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    state_counts = df['State'].value_counts().head(top_n)
    
    ax.bar(range(len(state_counts)), state_counts.values, color='teal')
    ax.set_xticks(range(len(state_counts)))
    ax.set_xticklabels(state_counts.index, rotation=45, ha='right')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title(f'Top {top_n} States by Accident Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(state_counts.values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of numeric features"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numeric columns
    numeric_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 
                   'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
                   'Precipitation(in)']
    
    # Filter to existing columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        ax.text(0.5, 0.5, 'Not enough numeric columns for correlation', 
                ha='center', va='center', fontsize=12)
        return fig
    
    # Calculate correlation
    corr_matrix = df[available_cols].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_monthly_trend(df):
    """Plot monthly trend of accidents"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'Month' not in df.columns:
        ax.text(0.5, 0.5, 'Month data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    monthly_counts = df['Month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax.plot(monthly_counts.index, monthly_counts.values, marker='o', 
            linewidth=2, markersize=8, color='green')
    ax.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.3, color='green')
    
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Number of Accidents', fontsize=12)
    ax.set_title('Monthly Accident Trend', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_severity_by_hour(df):
    """Plot average severity by hour"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'Hour' not in df.columns:
        ax.text(0.5, 0.5, 'Hour data not available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    hourly_severity = df.groupby('Hour')['Severity'].agg(['mean', 'count'])
    
    # Create double y-axis
    ax2 = ax.twinx()
    
    # Plot severity
    ax.plot(hourly_severity.index, hourly_severity['mean'], 
            marker='o', linewidth=2, markersize=6, color='red', label='Avg Severity')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Severity', fontsize=12, color='red')
    ax.tick_params(axis='y', labelcolor='red')
    
    # Plot count as bars
    ax2.bar(hourly_severity.index, hourly_severity['count'], 
            alpha=0.3, color='blue', label='Accident Count')
    ax2.set_ylabel('Accident Count', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax.set_title('Average Severity and Count by Hour', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig

def get_summary_stats(df):
    """Get summary statistics"""
    stats = {
        'Total Accidents': len(df),
        'Average Severity': round(df['Severity'].mean(), 2),
        'Most Common Weather': df['Weather_Condition'].mode()[0] if 'Weather_Condition' in df.columns else 'N/A',
        'Peak Hour': int(df['Hour'].mode()[0]) if 'Hour' in df.columns else 'N/A',
        'Most Affected State': df['State'].mode()[0] if 'State' in df.columns else 'N/A',
        'Date Range': f"{df['Start_Time'].min().date()} to {df['Start_Time'].max().date()}" if 'Start_Time' in df.columns else 'N/A'
    }
    return stats