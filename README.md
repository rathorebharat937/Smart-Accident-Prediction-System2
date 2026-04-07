# Traffic Accident Hotspot Analysis 🚦📊

A comprehensive Big Data Analytics project to identify, analyze, and visualize traffic accident hotspots across the United States using advanced data processing, interactive dashboards, and machine learning techniques.

---

## 🎯 Project Overview

This project leverages a large-scale dataset of US traffic accidents to:

- Identify **high-risk zones** using clustering algorithms  
- Predict **accident severity** under various conditions  
- Provide **interactive visualizations** for dashboards and maps  
- Perform **route safety analysis** for informed decision-making  

It is built with **Python**, **Streamlit**, and popular data science libraries for a responsive and user-friendly interface.

---

## ✨ Key Features

### 1. Dashboard
- Summary metrics: total accidents, average severity, peak hours  
- Visualizations: severity distribution, day/hour patterns, top cities/states  
- Weather impact analysis and feature correlation heatmaps  

### 2. Interactive Maps
- Basic marker map with clustering  
- Heatmap for accident density  
- Severity-based color coding with popups  
- Adjustable number of points displayed  

### 3. Route Safety Analysis
- Select start and destination cities  
- Search for accidents along route within a radius  
- Calculate safety scores  
- Visual route mapping with accident details  

### 4. Machine Learning Prediction
- Models: Random Forest & Gradient Boosting  
- Predict accident severity based on weather and temporal features  
- Confusion matrix and feature importance visualization  
- Real-time prediction interface  

### 5. Hotspot Detection
- DBSCAN clustering to detect accident hotspots  
- Configurable clustering radius & minimum accident count  
- Visual hotspot mapping with statistics  
- Top hotspot ranking by accident count & average severity  

### 6. Advanced Features
- Dynamic data filtering (city, severity)  
- Interactive, export-ready visualizations  
- Performance optimization using caching  
- Session-based persistence to avoid reloads  

---

## 🛠️ Technologies Used

### Data Analysis
- **Pandas**, **NumPy**: Data manipulation  
- **Scikit-learn**: Machine learning  
- **Matplotlib**, **Seaborn**, **Plotly**: Visualizations  

### Mapping & Visualization
- **Folium**: Interactive maps  
- **Streamlit**, **Streamlit-Folium**: Web app interface  
- **Geopy**: Geocoding for cities  

---

## 📁 Project Structure

```
traffic-accident-hotspot/
├── app.py                          # Main Streamlit application
├── data/
│   └── US_Accidents_March23.csv    # Dataset (download from Kaggle)
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & preprocessing
│   ├── eda_analysis.py             # EDA functions
│   ├── map_generator.py            # Map generation functions
│   ├── hotspot_detector.py         # DBSCAN hotspot detection
│   ├── route_analyzer.py           # Route safety analysis
│   ├── prediction_model.py         # Machine learning models
│   └── utils.py                    
├── requirements.txt                
└── README.md                       
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+  
- 8GB+ RAM recommended  

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/sneh-a-15/traffic-accident-hotspot.git
cd traffic-accident-hotspot
```

2. **Create a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place the dataset**
- Download [US Accidents Dataset (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- Put it in `data/US_Accidents_March23.csv`

5. **Run the Streamlit application**
```bash
streamlit run app.py
```

---

## 📊 Dataset Information

- **Source**: [Kaggle US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size**: ~7 million records (~3GB)

**Key Columns**:
- `ID`, `Severity` (1–4 scale)
- `Start_Time`, `End_Time`
- `Start_Lat`, `Start_Lng`, `End_Lat`, `End_Lng`
- `City`, `County`, `State`, `Zipcode`
- Weather & road conditions: `Temperature(F)`, `Humidity(%)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`, `Weather_Condition`
- Road features: `Amenity`, `Traffic_Signal`, `Junction`, etc.

---

## 🔧 Configuration

### Performance Tuning
- **Data Sample Size**: Adjust in sidebar (1,000 - 50,000 records)
- **Map Points**: Configure display limit (100 - 5,000 points)
- **Clustering Parameters**: Fine-tune eps and min_samples

### Filters
- **City Filter**: Focus on specific locations
- **Severity Filter**: Analyze by accident severity level
- **Date Range**: Filter by time period (if available)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add AmazingFeature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## ⭐ Star this repository if you find it helpful!

**Happy Analyzing! 🚀📊**