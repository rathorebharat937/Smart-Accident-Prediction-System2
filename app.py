import streamlit as st
import pandas as pd
from scripts import data_loader, eda_analysis, map_generator, hotspot_detector, route_analyzer, prediction_model, utils
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

DATA_PATH = "data/US_Accidents_March23.csv"

st.set_page_config(page_title="Traffic Accident Hotspot", page_icon="🚦", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .recommendation-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚦 Traffic Accident Hotspot Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Big Data Analytics & Prediction System with AI-Powered Insights</p>', unsafe_allow_html=True)

# Initialize session state for data loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data loading configuration
st.sidebar.subheader("📊 Data Loading")
use_full_dataset = st.sidebar.checkbox("Use Full Dataset (Big Data Mode)", value=False, 
                                        help="Load entire dataset for true big data analysis")

if use_full_dataset:
    data_limit = None
    st.sidebar.warning("⚠️ Loading full dataset - this may take time")
else:
    data_limit = st.sidebar.slider("Data Sample Size", 10000, 500000, 50000, 10000,
                                     help="For testing, use smaller samples")

# Load/Reload data button
if st.sidebar.button("🔄 Load/Reload Data", type="primary"):
    with st.spinner("Loading data... Please wait."):
        st.session_state.df = data_loader.load_data(DATA_PATH, limit=data_limit)
        st.session_state.data_loaded = True
        st.session_state.data_limit = data_limit
        # Clear any cached results
        for key in ['current_map', 'route_analyzed', 'hotspots_detected', 'model', 'scaler', 
                    'trained_model_type', 'route_map', 'spatial_map', 'temporal_map']:
            if key in st.session_state:
                del st.session_state[key]
    st.sidebar.success(f"✅ Loaded {len(st.session_state.df):,} records")

# Check if data is loaded
if not st.session_state.data_loaded:
    st.info("👈 Please load data from the sidebar to begin analysis")
    st.stop()

df = st.session_state.df

# Display data info
st.sidebar.info(f"📊 **Dataset Size:** {len(df):,} records")
if data_limit is None:
    st.sidebar.success("🎯 **Big Data Mode Active**")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Get unique values for filters
cities = ['All'] + sorted(df['City'].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("Select City", cities, key="city_filter")

severity_levels = ['All'] + sorted(df['Severity'].dropna().unique().tolist())
selected_severity = st.sidebar.selectbox("Select Severity", severity_levels, key="severity_filter")

# Apply filters efficiently
@st.cache_data
def apply_filters(df, city, severity):
    filtered = df.copy()
    if city != 'All':
        filtered = filtered[filtered['City'] == city]
    if severity != 'All':
        filtered = filtered[filtered['Severity'] == severity]
    return filtered

filtered_df = apply_filters(df, selected_city, selected_severity)
st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "🗺️ Interactive Maps", 
    "🛣️ Route Analysis", 
    "🤖 ML Prediction", 
    "🎯 Hotspot Detection",
    "📈 Temporal Analysis"
])

# TAB 1: Dashboard
with tab1:
    st.header("📊 Exploratory Data Analysis")
    
    # Summary metrics (cached)
    @st.cache_data
    def get_cached_stats(df):
        return utils.get_summary_statistics(df)
    
    stats = get_cached_stats(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accidents", f"{stats['total_accidents']:,}")
    with col2:
        st.metric("Average Severity", stats['avg_severity'])
    with col3:
        st.metric("High Severity (3+)", f"{stats['high_severity_count']:,}")
    with col4:
        st.metric("Peak Hour", f"{stats['peak_hour']}:00")

# TAB 2: Interactive Maps
with tab2:
    st.header("🗺️ Interactive Accident Maps")
    
    col1, col2 = st.columns(2)
    with col1:
        map_type = st.selectbox("Select Map Type", ["Basic Markers", "Heatmap", "Severity Map"])
    with col2:
        map_limit = st.slider("Number of Points to Display", 100, 10000, 1000, 100)
    
    # Generate map only when button is clicked
    if st.button("🗺️ Generate Map", type="primary", key="gen_map_btn"):
        with st.spinner("Generating map..."):
            if map_type == "Basic Markers":
                m = map_generator.create_basic_map(filtered_df, limit=map_limit)
            elif map_type == "Heatmap":
                m = map_generator.create_heatmap(filtered_df, limit=map_limit)
            else:
                m = map_generator.create_severity_map(filtered_df, limit=map_limit)
            
            st.session_state['current_map'] = m
            st.session_state['map_type'] = map_type
    
    # Display map if it exists in session state
    if 'current_map' in st.session_state:
        st.info(f"Displaying: {st.session_state.get('map_type', 'Map')}")
        st_folium(st.session_state['current_map'], width=1400, height=600, returned_objects=[])
    else:
        st.info("👆 Click 'Generate Map' to visualize accident locations")

# TAB 3: Route Analysis - FIXED TO PREVENT RELOADING
with tab3:
    st.header("🛣️ Enhanced Route Safety Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Location")
        start_city = st.selectbox("Select Start City", sorted(df['City'].dropna().unique()), key="start_city")
        start_coords = utils.get_city_coordinates(start_city, df)
        if start_coords:
            st.info(f"📍 Coordinates: {start_coords[0]:.4f}, {start_coords[1]:.4f}")
    
    with col2:
        st.subheader("Destination")
        end_city = st.selectbox("Select Destination City", sorted(df['City'].dropna().unique()), key="end_city")
        end_coords = utils.get_city_coordinates(end_city, df)
        if end_coords:
            st.info(f"📍 Coordinates: {end_coords[0]:.4f}, {end_coords[1]:.4f}")
    
    search_radius = st.slider("Search Radius (miles)", 5, 50, 10, key="route_radius")
    
    if st.button("🔍 Analyze Route", type="primary", key="analyze_route_btn"):
        if start_coords and end_coords:
            with st.spinner("Analyzing route with fusion-based intelligence..."):
                from geopy.distance import geodesic
                
                route_accidents = route_analyzer.find_accidents_on_route(
                    df, start_coords, end_coords, radius_miles=search_radius
                )
                
                distance_miles = geodesic(start_coords, end_coords).miles
                safety_score = route_analyzer.calculate_route_safety_score(route_accidents)
                route_stats = route_analyzer.get_route_statistics(route_accidents)
                
                # Get weather and traffic risks for triple-fusion calculation
                weather_risk = route_analyzer.get_route_weather_risk(start_coords, end_coords)
                traffic_risk = route_analyzer.get_route_traffic_risk(start_coords, end_coords)
                final_risk = route_analyzer.compute_final_route_risk(safety_score, weather_risk, traffic_risk)
                category = route_analyzer.classify_route(final_risk)
                
                # Create fusion-based route map
                route_map = route_analyzer.create_route_map(df, start_coords, end_coords, route_accidents)
                
                # Store results in session state
                st.session_state['route_accidents'] = route_accidents
                st.session_state['route_safety_score'] = safety_score
                st.session_state['route_stats'] = route_stats
                st.session_state['route_map'] = route_map
                st.session_state['route_distance'] = distance_miles
                st.session_state['weather_risk'] = weather_risk
                st.session_state['traffic_risk'] = traffic_risk
                st.session_state['final_risk'] = final_risk
                st.session_state['route_category'] = category
                st.session_state['route_analyzed'] = True
        else:
            st.error("Could not find coordinates for selected cities")
    
    # Display results if they exist in session state
    if st.session_state.get('route_analyzed', False):
        route_accidents = st.session_state['route_accidents']
        safety_score = st.session_state['route_safety_score']
        route_stats = st.session_state['route_stats']
        route_map = st.session_state['route_map']
        distance_miles = st.session_state['route_distance']
        weather_risk = st.session_state['weather_risk']
        traffic_risk = st.session_state['traffic_risk']
        final_risk = st.session_state['final_risk']
        category = st.session_state['route_category']
        
        # Key Metrics - Triple-Fusion System
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Route Distance", f"{distance_miles:.1f} mi")
        with col2:
            st.metric("Accidents on Route", route_stats['total_accidents'])
        with col3:
            st.metric("Weather Risk", f"{weather_risk:.2f}")
        with col4:
            st.metric("Traffic Risk", f"{traffic_risk:.2f}")
        with col5:
            st.metric("Final Risk", f"{final_risk:.2f}")
        with col6:
            st.metric("Route Category", category)
        
        # Fusion-Based Risk Assessment Box
        if category == "HIGH RISK":
            st.error("🚨 HIGH RISK ROUTE - Dangerous conditions detected")
        elif category == "MODERATE":
            st.warning("⚠️ MODERATE RISK ROUTE")
        else:
            st.success("✅ SAFE ROUTE")
        
        # Route Map - FIXED: Use returned_objects=[] to prevent reloading
        st.subheader("🗺️ Interactive Route Map")
        st_folium(route_map, width=1400, height=600, key="route_map_display", returned_objects=[])
        
        # Detailed Accident Information
        if not route_accidents.empty:
            with st.expander("📋 View Detailed Accident Data"):
                display_cols = ['City', 'Severity', 'Weather_Condition', 'Start_Time', 
                               'Temperature(F)', 'Visibility(mi)']
                available_cols = [col for col in display_cols if col in route_accidents.columns]
                st.dataframe(route_accidents[available_cols].head(50), use_container_width=True)

# TAB 4: ML Prediction
# ============================================================
# TAB 4: Accident Severity Prediction (Random Forest)
# ============================================================
with tab4:
    st.header("🤖 Accident Severity Prediction (Random Forest)")
    st.info("🌦️ Predict accident severity and risk based on weather and time conditions.")

    col1, col2 = st.columns([1, 1])

    # ---------------------------------------------
    # Model Training
    # ---------------------------------------------
    with col1:
        st.subheader("🎯 Train Random Forest Model")

        if st.button("Train Model", type="primary", key="train_rf_model"):
            with st.spinner("Training Random Forest on preprocessed crash data..."):
                ml_df = prediction_model.preprocess_for_ml(filtered_df)
                model, metrics, X_test, y_test, preds, pred_proba, feature_names = prediction_model.train_random_forest(ml_df)

                # Store in session
                st.session_state["rf_model"] = model
                st.session_state["rf_metrics"] = metrics
                st.session_state["rf_y_test"] = y_test
                st.session_state["rf_preds"] = preds

            st.success(f"✅ Model trained on {len(ml_df):,} records.")
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.metric("Cross-Validation", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

            # Confusion matrix + feature importance
            st.pyplot(prediction_model.plot_confusion_matrix(y_test, preds))
            st.pyplot(prediction_model.plot_feature_importance(model))

    # ---------------------------------------------
    # Prediction Section
    # ---------------------------------------------
    with col2:
        st.subheader("🔮 Predict Accident Risk")

        if "rf_model" in st.session_state:
            st.info("💡 Enter current or forecasted conditions to predict accident risk.")

            col_a, col_b = st.columns(2)
            with col_a:
                temp = st.number_input("Temperature (°F)", -20, 120, 70)
                humidity = st.slider("Humidity (%)", 0, 100, 50)
                pressure = st.number_input("Pressure (in)", 28.0, 31.0, 29.92, 0.01)
                visibility = st.number_input("Visibility (mi)", 0.0, 10.0, 10.0, 0.1)

            with col_b:
                wind_speed = st.number_input("Wind Speed (mph)", 0.0, 100.0, 5.0, 0.5)
                precipitation = st.number_input("Precipitation (in)", 0.0, 5.0, 0.0, 0.01)
                hour = st.slider("Hour of Day", 0, 23, 12)
                weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Heavy Rain', 'Snow', 'Fog', 
                                      'Thunderstorm', 'Ice', 'Hail']
                weather = st.selectbox("Weather Condition", weather_conditions)

            if st.button("🔍 Predict Risk", type="primary", key="predict_rf_btn"):
                conditions = {
                    "Temperature(F)": temp,
                    "Humidity(%)": humidity,
                    "Pressure(in)": pressure,
                    "Visibility(mi)": visibility,
                    "Wind_Speed(mph)": wind_speed,
                    "Precipitation(in)": precipitation,
                    "Hour": hour,
                    "Weather_Condition": weather
                }

                result = prediction_model.predict_accident_probability(st.session_state["rf_model"], conditions)

                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Severity", str(result["predicted_severity"]))
                with col2:
                    st.metric("Risk Score", f"{result['risk_score']:.2f}")
                with col3:
                    st.metric("Risk Level", result["risk_level"])

                # Probability chart
                probs = result["severity_probabilities"]
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(probs.keys(), probs.values(), color='skyblue')
                ax.set_xlabel("Severity Level")
                ax.set_ylabel("Probability")
                ax.set_title("Severity Probability Distribution")
                st.pyplot(fig)

                # Recommendation
                st.markdown(f"### 🚗 Driving Recommendation:\n{result['recommendation']}")
        else:
            st.info("👆 Train the model first to enable predictions.")


# TAB 5: Spatial Hotspot Detection - FIXED TO PREVENT RELOADING
with tab5:
    st.header("🎯 Spatial Hotspot Detection & Analysis")
    st.info("🔍 Identify high-risk crash zones using a simple grid-based density method.")

    col1, col2 = st.columns(2)
    grid_size = col1.slider("Grid Size (degrees)", 0.01, 0.2, 0.05, 0.01, key="grid_size_tab5")
    density_threshold = col2.slider("Min Crashes per Grid Cell", 3, 50, 10, 1, key="density_threshold_tab5")

    if st.button("🎯 Detect Hotspots", key="detect_hotspots_btn"):
        with st.spinner("Computing hotspot density..."):
            hotspots, spatial_df = hotspot_detector.detect_spatial_hotspots(
                filtered_df, grid_size=grid_size, density_threshold=density_threshold
            )
            
            # Store results
            st.session_state["hotspots"] = hotspots
            st.session_state["spatial_df"] = spatial_df
            
            # Generate map once and store it
            if not hotspots.empty:
                m = folium.Map(
                    location=[hotspots["Start_Lat"].mean(), hotspots["Start_Lng"].mean()],
                    zoom_start=6, tiles="CartoDB positron"
                )

                for _, row in hotspots.iterrows():
                    count = row["Count"]
                    color = "red" if count > 30 else "orange" if count > 15 else "yellow"
                    popup_id = row.get("Grid_ID", "Unknown")
                    folium.Circle(
                        location=[row["Start_Lat"], row["Start_Lng"]],
                        radius=count * 80,
                        popup=f"Grid {popup_id}<br>Crashes: {count}",
                        color=color, fill=True, fill_color=color, fill_opacity=0.5
                    ).add_to(m)
                
                st.session_state["spatial_map"] = m

    # Display persisted results
    hotspots = st.session_state.get("hotspots", pd.DataFrame())

    if not hotspots.empty:
        st.success(f"✅ Found {len(hotspots)} hotspot regions.")
        col1, col2 = st.columns(2)
        col1.metric("Total Hotspots", len(hotspots))
        col2.metric("Max Accidents in Grid", int(hotspots['Count'].max()))

        # Display stored map
        if "spatial_map" in st.session_state:
            st_folium(st.session_state["spatial_map"], width=1300, height=600, 
                     key="spatial_map_display", returned_objects=[])

        # Histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(hotspots["Count"], bins=20, color="salmon", edgecolor="black")
        ax.set_title("Accident Density Distribution")
        ax.set_xlabel("Accidents per Grid")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No hotspots detected yet. Click 'Detect Hotspots' above.")

# TAB 6: Temporal Hotspot Evolution - FIXED TO PREVENT RELOADING
with tab6:
    st.header("🕓 Spatiotemporal Hotspot Evolution")
    st.info("📅 Track hotspot emergence, persistence, and disappearance across time periods.")

    grid_size = st.slider("Grid Size", 0.01, 0.1, 0.05, 0.01, key="grid_size_tab6")
    density_threshold = st.slider("Density Threshold", 3, 20, 5, 1, key="density_threshold_tab6")
    time_unit = st.selectbox("Group By", ["Monthly", "Yearly"], key="time_unit_tab6")

    @st.cache_data(show_spinner=False)
    def compute_hotspot_evolution(df, grid_size, density_threshold, time_unit):
        return hotspot_detector.analyze_temporal_evolution(
            df,
            grid_size=grid_size,
            density_threshold=density_threshold,
            time_unit="M" if time_unit == "Monthly" else "Y"
        )

    if st.button("🔍 Analyze Evolution", key="analyze_evolution_btn"):
        with st.spinner("Evaluating hotspot transitions over time..."):
            evolution_df = compute_hotspot_evolution(filtered_df, grid_size, density_threshold, time_unit)

            if not evolution_df.empty:
                st.session_state["evolution_df"] = evolution_df
                
                # Generate map once and store it
                center_lat = filtered_df["Start_Lat"].mean()
                center_lng = filtered_df["Start_Lng"].mean()

                m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles="CartoDB positron")
                colors = {"New": "blue", "Persistent": "green", "Disappeared": "red"}

                for _, row in evolution_df.iterrows():
                    lat, lng = row.get("Start_Lat"), row.get("Start_Lng")
                    if pd.notnull(lat) and pd.notnull(lng):
                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=5,
                            color=colors.get(row["Status"], "gray"),
                            fill=True,
                            fill_color=colors.get(row["Status"], "gray"),
                            fill_opacity=0.7,
                            popup=f"<b>Grid:</b> {row.get('Grid_ID','N/A')}<br>"
                                  f"<b>Status:</b> {row['Status']}<br>"
                                  f"<b>Period:</b> {row.get('Period','N/A')}"
                        ).add_to(m)
                
                st.session_state["temporal_map"] = m
                st.success("✅ Temporal hotspot evolution analysis complete.")
            else:
                st.warning("⚠️ No hotspots identified in this period.")

    evolution_df = st.session_state.get("evolution_df", pd.DataFrame())

    if not evolution_df.empty:
        st.write("### Preview of Results")
        st.dataframe(evolution_df.head(10))

        st.subheader("📈 Hotspot Change Summary")

        status_counts = evolution_df["Status"].value_counts()
        total = int(status_counts.sum())
        emerging = int(status_counts.get("New", 0))
        persistent = int(status_counts.get("Persistent", 0))
        disappeared = int(status_counts.get("Disappeared", 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("🟦 New Hotspots", emerging, f"{(emerging / total * 100):.1f}%")
        c2.metric("🟩 Persistent", persistent, f"{(persistent / total * 100):.1f}%")
        c3.metric("🟥 Disappeared", disappeared, f"{(disappeared / total * 100):.1f}%")

        st.bar_chart(status_counts)

        st.markdown(f"""
        - **Total hotspots analyzed:** {total}  
        - **New:** {emerging} ({(emerging / total * 100):.1f}%)  
        - **Persistent:** {persistent} ({(persistent / total * 100):.1f}%)  
        - **Disappeared:** {disappeared} ({(disappeared / total * 100):.1f}%)  
        """)

        if emerging > disappeared:
            st.success("📈 More new hotspots are emerging — indicates increasing risk.")
        elif disappeared > emerging:
            st.info("📉 More hotspots have disappeared — signs of safety improvement.")
        else:
            st.warning("⚖️ Stable hotspot trend — similar distribution over time.")

        st.subheader("🗺️ Hotspot Evolution Map")
        
        # Display stored map
        if "temporal_map" in st.session_state:
            st_folium(st.session_state["temporal_map"], width=1200, height=600, 
                     key="temporal_map_display", returned_objects=[])

        st.markdown("### 🧭 Interpretation Guide")
        st.write("""
        - **🟦 Blue:** Newly emerging crash zones (recently risky areas).  
        - **🟩 Green:** Stable long-term hotspots (require sustained monitoring).  
        - **🟥 Red:** Areas where risk has reduced (positive change).  
        """)

    else:
        st.warning("⚠️ No temporal hotspot evolution detected yet. Run the analysis first.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><b>Enhanced Traffic Accident Analysis System</b> | Processing {len(df):,} accident records</p>
    <p>Features: Hotspot Detection • Route Safety Analysis • ML Prediction • Temporal Trends</p>
    <p>Data Source: US Accidents Dataset | Powered by Streamlit, Scikit-learn & Folium</p>
    <p style='font-size: 0.8em;'>{'🎯 Big Data Mode Active' if data_limit is None else f'Sample Mode: {data_limit:,} records'}</p>
</div>
""", unsafe_allow_html=True)