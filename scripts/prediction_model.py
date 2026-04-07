import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Preprocessing
# ============================================================
def preprocess_for_ml(df):
    """Preprocess accident data for Random Forest training."""
    df = df.copy()

    features = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                'Weather_Condition', 'Hour']
    df = df[[col for col in features if col in df.columns]].dropna()

    # Time features
    df['Is_Rush_Hour'] = df['Hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
    df['Is_Night'] = df['Hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)

    # Weather severity
    df['Weather_Severity'] = df['Weather_Condition'].apply(categorize_weather_severity)

    # Interaction features
    df['Poor_Conditions'] = ((df['Precipitation(in)'] > 0.1) & (df['Visibility(mi)'] < 5)).astype(int)
    df['Temp_Extreme'] = ((df['Temperature(F)'] < 32) | (df['Temperature(F)'] > 95)).astype(int)

    df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)
    return df


def categorize_weather_severity(weather_condition):
    """Categorize weather condition by severity."""
    if pd.isna(weather_condition):
        return 0
    weather_str = str(weather_condition).lower()
    if any(w in weather_str for w in ['snow', 'ice', 'blizzard', 'freezing']):
        return 3
    if any(w in weather_str for w in ['rain', 'fog', 'thunderstorm', 'hail']):
        return 2
    if any(w in weather_str for w in ['cloud', 'overcast', 'mist']):
        return 1
    return 0

# ============================================================
# 2. Training
# ============================================================
def train_random_forest(df):
    """Train Random Forest model for accident severity prediction."""
    X = df.drop(columns=['Severity'])
    y = df['Severity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'cv_mean': round(cv_scores.mean(), 4),
        'cv_std': round(cv_scores.std(), 4)
    }

    return model, metrics, X_test, y_test, preds, pred_proba, X.columns

# ============================================================
# 3. Prediction
# ============================================================
def predict_accident_probability(model, weather_conditions):
    """Predict accident severity and risk level from weather conditions."""
    input_df = pd.DataFrame([weather_conditions])

    # Feature engineering
    input_df['Is_Rush_Hour'] = input_df['Hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0)
    input_df['Is_Night'] = input_df['Hour'].apply(lambda x: 1 if x >= 20 or x <= 6 else 0)
    input_df['Weather_Severity'] = input_df['Weather_Condition'].apply(categorize_weather_severity)
    input_df['Poor_Conditions'] = ((input_df['Precipitation(in)'] > 0.1) & (input_df['Visibility(mi)'] < 5)).astype(int)
    input_df['Temp_Extreme'] = ((input_df['Temperature(F)'] < 32) | (input_df['Temperature(F)'] > 95)).astype(int)
    input_df = pd.get_dummies(input_df, columns=['Weather_Condition'], drop_first=True)

    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    probabilities = model.predict_proba(input_df)[0]
    predicted_severity = model.predict(input_df)[0]
    risk_score = sum(probabilities[i] * (i + 1) for i in range(len(probabilities)))
    risk_level = 'Low' if risk_score < 2 else 'Moderate' if risk_score < 3 else 'High'

    return {
        'predicted_severity': int(predicted_severity),
        'severity_probabilities': {i+1: round(float(prob), 4) for i, prob in enumerate(probabilities)},
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'recommendation': generate_driving_recommendation(risk_level, weather_conditions)
    }

def generate_driving_recommendation(risk_level, weather_conditions):
    """Generate basic recommendations."""
    recs = []
    if risk_level == 'High':
        recs.append("⚠ HIGH RISK: Postpone travel if possible.")
    elif risk_level == 'Moderate':
        recs.append("⚠ MODERATE RISK: Drive with caution.")
    else:
        recs.append("✓ LOW RISK: Safe driving conditions.")

    weather = str(weather_conditions.get('Weather_Condition', '')).lower()
    if 'rain' in weather:
        recs.append("Use headlights and maintain distance.")
    elif 'fog' in weather:
        recs.append("Use fog lights and reduce speed.")
    elif 'snow' in weather:
        recs.append("Drive slowly, use winter tires.")

    return " | ".join(recs)

# ============================================================
# 4. Visualization
# ============================================================
def plot_confusion_matrix(y_test, preds):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_feature_importance(model):
    """Plot Random Forest feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    features = np.array(model.feature_names_in_)[indices]
    scores = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(features, scores, color='steelblue')
    ax.set_title("Top 10 Feature Importances")
    ax.invert_yaxis()
    return fig
