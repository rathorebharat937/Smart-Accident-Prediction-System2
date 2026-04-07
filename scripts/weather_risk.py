import requests

API_KEY = "65f3c8f0b77b5bb5781bf686c78d7c01"

def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        res = requests.get(url, timeout=5)
        data = res.json()

        return {
            "condition": data["weather"][0]["main"],
            "visibility": data.get("visibility", 10000) / 1000,
            "precipitation": data.get("rain", {}).get("1h", 0),
            "wind_speed": data["wind"]["speed"]
        }

    except:
        return {
            "condition": "clear",
            "visibility": 10,
            "precipitation": 0,
            "wind_speed": 0
        }

def compute_weather_risk(weather):
    condition = weather["condition"].lower()
    visibility = weather["visibility"]
    precipitation = weather["precipitation"]
    wind = weather["wind_speed"]

    score = 0.0

    # CONDITION
    if "rain" in condition:
        score += 0.4
    elif "fog" in condition:
        score += 0.5
    elif "snow" in condition:
        score += 0.6
    elif "storm" in condition:
        score += 0.7
    else:
        score += 0.1

    # VISIBILITY
    if visibility < 2:
        score += 0.3
    elif visibility < 5:
        score += 0.2
    elif visibility < 8:
        score += 0.1

    # PRECIPITATION
    if precipitation > 1:
        score += 0.3
    elif precipitation > 0.3:
        score += 0.2
    elif precipitation > 0:
        score += 0.1

    # WIND
    if wind > 15:
        score += 0.2
    elif wind > 8:
        score += 0.1

    return min(score, 1.0)
