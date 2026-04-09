import requests

API_KEY = "uUi5KBIXB4HgXLaaqj4sdsvFmb2nln9R"

def get_traffic(lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point={lat},{lon}"
    
    try:
        res = requests.get(url, timeout=5)
        data = res.json()

        flow = data.get("flowSegmentData", {})

        current_speed = flow.get("currentSpeed", 0)
        free_flow_speed = flow.get("freeFlowSpeed", 1)

        return {
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed
        }

    except:
        return {
            "current_speed": 30,
            "free_flow_speed": 30
        }

def compute_traffic_risk(traffic):
    current = traffic["current_speed"]
    free = traffic["free_flow_speed"]

    if free == 0:
        return 0.5

    ratio = current / free

    if ratio > 0.8:
        return 0.1
    elif ratio > 0.5:
        return 0.4
    elif ratio > 0.3:
        return 0.7
    else:
        return 0.9
