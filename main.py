import pandas as pd
import os
import json

def load_data(file_path):
    """
    Load CSV, select relevant columns, drop nulls and duplicates.
    """
    df = pd.read_csv(file_path)
    cols = ["ID","Severity","Start_Time","Start_Lat","Start_Lng","City","State"]
    df = df[cols].dropna(subset=["Start_Lat","Start_Lng","City"]).drop_duplicates()
    return df

def create_map(df, limit=1000, output_file="accident_map.html"):
    """
    Generate interactive HTML map with accidents, source-destination selection,
    buffer distance, and route statistics.
    Shows all accidents initially, then filters to show only route accidents.
    """
    accident_data = df.head(limit).to_dict(orient="records")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>US Accidents Route Planner</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ height: 100vh; }}
        .route-control{{
            position: fixed; top:10px; right:10px; background:white; padding:15px;
            border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.3); z-index:1000; max-width:320px;
        }}
        .route-control h3{{margin-top:0; color:#333; font-size:18px;}}
        .route-control p{{margin:10px 0; font-size:14px; color:#666;}}
        .route-control button{{
            background:#ff4444;color:white;border:none;padding:10px 15px;
            border-radius:5px;cursor:pointer;margin-top:10px;width:100%;
            font-size:14px;font-weight:bold;transition:background 0.3s;
        }}
        .route-control button:hover{{background:#cc0000;}}
        .slider-container{{margin-top:15px;}}
        .slider-container label{{font-size:14px;color:#333;}}
        .slider-container input{{width:100%;margin-top:5px;}}
        #route-stats{{
            margin-top:15px;padding:12px;background:#f0f0f0;
            border-radius:5px;font-size:13px;line-height:1.6;
        }}
        #route-stats b{{color:#333;}}
        .info-badge{{
            display:inline-block;background:#4CAF50;color:white;
            padding:2px 8px;border-radius:3px;font-size:12px;margin-left:5px;
        }}
    </style>
</head>
<body>
<div id="map"></div>
<div class="route-control">
    <h3>🗺️ Route Planner</h3>
    <p>Click map to set <span style="color:green;font-weight:bold;">SOURCE</span> and <span style="color:red;font-weight:bold;">DESTINATION</span></p>
    <div class="slider-container">
        <label><b>Buffer Distance:</b> <span id="buffer-value">5 km</span></label>
        <input type="range" min="1" max="50" value="5" step="1" id="bufferSlider">
    </div>
    <div id="route-stats">
        <span style="color:#666;">📍 Click on map to set source and destination</span>
    </div>
    <button id="clearRoute">🗑️ Clear Route</button>
</div>

<script>
var map = L.map('map').setView([39.5, -98.35], 4);

// Add OSM layer
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '© OpenStreetMap contributors'
}}).addTo(map);

// Accident data
var accidentData = {json.dumps(accident_data)};
var accidentMarkers = [];

// Create all accident markers
accidentData.forEach(function(accident){{
    var m = L.circleMarker([accident.Start_Lat, accident.Start_Lng], {{
        radius: 3,
        color: '#ff4444',
        fillColor: '#ff0000',
        fillOpacity: 0.6,
        weight: 1
    }}).bindPopup(
        '<div style="font-size:13px;">' +
        '<b>Severity:</b> ' + accident.Severity + '<br>' +
        '<b>Location:</b> ' + accident.City + ', ' + accident.State + '<br>' +
        '<b>Time:</b> ' + accident.Start_Time +
        '</div>'
    ).addTo(map);
    
    m.accidentData = accident; // Store accident data with marker
    accidentMarkers.push(m);
}});

// Route variables
var sourceMarker = null;
var destMarker = null;
var routeLine = null;
var routeAccidents = []; // Accidents along the route
var bufferDistance = 5;
var isRouteActive = false;

// Distance calculation (Haversine formula)
function getDistance(lat1, lng1, lat2, lng2){{
    var R = 6371; // Earth radius in km
    var dLat = (lat2 - lat1) * Math.PI / 180;
    var dLng = (lng2 - lng1) * Math.PI / 180;
    var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
            Math.sin(dLng/2) * Math.sin(dLng/2);
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}}

// Calculate perpendicular distance from point to line segment
function pointToLineDistance(px, py, x1, y1, x2, y2){{
    var A = px - x1;
    var B = py - y1;
    var C = x2 - x1;
    var D = y2 - y1;
    
    var dot = A * C + B * D;
    var len_sq = C * C + D * D;
    var param = -1;
    
    if (len_sq != 0) param = dot / len_sq;
    
    var xx, yy;
    
    if (param < 0) {{
        xx = x1;
        yy = y1;
    }} else if (param > 1) {{
        xx = x2;
        yy = y2;
    }} else {{
        xx = x1 + param * C;
        yy = y1 + param * D;
    }}
    
    return getDistance(px, py, xx, yy);
}}

// Filter and show only accidents along the route
function filterRouteAccidents(){{
    if (!sourceMarker || !destMarker) return;
    
    var s = sourceMarker.getLatLng();
    var d = destMarker.getLatLng();
    
    routeAccidents = [];
    
    accidentMarkers.forEach(function(marker){{
        var p = marker.getLatLng();
        var distance = pointToLineDistance(p.lat, p.lng, s.lat, s.lng, d.lat, d.lng);
        
        if (distance <= bufferDistance) {{
            // Accident is within buffer zone - show and highlight it
            marker.setStyle({{
                radius: 5,
                color: '#FFA500',
                fillColor: '#FF8C00',
                fillOpacity: 0.9,
                weight: 2
            }});
            marker.addTo(map);
            routeAccidents.push(marker);
        }} else {{
            // Accident is outside buffer zone - hide it
            map.removeLayer(marker);
        }}
    }});
    
    updateStats();
}}

// Update statistics display
function updateStats(){{
    var statsDiv = document.getElementById('route-stats');
    
    if (sourceMarker && destMarker) {{
        var s = sourceMarker.getLatLng();
        var d = destMarker.getLatLng();
        var dist = getDistance(s.lat, s.lng, d.lat, d.lng);
        
        // Count severity distribution
        var severityCounts = {{}};
        routeAccidents.forEach(function(m){{
            var sev = m.accidentData.Severity;
            severityCounts[sev] = (severityCounts[sev] || 0) + 1;
        }});
        
        var sevText = '';
        for (var sev in severityCounts) {{
            sevText += '<br>&nbsp;&nbsp;• Severity ' + sev + ': ' + severityCounts[sev];
        }}
        
        statsDiv.innerHTML = 
            '<b>📏 Route Distance:</b> ' + dist.toFixed(2) + ' km<br>' +
            '<b>⚠️ Accidents on Route:</b> <span class="info-badge">' + routeAccidents.length + '</span>' +
            sevText + '<br>' +
            '<b>🎯 Buffer Zone:</b> ± ' + bufferDistance + ' km';
    }} else {{
        statsDiv.innerHTML = '<span style="color:#666;">📍 Click on map to set source and destination</span>';
    }}
}}

// Show all accidents (when no route is active)
function showAllAccidents(){{
    accidentMarkers.forEach(function(marker){{
        marker.setStyle({{
            radius: 3,
            color: '#ff4444',
            fillColor: '#ff0000',
            fillOpacity: 0.6,
            weight: 1
        }});
        marker.addTo(map);
    }});
}}

// Map click handler to set source and destination
map.on('click', function(e){{
    if (!sourceMarker) {{
        // Set source marker
        sourceMarker = L.marker(e.latlng, {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            }})
        }}).addTo(map).bindPopup('<b>SOURCE</b>').openPopup();
        
    }} else if (!destMarker) {{
        // Set destination marker
        destMarker = L.marker(e.latlng, {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41],
                popupAnchor: [1, -34],
                shadowSize: [41, 41]
            }})
        }}).addTo(map).bindPopup('<b>DESTINATION</b>').openPopup();
        
        // Draw route line
        routeLine = L.polyline(
            [sourceMarker.getLatLng(), destMarker.getLatLng()],
            {{
                color: '#2196F3',
                weight: 4,
                opacity: 0.7,
                dashArray: '10, 5'
            }}
        ).addTo(map);
        
        isRouteActive = true;
        
        // Filter accidents to show only those along the route
        filterRouteAccidents();
        
        // Fit map to show the route
        var bounds = L.latLngBounds([sourceMarker.getLatLng(), destMarker.getLatLng()]);
        map.fitBounds(bounds, {{padding: [50, 50]}});
    }}
}});

// Clear route button
document.getElementById('clearRoute').onclick = function(){{
    // Remove markers and line
    if (sourceMarker) {{
        map.removeLayer(sourceMarker);
        sourceMarker = null;
    }}
    if (destMarker) {{
        map.removeLayer(destMarker);
        destMarker = null;
    }}
    if (routeLine) {{
        map.removeLayer(routeLine);
        routeLine = null;
    }}
    
    isRouteActive = false;
    routeAccidents = [];
    
    // Show all accidents again
    showAllAccidents();
    
    // Reset stats
    updateStats();
    
    // Reset view
    map.setView([39.5, -98.35], 4);
}};

// Buffer distance slider
document.getElementById('bufferSlider').oninput = function(e){{
    bufferDistance = parseFloat(e.target.value);
    document.getElementById('buffer-value').innerText = bufferDistance + ' km';
    
    // Re-filter accidents if route is active
    if (isRouteActive) {{
        filterRouteAccidents();
    }}
}};
</script>
</body>
</html>
"""
    
    # Save HTML
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Map saved as {output_file}")