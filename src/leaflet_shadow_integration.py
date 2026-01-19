"""
Leaflet Shadow Simulator Integration for Solar Analysis
Provides real-time shadow visualization on interactive maps
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import json

def create_shadow_map_html(lat, lon, date=None, zoom=19, width=1000, height=600):
    """
    Create an HTML component with Leaflet Shadow Simulator.

    Args:
        lat: Latitude
        lon: Longitude
        date: datetime object for shadow calculation (defaults to now)
        zoom: Map zoom level
        width: Map width in pixels
        height: Map height in pixels

    Returns:
        HTML string for the shadow map
    """
    if date is None:
        date = datetime.now()

    # Convert datetime to JavaScript timestamp (milliseconds)
    js_timestamp = int(date.timestamp() * 1000)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
        <script src='https://www.unpkg.com/suncalc@1.9.0/suncalc.js'></script>
        <script src="https://unpkg.com/osmtogeojson/osmtogeojson.js"></script>
        <script src="https://unpkg.com/leaflet-shadow-simulator/dist/leaflet-shadow-simulator.umd.min.js"></script>
        <style>
            body {{
                padding: 0;
                margin: 0;
            }}
            #map {{
                width: 100%;
                height: {height}px;
            }}
            /* Hide Leaflet attribution */
            .leaflet-control-attribution {{
                display: none !important;
            }}
            .control-panel {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                z-index: 1000;
                max-width: 300px;
            }}
            .control-panel h3 {{
                margin: 0 0 10px 0;
                font-size: 16px;
                font-family: Arial, sans-serif;
            }}
            .control-panel button {{
                margin: 5px 2px;
                padding: 8px 12px;
                cursor: pointer;
                border: 1px solid #ccc;
                background: white;
                border-radius: 4px;
                font-size: 14px;
            }}
            .control-panel button:hover {{
                background: #f0f0f0;
            }}
            .info-box {{
                margin-top: 10px;
                padding: 10px;
                background: #f9f9f9;
                border-radius: 4px;
                font-family: Arial, sans-serif;
                font-size: 13px;
            }}
            .loader {{
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }}
            #exposure-info {{
                margin-top: 10px;
                padding: 8px;
                background: #e3f2fd;
                border-radius: 4px;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div class="control-panel">
            <h3>Shadow Simulator</h3>
            <div>
                <button id="decrement">-1 Hour</button>
                <button id="increment">+1 Hour</button>
            </div>
            <div>
                <button id="play">Play</button>
                <button id="stop">Stop</button>
            </div>
            <div style="margin-top: 10px;">
                <label>
                    <input type="checkbox" id="exposure">
                    <span style="font-size: 13px;">Full-day sun exposure</span>
                </label>
            </div>
            <div class="info-box" id="time-display"></div>
            <div class="loader" id="loader"></div>
            <div id="exposure-info"></div>
        </div>

        <script>
            // Initialize map
            var map = L.map('map').setView([{lat}, {lon}], {zoom});

            // Add satellite imagery
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                attribution: 'Esri',
                maxZoom: 19
            }}).addTo(map);

            // Add location marker
            L.marker([{lat}, {lon}], {{
                icon: L.icon({{
                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [1, -34],
                    shadowSize: [41, 41]
                }})
            }}).addTo(map).bindPopup('Selected Location');

            // Initialize shadow simulator
            let now = new Date({js_timestamp});
            const loaderEl = document.getElementById('loader');
            const timeDisplay = document.getElementById('time-display');
            const exposureInfo = document.getElementById('exposure-info');

            const shadeMap = L.shadeMap({{
                apiKey: "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6InRwcGlvdHJvd3NraUBzaGFkZW1hcC5hcHAiLCJjcmVhdGVkIjoxNjYyNDkzMDY2Nzk0LCJpYXQiOjE2NjI0OTMwNjZ9.ovCrLTYsdKFTF6TW3DuODxCaAtGQ3qhcmqj3DWcol5g",
                date: now,
                color: '#01112f',
                opacity: 0.7,
                terrainSource: {{
                    maxZoom: 15,
                    tileSize: 256,
                    getSourceUrl: ({{ x, y, z }}) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${{z}}/${{x}}/${{y}}.png`,
                    getElevation: ({{ r, g, b, a }}) => (r * 256 + g + b / 256) - 32768,
                }},
                getFeatures: async () => {{
                    try {{
                        if (map.getZoom() > 15) {{
                            const bounds = map.getBounds();
                            const north = bounds.getNorth();
                            const south = bounds.getSouth();
                            const east = bounds.getEast();
                            const west = bounds.getWest();
                            const query = `https://overpass-api.de/api/interpreter?data=%2F*%0AThis%20has%20been%20generated%20by%20the%20overpass-turbo%20wizard.%0AThe%20original%20search%20was%3A%0A%E2%80%9Cbuilding%E2%80%9D%0A*%2F%0A%5Bout%3Ajson%5D%5Btimeout%3A25%5D%3B%0A%2F%2F%20gather%20results%0A%28%0A%20%20%2F%2F%20query%20part%20for%3A%20%E2%80%9Cbuilding%E2%80%9D%0A%20%20way%5B%22building%22%5D%28${{south}}%2C${{west}}%2C${{north}}%2C${{east}}%29%3B%0A%29%3B%0A%2F%2F%20print%20results%0Aout%20body%3B%0A%3E%3B%0Aout%20skel%20qt%3B`;
                            const response = await fetch(query);
                            const json = await response.json();
                            const geojson = osmtogeojson(json);

                            // Default building height to 3 meters if not specified
                            geojson.features.forEach(feature => {{
                                if (!feature.properties) {{
                                    feature.properties = {{}};
                                }}
                                if (!feature.properties.height) {{
                                    feature.properties.height = 3;
                                }}
                            }});
                            return geojson.features;
                        }}
                    }} catch (e) {{
                        console.error('Error loading buildings:', e);
                    }}
                    return [];
                }}
            }}).addTo(map);

            // Update time display
            function updateTimeDisplay() {{
                const options = {{
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                }};
                timeDisplay.innerHTML = `<strong>Time:</strong> ${{now.toLocaleString('en-US', options)}}`;

                // Calculate sun position
                const sunPos = SunCalc.getPosition(now, {lat}, {lon});
                const altitude = sunPos.altitude * 180 / Math.PI;
                const azimuth = sunPos.azimuth * 180 / Math.PI;

                if (altitude > 0) {{
                    timeDisplay.innerHTML += `<br><strong>Sun Altitude:</strong> ${{altitude.toFixed(1)}}°`;
                }} else {{
                    timeDisplay.innerHTML += `<br><strong>Status:</strong> Nighttime`;
                }}
            }}

            updateTimeDisplay();

            // Track loading progress
            shadeMap.on('tileloaded', (loadedTiles, totalTiles) => {{
                loaderEl.innerText = `Loading: ${{(loadedTiles / totalTiles * 100).toFixed(0)}}%`;
                if (loadedTiles === totalTiles) {{
                    setTimeout(() => loaderEl.innerText = '', 1000);
                }}
            }});

            // Controls
            let intervalTimer;
            const increment = document.getElementById('increment');
            const decrement = document.getElementById('decrement');
            const play = document.getElementById('play');
            const stop = document.getElementById('stop');
            const exposure = document.getElementById('exposure');

            increment.addEventListener('click', () => {{
                now = new Date(now.getTime() + 3600000);
                shadeMap.setDate(now);
                updateTimeDisplay();
            }});

            decrement.addEventListener('click', () => {{
                now = new Date(now.getTime() - 3600000);
                shadeMap.setDate(now);
                updateTimeDisplay();
            }});

            play.addEventListener('click', () => {{
                intervalTimer = setInterval(() => {{
                    now = new Date(now.getTime() + 300000); // 5 minutes
                    shadeMap.setDate(now);
                    updateTimeDisplay();
                }}, 200);
            }});

            stop.addEventListener('click', () => {{
                clearInterval(intervalTimer);
            }});

            exposure.addEventListener('click', (e) => {{
                clearInterval(intervalTimer);
                const target = e.target;
                if (!target.checked) {{
                    shadeMap && shadeMap.setSunExposure(false);
                    increment.disabled = false;
                    decrement.disabled = false;
                    play.disabled = false;
                    stop.disabled = false;
                    exposureInfo.style.display = 'none';
                }} else {{
                    const {{ sunrise, sunset }} = SunCalc.getTimes(now, {lat}, {lon});
                    shadeMap && shadeMap.setSunExposure(true, {{
                        startDate: sunrise,
                        endDate: sunset,
                        iterations: 32
                    }});
                    increment.disabled = true;
                    decrement.disabled = true;
                    play.disabled = true;
                    stop.disabled = true;

                    const hours = (sunset - sunrise) / 1000 / 3600;
                    exposureInfo.style.display = 'block';
                    exposureInfo.innerHTML = `<strong>Daylight Hours:</strong> ${{hours.toFixed(1)}} hours<br><small>Blue = less sun, Red = more sun</small>`;
                }}
            }});
        </script>
    </body>
    </html>
    """

    return html


def render_shadow_map(lat, lon, date=None, zoom=19, height=600):
    """
    Render the shadow map in Streamlit.

    Args:
        lat: Latitude
        lon: Longitude
        date: datetime object for shadow calculation
        zoom: Map zoom level
        height: Map height in pixels
    """
    html = create_shadow_map_html(lat, lon, date, zoom, height=height)
    components.html(html, height=height + 20, scrolling=False)


def calculate_shadow_metrics(lat, lon, date=None):
    """
    Calculate shadow-related metrics for a location.

    Args:
        lat: Latitude
        lon: Longitude
        date: datetime object (defaults to today)

    Returns:
        dict: Shadow metrics including sunrise, sunset, daylight hours, etc.
    """
    if date is None:
        date = datetime.now()

    try:
        from pysolar import solar
        import pytz

        # Ensure date has timezone
        if date.tzinfo is None:
            date = pytz.UTC.localize(date)

        # Calculate sun position
        altitude = solar.get_altitude(lat, lon, date)
        azimuth = solar.get_azimuth(lat, lon, date)

        # Calculate sunrise/sunset times
        # Approximate by checking hourly
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        sunrise = None
        sunset = None

        for hour in range(24):
            check_time = day_start + timedelta(hours=hour)
            alt = solar.get_altitude(lat, lon, check_time)

            if alt > 0 and sunrise is None:
                sunrise = check_time
            elif alt <= 0 and sunrise is not None and sunset is None:
                sunset = check_time
                break

        daylight_hours = 0
        if sunrise and sunset:
            daylight_hours = (sunset - sunrise).total_seconds() / 3600

        return {
            'current_altitude': altitude,
            'current_azimuth': azimuth,
            'is_daytime': altitude > 0,
            'sunrise': sunrise,
            'sunset': sunset,
            'daylight_hours': daylight_hours,
            'date': date
        }

    except Exception as e:
        return {
            'error': str(e),
            'date': date
        }