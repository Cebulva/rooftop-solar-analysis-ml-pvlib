import sys
import os
from pathlib import Path

# Get absolute path to src folder
current_file = Path(__file__).resolve()
ui_folder = current_file.parent
project_root = ui_folder.parent
src_folder = project_root / 'src'

# Add src to path and verify it exists
if src_folder.exists():
    sys.path.insert(0, str(src_folder))
    print(f"Added to sys.path: {src_folder}")
else:
    raise FileNotFoundError(f"src folder not found at: {src_folder}")

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import ssl
import tempfile
from datetime import datetime

# Import the modules
try:
    from free_downloader import get_free_satellite_image, calculate_gsd
    from solar_analysis import analyze_solar_potential
    from shadow_analysis import calculate_sun_angles, calculate_daily_sun_exposure, get_optimal_panel_angle
    from building_analysis import analyze_building_solar_potential
    from leaflet_shadow_integration import render_shadow_map, calculate_shadow_metrics
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error(f"sys.path: {sys.path}")
    st.error(f"Files in src: {list(src_folder.glob('*.py'))}")
    raise

# --- SSL CERTIFICATE PATCH (Fixes 'CERTIFICATE_VERIFY_FAILED' errors) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- APP CONFIG ---
st.set_page_config(layout="wide", page_title="Solar Potential Map")

st.title("🌍 Satellite Solar Potential Selector")

# --- 1. SESSION STATE SETUP ---
if "map_center" not in st.session_state:
    st.session_state["map_center"] = [37.7749, -122.4194] # Default: San Francisco
if "selected_pos" not in st.session_state:
    st.session_state["selected_pos"] = None

# --- 2. SEARCH BAR ---
geolocator = Nominatim(user_agent="solar_project_team_app_v1")

with st.form("search_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        address_input = st.text_input("Search Address", placeholder="e.g. 1600 Amphitheatre Parkway, Mountain View, CA")
    with col2:
        search_submitted = st.form_submit_button("🔍 Search")

if search_submitted and address_input:
    try:
        with st.spinner("Locating..."):
            location = geolocator.geocode(address_input, timeout=10)
            if location:
                st.session_state["map_center"] = [location.latitude, location.longitude]
                st.session_state["selected_pos"] = None 
                st.success(f"Moved to: {location.address}")
            else:
                st.error("Address not found.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- 3. MAP RENDERING ---
center_lat, center_lon = st.session_state["map_center"]
m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

# Satellite Tiles
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

# Add Pin if selected
if st.session_state["selected_pos"]:
    pin_lat, pin_lon = st.session_state["selected_pos"]
    folium.Marker(
        [pin_lat, pin_lon],
        popup="Selected Property",
        icon=folium.Icon(color="red", icon="home"),
        draggable=False
    ).add_to(m)

st.write("Click specifically on the **center of your roof**.")
output = st_folium(m, width=1000, height=600)

# --- 4. CLICK HANDLING ---
if output["last_clicked"]:
    new_lat = output["last_clicked"]["lat"]
    new_lng = output["last_clicked"]["lng"]
    
    current_pos = st.session_state["selected_pos"]
    
    # Only update if the click is actually new/different
    if current_pos is None or (new_lat != current_pos[0] or new_lng != current_pos[1]):
        st.session_state["selected_pos"] = (new_lat, new_lng)
        st.rerun()

# --- 5. RESULTS & ANALYSIS ---
if st.session_state["selected_pos"]:
    lat, lon = st.session_state["selected_pos"]
    
    st.divider()
    st.header("✅ Selected Coordinates")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latitude", f"{lat:.6f}")
    with c2:
        st.metric("Longitude", f"{lon:.6f}")

    # --- REAL-TIME SHADOW VISUALIZATION ---
    st.divider()
    st.header("🌓 Real-Time Shadow & Sun Exposure Simulator")

    st.info("""
    **Interactive Shadow Visualization**

    This map shows real-time shadows cast by buildings and terrain at your location. You can:
    - Use +1/-1 Hour buttons to see shadows at different times of day
    - Click Play to animate shadow movement throughout the day
    - Enable "Full-day sun exposure" to see cumulative sunlight (blue = less sun, red = more sun)
    - Zoom in to see detailed building shadows (zoom > 15 for building data)

    This helps identify optimal solar panel placement by showing which roof areas receive the most sunlight!
    """)

    # Shadow metrics in columns
    shadow_metrics = calculate_shadow_metrics(lat, lon)

    if 'error' not in shadow_metrics:
        col_shadow1, col_shadow2, col_shadow3, col_shadow4 = st.columns(4)

        with col_shadow1:
            st.metric(
                "Current Sun Altitude",
                f"{shadow_metrics['current_altitude']:.1f}°",
                help="Angle of sun above horizon"
            )

        with col_shadow2:
            st.metric(
                "Current Sun Azimuth",
                f"{shadow_metrics['current_azimuth']:.1f}°",
                help="Direction from North (0°=N, 90°=E, 180°=S, 270°=W)"
            )

        with col_shadow3:
            st.metric(
                "Daylight Hours Today",
                f"{shadow_metrics['daylight_hours']:.1f} hrs",
                help="Total hours of sunlight today"
            )

        with col_shadow4:
            status = "☀️ Daytime" if shadow_metrics['is_daytime'] else "🌙 Nighttime"
            st.metric("Current Status", status)

        # Display sunrise/sunset times
        if shadow_metrics['sunrise'] and shadow_metrics['sunset']:
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                st.write(f"🌅 **Sunrise:** {shadow_metrics['sunrise'].strftime('%I:%M %p')}")
            with col_time2:
                st.write(f"🌇 **Sunset:** {shadow_metrics['sunset'].strftime('%I:%M %p')}")

    # Render the interactive shadow map
    st.subheader("📍 Interactive Shadow Map")
    render_shadow_map(lat, lon, height=600)

    st.markdown("""
    <small>
    💡 **Tips:**
    - Areas that stay bright throughout the day are best for solar panels
    - Shadows from nearby buildings significantly reduce solar efficiency
    - The "Full-day sun exposure" mode shows cumulative sunlight - red areas get the most sun!
    </small>
    """, unsafe_allow_html=True)

    # --- THE ANALYZE BUTTON ---
    st.divider()
    st.header("📊 Detailed Solar Analysis")

    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("Confirm & Analyze", key="analyze_btn"):
            
            with st.spinner("Fetching satellite weather data & running simulation..."):
                
                # 1. Solar Analysis
                data, error_msg = analyze_solar_potential(lat, lon)
                
                # 2. Shadow/Sun Analysis
                sun_angles = calculate_sun_angles(lat, lon)
                daily_sun = calculate_daily_sun_exposure(lat, lon)
                optimal_angles = get_optimal_panel_angle(lat)
                
                # 3. Building/Rooftop Analysis
                building_info = analyze_building_solar_potential(lat, lon)
                
                if error_msg:
                    st.error(f"Simulation Failed: {error_msg}")
                    st.info("Try a location on land.")
                
                elif data is not None:
                    st.success("✅ Comprehensive Analysis Complete!")
                    st.divider()
                    
                    # === SUN POSITION & SHADOW ANALYSIS ===
                    st.subheader("☀️ Current Sun Position")
                    col_sun1, col_sun2, col_sun3 = st.columns(3)
                    
                    with col_sun1:
                        st.metric(
                            "Sun Altitude", 
                            f"{sun_angles['altitude']:.1f}°",
                            help="Angle above horizon (0° = horizon, 90° = directly overhead)"
                        )
                    with col_sun2:
                        st.metric(
                            "Sun Azimuth", 
                            f"{sun_angles['azimuth']:.1f}°",
                            help="Direction from North (0° = North, 90° = East, 180° = South, 270° = West)"
                        )
                    with col_sun3:
                        status = "☀️ Daytime" if sun_angles['is_daytime'] else "🌙 Nighttime"
                        st.metric("Sun Status", status)
                    
                    # Daily Sun Exposure Chart
                    if daily_sun:
                        st.subheader("📊 Daily Sun Exposure Pattern")
                        import pandas as pd
                        sun_df = pd.DataFrame(daily_sun)
                        
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            st.write("**Sun Altitude Throughout Day**")
                            st.line_chart(sun_df.set_index('hour')['altitude'])
                        with col_chart2:
                            st.write("**Solar Radiation Throughout Day**")
                            st.area_chart(sun_df.set_index('hour')['radiation'])
                    
                    # Optimal Panel Angles
                    st.subheader("🔧 Recommended Solar Panel Angles")
                    col_angle1, col_angle2, col_angle3 = st.columns(3)
                    with col_angle1:
                        st.metric("Year-Round Optimal", f"{optimal_angles['year_round_optimal']}°")
                    with col_angle2:
                        st.metric("Summer Optimal", f"{optimal_angles['summer_optimal']}°")
                    with col_angle3:
                        st.metric("Winter Optimal", f"{optimal_angles['winter_optimal']}°")
                    
                    st.info("💡 " + optimal_angles['note'])
                    
                    st.divider()
                    
                    # === BUILDING & ROOFTOP ANALYSIS ===
                    if building_info:
                        st.subheader("🏠 Building & Rooftop Analysis")
                        
                        col_bldg1, col_bldg2, col_bldg3, col_bldg4 = st.columns(4)
                        
                        with col_bldg1:
                            st.metric(
                                "Roof Area", 
                                f"{building_info['roof_analysis']['usable_area_m2']:.1f} m²",
                                help="Estimated usable roof space for solar panels"
                            )
                        with col_bldg2:
                            st.metric(
                                "Building Height", 
                                f"{building_info['building_info']['height']:.1f} m",
                                help="Estimated building height"
                            )
                        with col_bldg3:
                            st.metric(
                                "Solar Panels", 
                                f"{building_info['solar_potential']['num_panels']}",
                                help="Estimated number of panels that can fit"
                            )
                        with col_bldg4:
                            st.metric(
                                "System Capacity", 
                                f"{building_info['solar_potential']['total_capacity_kw']:.1f} kW",
                                help="Total solar system capacity"
                            )
                        
                        # Additional Building Info
                        with st.expander("📋 Detailed Building Information"):
                            st.write(f"**Building Type:** {building_info['building_info']['type']}")
                            st.write(f"**Building Name:** {building_info['building_info']['name']}")
                            if building_info['building_info']['levels']:
                                st.write(f"**Estimated Floors:** {building_info['building_info']['levels']}")
                            st.write(f"**Total Roof Area:** {building_info['roof_analysis']['total_area_m2']:.1f} m²")
                            st.write(f"**Usable Percentage:** {building_info['roof_analysis']['usable_percentage']:.0f}%")
                        
                        # Solar Installation Potential
                        with st.expander("⚡ Solar Installation Potential"):
                            st.write(f"**Number of Panels:** {building_info['solar_potential']['num_panels']}")
                            st.write(f"**Panel Size:** {building_info['solar_potential']['panel_size_m2']} m² each")
                            st.write(f"**Watts per Panel:** {building_info['solar_potential']['watts_per_panel']} W")
                            st.write(f"**Total System Capacity:** {building_info['solar_potential']['total_capacity_kw']:.2f} kW")
                            st.write(f"**Estimated Annual Production:** {building_info['solar_potential']['estimated_annual_production_kwh']:,.0f} kWh")
                            
                            # Calculate potential savings (estimate $0.12 per kWh)
                            annual_savings = building_info['solar_potential']['estimated_annual_production_kwh'] * 0.12
                            st.write(f"**Estimated Annual Savings:** ${annual_savings:,.0f} (at $0.12/kWh)")
                    else:
                        st.warning("⚠️ Building data not available for this location. The location may not have detailed building information in OpenStreetMap.")
                    
                    st.divider()
                    
                    # === SOLAR ENERGY ANALYSIS ===
                    st.subheader("⚡ Solar Energy Production Analysis")
                    
                    # Calculate Metrics
                    total_yearly_yield = data['Energy Output (kWh/m²)'].sum()
                    
                    if 'Sunny Days' in data.columns:
                        avg_sunny = data['Sunny Days'].mean() * 12
                    else:
                        avg_sunny = 0
                        
                    # Display
                    m1, m2 = st.columns(2)
                    m1.metric("Est. Annual Yield", f"{total_yearly_yield:.1f} kWh/m²")
                    m2.metric("Est. Sunny Days / Year", f"{int(avg_sunny)}")
                    
                    st.subheader("📅 Monthly Breakdown")
                    
                    col_chart3, col_chart4 = st.columns(2)
                    with col_chart3:
                        st.write("**Weather Conditions**")
                        st.bar_chart(data.filter(like='Days'))
                    with col_chart4:
                        st.write("**Energy Output**")
                        st.line_chart(data['Energy Output (kWh/m²)'])
                    
                    st.dataframe(data, use_container_width=True)

    st.divider()
    st.subheader("📷 Site Snapshot (Free)")

    if st.button("Download Image"):
        zoom_level = 19
        safe_filename = f"satellite_{lat}_{lon}.png"
        
        saved_file = get_free_satellite_image(lat, lon, zoom=zoom_level, filename=safe_filename)
        
        if saved_file:
            # 1. Calculate Resolution
            meters_per_pixel = calculate_gsd(lat, zoom_level)
            
            # 2. Calculate Total Image Width (Standard tile is 256x256 pixels)
            image_width_meters = meters_per_pixel * 256
            
            st.success(f"Image Downloaded! (Resolution: {meters_per_pixel:.3f} meters/pixel)")
            
            # 3. Display Image with Scale Info
            st.image(saved_file, width=400)
            
            st.info(f"📏 Scale: This image is approx. **{image_width_meters:.1f} meters** wide.")
    
