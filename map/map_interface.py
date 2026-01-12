import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from free_downloader import get_free_satellite_image
import ssl
from free_downloader import get_free_satellite_image, calculate_gsd

try:
    from solar_analysis import analyze_solar_potential
except ImportError:
    st.error("Could not find 'solar_analysis.py'. Please make sure it is in the same folder.")

# --- SSL CERTIFICATE PATCH (Fixes 'CERTIFICATE_VERIFY_FAILED' errors) ---

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- APP CONFIG ---
st.set_page_config(layout="wide", page_title="Solar Potential Map")

st.title("📍 Satellite Solar Potential Selector")

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
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latitude", f"{lat:.6f}")
    with c2:
        st.metric("Longitude", f"{lon:.6f}")
    
    # --- THE ANALYZE BUTTON ---
    with c3:
        # This is the ONLY button. Make sure no other button exists below this block!
        if st.button("Confirm & Analyze", key="analyze_btn"):
            
            with st.spinner("Fetching satellite weather data & running simulation..."):
                
                # Call the function from solar_analysis.py
                # It returns TWO values: (data, error_message)
                data, error_msg = analyze_solar_potential(lat, lon)
                
                if error_msg:
                    st.error(f"Simulation Failed: {error_msg}")
                    st.info("Try a location on land.")
                
                elif data is not None:
                    st.success("Analysis Complete!")
                    st.divider()
                    
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
                    
                    st.subheader("Monthly Breakdown")
                    st.bar_chart(data.filter(like='Days'))
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