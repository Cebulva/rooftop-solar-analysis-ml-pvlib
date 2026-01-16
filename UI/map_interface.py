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
    from roof_solar_heatmap import RoofSolarHeatmap
    from osm_to_roof_heatmap import OSMRoofHeatmap
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
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Latitude", f"{lat:.6f}")
    with c2:
        st.metric("Longitude", f"{lon:.6f}")
    
    # --- THE ANALYZE BUTTON ---
    with c3:
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
    
    # ========================================
    # ROOF SOLAR HEATMAP SECTION
    # ========================================
    st.divider()
    st.subheader("🔥 Advanced: Solar Exposure Heatmap")
    
    st.info("""
    **Automatically generate solar heatmap from building data!**
    
    This uses the building geometry from OpenStreetMap (same data as the Building Analysis above) 
    to create a roof segmentation and calculate solar exposure.
    
    No manual image upload needed! 🎉
    """)
    
    # Heatmap options
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        heatmap_type = st.radio(
            "Analysis Period:",
            ["Daily (Today)", "Yearly (Annual Average)"],
            key="osm_heatmap_type",
            help="Daily: Shows sun exposure for today. Yearly: Shows average exposure throughout the year."
        )
        
        multi_building = st.checkbox(
            "Include all nearby buildings",
            value=False,
            help="If checked, generates heatmap for all buildings in the area. Otherwise, only the clicked building."
        )
    
    with col_opt2:
        colormap_choice = st.selectbox(
            "Color Scheme:",
            ["hot", "plasma", "viridis", "inferno", "magma", "jet"],
            key="osm_colormap",
            help="Choose the color scheme for the heatmap"
        )
        
        image_resolution = st.select_slider(
            "Image Resolution:",
            options=[0.25, 0.5, 1.0, 2.0],
            value=0.5,
            format_func=lambda x: f"{x} m/pixel",
            help="Lower = higher detail but slower processing"
        )
    
    # Generate heatmap button
    if st.button("🔥 Generate Solar Heatmap from Building Data", key="generate_osm_heatmap"):
        with st.spinner("Generating solar heatmap from OpenStreetMap building data... This may take a moment."):
            try:
                # Initialize OSM heatmap generator
                osm_heatmap = OSMRoofHeatmap(
                    lat, 
                    lon, 
                    image_size=512, 
                    meters_per_pixel=image_resolution
                )
                
                # Generate heatmap
                results = osm_heatmap.generate_heatmap_from_osm(
                    heatmap_type='yearly' if 'Yearly' in heatmap_type else 'daily',
                    multi_building=multi_building,
                    search_radius=100,
                    samples_per_month=2
                )
                
                if not results['success']:
                    st.error(f"❌ {results['error']}")
                    if results['error'] == 'No buildings found in this area':
                        st.info("💡 This location may not have building data in OpenStreetMap. Try clicking on a building in an urban area, or upload a manual roof segmentation image below.")
                else:
                    st.success(f"✅ Heatmap generated successfully! Found {results['num_buildings']} building(s)")
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Show roof segmentation and heatmap side by side
                    col_img1, col_img2 = st.columns(2)
                    
                    with col_img1:
                        st.write("**Roof Segmentation (from OSM)**")
                        st.image(results['roof_image'], use_column_width=True, caption="Building footprints from OpenStreetMap")
                    
                    with col_img2:
                        st.write("**Solar Exposure Heatmap**")
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(6, 6))
                        im = ax.imshow(results['heatmap'], cmap=colormap_choice, vmin=0, vmax=1)
                        ax.axis('off')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Solar Exposure')
                        st.pyplot(fig)
                        plt.close()
                    
                    # Building information
                    if results['buildings_info']:
                        with st.expander("🏠 Building Details"):
                            for i, bldg in enumerate(results['buildings_info'], 1):
                                st.write(f"**Building {i}:**")
                                st.write(f"- Type: {bldg.get('type', 'unknown')}")
                                if bldg.get('height'):
                                    st.write(f"- Height: {bldg['height']} m")
                                if bldg.get('area_m2'):
                                    st.write(f"- Area: {bldg['area_m2']:.1f} m²")
                                if bldg.get('name') and bldg.get('name') != 'Unknown':
                                    st.write(f"- Name: {bldg['name']}")
                                st.write("---")
                    
                    # Exposure zones analysis
                    st.subheader("📊 Roof Exposure Analysis")
                    zones = results['zones']
                    
                    zone_data = []
                    for zone_name, data in zones.items():
                        zone_data.append({
                            'Zone': zone_name,
                            'Exposure Range': f"{data['exposure_range'][0]:.2f} - {data['exposure_range'][1]:.2f}",
                            'Coverage': f"{data['percentage']:.1f}%",
                            'Avg Exposure': f"{data['avg_exposure']:.3f}"
                        })
                    
                    import pandas as pd
                    zone_df = pd.DataFrame(zone_data)
                    st.dataframe(zone_df, use_container_width=True)
                    
                    # Optimal panel locations
                    st.subheader("⚡ Optimal Solar Panel Locations")
                    
                    num_panels = st.slider(
                        "Number of panels to place:",
                        min_value=1,
                        max_value=50,
                        value=10,
                        key="osm_panel_count",
                        help="Find the best locations for this many solar panels"
                    )
                    
                    # Recalculate with user-specified panel count
                    optimal_locs = results['generator'].find_optimal_panel_locations(
                        results['heatmap'], 
                        panel_count=num_panels,
                        min_exposure=0.5
                    )
                    
                    if optimal_locs:
                        # Create overlay
                        overlay = results['generator'].create_overlay_heatmap(
                            results['heatmap'], 
                            colormap=colormap_choice
                        )
                        
                        # Draw green circles at optimal locations
                        import cv2
                        for y, x in optimal_locs:
                            cv2.circle(overlay, (x, y), 8, (0, 255, 0), 2)
                            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
                        
                        st.image(
                            overlay, 
                            caption=f"Optimal locations for {len(optimal_locs)} solar panels (marked in green)",
                            use_column_width=True
                        )
                        st.success(f"✅ Found {len(optimal_locs)} optimal panel locations with good sun exposure")
                    else:
                        st.warning("⚠️ No suitable locations found with minimum exposure threshold of 0.5")
                    
                    # Download buttons
                    st.subheader("📥 Download Results")
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Download heatmap
                        import io
                        buf = io.BytesIO()
                        fig_dl = plt.figure(figsize=(10, 8))
                        plt.imshow(results['heatmap'], cmap=colormap_choice, vmin=0, vmax=1)
                        plt.title(f"Solar Exposure Heatmap - ({lat:.4f}, {lon:.4f})")
                        plt.colorbar(label='Solar Exposure')
                        plt.axis('off')
                        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        plt.close()
                        
                        st.download_button(
                            label="📥 Download Heatmap",
                            data=buf,
                            file_name=f"solar_heatmap_{lat}_{lon}.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        # Download roof segmentation
                        buf_roof = io.BytesIO()
                        import cv2
                        cv2.imwrite('temp_roof.png', results['roof_image'])
                        with open('temp_roof.png', 'rb') as f:
                            buf_roof = io.BytesIO(f.read())
                        os.remove('temp_roof.png')
                        
                        st.download_button(
                            label="📥 Download Roof Segmentation",
                            data=buf_roof,
                            file_name=f"roof_segmentation_{lat}_{lon}.png",
                            mime="image/png"
                        )
            
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Optional: Manual upload section for locations without OSM data
    st.divider()
    with st.expander("🎨 Alternative: Upload Custom Roof Segmentation"):
        st.write("""
        If building data is not available in OpenStreetMap for your location, 
        you can manually upload a black & white image where:
        - **White pixels** = Roof areas
        - **Black pixels** = Everything else
        """)
        
        uploaded_roof = st.file_uploader(
            "Upload Roof Segmentation Image (PNG/JPG)", 
            type=['png', 'jpg', 'jpeg'],
            key="manual_roof_upload"
        )
        
        if uploaded_roof is not None:
            st.image(uploaded_roof, width=300, caption="Your uploaded roof segmentation")
            
            manual_heatmap_type = st.radio(
                "Analysis Period:",
                ["Daily", "Yearly"],
                key="manual_heatmap_type"
            )
            
            manual_colormap = st.selectbox(
                "Color Scheme:",
                ["hot", "plasma", "viridis", "inferno", "magma", "jet"],
                key="manual_colormap"
            )
            
            if st.button("🔥 Generate from Uploaded Image", key="generate_manual_heatmap"):
                with st.spinner("Generating heatmap from uploaded image..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            tmp_file.write(uploaded_roof.getvalue())
                            temp_path = tmp_file.name
                        
                        # Initialize heatmap generator
                        generator = RoofSolarHeatmap(lat, lon, temp_path)
                        
                        # Generate heatmap
                        if manual_heatmap_type == "Daily":
                            heatmap = generator.create_daily_heatmap()
                            title = f"Daily Solar Exposure - {datetime.now().strftime('%Y-%m-%d')}"
                        else:
                            heatmap = generator.create_yearly_heatmap(samples_per_month=2)
                            title = "Annual Average Solar Exposure"
                        
                        # Visualize
                        import matplotlib.pyplot as plt
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
                        ax1.imshow(generator.roof_mask, cmap='gray')
                        ax1.set_title('Your Roof Segmentation')
                        ax1.axis('off')
                        
                        im = ax2.imshow(heatmap, cmap=manual_colormap, vmin=0, vmax=1)
                        ax2.set_title(title)
                        ax2.axis('off')
                        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Solar Exposure')
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Clean up
                        os.unlink(temp_path)
                        
                        st.success("✅ Manual heatmap generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())