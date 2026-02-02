import streamlit as st
import pandas as pd
from datetime import datetime

# Import from src
from src.pdf_generator import generate_solar_report_pdf
from src.dealer_finder import find_nearby_dealers, init_database, add_sample_dealers
from src.dealer_grader import get_top_dealers, get_grade_color, GradedDealer
from src.quote_generator import process_quote_submission
from src.german_solar_calculator import (
    GermanSolarCalculator,
    ORIENTATION_MAP,
    CONSTANTS
)
from src.solar_analysis import analyze_solar_potential
from src.shadow_analysis import get_optimal_panel_angle
from src.building_analysis import analyze_building_solar_potential

# Month order for proper sorting (full names to match solar_analysis output)
MONTH_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Environmental Impact Constants (Germany-specific)
ENV_IMPACT = {
    'GRID_CO2_KG_PER_KWH': 0.380,      # kg CO2 per kWh (German grid 2024)
    'CO2_PER_TREE_KG_YEAR': 21,         # kg CO2 absorbed per mature tree per year
    'CAR_CO2_KG_PER_YEAR': 2400,        # kg CO2 per avg car per year (12,000 km)
    'CO2_PER_KM_KG': 0.12,              # kg CO2 per km driven (avg car)
    'COAL_KG_PER_KWH': 0.9,             # kg coal per kWh (coal plant)
    'PHONE_CHARGE_WH': 12,              # Wh per smartphone charge
    'PANEL_LIFETIME_YEARS': 25,         # Solar panel lifespan
    'DEGRADATION_FACTOR': 0.92,         # 25-year avg accounting for efficiency loss
}


def azimuth_to_orientation(azimuth):
    """Convert azimuth angle to orientation name."""
    # Normalize azimuth to 0-360
    azimuth = azimuth % 360

    if 337.5 <= azimuth or azimuth < 22.5:
        return "North"
    elif 22.5 <= azimuth < 67.5:
        return "North-East"
    elif 67.5 <= azimuth < 112.5:
        return "East"
    elif 112.5 <= azimuth < 157.5:
        return "South-East"
    elif 157.5 <= azimuth < 202.5:
        return "South"
    elif 202.5 <= azimuth < 247.5:
        return "South-West"
    elif 247.5 <= azimuth < 292.5:
        return "West"
    else:
        return "North-West"


def tilt_to_category(tilt):
    """Convert tilt angle to category name."""
    if tilt <= 5:
        return "Flat (0)"
    elif tilt <= 25:
        return "Low (15)"
    elif tilt <= 40:
        return "Normal (35)"
    else:
        return "Steep (45)"


def show():
    st.subheader("Step 4: Final Report")

    # Get data from previous stages
    lat = st.session_state.data.get("confirmed_lat")
    lon = st.session_state.data.get("confirmed_lon")
    consumption_inputs = st.session_state.data.get("consumption_inputs")
    solar_results = st.session_state.data.get("solar_results")

    # Validate required data
    if not lat or not lon:
        st.error("Location data not found. Please start from Step 1.")
        if st.button("🔄 Start Over"):
            st.session_state.step = 1
            st.session_state.data = {}
            st.rerun()
        return

    if not consumption_inputs:
        st.error("Consumption data not found. Please complete Step 3a.")
        if st.button("⬅️ Go to Questionnaire"):
            st.session_state.step = 3
            st.rerun()
        return

    if not solar_results:
        st.error("Solar analysis data not found. Please complete Step 3b.")
        if st.button("⬅️ Go to Solar Analysis"):
            st.session_state.step = 4
            st.rerun()
        return

    # Back navigation buttons
    col_back1, col_back2, col_back3 = st.columns(3)
    with col_back1:
        if st.button("⬅️ Edit Solar Analysis", key="back_to_3b"):
            for key in ["final_analysis", "detailed_solar_data"]:
                st.session_state.data.pop(key, None)
            st.session_state.pop("rag_bot", None)
            st.session_state.step = 4
            st.rerun()
    with col_back2:
        if st.button("⬅️ Edit Questionnaire", key="back_to_3a"):
            for key in ["final_analysis", "detailed_solar_data"]:
                st.session_state.data.pop(key, None)
            st.session_state.pop("rag_bot", None)
            st.session_state.step = 3
            st.rerun()
    with col_back3:
        if st.button("⬅️ Edit Roof", key="back_to_2"):
            for key in ["final_analysis", "detailed_solar_data"]:
                st.session_state.data.pop(key, None)
            st.session_state.pop("rag_bot", None)
            st.session_state.step = 2
            st.rerun()

    # Run full analysis if not done yet
    if "final_analysis" not in st.session_state.data:
        with st.spinner("Generating your personalized solar report..."):
            run_final_analysis(lat, lon, consumption_inputs, solar_results)

    render_final_report(lat, lon)


def run_final_analysis(lat, lon, consumption_inputs, solar_results):
    """Run the German Solar Calculator with data from previous stages."""

    # Convert azimuth to orientation name
    roof_orientation = azimuth_to_orientation(solar_results['azimuth'])

    # Convert tilt to category
    roof_tilt = tilt_to_category(solar_results['tilt_angle'])

    # Initialize calculator
    calc = GermanSolarCalculator(lat, lon)

    # Run full analysis with data from stage 3a (consumption) and stage 3b (roof)
    analysis = calc.full_analysis(
        people=consumption_inputs['people'],
        building_type=consumption_inputs['b_type_code'],
        roof_orientation=roof_orientation,
        roof_tilt=roof_tilt,
        has_water_heater=consumption_inputs['has_water_heater'],
        has_ev=consumption_inputs['has_ev'],
        has_heat_pump=consumption_inputs['has_heat_pump']
    )

    # Override with the exact production value from stage 3b (single source of truth)
    actual_kwp = solar_results['system_kwp']
    stage3b_production = solar_results.get('annual_production_kwh')
    if stage3b_production is not None and stage3b_production > 0:
        analysis['production']['annual_kwh'] = stage3b_production
        analysis['production']['specific_yield_kwh_kwp'] = (
            stage3b_production / actual_kwp if actual_kwp > 0 else 0
        )

    # Store the analysis
    st.session_state.data['final_analysis'] = analysis
    st.session_state.data['roof_orientation_name'] = roof_orientation
    st.session_state.data['roof_tilt_name'] = roof_tilt


def render_final_report(lat, lon):
    """Render the complete final report."""

    analysis = st.session_state.data['final_analysis']
    solar_results = st.session_state.data['solar_results']
    consumption_inputs = st.session_state.data['consumption_inputs']

    # === SUCCESS HEADER ===
    st.success("☀️ Solar Analysis Complete!")
    st.divider()

    # === SYSTEM RECOMMENDATION ===
    # Use actual panel count from stage 3b instead of calculator recommendation
    actual_panels = solar_results['panel_count']
    actual_kwp = solar_results['system_kwp']

    st.header(f"Your System: {actual_kwp:.1f} kWp ({actual_panels} panels)")

    # Key Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            "Annual Consumption",
            f"{consumption_inputs['annual_kwh']:,} kWh",
            help="Based on Stromspiegel 2025"
        )

    with col_m2:
        # Use the exact value from stage 3b (single source of truth)
        actual_production = solar_results.get('annual_production_kwh', 0)
        specific_yield = actual_production / actual_kwp if actual_kwp > 0 else 0
        st.metric(
            "Solar Production",
            f"{actual_production:,.0f} kWh",
            help="Based on location and panel configuration"
        )

    with col_m3:
        # Calculate actual investment cost
        actual_invest = actual_kwp * CONSTANTS['INSTALLATION_COST_PER_KW']
        st.metric(
            "Investment Cost",
            f"{int(actual_invest):,} EUR",
            "0% VAT",
            help="Including installation, 0% VAT for residential solar"
        )

    with col_m4:
        # Calculate actual ROI
        annual_benefit = calculate_annual_benefit(
            actual_production,
            consumption_inputs['annual_kwh']
        )
        roi_years = actual_invest / annual_benefit if annual_benefit > 0 else 99
        if roi_years < 10:
            st.metric("Payback Period", f"{roi_years:.1f} years", delta="Good investment!")
        else:
            st.metric("Payback Period", f"{roi_years:.1f} years")

    # Coverage Progress Bar
    st.divider()
    coverage_ratio = min(actual_production / consumption_inputs['annual_kwh'], 1.0)
    coverage_percent = (actual_production / consumption_inputs['annual_kwh']) * 100
    st.progress(coverage_ratio, text=f"Coverage: {coverage_percent:.0f}% of your consumption")

    if coverage_percent >= 100:
        surplus = actual_production - consumption_inputs['annual_kwh']
        st.success(f"This system covers 100% of your annual consumption with {int(surplus):,} kWh surplus!")
    else:
        deficit = consumption_inputs['annual_kwh'] - actual_production
        st.info(f"This system covers {coverage_percent:.0f}% of your needs. You'll still need {int(deficit):,} kWh from the grid.")

    # Financial Details
    st.divider()
    col_fin1, col_fin2 = st.columns(2)

    with col_fin1:
        st.subheader("💶 Annual Savings Breakdown")

        # Self-consumption (assume 30% without battery)
        self_consumption_rate = 0.30
        self_consumed = min(actual_production * self_consumption_rate, consumption_inputs['annual_kwh'])
        savings_usage = self_consumed * CONSTANTS['ELECTRICITY_PRICE_DE']

        # Feed-in earnings
        fed_in = actual_production - self_consumed
        earnings_feedin = fed_in * CONSTANTS['FEED_IN_TARIFF']

        annual_benefit = savings_usage + earnings_feedin

        st.write(f"**Electricity cost savings:** {int(savings_usage):,} EUR/year")
        st.write(f"**Feed-in tariff (EEG):** {int(earnings_feedin):,} EUR/year")
        st.markdown(f"### Total: {int(annual_benefit):,} EUR/year")
        st.caption(f"Monthly benefit: ~{int(annual_benefit/12):,} EUR")

    with col_fin2:
        st.subheader("📊 System Details")
        st.write(f"**Number of panels:** {actual_panels} x {CONSTANTS['PANEL_POWER_W']}W")
        st.write(f"**Total roof area:** {solar_results['total_roof_area_m2']:.1f} m²")
        st.write(f"**Usable roof area:** {solar_results['usable_roof_area_m2']:.1f} m²")
        st.write(f"**Specific yield:** {specific_yield:.0f} kWh/kWp")
        st.write(f"**Orientation:** {st.session_state.data['roof_orientation_name']} ({solar_results['azimuth']:.0f}°)")
        st.write(f"**Tilt:** {st.session_state.data['roof_tilt_name']} ({solar_results['tilt_angle']:.0f}°)")
        st.write(f"**Roof type:** {solar_results['roof_form']}")

    # 25-year profit
    st.divider()
    lifetime_profit = (annual_benefit * 25) - actual_invest
    if lifetime_profit > 0:
        st.success(f"**25-Year Net Profit:** {int(lifetime_profit):,} EUR (after system pays itself off)")
    else:
        st.warning(f"**25-Year Net Result:** {int(lifetime_profit):,} EUR")

    # === ENVIRONMENTAL IMPACT ===
    st.divider()
    st.header("🌍 Environmental Impact")

    # Calculate environmental metrics
    env_metrics = calculate_environmental_impact(actual_production)

    # Display impact cards
    col_env1, col_env2, col_env3, col_env4 = st.columns(4)

    with col_env1:
        st.metric(
            "🌳 Trees Equivalent",
            f"{env_metrics['trees_equivalent']:.0f}",
            help="Number of mature trees needed to absorb the same CO₂"
        )

    with col_env2:
        st.metric(
            "💨 CO₂ Avoided",
            f"{env_metrics['co2_avoided_tonnes']:.1f} t/year",
            help="Tonnes of CO₂ emissions avoided annually"
        )

    with col_env3:
        st.metric(
            "🚗 Cars Off Road",
            f"{env_metrics['cars_equivalent']:.1f}",
            help="Equivalent cars removed from roads for a year"
        )

    with col_env4:
        st.metric(
            "�ite Coal Saved",
            f"{env_metrics['coal_avoided_kg']:.0f} kg",
            help="Kilograms of coal not burned annually"
        )

    # Lifetime impact
    st.markdown(f"""
    **📊 25-Year Lifetime Impact:**
    - **{env_metrics['lifetime_co2_tonnes']:.1f} tonnes** of CO₂ avoided
    - Equivalent to planting **{env_metrics['lifetime_trees']:.0f} trees**
    - Like driving **{env_metrics['km_avoided']:,.0f} km** less
    """)

    # Store for PDF
    st.session_state.data['env_metrics'] = env_metrics

    # Tips
    st.divider()
    st.info("""
    💡 **Tips to increase savings:**
    - Add a battery storage to increase self-consumption from 30% to 60-70%
    - Use high-power appliances (washing machine, dishwasher) during sunny hours
    - Consider an EV to utilize surplus solar production
    """)

    # === DETAILED SOLAR ANALYSIS (cached, loads once) ===
    render_detailed_analysis_section(lat, lon)

    # === FIND NEARBY DEALERS ===
    render_dealer_finder_section(lat, lon, solar_results, analysis)

    # === FINAL ACTIONS ===
    st.divider()
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        panel_image = st.session_state.data.get("pdf_panel_image")
        cached_detail = st.session_state.data.get("detailed_solar_data")
        monthly_data = cached_detail.get("data_sorted") if cached_detail else None
        env_metrics = st.session_state.data.get("env_metrics")

        try:
            pdf_bytes = generate_solar_report_pdf(
                solar_results=solar_results,
                consumption_inputs=consumption_inputs,
                final_analysis=analysis,
                location={'lat': lat, 'lon': lon},
                panel_image=panel_image,
                monthly_data=monthly_data,
                inquiry_id=st.session_state.get("inquiry_id"),
                env_metrics=env_metrics,
            )

            inquiry_id = st.session_state.get("inquiry_id")
            if inquiry_id:
                filename = f"solar_report_{inquiry_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            else:
                filename = f"solar_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            if st.button("📥 Download PDF Report", use_container_width=True, type="primary"):
                st.error(f"Error generating PDF: {str(e)}")

    with col_btn2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.step = 1
            st.session_state.data = {}
            st.session_state["selected_pos"] = None
            st.session_state.pop("rag_bot", None)
            st.rerun()


def render_detailed_analysis_section(lat, lon):
    """Render the detailed solar analysis section with session state caching."""
    st.divider()
    st.header("📊 Detailed Solar Analysis")

    # Cache results so they only fetch once per analysis
    if "detailed_solar_data" not in st.session_state.data:
        with st.spinner("☀️ Solar Assistant is analyzing satellite weather data..."):
            data, error_msg = analyze_solar_potential(lat, lon)
            optimal_angles = get_optimal_panel_angle(lat)
            building_info = analyze_building_solar_potential(lat, lon)

            data_sorted = None
            if data is not None and not error_msg:
                data_sorted = reindex_by_month(data)

            st.session_state.data["detailed_solar_data"] = {
                "data": data,
                "data_sorted": data_sorted,
                "error_msg": error_msg,
                "optimal_angles": optimal_angles,
                "building_info": building_info,
            }

    cached = st.session_state.data["detailed_solar_data"]

    if cached["error_msg"]:
        st.error(f"Simulation Failed: {cached['error_msg']}")
        st.info("Try a location on land.")

    elif cached["data"] is not None:
        optimal_angles = cached["optimal_angles"]
        building_info = cached["building_info"]
        data_sorted = cached["data_sorted"]
        data = cached["data"]

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

        # Building & Rooftop Analysis
        if building_info:
            st.subheader("🏠 Building & Rooftop Analysis (OpenStreetMap)")
            col_bldg1, col_bldg2, col_bldg3, col_bldg4 = st.columns(4)
            with col_bldg1:
                st.metric("OSM Roof Area", f"{building_info['roof_analysis']['usable_area_m2']:.1f} m²", help="Estimated from OpenStreetMap data")
            with col_bldg2:
                st.metric("Building Height", f"{building_info['building_info']['height']:.1f} m", help="Estimated building height")
            with col_bldg3:
                st.metric("Building Type", f"{building_info['building_info']['type']}")
            with col_bldg4:
                if building_info['building_info']['levels']:
                    st.metric("Floors", f"{building_info['building_info']['levels']}")
                else:
                    st.metric("Floors", "N/A")
            st.divider()

        # Solar Energy Analysis
        if data_sorted is not None:
            st.subheader("⚡ Solar Energy Production Analysis")
            total_yearly_yield = data['Energy Output (kWh/m²)'].sum()
            avg_sunny = data['Sunny Days'].mean() * 12 if 'Sunny Days' in data.columns else 0

            m1, m2 = st.columns(2)
            m1.metric("Est. Annual Yield", f"{total_yearly_yield:.1f} kWh/m²")
            m2.metric("Est. Sunny Days / Year", f"{int(avg_sunny)}")

            st.subheader("📅 Monthly Breakdown")
            col_chart3, col_chart4 = st.columns(2)
            with col_chart3:
                st.write("**Weather Conditions**")
                weather_cols = [col for col in data_sorted.columns if 'Days' in col]
                if weather_cols:
                    st.bar_chart(data_sorted[weather_cols])
            with col_chart4:
                st.write("**Energy Output**")
                st.line_chart(data_sorted['Energy Output (kWh/m²)'])

            st.dataframe(data_sorted, use_container_width=True)


def calculate_annual_benefit(production, consumption):
    """Calculate annual financial benefit from solar system."""
    # Self-consumption (assume 30% without battery)
    self_consumption_rate = 0.30
    self_consumed = min(production * self_consumption_rate, consumption)
    savings_usage = self_consumed * CONSTANTS['ELECTRICITY_PRICE_DE']

    # Feed-in earnings
    fed_in = production - self_consumed
    earnings_feedin = fed_in * CONSTANTS['FEED_IN_TARIFF']

    return savings_usage + earnings_feedin


def calculate_environmental_impact(annual_production_kwh: float) -> dict:
    """
    Calculate environmental impact metrics from solar production.

    Args:
        annual_production_kwh: Annual solar energy production in kWh

    Returns:
        Dictionary with environmental impact metrics
    """
    # Annual CO2 avoided (kg and tonnes)
    co2_avoided_kg = annual_production_kwh * ENV_IMPACT['GRID_CO2_KG_PER_KWH']
    co2_avoided_tonnes = co2_avoided_kg / 1000

    # Trees equivalent (number of mature trees to absorb same CO2)
    trees_equivalent = co2_avoided_kg / ENV_IMPACT['CO2_PER_TREE_KG_YEAR']

    # Cars off the road equivalent
    cars_equivalent = co2_avoided_kg / ENV_IMPACT['CAR_CO2_KG_PER_YEAR']

    # Kilometers of driving avoided
    km_avoided = co2_avoided_kg / ENV_IMPACT['CO2_PER_KM_KG']

    # Coal not burned (kg)
    coal_avoided_kg = annual_production_kwh * ENV_IMPACT['COAL_KG_PER_KWH']

    # Smartphone charges possible
    phone_charges = (annual_production_kwh * 1000) / ENV_IMPACT['PHONE_CHARGE_WH']

    # 25-year lifetime impact (with degradation factor)
    lifetime_years = ENV_IMPACT['PANEL_LIFETIME_YEARS']
    degradation = ENV_IMPACT['DEGRADATION_FACTOR']
    lifetime_co2_tonnes = co2_avoided_tonnes * lifetime_years * degradation
    lifetime_trees = trees_equivalent * lifetime_years

    return {
        'co2_avoided_kg': co2_avoided_kg,
        'co2_avoided_tonnes': co2_avoided_tonnes,
        'trees_equivalent': trees_equivalent,
        'cars_equivalent': cars_equivalent,
        'km_avoided': km_avoided,
        'coal_avoided_kg': coal_avoided_kg,
        'phone_charges': phone_charges,
        'lifetime_co2_tonnes': lifetime_co2_tonnes,
        'lifetime_trees': lifetime_trees,
    }


def reindex_by_month(data):
    """Reindex dataframe to proper month order instead of alphabetical.

    Streamlit charts sort x-axis alphabetically, so we prefix month names
    with numbers to force correct chronological order.
    """
    if len(data.index) > 0 and data.index[0] in MONTH_ORDER:
        # Create ordered dataframe by selecting rows in correct month order
        ordered_months = [m for m in MONTH_ORDER if m in data.index]
        data = data.loc[ordered_months].copy()

        # Rename index with numeric prefix so alphabetical = chronological
        # "January" -> "01 Jan", "February" -> "02 Feb", etc.
        month_abbrev = {
            'January': '01 Jan', 'February': '02 Feb', 'March': '03 Mar',
            'April': '04 Apr', 'May': '05 May', 'June': '06 Jun',
            'July': '07 Jul', 'August': '08 Aug', 'September': '09 Sep',
            'October': '10 Oct', 'November': '11 Nov', 'December': '12 Dec'
        }
        data.index = data.index.map(lambda x: month_abbrev.get(x, x))
    return data


def render_dealer_finder_section(lat, lon, solar_results, analysis):
    """Render the Find Nearby Dealers section."""
    st.divider()
    st.header("🏪 Find Nearby Solar Installers")

    # Initialize session state for dealers
    if "dealers_loaded" not in st.session_state:
        st.session_state.dealers_loaded = False
    if "graded_dealers" not in st.session_state:
        st.session_state.graded_dealers = []

    with st.expander("Find installation companies near you", expanded=False):
        st.markdown("""
        Find certified solar installers in your area. We'll show you the top 3 dealers
        ranked by service quality, price competitiveness, and delivery time.
        """)

        col_search1, col_search2 = st.columns([2, 1])
        with col_search1:
            search_radius = st.slider(
                "Search Radius (km)",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                key="dealer_search_radius"
            )
        with col_search2:
            st.write("")  # Spacer
            st.write("")
            search_btn = st.button("🔍 Search Dealers", type="primary", use_container_width=True)

        if search_btn:
            with st.spinner("Searching for solar installers..."):
                # Initialize database and add sample dealers for demo
                init_database()
                add_sample_dealers()

                # Find dealers
                system_kwp = solar_results.get("system_kwp", 5.0)
                dealers = find_nearby_dealers(lat, lon, radius_km=search_radius, limit=10)

                if dealers:
                    # Grade and rank dealers
                    graded = get_top_dealers(dealers, system_kwp=system_kwp, limit=3)
                    st.session_state.graded_dealers = graded
                    st.session_state.dealers_loaded = True
                else:
                    st.warning(f"No solar installers found within {search_radius} km. Try increasing the search radius.")
                    st.session_state.dealers_loaded = False

        # Display dealers if loaded
        if st.session_state.dealers_loaded and st.session_state.graded_dealers:
            st.divider()
            st.subheader("Top Recommended Installers")

            for i, graded_dealer in enumerate(st.session_state.graded_dealers, 1):
                render_dealer_card(i, graded_dealer, lat, lon, solar_results, analysis)

            st.divider()
            st.info("""
            💡 **Tips:**
            - Request quotes from multiple dealers to compare prices
            - Ask about warranties and maintenance services
            - Check if they handle permits and grid connection
            """)


def render_dealer_card(rank: int, graded_dealer: GradedDealer, lat, lon, solar_results, analysis):
    """Render a single dealer card with scores and actions."""
    dealer = graded_dealer.dealer
    grade_color = get_grade_color(graded_dealer.grade)

    with st.container():
        # Header row
        col_info, col_grade = st.columns([4, 1])

        with col_info:
            st.markdown(f"### #{rank} {graded_dealer.name}")
            distance_str = f"📍 {graded_dealer.distance_km:.1f} km away"
            if graded_dealer.rating:
                rating_str = f" | ⭐ {graded_dealer.rating:.1f}/5 ({graded_dealer.review_count} reviews)"
            else:
                rating_str = ""
            st.caption(distance_str + rating_str)

            if graded_dealer.address:
                st.write(f"📫 {graded_dealer.address}")

        with col_grade:
            st.markdown(
                f"<div style='text-align: center; padding: 10px; background-color: {grade_color}; "
                f"border-radius: 8px; color: white; font-size: 24px; font-weight: bold;'>"
                f"{graded_dealer.grade}</div>",
                unsafe_allow_html=True
            )

        # Score bars
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.caption("Quality")
            st.progress(graded_dealer.quality_score / 100)
            st.caption(f"{graded_dealer.quality_score:.0f}/100")
        with col_s2:
            st.caption("Price")
            st.progress(graded_dealer.price_score / 100)
            st.caption(f"{graded_dealer.price_score:.0f}/100")
        with col_s3:
            st.caption("Delivery")
            st.progress(graded_dealer.delivery_score / 100)
            st.caption(f"{graded_dealer.delivery_score:.0f}/100")

        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button("📧 Request Quote", key=f"quote_{rank}", use_container_width=True):
                st.session_state[f"show_quote_form_{rank}"] = True

        with col_btn2:
            if dealer.website:
                st.link_button("🌐 Website", dealer.website, use_container_width=True)
            else:
                st.button("🌐 Website", disabled=True, use_container_width=True, key=f"web_{rank}")

        with col_btn3:
            if dealer.phone:
                st.link_button("📞 Call", f"tel:{dealer.phone}", use_container_width=True)
            else:
                st.button("📞 Call", disabled=True, use_container_width=True, key=f"call_{rank}")

        # Quote form
        if st.session_state.get(f"show_quote_form_{rank}", False):
            render_quote_form(rank, graded_dealer, lat, lon, solar_results, analysis)

        st.divider()


def render_quote_form(rank: int, graded_dealer: GradedDealer, lat, lon, solar_results, analysis):
    """Render the quote request form for a dealer."""
    st.markdown(f"#### Request Quote from {graded_dealer.name}")

    with st.form(key=f"quote_form_{rank}"):
        st.markdown("**Pre-filled System Details:**")
        col_sys1, col_sys2 = st.columns(2)
        with col_sys1:
            st.write(f"System Size: **{solar_results.get('system_kwp', 0):.2f} kWp**")
            st.write(f"Panel Count: **{solar_results.get('panel_count', 0)} panels**")
        with col_sys2:
            st.write(f"Roof Area: **{solar_results.get('usable_roof_area_m2', 0):.1f} m²**")
            address = st.session_state.data.get("address", f"{lat:.4f}, {lon:.4f}")
            st.write(f"Location: **{address[:30]}...**" if len(address) > 30 else f"Location: **{address}**")

        st.divider()
        st.markdown("**Your Contact Information:**")

        customer_name = st.text_input("Name *", key=f"cust_name_{rank}")
        customer_email = st.text_input("Email *", key=f"cust_email_{rank}")
        customer_phone = st.text_input("Phone (optional)", key=f"cust_phone_{rank}")

        st.divider()
        st.markdown("**Additional Options:**")

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            include_battery = st.checkbox("Include battery storage", key=f"opt_battery_{rank}")
        with col_opt2:
            include_financing = st.checkbox("Financing options", key=f"opt_finance_{rank}")
        with col_opt3:
            include_permits = st.checkbox("Permit assistance", key=f"opt_permits_{rank}")

        additional_notes = st.text_area("Additional notes or questions", key=f"notes_{rank}")

        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("📤 Send Quote Request", type="primary", use_container_width=True)
        with col_cancel:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

        if submitted:
            if not customer_name or not customer_email:
                st.error("Please fill in your name and email address.")
            elif "@" not in customer_email:
                st.error("Please enter a valid email address.")
            else:
                # Process the quote submission
                customer_info = {
                    "name": customer_name,
                    "email": customer_email,
                    "phone": customer_phone if customer_phone else None,
                    "notes": additional_notes if additional_notes else None,
                    "include_battery": include_battery,
                    "include_financing": include_financing,
                    "include_permits": include_permits
                }

                location = {
                    "lat": lat,
                    "lon": lon,
                    "address": st.session_state.data.get("address", ""),
                    "orientation": st.session_state.data.get("roof_orientation_name", "South")
                }

                inquiry_id = st.session_state.get("inquiry_id", "")

                success, message, quote_id = process_quote_submission(
                    graded_dealer=graded_dealer,
                    solar_results=solar_results,
                    final_analysis=analysis,
                    location=location,
                    customer_info=customer_info,
                    inquiry_id=inquiry_id
                )

                if success:
                    st.success(f"Quote request sent to {graded_dealer.name}!")
                    st.balloons()
                    st.session_state[f"show_quote_form_{rank}"] = False
                else:
                    # If email not configured, still save and show success
                    if "not configured" in message.lower():
                        st.warning("Quote saved! Email sending is not configured. Please contact the dealer directly.")
                        st.info(f"Dealer contact: {graded_dealer.email or graded_dealer.phone or graded_dealer.website or 'Visit their website'}")
                    else:
                        st.error(f"Failed to send: {message}")

        if cancelled:
            st.session_state[f"show_quote_form_{rank}"] = False
            st.rerun()