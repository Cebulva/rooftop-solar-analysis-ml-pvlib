import streamlit as st
import pandas as pd

# Import from src
from src.german_solar_calculator import (
    GermanSolarCalculator,
    ORIENTATION_MAP,
    CONSTANTS
)
from src.solar_analysis import analyze_solar_potential
from src.shadow_analysis import get_optimal_panel_angle
from src.building_analysis import analyze_building_solar_potential

# Month order for proper sorting
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


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
            # Clear final analysis so it recalculates when user comes back
            if "final_analysis" in st.session_state.data:
                del st.session_state.data["final_analysis"]
            st.session_state.step = 4
            st.rerun()
    with col_back2:
        if st.button("⬅️ Edit Questionnaire", key="back_to_3a"):
            if "final_analysis" in st.session_state.data:
                del st.session_state.data["final_analysis"]
            st.session_state.step = 3
            st.rerun()
    with col_back3:
        if st.button("⬅️ Edit Roof", key="back_to_2"):
            if "final_analysis" in st.session_state.data:
                del st.session_state.data["final_analysis"]
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
        # Calculate actual production based on system size from stage 3b
        specific_yield = analysis['production']['specific_yield_kwh_kwp']
        actual_production = actual_kwp * specific_yield
        st.metric(
            "Solar Production",
            f"{int(actual_production):,} kWh",
            help="Based on PVGIS satellite data"
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

    # Tips
    st.info("""
    💡 **Tips to increase savings:**
    - Add a battery storage to increase self-consumption from 30% to 60-70%
    - Use high-power appliances (washing machine, dishwasher) during sunny hours
    - Consider an EV to utilize surplus solar production
    """)

    # === DETAILED SOLAR ANALYSIS ===
    st.divider()
    st.header("📊 Detailed Solar Analysis")

    with st.spinner("Fetching satellite weather data..."):
        # Solar Analysis
        data, error_msg = analyze_solar_potential(lat, lon)

        # Optimal Panel Angles
        optimal_angles = get_optimal_panel_angle(lat)

        # Building/Rooftop Analysis
        building_info = analyze_building_solar_potential(lat, lon)

        if error_msg:
            st.error(f"Simulation Failed: {error_msg}")
            st.info("Try a location on land.")

        elif data is not None:
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
                st.subheader("🏠 Building & Rooftop Analysis (OpenStreetMap)")

                col_bldg1, col_bldg2, col_bldg3, col_bldg4 = st.columns(4)

                with col_bldg1:
                    st.metric(
                        "OSM Roof Area",
                        f"{building_info['roof_analysis']['usable_area_m2']:.1f} m²",
                        help="Estimated from OpenStreetMap data"
                    )
                with col_bldg2:
                    st.metric(
                        "Building Height",
                        f"{building_info['building_info']['height']:.1f} m",
                        help="Estimated building height"
                    )
                with col_bldg3:
                    st.metric(
                        "Building Type",
                        f"{building_info['building_info']['type']}",
                    )
                with col_bldg4:
                    if building_info['building_info']['levels']:
                        st.metric("Floors", f"{building_info['building_info']['levels']}")
                    else:
                        st.metric("Floors", "N/A")

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

            # Reindex data to proper month order
            data_sorted = reindex_by_month(data)

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

    # === FINAL ACTIONS ===
    st.divider()
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("📥 Download PDF Report", use_container_width=True, type="primary"):
            st.info("PDF export coming soon!")

    with col_btn2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.step = 1
            st.session_state.data = {}
            st.session_state["selected_pos"] = None
            st.rerun()


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


def reindex_by_month(data):
    """Reindex dataframe to proper month order instead of alphabetical."""
    if data.index.name == 'Month' or data.index[0] in MONTH_ORDER:
        # Create a categorical index with proper month order
        data = data.copy()
        data.index = pd.Categorical(data.index, categories=MONTH_ORDER, ordered=True)
        data = data.sort_index()
    return data