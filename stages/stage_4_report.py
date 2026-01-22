import streamlit as st
import pandas as pd

# Import from src
from src.german_solar_calculator import (
    GermanSolarCalculator,
    ORIENTATION_MAP,
    TILT_MAP,
    get_consumption_breakdown,
    CONSTANTS
)
from src.solar_analysis import analyze_solar_potential
from src.shadow_analysis import get_optimal_panel_angle
from src.building_analysis import analyze_building_solar_potential

# Month order for proper sorting
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def show():
    st.subheader("Step 4: Solar Analysis & Report")

    # Get coordinates from session state
    lat = st.session_state.data.get("confirmed_lat")
    lon = st.session_state.data.get("confirmed_lon")

    if not lat or not lon:
        st.error("Location data not found. Please start from Step 1.")
        if st.button("🔄 Start Over"):
            st.session_state.step = 1
            st.session_state.data = {}
            st.rerun()
        return

    # Check if questionnaire has been completed
    if "questionnaire_complete" not in st.session_state.data:
        render_questionnaire(lat, lon)
    else:
        render_analysis_results(lat, lon)


def render_questionnaire(lat, lon):
    """Render the household & roof questionnaire."""

    st.header("🏠 Your Household & Roof Information")
    st.info("""
    **Tell us about your home to get personalized solar recommendations!**

    We'll estimate your electricity consumption based on German Stromspiegel 2025 data
    and calculate the optimal solar system size for your needs.
    """)

    with st.form("consumption_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🔌 Your Consumption")

            building_type = st.selectbox(
                "Building Type",
                ["Single Family House", "Apartment"],
                help="Houses typically have higher consumption than apartments"
            )

            people = st.slider(
                "Number of People in Household",
                min_value=1,
                max_value=6,
                value=3,
                help="More people = higher electricity consumption"
            )

            st.markdown("**High-Power Appliances:**")
            has_water_heater = st.checkbox(
                "Electric Water Heater (Durchlauferhitzer)",
                help="Adds ~1,000 kWh/year"
            )
            has_ev = st.checkbox(
                "Electric Vehicle (with home charging)",
                help="Adds ~2,500 kWh/year for ~15,000 km"
            )
            has_heat_pump = st.checkbox(
                "Heat Pump",
                help="Adds ~3,500 kWh/year"
            )

        with col_right:
            st.subheader("☀️ Your Roof")

            roof_orientation = st.selectbox(
                "Roof Orientation",
                list(ORIENTATION_MAP.keys()),
                index=0,  # Default to South
                help="South-facing roofs get the most sun in Germany"
            )

            roof_tilt = st.selectbox(
                "Roof Tilt Angle",
                list(TILT_MAP.keys()),
                index=2,  # Default to Normal (35)
                help="35° is optimal for most German locations"
            )

            # Show consumption breakdown preview
            st.markdown("---")
            st.markdown("**Estimated Consumption Preview:**")
            b_type = 'house' if 'House' in building_type else 'apt'
            breakdown = get_consumption_breakdown(
                people, b_type, has_water_heater, has_ev, has_heat_pump
            )
            for item, kwh in breakdown.items():
                if item == 'Total':
                    st.markdown(f"**{item}: {kwh:,} kWh/year**")
                else:
                    st.write(f"- {item}: {kwh:,} kWh")

        calculate_solar = st.form_submit_button(
            "☀️ Calculate Solar Recommendation",
            use_container_width=True
        )

    if calculate_solar:
        with st.spinner("Calculating your personalized solar recommendation..."):
            # Initialize calculator with selected location
            calc = GermanSolarCalculator(lat, lon)

            # Get building type code
            b_type = 'house' if 'House' in building_type else 'apt'

            # Run full analysis
            analysis = calc.full_analysis(
                people=people,
                building_type=b_type,
                roof_orientation=roof_orientation,
                roof_tilt=roof_tilt,
                has_water_heater=has_water_heater,
                has_ev=has_ev,
                has_heat_pump=has_heat_pump
            )

            # Store in session state
            st.session_state.data['solar_analysis'] = analysis
            st.session_state.data['questionnaire_complete'] = True
            st.session_state.data['questionnaire_inputs'] = {
                'building_type': building_type,
                'people': people,
                'has_water_heater': has_water_heater,
                'has_ev': has_ev,
                'has_heat_pump': has_heat_pump,
                'roof_orientation': roof_orientation,
                'roof_tilt': roof_tilt
            }
            st.rerun()


def render_analysis_results(lat, lon):
    """Render the detailed solar analysis results."""

    analysis = st.session_state.data.get('solar_analysis')

    if not analysis:
        st.error("Analysis data not found.")
        return

    # === RECOMMENDATION HEADER ===
    st.success("☀️ Solar Analysis Complete!")
    st.divider()

    st.header(f"Recommended System: {analysis['system']['recommended_kwp']} kWp")

    # Key Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            "Your Annual Consumption",
            f"{analysis['consumption']['annual_kwh']:,} kWh",
            help="Based on Stromspiegel 2025"
        )

    with col_m2:
        st.metric(
            "Solar Production",
            f"{int(analysis['production']['annual_kwh']):,} kWh",
            help="Based on PVGIS satellite data"
        )

    with col_m3:
        st.metric(
            "Investment Cost",
            f"{int(analysis['economics']['invest_cost']):,} EUR",
            "0% VAT",
            help="Including installation, 0% VAT for residential solar"
        )

    with col_m4:
        roi = analysis['economics']['roi_years']
        if roi < 10:
            st.metric("Payback Period", f"{roi:.1f} years", delta="Good investment!")
        else:
            st.metric("Payback Period", f"{roi:.1f} years")

    # Coverage Progress Bar
    st.divider()
    coverage = min(analysis['coverage']['ratio'], 1.0)
    st.progress(coverage, text=f"Coverage: {analysis['coverage']['percent']:.0f}% of your consumption")

    if analysis['coverage']['percent'] >= 100:
        st.success(f"This system covers 100% of your annual consumption with {int(analysis['coverage']['surplus_kwh']):,} kWh surplus!")
    else:
        st.info(f"This system covers {analysis['coverage']['percent']:.0f}% of your needs. You'll still need {int(analysis['coverage']['deficit_kwh']):,} kWh from the grid.")

    # Financial Details
    st.divider()
    col_fin1, col_fin2 = st.columns(2)

    with col_fin1:
        st.subheader("💶 Annual Savings Breakdown")
        st.write(f"**Electricity cost savings:** {int(analysis['economics']['savings_usage']):,} EUR/year")
        st.write(f"**Feed-in tariff (EEG):** {int(analysis['economics']['earnings_feedin']):,} EUR/year")
        st.markdown(f"### Total: {int(analysis['economics']['annual_benefit']):,} EUR/year")
        st.caption(f"Monthly benefit: ~{int(analysis['economics']['monthly_benefit']):,} EUR")

    with col_fin2:
        st.subheader("📊 System Details")
        st.write(f"**Number of panels:** {analysis['system']['num_panels']} x {analysis['system']['panel_power_w']}W")
        st.write(f"**Required roof area:** {analysis['system']['required_roof_area_m2']:.1f} m²")
        st.write(f"**Specific yield:** {analysis['production']['specific_yield_kwh_kwp']:.0f} kWh/kWp")
        st.write(f"**Orientation:** {analysis['system']['roof_orientation']}")
        st.write(f"**Tilt:** {analysis['system']['roof_tilt']}")

    # 25-year profit
    st.divider()
    lifetime_profit = analysis['economics']['lifetime_profit_25y']
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

    with st.spinner("Fetching satellite weather data & running simulation..."):
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

                    # Calculate potential savings
                    annual_savings = building_info['solar_potential']['estimated_annual_production_kwh'] * CONSTANTS['ELECTRICITY_PRICE_DE']
                    st.write(f"**Estimated Annual Savings:** €{annual_savings:,.0f} (at €{CONSTANTS['ELECTRICITY_PRICE_DE']}/kWh)")
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


def reindex_by_month(data):
    """Reindex dataframe to proper month order instead of alphabetical."""
    if data.index.name == 'Month' or data.index[0] in MONTH_ORDER:
        # Create a categorical index with proper month order
        data = data.copy()
        data.index = pd.Categorical(data.index, categories=MONTH_ORDER, ordered=True)
        data = data.sort_index()
    return data
