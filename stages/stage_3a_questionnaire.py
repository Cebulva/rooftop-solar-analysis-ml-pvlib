import streamlit as st
import ui_components as ui

# Import consumption calculation from german_solar_calculator
from src.german_solar_calculator import (
    get_consumption_breakdown,
    BASE_LOAD_PROFILES,
    ADD_ONS,
    CONSTANTS
)
from src.inquiry_manager import save_inquiry


def show():
    st.header("Step 3a: Your Household Energy Profile")

    # Back button
    if st.button("⬅️ Back to Roof Refinement", key="back_to_step2"):
        st.session_state.step = 2
        st.rerun()

    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.info("""
        **Tell us about your household to calculate the ideal solar system size.**

        We'll estimate your electricity consumption based on German Stromspiegel 2025 data.
        """)

        with st.form("consumption_form"):
            st.subheader("🏠 Building Information")

            building_type = st.selectbox(
                "Building Type",
                ["Single Family House", "Apartment"],
                help="Houses typically have higher consumption than apartments"
            )

            st.subheader("👥 Household Size")

            people = st.slider(
                "Number of People in Household",
                min_value=1,
                max_value=6,
                value=3,
                help="More people = higher electricity consumption"
            )

            st.subheader("🔌 High-Power Appliances")
            st.caption("Select appliances that significantly increase your electricity consumption:")

            col1, col2 = st.columns(2)

            with col1:
                has_water_heater = st.checkbox(
                    "Electric Water Heater (Durchlauferhitzer)",
                    help="Adds ~1,000 kWh/year"
                )
                has_ev = st.checkbox(
                    "Electric Vehicle (with home charging)",
                    help="Adds ~2,500 kWh/year for ~15,000 km"
                )

            with col2:
                has_heat_pump = st.checkbox(
                    "Heat Pump",
                    help="Adds ~3,500 kWh/year"
                )

            submit = st.form_submit_button(
                "Continue to Solar Analysis ☀️",
                type="primary",
                use_container_width=True
            )

        if submit:
            # Calculate consumption breakdown
            b_type = 'house' if 'House' in building_type else 'apt'
            breakdown = get_consumption_breakdown(
                people, b_type, has_water_heater, has_ev, has_heat_pump
            )
            annual_kwh = breakdown['Total']

            # Calculate recommended panel count for stage 3b
            kwh_per_panel = CONSTANTS['PANEL_POWER_W'] * 0.95 * 1.1  # ~460 kWh/panel/year
            recommended_panels = max(1, round(annual_kwh / kwh_per_panel))

            # Store all consumption data for stage 3b and stage 4
            st.session_state.data["consumption_inputs"] = {
                'building_type': building_type,
                'b_type_code': b_type,
                'people': people,
                'has_water_heater': has_water_heater,
                'has_ev': has_ev,
                'has_heat_pump': has_heat_pump,
                'annual_kwh': annual_kwh,
                'breakdown': breakdown
            }

            # Store recommended panel count for stage 3b
            st.session_state.data["recommended_count"] = recommended_panels

            # Auto-save inquiry
            if st.session_state.get("inquiry_id"):
                save_inquiry(
                    st.session_state.inquiry_id,
                    st.session_state.data,
                    step=4,
                    sub_step="verify"
                )

            st.session_state.step = 4
            st.rerun()