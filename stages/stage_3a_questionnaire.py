import streamlit as st

def show():
    st.header("Step 3a: Consumption And Requirements")
    
    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.markdown("### 🏠 Household Energy Profile")
        st.write("Tell us about your energy needs to calculate the ideal system size.")

        # Questionnaire Inputs
        ppl = st.number_input("Number of people in household", 1, 10, 2)
        has_ev = st.toggle("Do you own an Electric Vehicle?")
        has_heatpump = st.toggle("Do you use a Heat Pump for heating?")
        
        annual_kwh = st.slider("Estimated Annual Consumption (kWh)", 1000, 15000, 3500)

        # Logic for dummy recommendation
        # 1kWp usually produces ~1000kWh/year in many regions. 
        # 1 panel is ~0.4kWp. So 1 panel = ~400kWh/year.
        recommended = round(annual_kwh / 400)
        
        st.info(f"Based on your consumption, we recommend approximately **{recommended} panels**.")

        if st.button("Continue to Solar Mapping ☀️", type="primary", use_container_width=True):
            # Save the recommendation to session state for the next stage
            st.session_state.data["recommended_count"] = recommended
            st.session_state.step = 4
            st.rerun()