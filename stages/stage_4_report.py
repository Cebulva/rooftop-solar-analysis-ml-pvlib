import streamlit as st

def show():
    st.subheader("Step 4: Project Summary And Export")
    
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    
    with col_center:
        st.success("✅ Analysis Complete!")
        st.markdown("""
        ### Final Report
        * **Address:** [Fetched from Lat/Lon]
        * **Roof Area:** Measured in Step 2
        * **Solar Potential:** Calculated in Step 3
        """)
        
        st.button("📥 Download PDF Report", use_container_width=True, type="primary")
        
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.step = 1
            st.session_state.data = {}
            st.rerun()