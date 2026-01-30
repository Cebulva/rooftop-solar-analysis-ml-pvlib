"""
Stage 0: Start Screen
Allows users to start a new analysis or load a previous one.
"""
import streamlit as st
from src.inquiry_manager import (
    create_inquiry,
    load_inquiry,
    inquiry_exists
)


def show():
    """Display the start screen for inquiry selection."""

    st.title("Welcome to SolarSight")
    st.markdown("**Rooftop Solar Analysis Tool**")
    st.divider()

    col1, col2 = st.columns(2)

    # === NEW ANALYSIS ===
    with col1:
        st.subheader("Start New Analysis")
        st.markdown("""
        Begin a fresh solar analysis for your property:
        1. Locate your building on the map
        2. Refine the roof boundary
        3. Enter your energy profile
        4. Get solar panel recommendations
        5. Download your personalized report
        """)

        if st.button("Start New Analysis", type="primary", use_container_width=True):
            # Create new inquiry
            inquiry_id = create_inquiry()
            st.session_state.inquiry_id = inquiry_id
            st.session_state.data = {}
            st.session_state.step = 1
            st.session_state.sub_step = "verify"
            st.session_state["selected_pos"] = None
            st.rerun()

    # === LOAD PREVIOUS ===
    with col2:
        st.subheader("Load Previous Analysis")
        st.markdown("""
        Continue from where you left off:
        1. Enter your Inquiry ID (e.g., INQ-001)
        2. Your progress will be restored
        3. Continue from your last step

        *Your Inquiry ID is shown in the sidebar during analysis.*
        """)

        manual_id = st.text_input(
            "Inquiry ID",
            placeholder="INQ-001",
            key="manual_inquiry_id"
        ).strip().upper()

        if st.button("Load Inquiry", type="secondary", use_container_width=True):
            if manual_id:
                if inquiry_exists(manual_id):
                    _load_and_restore(manual_id)
                else:
                    st.error(f"Inquiry '{manual_id}' not found.")
            else:
                st.warning("Please enter an Inquiry ID.")


def _load_and_restore(inquiry_id: str):
    """Load inquiry data and restore session state."""
    with st.spinner(f"Loading {inquiry_id}..."):
        result = load_inquiry(inquiry_id)

        if result:
            # Restore session state
            st.session_state.inquiry_id = inquiry_id
            st.session_state.data = result["data"]
            st.session_state.step = result["step"]
            st.session_state.sub_step = result.get("sub_step", "verify")

            # Restore map position if available
            if "confirmed_lat" in result["data"] and "confirmed_lon" in result["data"]:
                st.session_state["selected_pos"] = (
                    result["data"]["confirmed_lat"],
                    result["data"]["confirmed_lon"]
                )
                st.session_state["map_center"] = [
                    result["data"]["confirmed_lat"],
                    result["data"]["confirmed_lon"]
                ]

            st.success(f"Loaded {inquiry_id} successfully!")
            st.rerun()
        else:
            st.error(f"Failed to load {inquiry_id}. The data may be corrupted.")