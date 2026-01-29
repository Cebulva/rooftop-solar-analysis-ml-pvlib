"""
Stage 0: Start Screen
Allows users to start a new analysis or load a previous one.
"""
import streamlit as st
from src.inquiry_manager import (
    create_inquiry,
    load_inquiry,
    list_inquiries,
    inquiry_exists,
    get_inquiry_summary
)


def show():
    """Display the start screen for inquiry selection."""

    st.title("Welcome to SolarSight AI")
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

        # Get list of saved inquiries
        inquiries = list_inquiries()

        if inquiries:
            st.markdown("Select from your recent analyses:")

            # Create display options
            options = []
            for inq in inquiries:
                # Format display string
                addr = inq.get("address", "")[:30] if inq.get("address") else "No address"
                step = inq.get("current_step", 1)
                step_names = {1: "Location", 2: "Roof", 3: "Profile", 4: "Solar", 5: "Report"}
                step_name = step_names.get(step, f"Step {step}")
                kwp = inq.get("system_kwp")
                kwp_str = f" - {kwp:.1f} kWp" if kwp else ""

                display = f"{inq['id']}: {addr} ({step_name}{kwp_str})"
                options.append((inq['id'], display))

            # Dropdown selection
            selected_display = st.selectbox(
                "Select Inquiry",
                options=[opt[1] for opt in options],
                key="inquiry_select"
            )

            # Get selected ID
            selected_id = None
            for opt_id, opt_display in options:
                if opt_display == selected_display:
                    selected_id = opt_id
                    break

            # Show summary of selected
            if selected_id:
                summary = get_inquiry_summary(selected_id)
                if summary:
                    with st.expander("Inquiry Details", expanded=False):
                        if summary.get("lat") and summary.get("lon"):
                            st.write(f"**Location:** {summary['lat']:.6f}, {summary['lon']:.6f}")
                        if summary.get("address"):
                            st.write(f"**Address:** {summary['address']}")
                        if summary.get("system_kwp"):
                            st.write(f"**System Size:** {summary['system_kwp']:.2f} kWp")
                        st.write(f"**Last Modified:** {summary.get('modified', 'Unknown')[:19]}")

            # Load button
            if st.button("Load Selected", use_container_width=True):
                if selected_id:
                    _load_and_restore(selected_id)
                else:
                    st.error("Please select an inquiry first.")

            st.divider()

        # Manual ID entry
        st.markdown("**Or enter Inquiry ID directly:**")
        manual_id = st.text_input(
            "Inquiry ID",
            placeholder="INQ-001",
            key="manual_inquiry_id"
        ).strip().upper()

        if st.button("Load by ID", use_container_width=True):
            if manual_id:
                if inquiry_exists(manual_id):
                    _load_and_restore(manual_id)
                else:
                    st.error(f"Inquiry '{manual_id}' not found.")
            else:
                st.warning("Please enter an Inquiry ID.")

        if not inquiries:
            st.info("No saved analyses found. Start a new analysis to begin!")


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