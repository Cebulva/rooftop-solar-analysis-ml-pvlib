import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import ssl

# --- SSL CERTIFICATE PATCH ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Global Constants
MODEL_PATH = "models/model_zoom19.pth"
AI_ZOOM = 19
GSD_19 = 0.298
DISPLAY_WIDTH = 700
BUTTON_SIZE_VERIFY = 350
SHADOW_BIAS = 0

# Default location: Cuvrystraße 1, 10997 Berlin, Germany
DEFAULT_LAT = 52.4985
DEFAULT_LON = 13.4379

def render_sidebar():
    """Renders the sidebar with address search and location display."""

    # Initialize session state for map
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = [DEFAULT_LAT, DEFAULT_LON]
    if "selected_pos" not in st.session_state:
        st.session_state["selected_pos"] = None

    with st.sidebar:
        st.title("SolarSight AI")

        # Show current inquiry ID if active
        if st.session_state.get("inquiry_id"):
            st.info(f"**Inquiry:** {st.session_state.inquiry_id}")

        # Address Search
        st.subheader("📍 Find Location")
        address_input = st.text_input(
            "Enter Address",
            placeholder="e.g. Musterweg 123, Berlin",
            key="address_search"
        )

        if st.button("🔍 Search", use_container_width=True):
            if address_input:
                try:
                    geolocator = Nominatim(user_agent="solarsight_ai_v1")
                    with st.spinner("Locating..."):
                        location = geolocator.geocode(address_input, addressdetails=True, timeout=10)
                        if location:
                            st.session_state["map_center"] = [location.latitude, location.longitude]

                            # Check if address has house number (complete address)
                            # If yes, automatically place the pin
                            address_details = location.raw.get('address', {})
                            has_house_number = 'house_number' in address_details

                            if has_house_number:
                                # Complete address - place pin automatically
                                st.session_state["selected_pos"] = (location.latitude, location.longitude)
                                st.success(f"📍 Pinned: {location.address[:60]}...")
                            else:
                                # Incomplete address - just move map, user can click manually
                                st.session_state["selected_pos"] = None
                                st.info(f"Found area: {location.address[:60]}... Click on map to select exact building.")
                            st.rerun()
                        else:
                            st.error("Address not found.")
                except (GeocoderTimedOut, GeocoderUnavailable) as e:
                    st.error(f"Geocoding error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # Display current coordinates
        if st.session_state["selected_pos"]:
            lat, lon = st.session_state["selected_pos"]
            st.metric("Latitude", f"{lat:.6f}")
            st.metric("Longitude", f"{lon:.6f}")
        else:
            lat, lon = st.session_state["map_center"]
            st.caption("Click on map to select building")
            st.metric("Center Lat", f"{lat:.6f}")
            st.metric("Center Lon", f"{lon:.6f}")

        st.divider()

        # New Search button
        if st.button("🔄 New Search", type="primary", use_container_width=True):
            st.session_state.step = 0
            st.session_state.sub_step = "verify"
            st.session_state.data = {}
            st.session_state.inquiry_id = None
            st.session_state["selected_pos"] = None
            st.cache_data.clear()
            st.rerun()

        # Return the selected position or map center
        if st.session_state["selected_pos"]:
            return st.session_state["selected_pos"]
        else:
            return tuple(st.session_state["map_center"])