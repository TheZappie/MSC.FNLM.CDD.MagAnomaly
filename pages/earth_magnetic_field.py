import streamlit as st
import folium
from folium.plugins import MousePosition
from streamlit_folium import st_folium

st.title("Interactive World Map - Location Picker")

# Default location (New York City)
default_lat = 40.7128
default_lon = -74.0060

# Initialize session state for coordinates if not exists
if "selected_coords" not in st.session_state:
    st.session_state.selected_coords = {"lat": default_lat, "lon": default_lon}

# Create the map with world view
m = folium.Map(
    location=[0, 0],  # Center of the world
    zoom_start=1,  # Zoom level to see whole world
    tiles="OpenStreetMap",
)

# Add a marker at the selected location
folium.Marker(
    location=[
        st.session_state.selected_coords["lat"],
        st.session_state.selected_coords["lon"],
    ],
    popup=f"Selected Location<br>Lat: {st.session_state.selected_coords['lat']:.4f}<br>Lon: {st.session_state.selected_coords['lon']:.4f}",
    tooltip="Selected Location",
    icon=folium.Icon(color="red", icon="info-sign"),
).add_to(m)

# Add mouse position display
MousePosition().add_to(m)

# Display the map and capture clicks
st.write("**Click anywhere on the map to select a location**")
map_data = st_folium(m, width=700, height=500, key="map")

# Update coordinates when map is clicked
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    # Update session state
    st.session_state.selected_coords["lat"] = clicked_lat
    st.session_state.selected_coords["lon"] = clicked_lon

    # Rerun to update marker
    st.rerun()

# Display selected coordinates
st.subheader("Selected Coordinates")
col1, col2 = st.columns(2)
with col1:
    st.metric("Latitude", f"{st.session_state.selected_coords['lat']:.6f}")
with col2:
    st.metric("Longitude", f"{st.session_state.selected_coords['lon']:.6f}")

# Example of using coordinates for further processing
st.subheader("Use Coordinates in Your App")
st.code(
    f"""
# Access the selected coordinates:
latitude = {st.session_state.selected_coords["lat"]:.6f}
longitude = {st.session_state.selected_coords["lon"]:.6f}

# Use them for further processing...
# For example: fetch weather data, calculate distances, etc.
""",
    language="python",
)

# Reset button
if st.button("Reset to Default Location"):
    st.session_state.selected_coords = {"lat": default_lat, "lon": default_lon}
    st.rerun()
