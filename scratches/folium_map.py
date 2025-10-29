import streamlit as st
from folium.plugins import MousePosition
from streamlit_folium import st_folium
import folium


def get_coords_from_map(default_lat=55.0, default_lon=3.0, sidebar=True):
    # Choose where to render based on sidebar parameter
    container = st.sidebar if sidebar else st

    # Initialize session state for coordinates if not exists
    if "selected_coords" not in st.session_state:
        st.session_state.selected_coords = {"lat": default_lat, "lon": default_lon}

    # Create the map
    m = folium.Map(
        location=[default_lat, default_lon], zoom_start=0, tiles="OpenStreetMap"
    )

    # Add a marker at the selected location
    folium.Marker(
        location=[
            st.session_state.selected_coords["lat"],
            st.session_state.selected_coords["lon"],
        ],
        popup=f"Selected Location<br>Lat: {st.session_state.selected_coords['lat']:.4f}<br>Lon: {st.session_state.selected_coords['lon']:.4f}",
        tooltip="Selected Location",
        icon=folium.Icon(color="red", icon="target"),
    ).add_to(m)

    # Add mouse position display
    MousePosition().add_to(m)
    map_data = container.container().empty()
    with container.container():
        map_result = st_folium(m, width=400, height=250, key="map")

    # Update coordinates when map is clicked
    if map_result and map_result.get("last_clicked"):
        clicked_lat = map_result["last_clicked"]["lat"]
        clicked_lon = map_result["last_clicked"]["lng"]

        # Update session state
        st.session_state.selected_coords["lat"] = clicked_lat
        st.session_state.selected_coords["lon"] = clicked_lon

        # Rerun to update marker
        st.rerun()

    return st.session_state.selected_coords["lat"], st.session_state.selected_coords[
        "lon"
    ]
