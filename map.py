import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


@st.cache_data
def load_map_image():
    return Image.open("assets/equirectangular_world.jpg")


def pixel_to_latlon(x, y, img_width, img_height):
    """Convert pixel coordinates to latitude/longitude
    Assumes equirectangular projection"""
    lon = (x / img_width) * 360 - 180
    lat = 90 - (y / img_height) * 180
    return lat, lon


def latlon_to_pixel(lat, lon, img_width, img_height):
    """Convert latitude/longitude to pixel coordinates
    Assumes equirectangular projection"""
    x = ((lon + 180) / 360) * img_width
    y = ((90 - lat) / 180) * img_height
    return int(x), int(y)


def get_coords_from_map(default_lat=55.0, default_lon=3.0, sidebar=True):
    # Initialize session state
    base_image = load_map_image()
    img_width, img_height = base_image.size

    if "marker_pos" not in st.session_state:
        st.session_state.lat = default_lat
        st.session_state.lon = default_lon
        x, y = latlon_to_pixel(default_lat, default_lon, img_width, img_height)
        st.session_state.marker_pos = {"x": x, "y": y}

    # Draw marker on image if position exists
    display_image = base_image.copy()
    if st.session_state.marker_pos:
        draw = ImageDraw.Draw(display_image)
        x = st.session_state.marker_pos["x"]
        y = st.session_state.marker_pos["y"]
        # x, y = latlon_to_pixel(default_lat, default_lon, img_width, img_height)
        # Draw red circle marker
        radius = 10
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="darkred",
            width=3,
        )
        # Draw center dot
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="white")

    def on_click():
        x = st.session_state.marker_pos["x"]
        y = st.session_state.marker_pos["y"]
        # Convert to lat/lon
        lat, lon = pixel_to_latlon(x, y, img_width, img_height)
        st.session_state.lat = lat
        st.session_state.lon = lon

    container = st.sidebar if sidebar else st

    with container.container():
        streamlit_image_coordinates(
            display_image, key="marker_pos", on_click=on_click, cursor="crosshair"
        )
    return st.session_state.lat, st.session_state.lon
