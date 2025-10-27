def display_map():
    import pydeck as pdk
    # Default location (New York City)
    default_lat = 40.7128
    default_lon = -74.0060

    # Initialize session state for coordinates if not exists
    if 'selected_coords' not in st.session_state:
        st.session_state.selected_coords = {'lat': default_lat, 'lon': default_lon}

    # Create dataframe for the selected location marker
    marker_df = pd.DataFrame([{
        'lat': st.session_state.selected_coords['lat'],
        'lon': st.session_state.selected_coords['lon']
    }])
    # Create the view state (globe view)
    # Define the pydeck layer for the marker
    marker_layer = pdk.Layer(
        'ScatterplotLayer',
        data=marker_df,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=100000,
        pickable=True
    )

    # Create the view state (globe view)
    view_state = pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=1,
        pitch=0,
        bearing=0
    )
    # Display the map and capture clicks
    deck = pdk.Deck(
        layers=[marker_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v10',
        tooltip={
            'text': 'Selected Location\nLat: {lat:.4f}\nLon: {lon:.4f}'
        }
    )
    selection = st.sidebar.pydeck_chart(deck, on_select="rerun")

    print(selection)
    if selection and selection.get('objects'):
        # Get clicked coordinates from the map
        clicked_objects = selection.get('objects', [])
        if clicked_objects:
            clicked_coords = clicked_objects[0].get('coordinate')
            if clicked_coords:
                st.session_state.selected_coords['lon'] = clicked_coords[0]
                st.session_state.selected_coords['lat'] = clicked_coords[1]
                st.rerun()