import streamlit as st

# logging.basicConfig(format="Conversion %(levelname)s: %(message)s", level=logging.WARNING)
page1 = st.Page("pages/home.py", title="Home", icon=":material/home:")
page2 = st.Page(
    "pages/detection_limit.py", title="Detection Limit", icon=":material/sensors:"
)
page3 = st.Page("pages/model_type.py", title="Model Type", icon=":material/radar:")
page4 = st.Page(
    "pages/dynamic_range.py", title="Dynamic Range", icon=":material/arrow_range:"
)
st.logo("assets/fugro.png", link="https://www.fugro.com/")
pg = st.navigation([page1, page2, page3, page4])
pg.run()
