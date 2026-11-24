from pathlib import Path

from streamlit.components.v1 import declare_component

survey_line_editor = declare_component(
    "survey_line_editor", path=str(Path(__file__).resolve().parent)
)
