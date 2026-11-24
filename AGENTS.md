# AI context

## Project

This is a Python 3.12 Streamlit application for interactive magnetic-anomaly visualization and magnetometer survey-coverage scenarios. The dependency list is in `pyproject.toml`; `uv.lock` pins the environment.

## Application structure

- `main.py` is the Streamlit entry point and defines the sidebar navigation.
- `pages/home.py` is the landing page.
- `pages/model_type.py` visualizes simple induced and oriented dipole anomalies, including map and profile plots.
- `pages/detection_limit.py` implements **Line simulation/sampling**: it samples one or two editable survey segments at the selected survey altitude and plots the anomalies.
- `components\survey_line_editor\` contains the Streamlit component for editing survey lines. It overlays plain JavaScript/SVG controls on a Matplotlib heatmap; there is no frontend build step.
- `pages/dynamic_range.py` plots sensor-array coverage against altitude, burial depth, and detection range.
- `dipole.py` contains dipole and magnetic-point models and field calculations.
- `earths_field.py` converts between spherical and Cartesian field directions and retrieves the Earth's inducing field using the external British Geological Survey magnetic-field service.
- `map.py` handles the equirectangular location-picker image and its pixel/latitude/longitude conversion.
- `pages/earth_magnetic_field.py` exists but is not currently registered in `main.py`; check navigation before assuming a page is user-facing.
- `assets/` contains images used by the app. `scratches/` contains experimental code and is not part of the main navigation.

## Domain conventions

- Cartesian coordinates are `[x, y, z] = [East, North, Up]`; positions and survey distances are in meters.
- Magnetic declination is clockwise from geographic North. Inclination is positive downward.
- `spher2cart` and related calculations use radians by default; pass `degrees=True` when inputs are in degrees.
- Dipole field calculations use SI units internally; anomaly projections are converted to nano tesla (`nT`).
- Preserve these conventions and the displayed units when changing calculations, coordinate transforms, or plots.

## Survey line editor

- Each line is a pair of `[East, North]` endpoints in meters. Dragging an endpoint edits it independently; dragging the dashed line translates both endpoints without changing length; the curved arrow rotates the line around its midpoint. Keep endpoints within the map bounds.
- Pointer drags preview locally; the component sends the final endpoints on release. Python validates the two endpoint pairs and stores them in `st.session_state["survey_lines"]`. Hiding Line 2 suppresses its map controls, profile, and metrics without discarding its endpoints. **Reset lines** resets only visible lines.
- `sample_line` generates coordinates at the selected **Survey altitude [m]**; `evaluate_anomaly` computes the magnetic anomaly at those points. Profiles use distance from the first endpoint as the x-axis; peak-to-peak is the sampled maximum minus minimum in nT. `closest_approach` uses the finite segment (including its endpoints) and reports the horizontal distance and the straight-line range, `hypot(horizontal_distance, altitude)`, for each visible line.
- Cache the heatmap and Earth-field lookup independently of endpoint edits. When changing `editor.js`, bump its version query string in `index.html` so the browser loads the updated script.

## Development

- Run locally with `uv run streamlit run main.py` from the repository root.
- Format Python with `ruff format` (the project README documents this convention).
- No test suite or test-runner configuration is currently present. For UI changes, run the app and smoke-test the affected page. Pages that request the Earth's field may need internet access to reach the external service.
- Keep dependencies declared in `pyproject.toml` and update `uv.lock` when dependency versions change.
