from base64 import b64encode
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from components.survey_line_editor import survey_line_editor
from dipole import Dipole, MagneticPoint
from earths_field import EarthsInducingField
from map import get_coords_from_map

LINE_COLORS = ("green", "limegreen")
MAP_MIN, MAP_MAX = -10, 10
DEFAULT_LINES = [[[0, -8], [0, 8]], [[2, -8], [2, 8]]]


def create_oasis_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.0, "#0000ff"),
        (0.2, "#00e9ff"),
        (0.4, "#00ff24"),
        (0.5, "#ffff00"),
        (0.6, "#ffaa00"),
        (0.8, "#ff0037"),
        (1.0, "#ff9fff"),
    ]
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)
    return cmap


def add_north_arrow(ax):
    py = 0.8
    px = 0.92

    # Draw an arrow with a text "N" above it using annotation
    ax.annotate("N", xy=(px, py), fontsize=15, xycoords="axes fraction", ha="center")
    ax.annotate(
        "",
        xy=(px, py),
        xytext=(px, py - 0.07),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", facecolor="black"),
    )


def centered_arrow(
    ax,
    angle: float,
    pos=(0, 0),
    arrow_size: float = 1,
    degrees=False,
    text="",
    linewidth=5,
    **kwargs,
):
    """
    Angle clockwise from North

    Arrow size in map units
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.arrow.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate
    """
    x, y = pos
    if degrees:
        angle = np.deg2rad(angle)
    xy_back = x + np.sin(angle) * -arrow_size, y + np.cos(angle) * -arrow_size
    xy_front = x + np.sin(angle) * arrow_size, y + np.cos(angle) * arrow_size
    arrowprops = dict(arrowstyle="->", linewidth=linewidth, mutation_scale=20, **kwargs)
    ax.annotate(text, xytext=xy_back, xy=xy_front, arrowprops=arrowprops)


def annotate(ax, x_value, y_values):
    # Find maximum and minimum
    max_idx = np.argmax(y_values)
    min_idx = np.argmin(y_values)

    max_x = x_value[max_idx]
    max_y = y_values[max_idx]
    min_x = x_value[min_idx]
    min_y = y_values[min_idx]

    # Plot points at max and min
    ax.plot(max_x, max_y, "ro", markersize=8, label="Maximum")
    ax.plot(min_x, min_y, "go", markersize=8, label="Minimum")

    # Annotate maximum
    ax.annotate(
        f"{max_y:.2f} nT",
        xy=(max_x, max_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red"),
    )

    # Annotate minimum

    ax.annotate(
        f"{min_y:.2f} nT",
        xy=(min_x, min_y),
        xytext=(8, -8),
        textcoords="offset points",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="green", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="green"),
    )


def evaluate_anomaly(target, earth, coords):
    if isinstance(target, Dipole):
        return target.induced_anomaly(coords, earth.inclination, earth.declination)
    return target.induced_anomaly_along_vector(coords, earth.vector())


def sample_line(endpoints, altitude):
    start, end = np.asarray(endpoints, dtype=float)
    length = np.linalg.norm(end - start)
    # Keep the original sampling density, with finer sampling near the source.
    spacing = min(16 / 99, altitude / 10)
    count = max(100, int(np.ceil(length / spacing)) + 1)
    xy = np.linspace(start, end, count)
    coords = np.column_stack((xy, np.full(count, altitude)))
    return np.linspace(0, length, count), coords


def closest_approach(endpoints, target_xy, altitude):
    start, end = np.asarray(endpoints, dtype=float)
    segment = end - start
    offset = target_xy - start
    length_squared = segment @ segment
    fraction = (
        np.clip((offset @ segment) / length_squared, 0, 1) if length_squared else 0
    )
    horizontal_distance = np.linalg.norm(offset - fraction * segment)
    return horizontal_distance, np.hypot(horizontal_distance, altitude)


def validate_lines(value):
    lines = np.asarray(value, dtype=float)
    if (
        lines.shape != (2, 2, 2)
        or not np.isfinite(lines).all()
        or np.any((lines < MAP_MIN) | (lines > MAP_MAX))
    ):
        raise ValueError(
            "Expected two lines with two finite endpoints each on the map."
        )
    return lines.tolist()


@st.cache_data(max_entries=32)
def create_map(alt, earth, dipole):
    n = 100
    axis = np.linspace(MAP_MIN, MAP_MAX, n)
    xx, yy = np.meshgrid(axis, axis)
    coords = np.column_stack((xx.ravel(), yy.ravel(), np.full(n * n, alt)))
    grid_anomaly = evaluate_anomaly(dipole, earth, coords).reshape(n, n)
    limit = np.max(np.abs(grid_anomaly))

    map_fig, map_ax = plt.subplots(figsize=(7, 7), dpi=100)
    im = map_ax.imshow(
        grid_anomaly,
        extent=(MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX),
        cmap=create_oasis_cmap(),
        vmin=-limit,
        vmax=limit,
        interpolation="bicubic",
        origin="lower",
    )
    map_ax.set_axis_off()
    add_north_arrow(map_ax)
    xy = 0.1 + np.sin(earth.declination) * 0.08, 0.1 + np.cos(earth.declination) * 0.08
    map_ax.annotate(
        "",
        xytext=(0.1, 0.1),
        xy=xy,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->"),
        label="Mag moment",
    )
    map_ax.annotate(
        f"Earth field\nDec {earth.dec_deg():.1f}°\nInc: {earth.inc_deg():.1f}°",
        (0.02, 0.02),
        xycoords="axes fraction",
    )
    map_ax.scatter(0, 0, marker="x", zorder=10, label="Anomaly position", s=80)
    map_ax.set_aspect("equal")
    map_ax.text(
        0.98,
        0.98,
        f"Theoretical Peak-to-peak: {np.max(grid_anomaly) - np.min(grid_anomaly):.2f} nT",
        transform=map_ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="grey", alpha=0.8, edgecolor="black"),
    )

    cax = inset_axes(
        map_ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02, 1, 1),
        bbox_transform=map_ax.transAxes,
        borderpad=0,
    )
    map_fig.colorbar(im, cax=cax, orientation="vertical", fraction=0.016, label="nT")

    if isinstance(dipole, Dipole):
        decl = dipole.get_declination()
        centered_arrow(
            map_ax, decl, (dipole.position[0], dipole.position[1]), degrees=True
        )

    with BytesIO() as png:
        map_fig.savefig(png, format="png", dpi=map_fig.dpi)
        image = b64encode(png.getvalue()).decode("ascii")
    width, height = map_fig.canvas.get_width_height()
    bounds = map_ax.get_window_extent()
    # Matplotlib measures from the bottom; the overlay measures from the top.
    plot = [bounds.x0, height - bounds.y1, bounds.width, bounds.height]
    plt.close(map_fig)
    return f"data:image/png;base64,{image}", [width, height], plot


def main():
    st.title("Visualize detection limits!")
    alt, earth, dipole = get_input()
    image, size, plot = create_map(alt, earth, dipole)

    show_second_line = st.toggle(
        "Show second line", value=True, key="show_second_survey_line"
    )
    visible_line_count = 2 if show_second_line else 1
    st.caption(
        "Drag a dashed line to move it, its curved arrow to rotate it, "
        "or either endpoint to reposition it. Release to update the profiles. "
        "You can also focus a control and use the arrow keys."
    )
    saved_lines = st.session_state.get("survey_lines", DEFAULT_LINES)
    value = survey_line_editor(
        image=image,
        size=size,
        plot=plot,
        limits=[MAP_MIN, MAP_MAX],
        endpoints=saved_lines,
        defaults=DEFAULT_LINES,
        colors=LINE_COLORS,
        visible_line_count=visible_line_count,
        key="survey_line_editor",
        default=saved_lines,
    )
    try:
        lines = validate_lines(value)
    except (ValueError, TypeError):
        st.warning("Invalid survey line coordinates. Keeping the last valid lines.")
        lines = saved_lines
    st.session_state["survey_lines"] = lines

    visible_lines = lines[:visible_line_count]
    fig_profile, ax_plot = plt.subplots(figsize=(9, 3), layout="constrained")
    metrics = st.columns(visible_line_count)
    for i, (endpoints, color) in enumerate(zip(visible_lines, LINE_COLORS)):
        distance, coords = sample_line(endpoints, alt)
        anomaly = evaluate_anomaly(dipole, earth, coords)
        peak_to_peak = np.ptp(anomaly)
        ax_plot.plot(
            distance,
            anomaly,
            color=color,
            label=f"Line {i + 1}",
            marker="." if distance[-1] == 0 else None,
        )
        if i == 0 and peak_to_peak > 0:
            annotate(ax_plot, distance, anomaly)
        metrics[i].metric(f"Line {i + 1} peak-to-peak", f"{peak_to_peak:.2f} nT")

    ax_plot.axhline(0, color="blue", linestyle="--", linewidth=1)
    ax_plot.set_xlabel("Distance from first endpoint [m]")
    ax_plot.set_ylabel("Magnetic anomaly [nT]")
    ax_plot.set_title(
        "Anomaly along survey lines"
        if show_second_line
        else "Anomaly along survey line"
    )
    ax_plot.legend()
    st.pyplot(fig_profile, use_container_width=True)
    plt.close(fig_profile)

    for i, (endpoints, column) in enumerate(
        zip(visible_lines, st.columns(visible_line_count))
    ):
        horizontal, target_range = closest_approach(endpoints, dipole.position[:2], alt)
        column.markdown(f"**Line {i + 1}**")
        column.metric("Horizontal distance of closest approach", f"{horizontal:.2f} m")
        column.metric("Range to target at closest approach", f"{target_range:.2f} m")


@st.cache_data(ttl=3600, max_entries=128)
def get_earth_field(lat, lon):
    return EarthsInducingField.from_coords(lat, lon)


TARGET_TYPES = ("Simple", "Dipole")


def get_input() -> tuple[float, EarthsInducingField, MagneticPoint | Dipole]:
    alt = st.sidebar.slider("Survey altitude [m]", 0.1, 15.0, 3.0, step=0.01)

    kind = st.sidebar.selectbox("target type", TARGET_TYPES, index=1)
    position = np.array([0, 0, 0])
    if kind == "Simple":
        susceptibility = 1e-4
        dipole = MagneticPoint(position, susceptibility)
        st.sidebar.text(
            f"Induced magnetism only without preferential axis\nAssuming magnetic susceptibility: {susceptibility:.1e}"
        )
    else:
        st.sidebar.text("Ferromagnetic dipole")
        magnitude = st.sidebar.slider(
            "Anomaly strength [Am2]", 1.0, 30.0, 10.0, step=0.1
        )
        m_incl = st.sidebar.slider(
            "Anomaly inclination [°]", -90.0, 90.0, 0.0, step=5.0
        )
        m_decl = st.sidebar.slider(
            "Anomaly declination [°]", -180.0, 180.0, -45.0, step=5.0
        )
        dipole = Dipole.from_spherical_moment(
            position=position,
            spherical_moment=[magnitude, m_incl, m_decl],
            degrees=True,
        )

    lat, lon = get_coords_from_map()
    return alt, get_earth_field(lat, lon), dipole


if __name__ == "__main__":
    main()
