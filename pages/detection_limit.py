import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dipole import Dipole, MagneticPoint
from earths_field import EarthsInducingField
from map import get_coords_from_map

COLOR_2ND_LINE = "limegreen"
COLOR_1ST_LINE = "green"

logger = logging.getLogger(__name__)


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

    x_diff = max_x - min_x
    y_diff = max_y - min_y
    # Annotate maximum
    ax.annotate(
        f"{max_y:.2f} nT",
        xy=(max_x, max_y),
        xytext=(max_x + 1, max_y + 0.0 * y_diff),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red"),
    )

    # Annotate minimum

    ax.annotate(
        f"{min_y:.2f} nT",
        xy=(min_x, min_y),
        xytext=(min_x + 0.15, min_y - 0.15 * y_diff),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="green", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="green"),
    )


def main():
    st.title("Visualize detection limits!")

    alt, earth, dipole, first_line_offset, second_line_offset = get_input()

    map_ax, map_fig = create_fig()

    n = 100
    xmin = -10
    xmax = 10
    x_axis = np.linspace(xmin, xmax, n)

    line_xmin = -8
    line_xmax = 8
    line_axis = np.linspace(line_xmin, line_xmax, n)
    line1_points = np.vstack(
        [np.full(n, first_line_offset), line_axis, np.full(n, alt)]
    ).T
    line2_points = np.vstack(
        [np.full(n, second_line_offset), line_axis, np.full(n, alt)]
    ).T
    if isinstance(dipole, Dipole):
        line_anomaly1 = dipole.induced_anomaly(
            line1_points, earth.inclination, earth.declination
        )
        line_anomaly2 = dipole.induced_anomaly(
            line2_points, earth.inclination, earth.declination
        )
    elif isinstance(dipole, MagneticPoint):
        line_anomaly1 = dipole.induced_anomaly_along_vector(
            line1_points, earth.vector()
        )
        line_anomaly2 = dipole.induced_anomaly_along_vector(
            line2_points, earth.vector()
        )
    else:
        raise
    xx, yy = np.meshgrid(x_axis, x_axis)

    grid_coords = np.vstack([xx.ravel(), yy.ravel(), np.full(n * n, alt)]).T
    if isinstance(dipole, Dipole):
        grid_anomaly = dipole.induced_anomaly(
            grid_coords, earth.inclination, earth.declination
        ).reshape(n, n)
    elif isinstance(dipole, MagneticPoint):
        grid_anomaly = dipole.induced_anomaly_along_vector(
            grid_coords, earth.vector()
        ).reshape(n, n)
    else:
        raise

    q = 100
    vmin: float = -np.percentile(np.abs(grid_anomaly), q)
    vmax: float = np.percentile(np.abs(grid_anomaly), q)
    im = map_ax.imshow(
        grid_anomaly,
        extent=(xmin, xmax, xmin, xmax),
        cmap=create_oasis_cmap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        origin="lower",
    )
    map_ax.vlines(
        first_line_offset,
        ymin=line_xmin,
        ymax=line_xmax,
        color=COLOR_1ST_LINE,
        linestyle="--",
        label="survey line",
    )
    map_ax.vlines(
        second_line_offset,
        ymin=line_xmin,
        ymax=line_xmax,
        color=COLOR_2ND_LINE,
        linestyle="--",
        label="survey line",
    )
    map_ax.set_axis_off()
    add_north_arrow(map_ax)
    # clb = fig.colorbar(im)
    # clb.map_ax.set_ylabel('nT', fontsize=10, rotation=270)
    xy = 0.1 + np.sin(earth.declination) * 0.08, 0.1 + np.cos(earth.declination) * 0.08
    map_ax.annotate(
        f"",
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
    map_fig.colorbar(im, cax=cax, orientation="vertical", fraction=0.016)

    if isinstance(dipole, Dipole):
        decl = dipole.get_declination()
        centered_arrow(
            map_ax, decl, (dipole.position[0], dipole.position[1]), degrees=True
        )

    fig_profile, ax_plot = plt.subplots(
        figsize=(9, 3),
    )
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["bottom"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_visible(False)
    ax_plot.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )

    ax_plot.plot(line_axis, line_anomaly1, color=COLOR_1ST_LINE)
    ax_plot.plot(line_axis, line_anomaly2, color=COLOR_2ND_LINE)

    annotate(ax_plot, line_axis, line_anomaly1)

    ax_plot.text(
        0.98,
        0.98,
        f"Peak-to-peak: {max(line_anomaly1) - min(line_anomaly1):.2f} nT",
        transform=ax_plot.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor=COLOR_1ST_LINE, alpha=0.8, edgecolor="black"
        ),
    )
    ax_plot.text(
        0.98,
        0.85,
        f"Peak-to-peak: {max(line_anomaly2) - min(line_anomaly2):.2f} nT",
        transform=ax_plot.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor=COLOR_2ND_LINE, alpha=0.8, edgecolor="black"
        ),
    )

    ax_plot.hlines(
        0, xmin=-8, xmax=8, color="blue", linestyle="--", label="survey line"
    )
    # ax_plot.set_xlabel("Horizontal [m]")
    ax_plot.set_ylabel("Magnetic anomaly [nT]")
    ax_plot.set_title("Anomaly along survey lines")

    st.pyplot(map_fig, use_container_width=False)
    st.pyplot(fig_profile, use_container_width=False)


@st.cache_data
def create_fig():
    map_fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    return ax, map_fig


CONFIGS = {
    "North Sea": (55.0, 3.0),
    "Equator": (0.0, 0.0),
}

TARGET_TYPES = {
    "Simple",
    "Dipole",
}


def get_input() -> tuple[
    float, EarthsInducingField, MagneticPoint | Dipole, float, float
]:
    alt = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)

    kind = st.sidebar.selectbox("target type", TARGET_TYPES, index=1)
    position = np.array([0, 0, 0])
    if kind == "Simple":
        susceptibility = 1e-4
        dipole = MagneticPoint(position, susceptibility)
        st.sidebar.text(
            f"Induced magnetism only without preferential axis\nAssuming magnetic susceptibility: {susceptibility:.1e}"
        )
    else:
        st.sidebar.text(f"Ferromagnetic dipole")
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

    first_line = st.sidebar.slider(
        "Offset distance line [m]", -10.0, 10.0, 0.0, step=0.1
    )
    second_line = first_line + st.sidebar.slider(
        "Line spacing [m]", -10.0, 10.0, 2.0, step=0.1
    )
    lat, long = get_coords_from_map()
    logger.info(f"{lat}, {long}")
    earths_field = EarthsInducingField.from_coords(lat, long)

    return alt, earths_field, dipole, first_line, second_line


if __name__ == "__main__":
    main()
