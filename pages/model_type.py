import textwrap

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dipole import Dipole, MagneticPoint
from earths_field import EarthsInducingField
from map import get_coords_from_map


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


def main():
    st.title("Visualize magnetic anomalies!")
    st.text(
        textwrap.dedent("""
    The classic example of a magnetometer anomaly is one in which the magnetic moment of the dipole aligns with the Earth's field. 

    But in real surveys, we find anomalies that don’t look like this. That has to be caused by objects that are more complicated than the textbook example. The next step up in complexity is a point contact with a magnetic moment vector.

    This application shows how anomalies looks like for both the simple and the more advanced case. It shows that the magnetic anomaly can look very different depending both on the orientation of magnetic moment, but also the location on the earth. Feel free to try it!""")
    )
    alt, earth, dipole = get_input()

    ax, map_fig = create_fig()

    n = 100
    xmin = -10
    xmax = 10
    x_axis = np.linspace(xmin, xmax, n)

    line_xmin = -8
    line_xmax = 8
    line_axis = np.linspace(line_xmin, line_xmax, n)
    line_points = np.vstack([np.zeros(n), line_axis, np.full(n, alt)]).T
    if isinstance(dipole, Dipole):
        line_anomaly = dipole.induced_anomaly(
            line_points, earth.inclination, earth.declination
        )
    elif isinstance(dipole, MagneticPoint):
        line_anomaly = dipole.induced_anomaly_along_vector(line_points, earth.vector())
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
    vmin = -np.percentile(np.abs(grid_anomaly), q)
    vmax = np.percentile(np.abs(grid_anomaly), q)
    im = ax.imshow(
        grid_anomaly,
        extent=(xmin, xmax, xmin, xmax),
        cmap=create_oasis_cmap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="bicubic",
        origin="lower",
    )
    ax.vlines(
        0,
        ymin=line_xmin,
        ymax=line_xmax,
        color="red",
        linestyle="--",
        label="survey line",
    )
    ax.set_axis_off()
    add_north_arrow(ax)
    # clb = fig.colorbar(im)
    # clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    xy = 0.1 + np.sin(earth.declination) * 0.08, 0.1 + np.cos(earth.declination) * 0.08
    ax.annotate(
        f"",
        xytext=(0.1, 0.1),
        xy=xy,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->"),
        label="Mag moment",
    )
    ax.annotate(
        f"Earth field\nDec {earth.dec_deg():.1f}°\nInc: {earth.inc_deg():.1f}°",
        (0.02, 0.02),
        xycoords="axes fraction",
    )
    ax.scatter(0, 0, marker="x", zorder=10, label="Anomaly position", s=80)
    ax.set_aspect("equal")
    cax = inset_axes(
        ax,
        width="5%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    clb = map_fig.colorbar(im, cax=cax, orientation="vertical", fraction=0.016)

    if isinstance(dipole, Dipole):
        decl = dipole.get_declination()
        centered_arrow(ax, decl, (dipole.position[0], dipole.position[1]), degrees=True)

    n_per_meter = 2
    radius = 3
    y_n = (xmax - xmin) * n_per_meter
    z_max = int(np.round(alt)) + 3
    z_min = z_max - 10
    z_n = (z_max - z_min) * n_per_meter
    y_axis = np.linspace(xmin, xmax, y_n)
    z_axis = np.linspace(z_min, z_max, z_n)
    yy, zz = np.meshgrid(y_axis, z_axis, indexing="ij")
    mask = np.sqrt(yy ** 2 + zz ** 2) > radius  # Keep points outside the circle
    mask_1d = np.sqrt(y_axis ** 2 + alt ** 2) > radius

    profile_coords = np.vstack(
        [np.full(y_n * z_n, 0), yy.ravel(), zz.ravel()]
    ).T.reshape(y_n, z_n, 3)
    if isinstance(dipole, MagneticPoint):
        profile_field = dipole.induced_field(profile_coords, earth.vector())
    else:
        profile_field = dipole.induced_field(profile_coords)

    # ax_profile, fig_profile = create_fig()
    fig_profile, (ax_plot, ax_profile) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True, height_ratios=(1, 3)
    )
    fig_profile.subplots_adjust(hspace=-0.3)
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["bottom"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    ax_plot.spines["left"].set_visible(False)
    ax_plot.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )

    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)
    ax_profile.quiver(
        yy[mask],
        zz[mask],
        profile_field[:, :, 1][mask],
        profile_field[:, :, 2][mask],
        pivot="middle",
    )
    ax_profile.set_aspect("equal")
    yy, zz = np.meshgrid(y_axis, [alt])
    profile_coords = np.vstack([np.zeros(y_n), yy.ravel(), zz.ravel()]).T.reshape(
        y_n, 1, 3
    )
    if isinstance(dipole, MagneticPoint):
        profile_field = dipole.induced_field(profile_coords, earth.vector())
    else:
        profile_field = dipole.induced_field(profile_coords)

    profile_anomaly = Dipole.along_inc_dec(
        profile_field, earth.inclination, earth.declination
    )

    U = profile_anomaly * earth.unit_vector()[1]
    V = profile_anomaly * earth.unit_vector()[2]

    Q = ax_profile.quiver(
        yy[0][mask_1d],
        zz[0][mask_1d],
        profile_field[:, :, 1][mask_1d],
        profile_field[:, :, 2][mask_1d],
        pivot="middle",
        color="brown",
        width=0.005,
        label="Anomaly",
    )
    Q._init()
    # values are too low for target declination=90 and earth declination=0

    a = np.where(profile_anomaly > 0)[0]
    ax_profile.quiver(
        yy[0][mask_1d][a],
        zz[0][mask_1d][a],
        U[mask_1d][a],
        V[mask_1d][a],
        pivot="middle",
        color="green",
        scale=Q.scale,
        label="Along earth's field",
    )
    b = np.where(profile_anomaly < 0)[0]
    ax_profile.quiver(
        yy[0][mask_1d][b],
        zz[0][mask_1d][b],
        U[mask_1d][b],
        V[mask_1d][b],
        pivot="middle",
        color="red",
        scale=Q.scale,
        label="Against earth's field",
    )

    if isinstance(dipole, MagneticPoint):
        centered_arrow(ax_profile, earth.inclination + 0.5 * np.pi, color="grey")
    else:
        if -90 < dipole.get_declination() < 90:
            centered_arrow(
                ax_profile, dipole.get_inclination() + 90, color="grey", degrees=True
            )
        else:
            centered_arrow(
                ax_profile, dipole.get_inclination() - 90, color="grey", degrees=True
            )
    centered_arrow(
        ax_profile,
        earth.inc_deg() + 90,
        color="green",
        pos=(-9, alt + 1),
        degrees=True,
        text="B\u2080",
        linewidth=3,
    )

    ax_profile.set_xlabel("South ←   Horizontal [m]   → North")
    ax_profile.legend()

    ax_plot.plot(line_axis, line_anomaly, color="green")
    ax_plot.hlines(
        0, xmin=-8, xmax=8, color="blue", linestyle="--", label="survey line"
    )
    # ax_plot.set_xlabel("Horizontal [m]")
    ax_plot.set_ylabel("Magnetic anomaly [nT]")
    ax_plot.set_title("Anomaly along survey line")

    clb.ax.set_ylabel("nT", fontsize=10, rotation=270)
    st.pyplot(map_fig, use_container_width=False)
    st.pyplot(fig_profile, use_container_width=False)
    st.divider()
    url = r"https://intermagnet.org/faq/10.geomagnetic-comp"
    st.markdown("Inc - Inclination defined positive is downward. \n")
    st.markdown("Dec - Declination defined clockwise from geographic North. \n")
    st.markdown("Definition aligned with: [common geomagnetic definition](%s)" % url)
    st.divider()
    st.markdown(
        "Static examples with simple parametric target (taken from [here](%s))"
        % r"https://www.researchgate.net/publication/292966740_Airborne_Magnetic_Surveys_to_Investigate_High_Temperature_Geothermal_Reservoirs"
    )
    st.image("assets/mag_image.png")


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


def get_input() -> tuple[float, EarthsInducingField, MagneticPoint | Dipole]:
    alt: float = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)

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

    lat, long = get_coords_from_map()
    earths_field = EarthsInducingField.from_coords(lat, long)

    return alt, earths_field, dipole


if __name__ == "__main__":
    main()
