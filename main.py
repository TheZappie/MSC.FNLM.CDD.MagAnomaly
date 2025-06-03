import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dipole import Dipole
from earths_field import EarthsInducingField


def create_oasis_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.0, '#0000ff'),
        (0.2, '#00e9ff'),
        (0.4, '#00ff24'),
        (0.5, '#ffff00'),
        (0.6, '#ffaa00'),
        (0.8, '#ff0037'),
        (1.0, '#ff9fff')
    ]
    cmap = LinearSegmentedColormap.from_list('custom_rainbow', colors)
    return cmap


def add_north_arrow(ax):
    py = 0.8
    px = 0.92

    # Draw an arrow with a text "N" above it using annotation
    ax.annotate("N", xy=(px, py), fontsize=15, xycoords="axes fraction", ha='center')
    ax.annotate("", xy=(px, py), xytext=(px, py - 0.07), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", facecolor="black"))


def main():
    st.title("3-D Magnetic dipole anomaly Visualizer")
    # alt, lat, long, magnitude, direction = get_input()
    alt, earth, m_magnitude, m_inc, m_dec = get_input()
    dipole = Dipole.from_spherical_moment(
        position=np.array([0, 0, 0]),
        spherical_moment=np.array([m_magnitude, m_inc, m_dec]))

    ax, map_fig = create_fig()

    n = 100
    xmin = -10
    xmax = 10
    x_axis = np.linspace(xmin, xmax, n)

    line_xmin = -8
    line_xmax = 8
    line_axis = np.linspace(line_xmin, line_xmax, n)
    line_points = np.vstack([line_axis, np.zeros(n), np.full(n, alt)]).T
    line_anomaly = dipole.induced_field(line_points, earth.inclination, earth.declination)
    profile_fig = plot_profile(line_anomaly, line_axis)

    # arrow_fix, _ = plot_arrow(m_magnitude, m_inc)

    xx, yy = np.meshgrid(x_axis, x_axis)

    grid_coords = np.vstack([xx.ravel(), yy.ravel(), np.full(n * n, alt)]).T.reshape(n, n, 3)
    grid_anomaly = dipole.induced_field(grid_coords, earth.inclination, earth.declination).reshape(n, n)

    q = 100
    vmin = -np.percentile(np.abs(grid_anomaly), q)
    vmax = np.percentile(np.abs(grid_anomaly), q)
    im = ax.imshow(grid_anomaly, extent=(xmin, xmax, xmin, xmax),
                   cmap=create_oasis_cmap(),
                   vmin=vmin, vmax=vmax, interpolation='bicubic')
    ax.hlines(0, xmin=line_xmin, xmax=line_xmax, color='red', linestyle='--', label='survey line')
    ax.set_axis_off()
    add_north_arrow(ax)
    # clb = fig.colorbar(im)
    # clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    xy = 0.1 + np.sin(earth.declination) * 0.08, 0.1 + np.cos(earth.declination) * 0.08
    ax.annotate(f"", xytext=(0.1, 0.1), xy=xy, xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->"), label='Mag moment')
    ax.annotate(f"Earth field\nDec {earth.dec_deg():.1f}°\nInc: {earth.inc_deg():.1f}°", (0.02, 0.02),
                xycoords='axes fraction')
    ax.scatter(0, 0, marker='x', zorder=10, label="Anomaly position", s=80)
    ax.set_aspect('equal')
    cax = inset_axes(ax, width="5%", height="30%", loc='lower left',
                     bbox_to_anchor=(1.02, 0.02, 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    clb = map_fig.colorbar(im, cax=cax, orientation='vertical', fraction=0.016)

    decl = np.deg2rad(dipole.get_declination())
    arrow_size = 1  # map units
    xy1 = dipole.position[0] + np.sin(decl) * -arrow_size, dipole.position[1] + np.cos(decl) * -arrow_size
    xy2 = dipole.position[0] + np.sin(decl) * arrow_size, dipole.position[1] + np.cos(decl) * arrow_size
    ax.annotate(f"", xytext=xy1, xy=xy2,
                arrowprops=dict(arrowstyle="->", linewidth=2, mutation_scale=20))
    # ax.annotate(
    #     f"dec {round(dipole.get_declination())}°\n"
    #     f"inc {round(dipole.get_inclination())}°\n"
    #     f"{round(dipole.get_magnitude())}nT/m",
    #     (dipole.position[0] + 0.3, dipole.position[1] + 0.3))

    # ax.annotate(
    #     f"Dipole moment, Dec {round(dipole.get_declination())}°\n",
    #     (dipole.position[0] + 0.3, dipole.position[1] + 0.3))
    ax.legend()

    clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    st.pyplot(map_fig, use_container_width=False)
    st.pyplot(profile_fig, use_container_width=False)
    # st.pyplot(arrow_fix)
    st.divider()
    url = r"https://intermagnet.org/faq/10.geomagnetic-comp"
    st.markdown("Inc - Inclination defined positive is downward. \n")
    st.markdown("Dec - Declination defined clockwise from geographic North. \n")
    st.markdown("Definition aligned with: [common geomagnetic definition](%s)" % url)
    st.divider()
    st.markdown("Static examples (taken from [here](%s))" % r"https://www.researchgate.net/publication/292966740_Airborne_Magnetic_Surveys_to_Investigate_High_Temperature_Geothermal_Reservoirs")
    st.image("mag_image.png")


@st.cache_data
def create_fig():
    map_fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    return ax, map_fig


def plot_profile(line_anomaly, x_survey):
    profile_fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(x_survey, line_anomaly)
    # ax.axhline(y=0, color='red', linestyle='--')
    ax.hlines(0, xmin=-8, xmax=8, color='red', linestyle='--', label='survey line')
    ax.set_xlabel("Horizontal [m]")
    ax.set_ylabel("Magnetic anomaly [nT]")
    ax.set_title("Anomaly along survey line")
    return profile_fig


CONFIGS = {
    'North Sea': (55.0, 3.0),
    'Equator': (0.0, 0.0),
}


def get_input():
    alt = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)
    magnitude = st.sidebar.slider("Anomaly strength [Am2]", 1.0, 30.0, 10.0, step=0.1)
    m_incl = st.sidebar.slider("Anomaly inclination [°]", -180.0, 180.0, 0.0, step=5.0)
    m_decl = st.sidebar.slider("Anomaly declination [°]", -180.0, 180.0, -45.0, step=5.0)

    config = st.sidebar.selectbox('Location', CONFIGS.keys(), index=0)
    lat_default, long_default = CONFIGS[config]

    lat = st.sidebar.slider("Location Latitude [°]", -90.0, 90.0, lat_default, step=1.0)
    long = st.sidebar.slider("Location longitude [°]", -180.0, 180.0, long_default, step=1.0)
    earths_field = EarthsInducingField.from_coords(lat, long)
    return alt, earths_field, magnitude, m_incl, m_decl


if __name__ == '__main__':
    main()
