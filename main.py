import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
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
    py = 0.8 * ax.figure.bbox.height
    px = 0.05 * ax.figure.bbox.width

    # Draw an arrow with a text "N" above it using annotation
    ax.annotate("N", xy=(px, py), fontsize=15, xycoords="figure pixels", ha='center')
    ax.annotate("", xy=(px, py), xytext=(px, py - 80), xycoords="figure pixels",
                arrowprops=dict(arrowstyle="-|>", facecolor="black"))


def main():
    st.title("Magnetic Anomaly Visualizer")
    # alt, lat, long, magnitude, direction = get_input()
    alt, earth, m_magnitude, m_inc, m_dec = get_input()
    dipole = Dipole.from_spherical_moment(
        position=np.array([0, 0, 0]),
        spherical_moment=np.array([m_magnitude, m_inc, m_dec]))

    map_fig, ax = plt.subplots(1, 1, figsize=(6, 6))

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
                   # cmap='turbo',
                   vmin=vmin, vmax=vmax, interpolation='bicubic')
    ax.hlines(0, xmin=line_xmin, xmax=line_xmax, color='red', linestyle='--', label='survey line')
    ax.set_axis_off()
    add_north_arrow(ax)
    # clb = fig.colorbar(im)
    # clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    xy = 0.1 + np.sin(earth.declination) * 0.08, 0.1 + np.cos(earth.declination) * 0.08
    ax.annotate(f"", xytext=(0.1, 0.1), xy=xy, xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"Earth\ndeclination {earth.dec_deg():.1f}°\ninclination: {earth.inc_deg():.1f}°", (0.02, 0.02), xycoords='axes fraction')

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
    ax.annotate(
        f"dec {round(dipole.get_declination())}°\n"
        f"inc {round(dipole.get_inclination())}°\n"
        f"{round(dipole.get_magnitude())}nT/m",
        (dipole.position[0] + 0.3, dipole.position[1] + 0.3))
    ax.legend()
    clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    st.pyplot(map_fig, use_container_width=False)
    st.pyplot(profile_fig, use_container_width=False)
    # st.pyplot(arrow_fix)


def plot_profile(line_anomaly, x_survey):
    profile_fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x_survey, line_anomaly)
    ax.axhline(y=0, color='green', linestyle='--')
    ax.set_xlabel("Horizontal [m]")
    ax.set_ylabel("Magnetic anomaly [nT]")
    return profile_fig


CONFIGS = {
    'North Sea': (55.0, 3.0),
    'Equator': (0.0, 0.0),
}


def get_input():
    alt = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)
    magnitude = st.sidebar.slider("Anomaly strength [Am2]", 10.0, 100.0, 50.0, step=1.0)
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
