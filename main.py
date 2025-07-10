import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dipole import Dipole, compute_mag_field_cartesian_vectorized, MagneticPoint, MagneticThing
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


def centered_arrow(ax, angle, pos=(0,0), **kwargs):
    arrow_size = 1  # map units
    x,y = pos
    xy1 = x + np.sin(angle) * -arrow_size, y + np.cos(angle) * -arrow_size
    xy2 = x + np.sin(angle) * arrow_size, y + np.cos(angle) * arrow_size
    ax.annotate(f"", xytext=xy1, xy=xy2,
                arrowprops=dict(arrowstyle="->", linewidth=5, mutation_scale=20, **kwargs), )


def main():
    st.title("3-D Magnetic dipole anomaly Visualizer")
    # alt, lat, long, magnitude, direction = get_input()
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
        line_anomaly = dipole.induced_anomaly(line_points, earth.inclination, earth.declination)
    else:
        line_anomaly = dipole.induced_anomaly(line_points, earth.vector())
    profile_fig = plot_profile(line_anomaly, line_axis)

    # arrow_fix, _ = plot_arrow(m_magnitude, m_inc)

    xx, yy = np.meshgrid(x_axis, x_axis)

    grid_coords = np.vstack([xx.ravel(), yy.ravel(), np.full(n * n, alt)]).T
    if isinstance(dipole, Dipole):
        grid_anomaly = dipole.induced_anomaly(grid_coords, earth.inclination, earth.declination).reshape(n, n)
    else:
        grid_anomaly = dipole.induced_anomaly(grid_coords, earth.vector()).reshape(n, n)

    q = 100
    vmin = -np.percentile(np.abs(grid_anomaly), q)
    vmax = np.percentile(np.abs(grid_anomaly), q)
    im = ax.imshow(grid_anomaly, extent=(xmin, xmax, xmin, xmax),
                   cmap=create_oasis_cmap(),
                   vmin=vmin, vmax=vmax, interpolation='bicubic')
    # ax.hlines(0, xmin=line_xmin, xmax=line_xmax, color='red', linestyle='--', label='survey line')
    ax.vlines(0, ymin=line_xmin, ymax=line_xmax, color='red', linestyle='--', label='survey line')
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

    if isinstance(dipole, Dipole):
        decl = np.deg2rad(dipole.get_declination())
        centered_arrow(ax, decl, (dipole.position[0], dipole.position[1]))

    n_profile = 30
    radius = 3

    y_axis = np.linspace(xmin, xmax, n_profile)
    z_axis = np.linspace(-5, 10, n_profile)
    yy, zz = np.meshgrid(y_axis, z_axis)
    mask = np.sqrt(yy ** 2 + zz ** 2) > radius  # Keep points outside the circle

    profile_coords = np.vstack([np.full(n_profile**2, 0), yy.ravel(), zz.ravel()]).T.reshape(n_profile, n_profile, 3)
    if isinstance(dipole, MagneticPoint):
        profile_field = dipole.induced_field(profile_coords, earth.vector())
    else:
        profile_field = dipole.induced_field(profile_coords)

    ax_profile, fig_profile = create_fig()
    ax_profile.set_axis_off()
    ax_profile.quiver(yy[mask], zz[mask], profile_field[:,:,1][mask], profile_field[:,:,2][mask], pivot='middle')

    yy, zz = np.meshgrid(y_axis, [alt])
    profile_coords = np.vstack([np.zeros(n_profile), yy.ravel(), zz.ravel()]).T.reshape(n_profile, 1, 3)
    if isinstance(dipole, MagneticPoint):
        profile_field = dipole.induced_field(profile_coords, earth.vector())
    else:
        profile_field = dipole.induced_field(profile_coords)

    profile_anomaly = Dipole.along_inc_dec(profile_field,  earth.inclination, earth.declination)

    U = profile_anomaly * earth.unit_vector()[1]
    V = profile_anomaly * earth.unit_vector()[2]
    Q = ax_profile.quiver(yy, zz, profile_field[:,:,1], profile_field[:,:,2], pivot='middle', color='brown')
    Q._init()
    ax_profile.quiver(yy, zz, U, V, pivot='middle', color='red', scale=Q.scale)

    if isinstance(dipole, MagneticPoint):
        centered_arrow(ax_profile, earth.inclination + 0.5 * np.pi, color='grey')
    else:
        centered_arrow(ax_profile, np.deg2rad(dipole.get_inclination()) + 0.5 * np.pi, color='grey')

    clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    st.pyplot(map_fig, use_container_width=False)
    st.pyplot(fig_profile, use_container_width=False)
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

TARGET_TYPES = {
    'Paramagnetic',
    'Complex',
}

def get_input() -> tuple[float, EarthsInducingField, MagneticThing]:
    alt = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)

    kind = st.sidebar.selectbox('target type', TARGET_TYPES, index=0)
    position = np.array([0, 0, 0])
    if kind == 'Paramagnetic':
        dipole = MagneticPoint(position, 1E-5)
    else:
        magnitude = st.sidebar.slider("Anomaly strength [Am2]", 1.0, 30.0, 10.0, step=0.1)
        m_incl = st.sidebar.slider("Anomaly inclination [°]", -180.0, 180.0, 0.0, step=5.0)
        m_decl = st.sidebar.slider("Anomaly declination [°]", -180.0, 180.0, -45.0, step=5.0)
        dipole = Dipole.from_spherical_moment(
            position=position,
            spherical_moment=np.array([magnitude, m_incl, m_decl]))

    config = st.sidebar.selectbox('Location', CONFIGS.keys(), index=0)
    lat_default, long_default = CONFIGS[config]

    lat = st.sidebar.slider("Location Latitude [°]", -90.0, 90.0, lat_default, step=1.0)
    long = st.sidebar.slider("Location longitude [°]", -180.0, 180.0, long_default, step=1.0)
    earths_field = EarthsInducingField.from_coords(lat, long)
    return alt, earths_field, dipole


if __name__ == '__main__':
    main()
