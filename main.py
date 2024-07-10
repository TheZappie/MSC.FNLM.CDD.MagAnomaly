from typing import Tuple
from math import sin, cos, atan2, sqrt

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import magpylib
from matplotlib.colors import LinearSegmentedColormap

from compute_anomaly import Anomaly, compute_mag_field
from magnetic_field_calculator import MagneticFieldCalculator

Vector = Tuple[float, float, float]


def along_background_field(vector_array, vector):
    Incl_m, Decl_m = vector
    return np.cos(Incl_m) * np.cos(Decl_m) * vector_array[:, 0] + np.cos(Incl_m) * np.sin(Decl_m) * vector_array[:,
                                                                                                    1] + np.sin(
        Incl_m) * vector_array[:, 2]


def get_mag_field(lat, lon):
    a = MagneticFieldCalculator().calculate(
        latitude=lat,
        longitude=lon,
        depth=0,
        date='2024-06-01',
    )
    dec = a['field-value']['declination']['value']  # declination of the Earth's magnetic field in this location
    inc = a['field-value']['inclination']['value']  # inclination of the Earth's magnetic field in this location
    return dec, inc

def create_custom_colormap():
    # Define the colors and their positions
    colors = [
        (0.0, 'blue'),
        (0.2, 'cyan'),
        (0.4, 'green'),
        (0.5, 'yellow'),
        (0.6, 'orange'),
        (0.8, 'red'),
        (1.0, 'magenta')
    ]

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('custom_rainbow', colors)

    return cmap


def main():
    st.title("Magnetic Anomaly Visualizer")
    alt, lat, long, magnitude, direction = get_input()
    anomaly = Anomaly(*to_cartesian(magnitude, direction, 0), 0, 0, -alt)
    dec, inc = get_mag_field(lat, long)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    bounds = [-13, 12, -8, 13]
    n = 100
    survey_points = np.linspace(-10, 10, n)
    survey_anomaly = [compute_mag_field(anomaly, x, 0, 0) for x in survey_points]
    ax1.plot(survey_points, survey_anomaly)
    # ax.axhline(y=0, color='b', linestyle='-')
    ax1.set_xlabel("Horizontal [m]")
    # ax.set_ylabel("Altitude [m]")
    # ax.set_aspect('equal')
    ax1.set_ylabel("Magnetic anomaly [unit]")

    arrow_fix, _ = plot_arrow(magnitude, direction)

    xv, yv = np.meshgrid(survey_points, survey_points)

    B = magpylib.getB(
        sources="Dipole",
        observers=np.stack([xv.ravel(), yv.ravel(), np.full(len(survey_points) ** 2, alt)]).T,
        moment=to_cartesian(magnitude, direction, 0),
    ) * 1E9

    field = along_background_field(B, (inc, dec)).reshape(xv.shape)
    im = ax2.imshow(field, extent=(survey_points[0], survey_points[-1], survey_points[0], survey_points[-1]), cmap='turbo')
    ax2.axhline(0, color='red', linestyle='--')
    clb = fig.colorbar(im)
    clb.ax.set_ylabel('nT', fontsize=10, rotation=270)
    st.pyplot(fig)
    st.pyplot(arrow_fix)


def plot_arrow(size, direction):
    size = 1
    plot_size = 1.5
    # Convert direction from degrees to radians
    direction_rad = np.deg2rad(direction)
    end_x = size * np.cos(direction_rad)
    end_y = size * np.sin(direction_rad)

    fig, ax = plt.subplots()

    # Plot the arrow using the axes object
    ax.arrow(0, 0, end_x, end_y, head_width=0.1 * size, head_length=0.2 * size, fc='blue',
             ec='blue')
    ax.axis('off')
    # Set the limits and labels of the plot
    ax.set_xlim(-plot_size, plot_size)
    ax.set_ylim(-plot_size, plot_size)
    ax.grid()
    ax.set_aspect('equal', adjustable='box')
    return fig, ax


def distance(a: Vector, b: Vector) -> float:
    """Returns the distance between two cartesian points."""
    x = (b[0] - a[0]) ** 2
    y = (b[1] - a[1]) ** 2
    z = (b[2] - a[2]) ** 2
    return (x + y + z) ** 0.5


def magnitude(x: float, y: float, z: float) -> float:
    """Returns the magnitude of the vector."""
    return np.sqrt(x * x + y * y + z * z)


def to_spherical(x: float, y: float, z: float) -> Vector:
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
    radius = magnitude(x, y, z)
    theta = atan2(sqrt(x * x + y * y), z)
    phi = atan2(y, x)
    return radius, theta, phi


def to_cartesian(radius: float, theta: float, phi: float) -> Vector:
    """Converts a spherical coordinate (radius, theta, phi) into a cartesian one (x, y, z)."""
    x = radius * cos(phi) * sin(theta)
    y = radius * sin(phi) * sin(theta)
    z = radius * cos(theta)
    return x, y, z


def get_input():
    alt = st.sidebar.slider("Distance from anomaly [m]", 0.1, 10.0, 3.0, step=0.01)
    lat = st.sidebar.slider("Location Latitude [°]", -90.0, 90.0, 52.0, step=1.0)
    long = st.sidebar.slider("Location longitude [°]", -180.0, 180.0, 5.0, step=1.0)
    magnitude = st.sidebar.slider("Anomaly strength [Am2]", 0.0, 1E3, 10.0, step=1.0)
    direction = st.sidebar.slider("Anomaly direction [°]", -180.0, 180.0, 3.0, step=10.0)

    return alt, lat, long, magnitude, direction


if __name__ == '__main__':
    main()
