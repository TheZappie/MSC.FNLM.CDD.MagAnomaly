import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from compute_anomaly import Anomaly, compute_mag_field


def main():
    st.title("MAG Coverage")
    alt = get_input()

    fig, ax = plt.subplots()
    bounds = [-13, 12, -8, 13]

    anomaly = Anomaly(10, 0, 0, 0, 0, alt)
    survey_points = np.linspace(-10, 10)
    survey_anomaly = [compute_mag_field(anomaly, x, 0, 0) for x in survey_points]
    ax.plot(survey_anomaly)
    # ax.axhline(y=0, color='b', linestyle='-')
    ax.set_xlabel("Horizontal [m]")
    # ax.set_ylabel("Altitude [m]")
    # ax.set_aspect('equal')
    ax.set_ylabel("Magnetic anomaly [unit]")
    st.pyplot(fig)


def get_input():
    BD = st.slider("Distance from anomaly [m]", 0.0, 10.0, 3.0, step=0.1)
    return BD


if __name__ == '__main__':
    main()
