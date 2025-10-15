from dataclasses import dataclass
import datetime
from math import floor

import magnetic_field_calculator
import numpy as np
from numpy import ndarray
from dateutil.utils import today
from magnetic_field_calculator import MagneticFieldCalculator


def custom_round(x, base=1):
    return base * floor(x / base)


@dataclass
class EarthsInducingField:
    """
    Earth's inducing field from the UK geological survey
    Directions defined here: https://intermagnet.org/faq/10.geomagnetic-comp
    """

    inclination: float  # radians, Positive is downward
    declination: float  # radians, Clockwise from geographic North
    strength: float  # nT

    def inc_deg(self):
        return np.rad2deg(self.inclination)

    def dec_deg(self):
        return np.rad2deg(self.declination)

    def __str__(self):
        return (
            f"Field strength {self.strength:.3f} nT\n"
            f"inclination {np.rad2deg(self.inclination):.3f}° (Positive is downward)\n"
            f"declination {np.rad2deg(self.declination):.3f}° (Clockwise from geographic North)"
        )

    @classmethod
    def from_coords(cls, lat, lon, date: str | datetime.date = today()):
        """
        latitude [deg]
        longitude [deg]

        lat lon in WGS84
        """
        if isinstance(date, datetime.date):
            year = date.year
            date = date.strftime("%Y-%m-%d")
        else:
            year = date[:4]
        # revisions are made every 5 year. Near the end of the year,
        # default values give error when not specifying the revision explicitly
        calculator = MagneticFieldCalculator(revision=str(custom_round(float(year), 5)))
        try:
            model: dict = calculator.calculate(
                latitude=lat,
                longitude=lon,
                date=str(date),
            )
        except magnetic_field_calculator.ApiError as e:
            raise RuntimeError(
                f"Error getting the magnetic model from British Geological Survey via the internet. "
                f"Original error message: {e}"
            )

        intensity = model["field-value"]["total-intensity"][
            "value"
        ]  # strength of the Earth's magnetic field
        dec = model["field-value"]["declination"][
            "value"
        ]  # declination of the Earth's magnetic field
        inc = model["field-value"]["inclination"][
            "value"
        ]  # inclination of the Earth's magnetic field

        return cls(
            inclination=np.deg2rad(inc), declination=np.deg2rad(dec), strength=intensity
        )

    def vector(self) -> ndarray:
        """
        Coordinate system
        1:      Easting
        2:      Northing
        3:      Height, positive up

        output:
           array = [B0x, B0y, B0z] in nT based on the matching field strength [nT], inclination and declination [rad] of the
           Earth's magnetic field.
        """
        return self.strength * self.unit_vector()

    def unit_vector(self) -> ndarray:
        return np.array(spher2cart(self.inclination, self.declination))


def spher2cart(inclination, declination, geographic=True, degrees=False):
    """
    Spherical to cartesian transformation

    geographic system = True: declination is angle clockwise from geographic North.
    if False, declination is counterclockwise from X axis (mathematical definition).
    If degrees = False, radians are assumed

    return 3-dimensional unit vector
    """
    if degrees:
        inclination = np.deg2rad(inclination)
        declination = np.deg2rad(declination)
    if geographic:
        a = np.cos(inclination) * np.sin(declination)
        b = np.cos(inclination) * np.cos(declination)
        c = -np.sin(inclination)
        return a, b, c
    else:
        a = np.cos(inclination) * np.cos(declination)
        b = np.cos(inclination) * np.sin(declination)
        c = -np.sin(inclination)
        return a, b, c


def cart2spher(a, b, c, geographic=True, degrees=False):
    magnitude = (a**2 + b**2 + c**2) ** (1 / 2)
    if geographic:
        inc = np.arcsin(-c / magnitude)
        dec = np.arctan2(a, b)
    else:
        inc = np.arcsin(-c / magnitude)
        dec = np.arctan2(b, a)
    if degrees:
        inc = np.rad2deg(inc)
        dec = np.rad2deg(dec)
    return magnitude, inc, dec
