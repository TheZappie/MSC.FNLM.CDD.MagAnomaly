from dataclasses import dataclass
from typing import Iterable, Self, Protocol

import numpy as np
from numpy import ndarray

from earths_field import spher2cart

PI4 = 4 * np.pi
mu0 = 4e-7 * np.pi  # vacuum magnetic permeability
t2nt = 1e9


class MagneticThing(Protocol):
    def induced_field(self): ...
    def induced_anomaly(self): ...

    def induced_anomaly_along_vector(self, coordinates: ndarray, vector) -> ndarray: ...


@dataclass
class MagneticPoint:
    position: ndarray  # length 3
    susceptibility: float

    def induced_field(self, coordinates: ndarray, vector) -> ndarray:
        dipole = Dipole(self.position, moment=self.susceptibility * vector)
        field = compute_mag_field_cartesian_vectorized(dipole, coordinates)
        return field

    def induced_anomaly_along_vector(self, coordinates: ndarray, vector) -> ndarray:
        field = self.induced_field(coordinates, vector).T
        unit_vector = vector / np.linalg.norm(vector)
        f_anomaly = np.dot(unit_vector, field) * t2nt
        return f_anomaly


@dataclass
class Dipole:
    """
    Cartesian system

    X = positive Easting
    Y = positive Northing
    Z = positive up

    Magnetic moment same directions as coordinates
    """

    position: ndarray[3]
    moment: ndarray[3]

    def __str__(self):
        return (
            f"Position (XYZ): {self.x:.2f}, {self.y:.2f}, {self.z:.2f}\n"
            f"Magnetic moment magnitude {self.get_magnitude():.3g}\n"
            f"Inclination {self.get_inclination():.1f}� (Positive is downward)\n"
            f"Declination {self.get_declination():.1f}� (Clockwise from geographic North)"
        )

    def is_valid(self):
        return ~np.any(np.isnan(np.hstack([self.position, self.moment])))

    @property
    def xy(self):
        return self.position[:2]

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def get_magnitude(self):
        return (self.moment[0] ** 2 + self.moment[1] ** 2 + self.moment[2] ** 2) ** (
            1 / 2
        )

    def get_declination(self):
        """Clockwise from geographic North in degrees"""
        return np.rad2deg(np.arctan2(self.moment[0], self.moment[1]))

    def get_inclination(self):
        """Positive is downward"""
        magnitude = self.get_magnitude()
        if magnitude == 0:
            return 0
        return np.rad2deg((np.arcsin(-self.moment[2] / magnitude)))

    def get_spherical_moment(self):
        return self.get_magnitude(), self.get_inclination(), self.get_declination()

    @classmethod
    def from_spherical_moment(
        cls, position: ndarray, spherical_moment, degrees=False
    ) -> Self:
        """
        Create CartesianDipole from spherical moment representation

        Parameters:
        -----------
        location: ndarray
            Location in cartesian coordinates [x, y, z]
        spherical_moment: ndarray
            Moment in spherical coordinates [magnitude, inclination, declination]
            Inclination: positive downward
            Declination geographic definition: Clockwise from geographic North

        Returns:
        --------
        CartesianDipole
        """

        magnitude, inc, dec = spherical_moment
        cartesian_moment = magnitude * np.array(spher2cart(inc, dec, degrees=degrees))
        # noinspection PyTypeChecker
        return cls(position, cartesian_moment)

    def dBdZ(
        self, coordinates: ndarray, inc: float, dec: float, degrees: bool = False
    ) -> ndarray:
        """
        Coordinates 1d or 2d. If 2D, first axis required to be size 3.
        """
        field = self.dBdZ_field(coordinates)
        f_anomaly = self.along_inc_dec(field.T, inc, dec, degrees) * t2nt
        return f_anomaly

    @staticmethod
    def along_inc_dec(field, inc: float, dec: float, degrees: bool = False):
        if degrees:
            inc = np.deg2rad(inc)
            dec = np.deg2rad(dec)

        # bx, by, bz = field
        # a, b, c = spher2cart(inc, dec)
        # f_anomaly = (a * bx +
        #              b * by +
        #              c * bz)
        # return f_anomaly

        # might be faster in numpy:
        return np.dot(field, np.array(spher2cart(inc, dec)))

    def induced_anomaly(
        self, coordinates: ndarray, inc: float, dec: float, degrees=False
    ) -> ndarray:
        """
        Induced magnetic field

        Parameters:
        -----------
        coordinates : ndarray
            Array of shape (N, 3) containing (x, y, z) coordinates
        inc : float
            Inclination
        dec : float
            Declination
        degrees : bool
            Toggle for inclination declination unit: radians <-> degrees

        Returns:
        --------
        f_anomaly : ndarray
            Magnetic field anomaly at each coordinate point, in nano tesla
        """
        field = compute_mag_field_cartesian_vectorized(self, coordinates)
        f_anomaly = self.along_inc_dec(field, inc, dec, degrees) * t2nt
        return f_anomaly

    def induced_field(self, coordinates):
        return compute_mag_field_cartesian_vectorized(self, coordinates)

    def induced_field_along_vector(
        self, coordinates: ndarray, vector, normalize=False
    ) -> ndarray:
        field = compute_mag_field_cartesian_vectorized(self, coordinates).T
        if normalize:
            vector = vector / np.linalg.norm(vector)
        f_anomaly = np.dot(vector, field) * t2nt
        return f_anomaly


def compute_mag_field_cartesian_vectorized(
    source: Dipole, coordinates: ndarray
) -> ndarray:
    """
    Vectorized computation of magnetic field in cartesian coordinates

    Parameters:
    -----------
    source : CartesianDipole
        Source dipole object with location and moment attributes
    coordinates : ndarray
        Array of shape (N, 3) containing (x, y, z) coordinates

    Returns:
    --------
    bx, by, bz : ndarray
        Components of the magnetic field at each coordinate point
    """
    # Compute relative positions
    r = coordinates - source.position  # shape: N * 3
    r2 = np.sum(r**2, axis=-1)  # shape: N
    r5 = np.power(r2, 5 / 2)  # shape: N
    dot = np.dot(r, source.moment)  # shape: N
    b = (
        mu0
        / PI4
        * (3 * dot[..., np.newaxis] * r - r2[..., np.newaxis] * source.moment)
        / r5[..., np.newaxis]
    )
    return b


def induced_field(sources: Iterable[Dipole], coords, inc, dec, degrees=False):
    """
    Same as Dipole.induced_field but for multiple dipole sources

    coords: ndarray[... , 3] , last axis required to be length 3, XYZ
    """
    sources = list(sources)
    if len(sources) == 0:
        return np.zeros(coords.shape[0])
    # precomputing the cartesian vector
    vector = np.array(spher2cart(inc, dec, degrees=degrees))
    calculated_field = [
        source.induced_field_along_vector(coords, vector, degrees) for source in sources
    ]
    # calculated_field = [source.induced_field(coords, inc, dec, degrees) for source in sources]
    return np.sum(np.array(calculated_field), axis=0)


def induced_field_multiline(
    sources: Iterable[Dipole],
    lines: Iterable[ndarray],
    inc,
    dec,
    degrees=False,
    xyz_last=True,
):
    if xyz_last:
        return [induced_field(sources, coords, inc, dec, degrees) for coords in lines]
    else:
        return [induced_field(sources, coords.T, inc, dec, degrees) for coords in lines]


def induced_field_multiline_along_vector(
    sources: Iterable[Dipole], lines: Iterable[ndarray], vector, normalize=True
):
    if normalize:
        vector = vector / np.linalg.norm(vector)
    return [
        induced_field_along_vector(sources, coords, vector, normalize=False)
        for coords in lines
    ]


def induced_field_along_vector(
    sources: Iterable[Dipole], coords, vector, normalize=True
):
    """
    Dipole.induced_field for multiple dipoles
    """
    if normalize:
        vector = vector / np.linalg.norm(vector)

    calculated_field = [
        source.induced_field_along_vector(coords, vector) for source in sources
    ]
    # sum contributions from every source
    return np.sum(np.array(calculated_field), axis=0)
