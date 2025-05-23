from dataclasses import dataclass
from typing import Iterable, Self

import numpy as np
from numpy import ndarray

from earths_field import spher2cart

PI4 = 4 * np.pi
mu0 = 4e-7 * np.pi  # vacuum magnetic permeability
deg2rad = np.pi / 180
t2nt = 1e9


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
        return ((self.moment[0] ** 2
                 + self.moment[1] ** 2
                 + self.moment[2] ** 2) ** (1 / 2))

    def get_declination(self):
        """Clockwise from geographic North in degrees"""
        return np.rad2deg(np.arctan2(self.moment[0], self.moment[1]))

    def get_inclination(self):
        """Positive is upward"""
        magnitude = self.get_magnitude()
        if magnitude == 0:
            return 0
        return np.rad2deg((np.arcsin(self.moment[2] / magnitude)))

    def get_spherical_moment(self):
        return self.get_magnitude(), self.get_inclination(), self.get_declination()

    @classmethod
    def nan(cls):
        return cls(position=np.full(3, np.nan), moment=np.full(3, np.nan))

    @classmethod
    def from_magpylib(cls, dipole):
        return cls(moment=dipole.moment, position=dipole.position)

    @classmethod
    def from_spherical_moment(cls, position: ndarray, spherical_moment: ndarray) -> Self:
        """
        Create CartesianDipole from spherical moment representation

        Parameters:
        -----------
        location: ndarray
            Location in cartesian coordinates [x, y, z]
        spherical_moment: ndarray
            Moment in spherical coordinates [magnitude, inclination, declination]
            where inclination and declination are in degrees.
            Declination geographic definition: Clockwise from geographic North

        Returns:
        --------
        CartesianDipole
            New instance with converted cartesian moment
        """

        magnitude = spherical_moment[0]
        inc_rad = np.deg2rad(spherical_moment[1])
        dec_rad = np.deg2rad(spherical_moment[2])
        cartesian_moment = magnitude * np.array(spher2cart(inc_rad, dec_rad))
        # noinspection PyTypeChecker
        return cls(position, cartesian_moment)

    def dBdZ(self, coordinates: ndarray, inc: float, dec: float, degrees: bool = False) -> ndarray:
        """
        Coordinates 1d or 2d. If 2D, first axis required to be size 3.
        """
        field = self.dBdZ_field(coordinates)
        f_anomaly = self.along_inc_dec(field.T, inc, dec, degrees) * t2nt
        return f_anomaly

    @staticmethod
    def along_inc_dec(field, inc: float, dec: float, degrees: bool = False):
        if degrees:
            inc = inc * deg2rad
            dec = dec * deg2rad

        # bx, by, bz = field
        # a, b, c = spher2cart(inc, dec)
        # f_anomaly = (a * bx +
        #              b * by +
        #              c * bz)
        # return f_anomaly

        # might be faster in numpy:
        return np.dot(field, np.array(spher2cart(inc, dec)))

    def induced_field(self, coordinates: ndarray, inc: float, dec: float, degrees=False) -> ndarray:
        """
        Induced magnetic field

        Parameters:
        -----------
        coordinates : ndarray
            Array of shape (N, 3) containing (x, y, z) coordinates
        inc : float
            Inclination
        dec : float
            Declination anticlockwise from easting-axis
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

    def induced_field_along_vector(self, coordinates: ndarray, vector, normalize=False) -> ndarray:
        field = compute_mag_field_cartesian_vectorized(self, coordinates).T
        if normalize:
            vector = vector / np.linalg.norm(vector)
        f_anomaly = np.dot(vector, field) * t2nt
        return f_anomaly


def compute_mag_field_cartesian_vectorized(source: Dipole, coordinates: ndarray) -> ndarray:
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
    r2 = np.sum(r ** 2, axis=-1)  # shape: N
    r5 = np.power(r2, 5 / 2)  # shape: N
    dot = np.dot(r, source.moment)  # shape: N
    b = mu0 / PI4 * (3 * dot[..., np.newaxis] * r - r2[..., np.newaxis] * source.moment) / r5[..., np.newaxis]
    return b


def induced_field(sources: Iterable[Dipole], coords, inc, dec, degrees=False):
    """
    Same as Dipole.induced_field but for multiple dipole sources

    coords: ndarray[... , 3] , last axis required to be length 3, XYZ
    """
    sources = list(sources)
    if len(sources) == 0:
        return np.zeros(coords.shape[0])
    calculated_field = [source.induced_field(coords, inc, dec, degrees) for source in sources]
    return np.sum(np.array(calculated_field), axis=0)


def induced_field_multiline(sources: Iterable[Dipole], lines: Iterable[ndarray], inc, dec, degrees=False,
                            xyz_last=True):
    if xyz_last:
        return [induced_field(sources, coords, inc, dec, degrees) for coords in lines]
    else:
        return [induced_field(sources, coords.T, inc, dec, degrees) for coords in lines]


def induced_field_multiline_along_vector(sources: Iterable[Dipole], lines: Iterable[ndarray], vector, normalize=True):
    if normalize:
        vector = vector / np.linalg.norm(vector)
    return [induced_field_along_vector(sources, coords, vector, normalize=False) for coords in lines]


def induced_field_along_vector(sources: Iterable[Dipole], coords, vector, normalize=True):
    """
    Dipole.induced_field for multiple dipoles
    """
    if normalize:
        vector = vector / np.linalg.norm(vector)

    calculated_field = [source.induced_field_along_vector(coords, vector) for source in sources]
    # sum contributions from every source
    return np.sum(np.array(calculated_field), axis=0)
