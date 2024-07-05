from dataclasses import dataclass

import numpy as np

# # Input parameters:

# # m: model parameters 
# 0,1,2 = mx, my ,mz ( magnetic moment vector components)
# 3,4 = X,Y position of moment
# 5 = Depth BSB of source

# X,Y,Z = Obervation points (Z given defined as altiude ASB)

# SN = Add random gaussian noise to the data (based on SNR)
inclf = 67
declf = 2


@dataclass
class Anomaly:
    mx: float
    my: float
    mz: float
    x: float
    y: float
    z: float


def compute_mag_field(m: Anomaly, x: float, y: float, z: float, noise: float = 0):
    my0 = 4e-7 * np.pi
    rad = np.pi / 180
    t2nt = 1e9

    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    z = np.reshape(z, -1)

    def dircos(incl, decl, azim):
        d2rad = 0.017453293  # Converting from degrees to radians
        xincl = incl * d2rad
        xdecl = decl * d2rad
        xazim = azim * d2rad
        a = np.cos(xincl) * np.cos(xdecl - xazim)
        b = np.cos(xincl) * np.sin(xdecl - xazim)
        c = np.sin(xincl)
        return a, b, c  # return new values

    # X AXIS : POSITIVE East
    # Y AXIS : Positive North

    # Define magnetic moment incl, decl from input vector
    mF = np.sqrt(m.mx ** 2 + m.my ** 2 + m.mz ** 2)
    mi = np.arctan2(m.mz, np.sqrt(m.mx ** 2 + m.my ** 2))  # I = atan(Z/H)
    md = np.arctan2(m.my, m.mx)  # D = atan(Y/X)

    mi = mi / rad
    md = md / rad

    [mx, my, mz] = dircos(mi, md, 0)

    xp = x
    yp = y
    zp = z - np.mean(z)

    rx = np.double(xp - m.x)
    ry = np.double(yp - m.y)
    rz = np.double(zp - m.z)
    r2 = rx ** 2 + ry ** 2 + rz ** 2
    r = np.sqrt(r2)
    r5 = np.power(r, 5)

    dot = rx * mx + ry * my + rz * mz

    bx = my0 / (4 * np.pi) * mF * (3 * dot * rx - r2 * mx) / r5
    by = my0 / (4 * np.pi) * mF * (3 * dot * ry - r2 * my) / r5
    bz = my0 / (4 * np.pi) * mF * (3 * dot * rz - r2 * mz) / r5

    # Geomagnetic field input parameters(Inclination, Declination)
    Incl_m = inclf * rad  # Should update IGRF delcf and inclf
    Decl_m = declf * rad

    # Calculate the anomaly with respect to the geomagnetic field. 

    f_anomaly = np.cos(Incl_m) * np.cos(Decl_m) * bx + np.cos(Incl_m) * np.sin(Decl_m) * by + np.sin(Incl_m) * bz

    f_anomaly = (f_anomaly) * t2nt

    if noise > 0:
        signal_std = np.std(f_anomaly)
        noise_std = signal_std / noise
        data_std = noise_std * np.ones_like(f_anomaly)
        data_noise = np.random.rand(data_std.size) * data_std

        f_anomaly = f_anomaly + data_noise

    return f_anomaly
