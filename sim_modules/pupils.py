import numpy as np
from scipy.special import jacobi
from typing import Tuple
from numpy.typing import NDArray
from enum import Enum, auto


class PupilType(Enum):
    """Enum for pupil types"""
    CIRC = auto()
    """Circular pupil
    """
    RECT = auto()
    """Rectangular pupil
    """
    GAUSS = auto()
    """Gaussian pupil
    """
    HANN = auto()
    """2D Hann pupil
    """
    BESSEL = auto()
    """Bessel beam pupil
    """


class Pupil3D:
    def __init__(self, pupil_type: PupilType, na_co: float | Tuple[float], **kwargs):
        """Initialize the Pupil class

        Parameters
        ----------
        pupil_type : PupilType
            Pupil type, can be 'Gauss', 'Hann', 'Bessel', 'circ', or 'rect'
        na_co : float
            Cut-off NA
        kwargs : dict
            na_w : float
                Width of the Gaussian pupil (only for 'Gauss' mode)
            w : float
                Width of the Bessel pupil ring (only for 'Bessel' mode)
        """

        match pupil_type:
            case PupilType.CIRC:
                def pupil_func(σx, σy):
                    return circ(
                        σx,
                        σy,
                        na_co
                    )
            case PupilType.RECT:
                def pupil_func(σx, σy):
                    return rect(
                        σx,
                        σy,
                        na_co
                    )
            case PupilType.GAUSS:
                try:
                    na_w = kwargs['na_w']
                except KeyError:
                    raise ValueError("Gaussian mode requires na_w parameter")
                self.na_w = na_w

                def pupil_func(σx, σy):
                    return gauss_trunc(
                        σx,
                        σy,
                        na_w,
                        na_co
                    )
            case PupilType.HANN:
                def pupil_func(σx, σy):
                    return Hann_2d_circ(
                        σx,
                        σy,
                        na_co
                    )
            case PupilType.BESSEL:
                try:
                    w = kwargs['w']
                except KeyError:
                    raise ValueError("Bessel mode requires w parameter")
                self.w = w

                def pupil_func(σx, σy):
                    return Bessel_beam_pupil(
                        σx,
                        σy,
                        na_co,
                        w
                    )
            case _:
                raise ValueError("Invalid pupil type")

        self.pupil2D = pupil_func
        self.na_co = na_co
        if np.isscalar(self.na_co):
            self.na_co_max: float = self.na_co
        else:
            self.na_co_max = max(self.na_co)
        self.pupil_type = pupil_type
        self.reverse = False

    def __call__(self, σx: NDArray, σy: NDArray, k: float, z: NDArray,
                 PARAXIAL=False) -> NDArray:
        """Call the 3D pupil function with the given arguments

        Parameters
        ----------
        σx : 2D NDArray
            Horizontal pupil coordinate
        σy : 2D NDArray
            Vertical pupil coordinate
        k : float
            Wave number
        z : 1D NDArray
            Axial coordinate

        Returns
        -------
        NDArray
            3D pupil function, h-tilde
        """
        return self.calc_ftilde(σx, σy, k, z, PARAXIAL)

    def calc_sigma_z(self, σx: NDArray, σy: NDArray, PARAXIAL=False) -> None:
        """Calculate the sigma_z value for the pupil function

        Parameters
        ----------
        σx : NDArray
            Horizontal pupil coordinate
        σy : NDArray
            Vertical pupil coordinate

        Returns
        -------
        NDArray
            Sigma_z value for the pupil function
        """
        if PARAXIAL:
            σz = np.ones_like(σx) - (σx ** 2 + σy ** 2) / 2
        else:
            σr2 = σx ** 2 + σy ** 2
            σr2[σr2 >= 1] = np.nan
            σz = np.sqrt(1 - σr2)
        self.𝛔 = (σx, σy, σz)

    def propagation_factor(self, σz: NDArray, k: float, z: NDArray) -> NDArray:
        """Calculate the propagation factor for the pupil function

        Parameters
        ----------
        σz : 2D NDArray
            Horizontal pupil coordinate
        k : float
            Wave number
        z : 1D NDArray
            Axial coordinate

        Returns
        -------
        NDArray
            Propagation factor for the pupil function
        """
        if self.reverse:
            a = -1.0
        else:
            a = 1.0
        return np.exp(-1j * k * a * σz[..., None] * z[None, None, ...])

    def calc_ftilde(self, σx: NDArray, σy: NDArray, k: float, z: NDArray,
                    PARAXIAL=False) -> NDArray:
        """Calculate the h-tilde value for the pupil function

        Parameters
        ----------
        σx : 2D NDArray
            Horizontal pupil coordinate
        σy : 2D NDArray
            Vertical pupil coordinate
        k : float
            Wave number
        z : 1D NDArray
            Axial coordinate

        Returns
        -------
        3D NDArray
            h-tilde value for the pupil function
        """
        self.calc_sigma_z(σx, σy, PARAXIAL)
        return (2 * np.pi * 1j) / k * (
            (self.pupil2D(σx, σy) / self.σ[2])[..., None] *
            self.propagation_factor(self.σ[2], k, z)
        )


class AberratedPupil3D(Pupil3D):
    def __init__(self, pupil_type: PupilType, na_co: float | Tuple,
                 ns: Tuple[Tuple], coeff: Tuple[float | Tuple],
                 ca: Tuple[float, float, float] = (0, 0, 0), kc: float = 0,
                 **kwargs):
        """Initialize the AberratedPupil class

        Parameters
        ----------
        pupil_type : PupilType
            Pupil type, can be 'Gauss', 'Hann', 'Bessel', 'circ', or 'rect'
        na_co : float
            Cut-off NA
        ns : Tuple of Tuples
            The order of the Zernike polynomials
        coeff : Tuple of floats or Tuples
            The coefficients of the Zernike polynomials
            corresponding to the order in `ns`.
        ca : Tuple[float, float, float], optional
            Coefficients for the chromatic aberration for the three axes
            They are in the rate of the focal shift per the wavenumber offset
            from the central wavenumber, `kc`.
            (horizonta, vertical, axial) in the order of (x, y, z),
            by default (0, 0, 0)
        kc : float, optional
            Central wavenumber for the chromatic aberration, by default 0
        kwargs : dict
            na_w : float
                Width of the Gaussian pupil (only for 'Gauss' mode)
            w : float
                Width of the Bessel pupil ring (only for 'Bessel' mode)
        """
        super().__init__(pupil_type, na_co, **kwargs)
        self.ns = ns
        self.coeff = coeff
        self.ca = ca
        self.kc = kc

    def set_wavefront_error(self, σx: NDArray, σy: NDArray):

        self.we = sim_wavefront_error(
            σx, σy,
            self.na_co_max,
            self.ns,
            self.coeff
        )

    def aberrated_pupil2D(self, σx: NDArray, σy: NDArray, k: float) -> NDArray:
        """Calculate the aberrated pupil function

        Parameters
        ----------
        σx : 2D NDArray
            Horizontal pupil coordinate
        σy : 2D NDArray
            Vertical pupil coordinate
        k : float
            Wave number

        Returns
        -------
        2D NDArray
            Aberrated 2D pupil function
        """
        self.set_wavefront_error(σx, σy)

        return self.pupil2D(σx, σy) * np.exp(1j * k * self.we)

    def calc_ftilde(self, σx: NDArray, σy: NDArray, k: float, z: NDArray,
                    PARAXIAL=False) -> NDArray:
        """Calculate the h-tilde value for the pupil function

        Parameters
        ----------
        σx : 2D NDArray
            Horizontal pupil coordinate
        σy : 2D NDArray
            Vertical pupil coordinate
        k : float
            Wave number
        z : 1D NDArray
            Axial coordinate

        Returns
        -------
        3D NDArray
            h-tilde value for the pupil function
        """
        self.calc_sigma_z(σx, σy, PARAXIAL)
        return (2 * np.pi * 1j) / k * (
            (self.aberrated_pupil2D(σx, σy, k) / self.σ[2])[..., None] *
            self.propagation_factor(self.σ[2], k, z)
        ) * (np.exp(-1j * k * (k - self.kc) * (
            σx * self.ca[0] + σy * self.ca[1] + self.σ[2] * self.ca[2]
            ))  # Chromatic aberration
            * np.exp(1j * k * self.kc * self.ca[2])  # Correct the shift due to LCA
        )[..., None]


def circ(σx: NDArray, σy: NDArray, c: float) -> NDArray:
    """Calculate 2D circular window

    Parameters
    ----------
    yx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    c : float
        Cut-off

    Returns
    -------
    NDArray
        2D array of the pupil
    """
    r = np.sqrt(σx ** 2 + σy ** 2)
    out = np.ones_like(r, dtype=np.float32)
    out[σx ** 2 + σy ** 2 > c ** 2] = 0
    return out


def rect(σx: NDArray, σy: NDArray, c: float | Tuple[float]) -> NDArray:
    """Calculate 2D rectangular window

    Parameters
    ----------
    yx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    c : float
        Cut-off

    Returns
    -------
    NDArray
        2D array of the pupil
    """
    r = np.sqrt(σx ** 2 + σy ** 2)
    out = np.ones_like(r, dtype=np.float32)
    if np.isscalar(c):
        out[np.logical_or(np.abs(σx) > c, np.abs(σy) > c)] = 0
    else:
        cx, cy = c
        out[np.logical_or(np.abs(σx) > cx, np.abs(σy) > cy)] = 0
    return out


def gauss_trunc(σx: NDArray, σy: NDArray, w: float | Tuple, c: float | Tuple
                ) -> NDArray:
    """Calculate truncated Gaussian distribution

    Parameters
    ----------
    σx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    w : float or array_like
        Width
    c : float or array_like
        Cut-off

    Returns
    -------
    NDArray
        2D array of the pupil
    """
    if np.isscalar(w):
        out = np.exp(- (σx ** 2 + σy ** 2) / w ** 2)
    else:
        wx, wy = w
        out = np.exp(- (σx ** 2 / wx ** 2 + σy ** 2 / wy ** 2))
    if np.isscalar(c):
        out[σx ** 2 + σy ** 2 > c ** 2] = 0
    else:
        cx, cy = c
        out[σx ** 2 / cx ** 2 + σy ** 2 / cy ** 2 > 1] = 0
    return out


def Hann_2d_circ(σx: NDArray, σy: NDArray, c: float) -> NDArray:
    """Calculate 2D circular Hann window

    Parameters
    ----------
    yx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    c : float
        Cut-off

    Returns
    -------
    NDArray
        2D array of the pupil
    """
    r = np.sqrt(σx ** 2 + σy ** 2)
    out = np.cos(np.pi * r / (2 * c)) ** 2  # / (2 * c)
    out[σx ** 2 + σy ** 2 > c ** 2] = 0
    return out


def Bessel_beam_pupil(σx: NDArray, σy: NDArray, c: float, w: float) -> NDArray:
    """Pupil for Bessel beam

    Parameters
    ----------
    σx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    c : float
        Cut-off
    w : float
        Width of the ring

    Returns
    -------
    NDArray
        2D array of the pupil
    """
    r = np.sqrt(σx ** 2 + σy ** 2)
    out = np.zeros_like(r, dtype=np.float32)
    out[np.abs(r - c) < w] = 1.0
    return out


def sim_wavefront_error(
        σx: NDArray, σy: NDArray, c: float,
        ns: Tuple[Tuple], coeff: Tuple[float | Tuple]
        ) -> NDArray:
    """Simulate wavefront error using Zernike polynomials
    This function calculates the wavefront error using Zernike polynomials

    Parameters
    ----------
    σx : NDArray
        Horizontal pupil coordinate
    σy : NDArray
        Vertical pupil coordinate
    c : float
        Cut-off
    ns : Tuple of Tuples
        The order of the Zernike polynomials
    coeff : Tuple of floats or Tuples
        The coefficients of the Zernike polynomials
        corresponding to the order in `ns`.

    Returns
    -------
    NDArray
        2D array of the wavefront error
    """
    W = np.zeros_like(σx)

    fz = (σx - 1j * σy) / c
    for os, co in zip(ns, coeff):
        joc_n = jacobi((os[0] - os[1]) / 2, 0, os[1])
        zer = fz ** os[1] * joc_n(2 * np.abs(fz) ** 2 - 1)
        zer[np.abs(fz) > 1] = np.nan + 1j * np.nan
        if os[1] == 0:
            assert np.isscalar(co)
            W += co * np.sqrt(os[0] + 1) * zer.real
        else:
            assert len(co) == 2
            W += co[0] * np.sqrt(2 * (os[0] + 1)) * zer.real
            W += co[1] * np.sqrt(2 * (os[0] + 1)) * zer.imag

    return W
