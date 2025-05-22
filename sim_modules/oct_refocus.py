import numpy as np
from numpy.typing import NDArray
import scipy as sp
from typing import Tuple
from enum import Enum, auto


class RefocusMode(Enum):
    """Enum for pupil types"""

    PSFO = auto()
    """Refocus for PSFO (point-scanning fiber-optic) imaging system

    The same 2D Gaussian illumination and collection pupils.
    """
    SCFF = auto()
    """Refocus for SCFF (spatially coherent full-field) imaging system

    Delta-function illumination pupil and 2D circular collection pupil.
    """
    PSPinhole = auto()
    """Refocus for PS (point-scanning) imaging system with pinhole detection

    2D Gaussian illumination pupil and 1D circular collection pupil.
    """
    LF = auto()
    """Refocus for LF (line-field) imaging system.

    1D Gaussian illumination pupil and 2D circular collection pupil.
    """
    GaussColLF = auto()
    """Refocus for LF (line-field) imaging system with Gaussian collection pupil

    1D Gaussian illumination pupil and 2D Gaussian collection pupil.
    """


def forward_refocus_filter(
        l: NDArray, kbc: float, nb: float, z0: float,
        ν_para: Tuple[NDArray, NDArray],
        refocus_mode: RefocusMode,
        Df_ill: float | None = None
) -> NDArray:
    """
    Forward filter for the oct_refocus module.

    Parameters
    ----------
    l : NDArray
        optical path length
    kbc : float
        Wavenumber in the medium
    nb : float
        Refractive index of the medium
    z0 : float
        Axial location of the focal plane
    ν_para : Tuple[NDArray, NDArray]
        Spatial frequency in the x and y directions
    refocus_mode : RefocusMode
        Refocus mode, can be one of the following:
        'PSFD', 'SCFF', 'PinholePSFD', 'LF', 'GaussColLF'
    Df_ill : float, optional
        The spatial frequency width of the illumination, by default None

    Returns
    -------
    NDArray
        Refocus filter in the spatial frequency domain
    """

    ν_xx, ν_yy = ν_para

    phi = 2 * np.pi ** 2 / kbc * (l / nb - z0)
    i_phi_PSFD = phi / 2  # focused beam illumination and collection
    i_phi_FFSS = phi  # plane wave illumination
    if Df_ill is not None:
        i_phi_FrCol = (1 + 2 * Df_ill ** 4 * phi ** 2) / (1 + 4 * Df_ill ** 4 * phi ** 2) * phi  # focused beam illumination and point collection

    match refocus_mode:
        case RefocusMode.PSFO:
            defocus_phase = i_phi_PSFD * (ν_xx ** 2 + ν_yy ** 2)
        case RefocusMode.SCFF:
            defocus_phase = i_phi_FFSS * (ν_xx ** 2 + ν_yy ** 2)
        case RefocusMode.PSPinhole:
            defocus_phase = i_phi_FrCol * (ν_xx ** 2 + ν_yy ** 2)
        case RefocusMode.LF:
            defocus_phase = (i_phi_FFSS * ν_yy ** 2 + i_phi_FrCol * ν_xx ** 2)
        case RefocusMode.GaussColLF:
            defocus_phase = (i_phi_FFSS * ν_yy ** 2 + i_phi_PSFD * ν_xx ** 2)
        case _:
            defocus_phase = None

    return np.exp(1j * defocus_phase)


def isam_resampling_points(
        kb: NDArray, ν_para: Tuple[NDArray, NDArray],
        na_co_ill: float, na_co_col: float,
        refocus_mode: RefocusMode
) -> Tuple[NDArray, NDArray]:
    """
    Returns the resampling points for ISAM.

    Parameters
    ----------
    kb : NDArray
        Wavenumbers in the medium
    ν_para : Tuple[NDArray, NDArray]
        Spatial frequency in the x and y directions
        (ν_xx, ν_yy) tuple of 2D arrays
    na_co_ill : float
        Cut-off NA of the illumination system
    na_co_col : float
        Cut-off NA of the collection system
    refocus_mode : RefocusMode
        Refocus mode, can be one of the following:
        'PSFD', 'SCFF'

    Returns
    -------
    Tuple[NDArray, NDArray]
        The resampling points in the wavenumber and corresponding axial frequency.
    """
    ν_xx, ν_yy = ν_para

    match refocus_mode:
        case RefocusMode.PSFO:
            νz_min = 2 * kb.min() * np.sqrt(1 - (na_co_ill + na_co_ill) ** 2) / (2 * np.pi)
        case RefocusMode.SCFF:
            νz_min = (kb.min() + kb.min() * np.sqrt(1 - na_co_col ** 2)) / (2 * np.pi)
        case _:
            pass

    νz_max = 2 * kb.max() / (2 * np.pi)

    νz_num = int((νz_max - νz_min) / (2 * (kb[1] - kb[0]) / (2 * np.pi)))

    νz_re = np.linspace(νz_min, νz_max, num=νz_num)

    # Calculate νz-linear resampling points in the wavenumber
    match refocus_mode:
        case RefocusMode.PSFO:
            k_re = np.pi * np.sqrt((νz_re[None, None, :]) ** 2 +
                                   (ν_xx[..., None] ** 2 + ν_yy[..., None] ** 2))
        case RefocusMode.SCFF:
            k_re = np.pi * ((νz_re[None, None, :]) ** 2 +
                            (ν_xx[..., None] ** 2 + ν_yy[..., None] ** 2)) / νz_re[None, None, :]
        case _:
            k_re = None

    return k_re, νz_re


def isam(
    H: NDArray, νx: NDArray, νy: NDArray,
    kb: NDArray, k_re: NDArray
) -> NDArray:
    """ISAM interpolation.
    This function performs interpolation of the input data H using the
    specified spatial frequencies qx and qy, and the wavenumber k_re.

    Parameters
    ----------
    H : NDArray
        3D array of frequency domain data of OCT signal.
    νx : NDArray
        1D array of horizontal spatial frequency.
    νy : NDArray
        1D array of vertical spatial frequency.
    kb : NDArray
        1D array of wavenumber in the medium.
    k_re : NDArray
        3D array of resampling points in wavenumber which is linear to the axial spatial frequency.

    Returns
    -------
    NDArray
        3D array of interpolated data.
        3D spatial frequency spectrum.
    """
    points = (νy, νx, kb)
    points_re = np.stack(
        np.meshgrid(
            νy, νx, np.zeros(k_re.shape[-1], dtype=np.float64),
            indexing='ij'
        )[:2] + [k_re,],
        axis=-1
    )

    out = sp.interpolate.interpn(
        points,
        H,
        points_re,
        bounds_error=False,
        fill_value=0.0
    )
    return out
