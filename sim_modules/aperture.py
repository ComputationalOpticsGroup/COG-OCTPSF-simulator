import numpy as np
import scipy as sp
from typing import Tuple
from numpy.typing import NDArray
from .pupils import Pupil3D


class Aperture:
    """
    Class to handle the aperture of an optical system.
    """

    def __init__(self,
                 ill: Pupil3D, col: Pupil3D,
                 IMG_MODE: str,
                 FIELD_TYPE: str = 'scalar',
                 ψ: NDArray | float = 1.0
                 ) -> None:
        """
        Initialize the aperture with a given radius.

        Parameters
        ----------
        ill : Pupil3D
            Pupil function for the illumination system.
        col : Pupil3D
            Pupil function for the collection system.
        IMG_MODE : str
            Imaging mode, can be one of the following:
            'PSFD', 'SCFF', 'LF'
        FIELD_TYPE : str, optional
            Type of aperture function {'scalar'}, by default 'scalar'.
        ψ : NDArray | float, optional
            Susceptibility tensor of dipoles, by default 1.0.
        """
        self.ill = ill
        self.col = col
        self.IMG_MODE = IMG_MODE
        self.FIELD_TYPE = FIELD_TYPE

        match self.FIELD_TYPE:
            case 'scalar':
                if not np.isscalar(ψ):
                    raise ValueError("ψ must be a scalar for scalar aperture.")
                self.ψ = np.array([[ψ]])
            case _:
                raise ValueError("Invalid aperture type. Must be 'scalar'.")

    def __call__(self, ν_para_ill: Tuple[NDArray, NDArray],
                 ν_para_col: Tuple[NDArray, NDArray],
                 k: float, z: NDArray,
                 NORMALIZE: bool = False, idx_z0: int = 0,
                 PARAXIAL: bool = False) -> NDArray:
        """
        Obtain the aperture function to the given coordinates.

        Parameters
        ----------
        ν_para : Tuple[NDArray, NDArray]
            Spatial frequency in the x and y directions
            for the illumination pupil (ν_xx, ν_yy) tuple of 2D arrays.
        ν_para_col : Tuple[NDArray, NDArray]
            Spatial frequency in the x and y directions
            for the collection pupil (ν_xx, ν_yy) tuple of 2D arrays.
        k : float
            Wavenumber in the medium.
        z : NDArray
            Axial coordinate.
        NORMALIZE : bool, optional
            If True, normalize the pupil function, by default False.
        idx_z0 : int, optional
            Index for the axial coordinate for normalization, by default 0.
        PARAXIAL : bool, optional
            If True, use paraxial approximation, by default False.

        Returns
        -------
        h-tilde : NDArray
            Aperture function evaluated at the given coordinates.
        """

        dνx_col = np.abs(ν_para_col[0][0, 1] - ν_para_col[0][0, 0])
        dνy_col = np.abs(ν_para_col[1][1, 0] - ν_para_col[1][0, 0])

        match self.IMG_MODE:
            case 'PSFD':
                dνx = np.abs(ν_para_ill[0][0, 1] - ν_para_ill[0][0, 0])
                dνy = np.abs(ν_para_ill[1][1, 0] - ν_para_ill[1][0, 0])
                assert dνx == dνx_col and dνy == dνy_col, "Sampling steps in spatial frequency between illumination and collection must be equal."
            case 'LF':
                dνx = np.abs(ν_para_ill[0][0, 1] - ν_para_ill[0][0, 0])
                assert dνx == dνx_col, "Sampling steps in spatial frequency between illumination and collection must be equal."
                dνy = dνy_col
            case 'SCFF':
                dνx = dνx_col
                dνy = dνy_col
            case _:
                raise ValueError("Invalid imaging mode. Must be 'PSFD', 'SCFF', or 'LF'.")

        self.dνx = dνx
        self.dνy = dνy

        self.calc_ftiles(
            ν_para_ill, ν_para_col, k, z, PARAXIAL
        )

        # Normalize the pupils
        if NORMALIZE:
            self.normalize_ftildes(idx_z0)

        self.calc_htilde()

        return self.htilde

    def calc_ftiles(self, ν_para_ill: Tuple[NDArray, NDArray],
                    ν_para_col: Tuple[NDArray, NDArray],
                    k: float, z: NDArray,
                    PARAXIAL: bool = False):
        """
        Calculate the 3D pupil functions for a given set of parameters.

        Parameters
        ----------
        ν_para : Tuple[NDArray, NDArray]
            Spatial frequency in the x and y directions
            for the illumination pupil (ν_xx, ν_yy) tuple of 2D arrays.
        ν_para_col : Tuple[NDArray, NDArray]
            Spatial frequency in the x and y directions
            for the collection pupil (ν_xx, ν_yy) tuple of 2D arrays.
        k : float
            Wavenumber in the medium.
        z : NDArray
            Axial coordinate.
        PARAXIAL : bool, optional
            If True, use paraxial approximation, by default False.

        Returns
        -------
        f_tilde_ill : NDArray
            3D pupil function evaluated at the given coordinates.
        f_tilde_col : NDArray
            3D pupil function evaluated at the given coordinates.
        """
        if self.IMG_MODE == 'SCFF':
            f_tilde_ill = np.exp(- 1j * k * z[None, None, ...])
        else:
            f_tilde_ill = self.ill(
                - (2 * np.pi / k) * ν_para_ill[0],
                - (2 * np.pi / k) * ν_para_ill[1],
                k, z, PARAXIAL
            )
        f_tilde_col = self.col(
            - (2 * np.pi / k) * ν_para_col[0],
            - (2 * np.pi / k) * ν_para_col[1],
            k, z, PARAXIAL
        )

        self.ftilde_ill = f_tilde_ill
        self.ftilde_col = f_tilde_col

    def get_dyad_for_ftilde(self) -> Tuple[NDArray, NDArray]:
        """
        Convert the 3D pupil function to a dyadic form.

        Returns
        -------
        d_ill : NDArray
            Dyad for the illumination 3D pupil function.
        d_col : NDArray
            Dyad for the collection 3D pupil function.
        """

        match self.FIELD_TYPE:
            case 'scalar':
                # Polarization-insensitive
                d_ill = d_col = np.ones((1, ), dtype=np.float32)

        return d_ill, d_col

    def normalize_ftildes(self, idx_z0: int):
        """
        Normalize the 3D pupil functions.

        Parameters
        ----------
        idx_z : Int
            Index for the axial coordinate.
        """
        f_tilde_ill = self.ftilde_ill
        f_tilde_col = self.ftilde_col

        match self.IMG_MODE:
            case 'PSFD':
                f_tilde_ill /= np.sqrt(
                    np.nansum(np.abs(f_tilde_ill[..., idx_z0]) ** 2,
                              axis=(0, 1)) * self.dνx * self.dνy
                    )[None, None, None, ...]
                ill_amp = np.abs(np.sum(
                    np.nan_to_num(f_tilde_ill[..., idx_z0])
                    ))
            case 'LF':
                f_tilde_ill /= np.sqrt(
                    np.nansum(np.abs(f_tilde_ill[..., idx_z0]) ** 2,
                              axis=(1, )) * self.dνx
                    )[None, :, None, ...]
                ill_amp = np.abs(np.sum(
                    np.nan_to_num(f_tilde_ill[0, ..., idx_z0])
                    ))
            case 'SCFF':
                ill_amp = np.abs(np.sum(
                    np.nan_to_num(f_tilde_ill[..., idx_z0])
                    ))
        f_tilde_col /= np.sqrt(
            np.nansum(np.abs(f_tilde_col[..., idx_z0]) ** 2,
                      axis=(0, 1)) * self.dνx * self.dνy
            )[None, None, None, ...]
        col_amp = np.abs(np.sum(np.nan_to_num(f_tilde_col[..., idx_z0])))

        self.ftilde_ill = f_tilde_ill
        self.ftilde_col = f_tilde_col
        self.ill_amp = ill_amp
        self.col_amp = col_amp

    def calc_htilde(self):
        """
        Calculate the system transfer function.

        Returns
        -------
        h_tilde : NDArray
            System transfer function.
        """

        d_ill, d_col = self.get_dyad_for_ftilde()

        # f_tilde_ill_d[:, x_num // 4:(x_num * 3) // 4,
        #       x_num // 4:(x_num * 3) // 4, ...] = np.nan_to_num(
        #            d[..., None] * f_tilde_ill[None, ...]
        #         )
        f_tilde_ill_d = np.nan_to_num(
            d_ill[..., None] * self.ftilde_ill[None, ...]
        )

        # f_tilde_col_d[:, x_num // 4:(x_num * 3) // 4,
        #         x_num // 4:(x_num * 3) // 4, ...] = np.nan_to_num(
        #             d[..., None] * f_tilde_col[None, ...]
        #         )
        f_tilde_col_d = np.nan_to_num(
            d_col[..., None] * self.ftilde_col[None, ...]
        )

        # Calculate the system transfer function
        # 2D convolution between f_tilde_ill and f_tilde_col.
        match self.IMG_MODE:
            case 'SCFF':
                h_tilde = np.sum(f_tilde_ill_d[None, ...] * f_tilde_col_d[:, None]
                                 * self.ψ[..., None, None, None],
                                 axis=(0, 1))
            case 'PSFD':
                h_tilde = np.sum(
                    sp.signal.fftconvolve(
                        f_tilde_ill_d[None, ...],
                        f_tilde_col_d[:, None],
                        mode='full',
                        axes=(2, 3)
                    ) * self.dνx * self.dνy
                    * self.ψ[..., None, None, None],
                    axis=(0, 1)
                )
            case 'LF':
                h_tilde = np.sum(
                    sp.signal.fftconvolve(
                        f_tilde_ill_d[None, ...],
                        f_tilde_col_d[:, None],
                        mode='full',
                        axes=3
                    ) * self.dνx
                    * self.ψ[..., None, None, None],
                    axis=(0, 1)
                )

        self.htilde = h_tilde
