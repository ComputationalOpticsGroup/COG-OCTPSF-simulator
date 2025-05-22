import numpy as np
from scipy.signal import zoom_fft
from numpy.typing import NDArray, ArrayLike


def zoom_fft_correct_phase(
        x: NDArray, fn: ArrayLike, s: float, m: int | None = None, *,
        fs: float = 2, endpoint: bool = False, axis: int = -1) -> NDArray:
    """
    Compute the DFT of `x` only for frequencies in range `fn`
    with a correction of phase shift due to not start from the origin of `x`.

    Parameters
    ----------
    x : array
        The signal to transform.
    fn : array_like
        A length-2 sequence [`f1`, `f2`] giving the frequency range, or a
        scalar, for which the range [0, `fn`] is assumed.
    s : float
        The coordinate of the starting point of the input `x[0]`.
    m : int, optional
        The number of points to evaluate.  The default is the length of `x`.
    fs : float, optional
        The sampling frequency.  If ``fs=10`` represented 10 kHz, for example,
        then `f1` and `f2` would also be given in kHz.
        The default sampling frequency is 2, so `f1` and `f2` should be
        in the range [0, 1] to keep the transform below the Nyquist
        frequency.
    endpoint : bool, optional
        If True, `f2` is the last sample. Otherwise, it is not included.
        Default is False.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.

    Returns
    -------
    out : ndarray
        The transformed signal.  The Fourier transform will be calculated
        at the points f1, f1+df, f1+2df, ..., f2, where df=(f2-f1)/m.
    """
    f = np.linspace(
        fn[0], fn[1],
        m if m is not None else x.shape[axis],
        endpoint=endpoint
    )

    axs = [i for i in range(x.ndim)]
    axs.pop(axis)

    out = zoom_fft(x, fn, m=m, fs=fs, endpoint=endpoint, axis=axis)

    # Cancel phase shift of zoom FFT
    out = out * np.exp(-2j * np.pi * np.expand_dims(f, axis=axs) * s)

    return out
