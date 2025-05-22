import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LogNorm
from typing import TypedDict, List
from .pupils import Pupil3D, AberratedPupil3D
from .aperture import IMG_MODE
from numpy.typing import NDArray

colors = plt.colormaps['viridis'].colors

AXIAL_FREQ_LABEL = r"Axial spatial frequency $\nu_z$ [µm$^{-1}$]"
HORIZONTAL_FREQ_LABEL = r"Horizontal spatial frequency $\nu_x$ [µm$^{-1}$]"
VERTICAL_FREQ_LABEL = r"Vertical spatial frequency $\nu_y$ [µm$^{-1}$]"

OPL_LABEL = r"Single-trip OPL $l$ [µm]"
RECONST_AXIAL_LABEL = r"Reconstructed depth $z_\mathrm{re}$ [µm]"
HORIZONTAL_LABEL = r"Horizontal $x_0$ [µm]"
VERTICAL_LABEL = r"Vertical $y_0$ [µm]"
AXIAL_LABEL = r"Axial $z_0$ [µm]"

HORIZONTAL_COS_LABEL = r"Horizontal directional cosine $\sigma_x$"
VERTICAL_COS_LABEL = r"Vertical directional cosine $\sigma_y$"

CTF_PHASE_LABEL = r"Phase $\angle H$ [rad]"
CTF_AMP_LABEL = r"CTF amplitude $|H|$ "
CTF_REAL_LABEL = r"CTF real part Re[$H$] "
CTF_IMAG_LABEL = r"CTF imaginary part Im[$H$] "

PSF_AMP_LABEL = r"PSF amplitude $|h|$ "

OCT_PSF_AMP_LABEL = r"OCT's PSF amplitude $|h_\mathrm{OCT}|$ "
OCT_SPECT_AMP_LABEL = r"OCT's spatial frequency amplitude $|\tilde{h}_\mathrm{OCT}|$ "
OCT_PSF_PHASE_LABEL = r"OCT's PSF phase $\angle h_\mathrm{OCT}$ [rad]"
OCT_PSF_REAL_LABEL = r"OCT's PSF real part Re[$h_\mathrm{OCT}$] "
OCT_PSF_IMAG_LABEL = r"OCT's PSF imaginary part Im[$h_\mathrm{OCT}$] "

DEFOCUS = r"Defocus: {} µm"

LOCATION_XY = r"($x_0$, $y_0$) = ({} µm, {} µm)"
LOCATION_X = r"$x_0$ = {} µm"
LOCATION_Y = r"$y_0$ = {} µm"
LOCATION_OPL = r"$l$ = {} µm"
LOCATION_ZRE = r"$z_\mathrm{{re}}$ = {} µm"
LOCATION_Z = r"$z_0$ = {} µm"

WAVELENGTH = r"Wavelength: {:.3f} µm"
HORIZONTAL_FREQ = r"Horizontal frequency: {:.3f} µm$^{{-1}}$"
VERTICAL_FREQ = r"Vertical frequency: {:.3f} µm$^{{-1}}$"
AXIAL_FREQ = r"Axial frequency: {:.3f} µm$^{{-1}}$"

WAVELENGTH_LEGEND = r"$\lambda$ = {:.3f} µm"


def ctf_unit(ABS_UNIT: bool = False) -> str:
    """Return the CTF's correct unit."""
    if ABS_UNIT:
        return r"[µm]"
    else:
        return r" [a.u.]"


def psf_unit(ABS_UNIT: bool = False) -> str:
    """Return the PSF's correct unit."""
    if ABS_UNIT:
        return r"[µm$^{{-2}}$]"
    else:
        return r" [a.u.]"


def oct_psf_unit(ABS_UNIT: bool = False) -> str:
    """Return the correct unit of OCT's PSF."""
    if ABS_UNIT:
        return r"[µm$^{{-1}}$]"
    else:
        return r" [a.u.]"


def oct_htilde_unit(ABS_UNIT: bool = False) -> str:
    """Return the correct unit of OCT's spatial frequency."""
    if ABS_UNIT:
        return r"[µm]"
    else:
        return r" [a.u.]"


class PSFDictRequired(TypedDict):
    psf: NDArray
    x: NDArray
    defocus: NDArray
    desc: str
    MODE: IMG_MODE
    NORMALIZE: bool


class PSFDict(PSFDictRequired, total=False):
    opl: NDArray
    z: NDArray


def plot_psf_xl(psf_dict: PSFDict,
                i=0, num=None, show_FWHM=True,
                log=False, y_i: int | None = None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size

    if y_i is None:
        y_i = xd_num // 2
    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    amp = np.abs(psf[
        y_i,
        xd_r,
        i,
        :
    ])
    pha = np.angle(psf[
        y_i,
        xd_r,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, (ax, ax_phase) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax.pcolormesh(
        ld_m,
        xd_m,
        amp,
        norm=LogNorm() if log else None,
    )
    fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    if show_FWHM:
        ax.contour(
            ld_m,
            xd_m,
            amp,
            [np.max(amp) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title("amplitude")

    pcm2 = ax_phase.pcolormesh(
        ld_m,
        xd_m,
        pha,
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    fig.colorbar(pcm2, label=OCT_PSF_PHASE_LABEL)
    ax_phase.set_aspect('equal')
    ax_phase.set_title("phase")

    fig.suptitle(
        ("{}-OCT X-slice {} PSF.\n" +
         DEFOCUS + ", " + LOCATION_Y).format(
            MODE.name, desc, d[i], xd[y_i])
    )
    if ZR:
        fig.supxlabel(RECONST_AXIAL_LABEL)
    else:
        fig.supxlabel(OPL_LABEL)
    fig.supylabel(HORIZONTAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_psf_xl_reim(psf_dict: PSFDict,
                     i=0, num=None, y_i: int | None = None,
                     hmax: float | None = None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size

    if y_i is None:
        y_i = xd_num // 2
    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    if hmax is None:
        hmax = np.max(np.abs(psf[..., i, :]))

    re = np.real(psf[
        y_i,
        xd_r,
        i,
        :
    ])
    im = np.imag(psf[
        y_i,
        xd_r,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax_re.pcolormesh(
        ld_m,
        xd_m,
        re,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm, label=OCT_PSF_REAL_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_re.set_aspect('equal')
    ax_re.set_title("real")

    pcm2 = ax_im.pcolormesh(
        ld_m,
        xd_m,
        im,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm2, label=OCT_PSF_IMAG_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_im.set_aspect('equal')
    ax_im.set_title("imaginary")

    fig.suptitle(
        ("{}-OCT X-slice {} PSF.\n" +
         DEFOCUS + ", " + LOCATION_Y).format(
            MODE.name, desc, d[i], xd[y_i])
    )
    if ZR:
        fig.supxlabel(RECONST_AXIAL_LABEL)
    else:
        fig.supxlabel(OPL_LABEL)
    fig.supylabel(HORIZONTAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_psf_yl(psf_dict: PSFDict,
                i=0, num=None, show_FWHM=True,
                log=False, x_i: int | None = None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    if x_i is None:
        x_i = xd_num // 2

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    amp = np.abs(psf[
        xd_r,
        x_i,
        i,
        :
    ])
    pha = np.angle(psf[
        xd_r,
        x_i,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, (ax, ax_pha) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax.pcolormesh(
        ld_m,
        xd_m,
        amp,
        norm=LogNorm() if log else None,
    )
    fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    if show_FWHM:
        ax.contour(
            ld_m,
            xd_m,
            amp,
            [np.max(amp) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title("amplitude")

    pcm2 = ax_pha.pcolormesh(
        ld_m,
        xd_m,
        pha,
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    fig.colorbar(pcm2, label=OCT_PSF_PHASE_LABEL)
    ax_pha.set_aspect('equal')
    ax_pha.set_title("phase")

    fig.suptitle(
        ("{}-OCT Y-slice {} PSF.\n" +
         DEFOCUS + ", " + LOCATION_X).format(
            MODE.name, desc, d[i], xd[x_i])
    )
    if ZR:
        fig.supxlabel(RECONST_AXIAL_LABEL)
    else:
        fig.supxlabel(OPL_LABEL)
    fig.supylabel(VERTICAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_psf_yl_reim(psf_dict: PSFDict,
                     i=0, num=None, x_i: int | None = None,
                     hmax: float | None = None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    if x_i is None:
        x_i = xd_num // 2

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    if hmax is None:
        hmax = np.max(np.abs(psf[..., i, :]))

    re = np.real(psf[
        xd_r,
        x_i,
        i,
        :
    ])
    im = np.imag(psf[
        xd_r,
        x_i,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax_re.pcolormesh(
        ld_m,
        xd_m,
        re,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm, label=OCT_PSF_REAL_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_re.set_aspect('equal')
    ax_re.set_title("real")

    pcm2 = ax_im.pcolormesh(
        ld_m,
        xd_m,
        im,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm2, label=OCT_PSF_IMAG_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_im.set_aspect('equal')
    ax_im.set_title("imaginary")

    fig.suptitle(
        ("{}-OCT Y-slice {} PSF.\n" +
         DEFOCUS + ", " + LOCATION_X).format(
            MODE.name, desc, d[i], xd[x_i])
    )
    if ZR:
        fig.supxlabel(RECONST_AXIAL_LABEL)
    else:
        fig.supxlabel(OPL_LABEL)
    fig.supylabel(VERTICAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_psf_xy(psf_dict: PSFDict,
                i=0, num=None, l_i_s=0, show_FWHM=True, ref=None,
                log=False):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    ld_num = ld.shape[0]

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    amp = np.abs(psf[
        xd_r,
        xd_r,
        i,
        ld_num // 2 + l_i_s
    ])
    pha = np.angle(psf[
        xd_r,
        xd_r,
        i,
        ld_num // 2 + l_i_s
    ])
    xd_m = xd[xd_r]

    xy_max = np.unravel_index(np.argmax(amp), amp.shape)
    max_val = amp[*xy_max]

    if np.isscalar(ref):
        msg = 'Strehl ratio: {}\n'.format(max_val / ref)
    else:
        msg = ''

    fig, (ax, ax_pha) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax.pcolormesh(
        xd_m, xd_m,
        amp,
        norm=LogNorm() if log else None
    )
    fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    if show_FWHM:
        ax.contour(
            xd_m, xd_m,
            amp,
            [np.max(amp) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title("amplitude")

    pcm2 = ax_pha.pcolormesh(
        xd_m, xd_m,
        pha,
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    fig.colorbar(pcm2, label=OCT_PSF_PHASE_LABEL)
    ax_pha.set_aspect('equal')
    ax_pha.set_title("phase")

    fig.suptitle(
        ("{}-OCT en-face {} PSF.\n" +
         DEFOCUS + ", " +
         (LOCATION_ZRE if ZR else LOCATION_OPL)
         # "\nMax at: ({}, {}) µm."
         ).format(
             MODE.name, desc, d[i],
             ld[ld_num // 2 + l_i_s, i],
             # xd_m[xy_max[0]], xd_m[xy_max[1]]
         ) + msg)
    fig.supxlabel(HORIZONTAL_LABEL)
    fig.supylabel(VERTICAL_LABEL)

    plt.show()

    return max_val


def plot_psf_xy_reim(psf_dict: PSFDict,
                     i=0, num=None, l_i_s=0,
                     hmax: float | None = None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    ld_num = ld.shape[0]

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    if hmax is None:
        hmax = np.max(np.abs(psf[..., i, ld_num // 2 + l_i_s]))

    re = np.real(psf[
        xd_r,
        xd_r,
        i,
        ld_num // 2 + l_i_s
    ])
    im = np.imag(psf[
        xd_r,
        xd_r,
        i,
        ld_num // 2 + l_i_s
    ])
    xd_m = xd[xd_r]

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm = ax_re.pcolormesh(
        xd_m, xd_m,
        re,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm, label=OCT_PSF_REAL_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_re.set_aspect('equal')
    ax_re.set_title("real")

    pcm2 = ax_im.pcolormesh(
        xd_m, xd_m,
        im,
        vmin=-hmax, vmax=hmax,
        cmap='PiYG'
    )
    fig.colorbar(pcm2, label=OCT_PSF_IMAG_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_im.set_aspect('equal')
    ax_im.set_title("imaginary")

    fig.suptitle(
        ("{}-OCT en-face {} PSF.\n" +
         DEFOCUS + ", " +
         (LOCATION_ZRE if ZR else LOCATION_OPL)
         # "\nMax at: ({}, {}) µm."
         ).format(
             MODE.name, desc, d[i],
             ld[ld_num // 2 + l_i_s, i],
             # xd_m[xy_max[0]], xd_m[xy_max[1]]
        )
    )
    fig.supxlabel(HORIZONTAL_LABEL)
    fig.supylabel(VERTICAL_LABEL)

    plt.show()


def plot_psf_xy_3d(psf_dict: PSFDict,
                   i=0, num=None, l_i_s=0, show_FWHM=False, ref=None):

    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    ld_num = ld.shape[0]

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    data = np.abs(psf[
        xd_r,
        xd_r,
        i,
        ld_num // 2 + l_i_s
    ])
    xd_m = xd[xd_r]

    xy_max = np.unravel_index(np.argmax(data), data.shape)
    max_val = data[*xy_max]

    if np.isscalar(ref):
        msg = 'Strehl ratio: {}\n'.format(max_val / ref)
    else:
        msg = ''

    fig, ax = plt.subplots(1, 1,
                           layout='constrained',
                           subplot_kw={"projection": "3d"})

    pcm = ax.plot_surface(
        xd_m[None, :], xd_m[:, None],
        data,
        cmap='viridis',
    )
    fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    if show_FWHM:
        ax.contour(
            xd_m, xd_m,
            data,
            [np.max(data) / 2],
            colors=['red']
        )
    ax.set_title(("{}-OCT en-face {} PSF.\n" +
                  DEFOCUS + ", " +
                  (LOCATION_ZRE if ZR else LOCATION_OPL)
                 # "\nMax at: ({}, {}) µm."
                 ).format(
                    MODE.name, desc, d[i],
                    ld[ld_num // 2 + l_i_s, i],
                    # xd_m[xy_max[0]], xd_m[xy_max[1]]
                 ) + msg)

    plt.show()

    return max_val


def plot_axial_psf(psf_dict: PSFDict, i=None, NORM=True,
                   x_i: int | None = None, y_i: int | None = None,
                   log: bool = False):
    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    if x_i is None:
        x_i = xd_num // 2
    if y_i is None:
        y_i = xd_num // 2

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    def extract(i):
        x = ld[:, i]
        y = psf[
            y_i,
            x_i,
            i,
            :
        ]
        return x, y

    LOCATION = LOCATION_XY.format(
        xd[x_i], xd[y_i]
    )

    fig, (ax, ax_pha) = plt.subplots(
        2, 1,
        layout='compressed',
        sharex=True
    )

    ax.set_title("Amplitude")
    ax_pha.set_title("Phase")

    if log:
        ax.set_yscale('log')
    if i is not None:
        x, y = extract(i)

        amp = np.abs(y)
        ang = np.angle(y)

        ax.plot(x, amp / np.max(amp) if NORM else amp)
        ax_pha.plot(x, ang)

        fig.suptitle(("{}-OCT {} axial PSF.\n" + DEFOCUS + ", " + LOCATION).format(
            MODE.name, desc, d[i]
            ))
    else:
        for i in range(d.size):
            x, y = extract(i)
            amp = np.abs(y)
            ang = np.angle(y)
            ax.plot(
                x, amp / np.max(amp) if NORM else amp,
                label=DEFOCUS.format(d[i])
            )
            ax_pha.plot(
                x, ang,
                label=DEFOCUS.format(d[i])
            )
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax_pha.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.suptitle("{}-OCT {} axial PSFs.\n".format(MODE.name, desc) + LOCATION)

    ax.set_ylabel("Normalized amplitude [a.u.]" if NORM else OCT_PSF_AMP_LABEL + oct_psf_unit(NORMALIZE and MODE == IMG_MODE.PSFD))
    ax_pha.set_ylabel(OCT_PSF_PHASE_LABEL)

    if ZR:
        fig.supxlabel(RECONST_AXIAL_LABEL)
    else:
        fig.supxlabel(OPL_LABEL)

    plt.show()


def plot_psfs_xl(psf_dict: PSFDict, y_i: int | None = None):
    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    if y_i is None:
        y_i = xd_num // 2

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    fig, ax = plt.subplots(1, 1, layout='compressed')

    for i in range(d.size):
        pcm = ax.pcolormesh(
            ld[:, i], xd,
            np.abs(psf[
                y_i, :, i, :
            ]),
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT X-slice {} PSFs\n" + LOCATION_Y).format(
        MODE.name, desc, xd[y_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(HORIZONTAL_LABEL)
    ax.set_facecolor(colors[0])
    cbar = fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(False))
    cbar.set_ticks([])

    plt.show()


def plot_psfs_yl(psf_dict: PSFDict, x_i: int | None = None):
    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
        ZR = False
    except KeyError:
        ld = psf_dict['z']
        ZR = True
    xd_num = xd.size
    if x_i is None:
        x_i = xd_num // 2

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']
    NORMALIZE = psf_dict['NORMALIZE']

    fig, ax = plt.subplots(1, 1, layout='compressed')

    for i in range(d.size):
        pcm = ax.pcolormesh(
            ld[:, i], xd,
            np.abs(psf[
                :, x_i, i, :
            ]),
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT Y-slice {} PSFs\n" + LOCATION_X).format(
        MODE.name, desc, xd[x_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(VERTICAL_LABEL)
    ax.set_facecolor(colors[0])
    cbar = fig.colorbar(pcm, label=OCT_PSF_AMP_LABEL + oct_psf_unit(False))
    cbar.set_ticks([])

    plt.show()


def plot_psfs_power(psf_dict: PSFDict):
    psf = psf_dict['psf']
    xd = psf_dict['x']
    d = psf_dict['defocus']
    try:
        ld = psf_dict['opl']
    except KeyError:
        ld = psf_dict['z']

    desc = psf_dict['desc']
    MODE = psf_dict['MODE']

    powers = np.zeros(d.size, dtype=np.float32)
    for i in range(d.size):
        powers[i] = np.sum(
            np.abs(psf[
                :, :, i, :
            ]) ** 2
        ) * (xd[1] - xd[0]) ** 2 * (ld[1, i] - ld[0, i])

    plt.figure()
    plt.semilogy(d, powers, 'x')
    # plt.ylim(0, None)
    plt.title("{}-OCT {} PSF energies".format(MODE.name, desc))
    plt.xlabel("Defocus [µm]")

    plt.show()


def plot_2Dpupil(pupil: Pupil3D, σx: NDArray, σy: NDArray,
                 NAME: str = ''):
    P = pupil.pupil2D(σx, σy)

    plt.figure()
    plt.pcolormesh(σx, σy, P)
    plt.gca().set_aspect('equal')
    plt.ylabel(VERTICAL_COS_LABEL)
    plt.colorbar()
    plt.title(
        "Pupil {0:}.\n"
        "Cut-off NA: {1:}".format(
            '(' + NAME + ')' if NAME else '',
            pupil.na_co
        )
    )
    plt.xlabel(HORIZONTAL_COS_LABEL)

    plt.show()


def plot_3Dpupil(pupil: Pupil3D, σx: NDArray, σy: NDArray, NAME: str = ''):
    P = pupil.pupil2D(σx, σy)
    pupil.calc_sigma_z(σx, σy)

    σz = - pupil.σ[2]
    if pupil.reverse:
        σz = - σz
    σz[P == 0.0] = np.nan

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), layout='compressed')
    ls = LightSource(270, 45)
    rgb = ls.shade(
        P,
        cmap=plt.cm.viridis,
        vert_exag=0.1,
        blend_mode='soft'
    )
    surf = ax.plot_surface(
        σx, σy, σz,
        facecolors=rgb, rstride=1, cstride=1, linewidth=0,
        antialiased=False, shade=False, alpha=0.1
    )
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m.set_array(P)
    fig.colorbar(m, ax=ax, label='Pupil amplitude [a.u.]')

    ax.view_init(elev=60, azim=20, roll=110)

    if pupil.reverse:
        ax.set_zlim(0, np.nanmax(σz))
    else:
        ax.set_zlim(np.nanmin(σz), 0)

    ax.set_box_aspect(
        (np.ptp(σx),
         np.ptp(σy),
         -(np.nanmax(np.abs(σz)) - 0))
    )
    ax.set_xlabel(HORIZONTAL_COS_LABEL)
    ax.set_ylabel(VERTICAL_COS_LABEL)
    ax.set_zlabel("Axial directional cosine $\sigma_z$")
    ax.set_title(
        "Pupil {0:}.\n"
        "Cut-off NA: {1:}".format(
            '(' + NAME + ')' if NAME else '',
            pupil.na_co
        )
    )

    plt.show()


def plot_wavefronterror(pupil: AberratedPupil3D, σx: NDArray, σy: NDArray,
                        NAME: str = ''):
    pupil.set_wavefront_error(σx, σy)
    W_ill = pupil.we

    plt.figure()
    plt.pcolormesh(
        σx, σy,
        W_ill,
    )
    plt.gca().set_aspect('equal')
    plt.ylabel(VERTICAL_COS_LABEL)
    plt.colorbar(label='Error [µm]')
    plt.title("Wavefront error of the {0:} pupil.".format(NAME))
    plt.xlabel(HORIZONTAL_COS_LABEL)

    plt.show()


def plot_ctf_xz_reim(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                     νy_i: int, λ: float,
                     NORMALIZE: bool, img_mode: IMG_MODE,
                     νz12: List | None = None,
                     Hmax: float | None = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm_re = ax_re.pcolormesh(
        νz, νx,
        (H.real[νy_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    if νz12 is not None:
        ax_re.set_xlim(νz12[0], νz12[1])
    ax_re.set_aspect('equal')
    fig.colorbar(pcm_re, label=CTF_REAL_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    ax_re.set_title("Real part")

    pcm_im = ax_im.pcolormesh(
        νz, νx,
        (H.imag[νy_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    if νz12 is not None:
        ax_im.set_xlim(νz12[0], νz12[1])
    ax_im.set_aspect('equal')
    fig.colorbar(pcm_im, label=CTF_IMAG_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    ax_im.set_title("Imaginary part")

    fig.supxlabel(AXIAL_FREQ_LABEL)
    fig.supylabel(HORIZONTAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i]
        )
    )

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_ctf_xz_amppha(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                       νy_i: int, λ: float,
                       NORMALIZE: bool, img_mode: IMG_MODE,
                       νz12: List | None = None,
                       Hmax: float | None = None,
                       log: bool = False):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    fig, (ax_amp, ax_pha) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True,
    )

    if log:
        norm = LogNorm(vmax=Hmax)
    else:
        norm = Normalize(vmin=0, vmax=Hmax)

    pcm_amp = ax_amp.pcolormesh(
        νz, νx,
        np.abs(H[νy_i]),
        norm=norm,
    )
    ax_amp.set_aspect('equal')
    fig.colorbar(pcm_amp, label=CTF_AMP_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    if νz12 is not None:
        ax_amp.set_xlim(νz12[0], νz12[1])
    ax_amp.set_title("Magnitude")

    pcm_pha = ax_pha.pcolormesh(
        νz, νx,
        np.angle(H[νy_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    ax_pha.set_aspect('equal')
    fig.colorbar(pcm_pha, label=CTF_PHASE_LABEL)
    if νz12 is not None:
        ax_pha.set_xlim(νz12[0], νz12[1])
    ax_pha.set_title("Phase")

    fig.supxlabel(AXIAL_FREQ_LABEL)
    fig.supylabel(HORIZONTAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i]
        )
    )

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_ctf_yz_reim(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                     νx_i: int, λ: float,
                     NORMALIZE: bool, img_mode: IMG_MODE,
                     νz12: List | None = None,
                     Hmax: float | None = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True,
        subplot_kw=dict(box_aspect=1)
    )

    pcm_re = ax_re.pcolormesh(
        νz, νy,
        (H.real[:, νx_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    if νz12 is not None:
        ax_re.set_xlim(νz12[0], νz12[1])
    ax_re.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm_re, label=CTF_REAL_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    ax_re.set_title("Real part")

    pcm_im = ax_im.pcolormesh(
        νz, νy,
        (H.imag[:, νx_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    if νz12 is not None:
        ax_im.set_xlim(νz12[0], νz12[1])
    ax_im.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm_im, label=CTF_IMAG_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    ax_im.set_title("Imaginary part")

    fig.supxlabel(AXIAL_FREQ_LABEL)
    fig.supylabel(VERTICAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_ctf_yz_amppha(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                       νx_i: int, λ: float,
                       NORMALIZE: bool, img_mode: IMG_MODE,
                       νz12: List | None = None,
                       Hmax: float | None = None,
                       log: bool = False):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    fig, (ax_amp, ax_pha) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    if log:
        norm = LogNorm(vmax=Hmax)
    else:
        norm = Normalize(vmin=0, vmax=Hmax)

    pcm_amp = ax_amp.pcolormesh(
        νz, νy,
        np.abs(H[:, νx_i]),
        norm=norm,
    )
    ax_amp.set_aspect('equal')
    fig.colorbar(pcm_amp, label=CTF_AMP_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    if νz12 is not None:
        ax_amp.set_xlim(νz12[0], νz12[1])
    ax_amp.set_title("Magnitude")

    pcm_pha = ax_pha.pcolormesh(
        νz, νy,
        np.angle(H[:, νx_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    ax_pha.set_aspect('equal')
    fig.colorbar(pcm_pha, label=CTF_PHASE_LABEL)
    if νz12 is not None:
        ax_pha.xlim(νz12[0], νz12[1])
    ax_pha.set_title("Phase")

    fig.supxlabel(AXIAL_FREQ_LABEL)
    fig.supylabel(VERTICAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)

    plt.show()


def plot_ctf_xy_reim(H: NDArray, νx: NDArray, νy: NDArray,
                     νz: NDArray,
                     νz_i: int, λ: float,
                     NORMALIZE: bool, img_mode: IMG_MODE,
                     Hmax: float | None = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H[..., νz_i]))

    fig, (ax_re, ax_im) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    pcm_re = ax_re.pcolormesh(
        νx, νy,
        (H.real[..., νz_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    ax_re.set_aspect('equal')
    ax_re.set_title("Real part")
    fig.colorbar(pcm_re, label=CTF_REAL_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))

    pcm_im = ax_im.pcolormesh(
        νx, νy,
        (H.imag[..., νz_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    ax_im.set_aspect('equal')
    ax_im.set_title("Imaginary part")
    fig.colorbar(pcm_im, label=CTF_IMAG_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    fig.supxlabel(HORIZONTAL_FREQ_LABEL)
    fig.supylabel(VERTICAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )

    plt.show()


def plot_ctf_xy_amppha(H: NDArray, νx: NDArray, νy: NDArray,
                       νz: NDArray,
                       νz_i: int, λ: float,
                       NORMALIZE: bool, img_mode: IMG_MODE,
                       Hmax: float | None = None,
                       log: bool = False):

    if Hmax is None:
        Hmax = np.max(np.abs(H[..., νz_i]))

    fig, (ax_amp, ax_pha) = plt.subplots(
        1, 2,
        layout='compressed',
        sharex=True, sharey=True
    )

    if log:
        norm = LogNorm(vmax=Hmax)
    else:
        norm = Normalize(vmin=0, vmax=Hmax)

    pcm_amp = ax_amp.pcolormesh(
        νx, νy,
        np.abs(H[..., νz_i]),
        norm=norm,
    )
    ax_amp.set_aspect('equal')
    fig.colorbar(pcm_amp, label=CTF_AMP_LABEL + ctf_unit(NORMALIZE and img_mode == IMG_MODE.PSFD))
    ax_amp.set_title("Magnitude")

    pcm_pha = ax_pha.pcolormesh(
        νx, νy,
        np.angle(H[..., νz_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='twilight'
    )
    ax_pha.set_aspect('equal')
    fig.colorbar(pcm_pha, label=CTF_PHASE_LABEL)
    ax_pha.set_title("Phase")

    fig.supxlabel(HORIZONTAL_FREQ_LABEL)
    fig.supylabel(VERTICAL_FREQ_LABEL)
    fig.suptitle(
        ("Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )

    plt.show()
