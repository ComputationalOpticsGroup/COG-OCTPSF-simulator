import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from typing import TypedDict, List
from .pupils import Pupil3D, AberratedPupil3D
from numpy.typing import NDArray

colors = plt.colormaps['viridis'].colors

display_gamma = 1.0

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
CTF_AMP_LABEL = r"CTF amplitude $|H|$ [µm]"
CTF_REAL_LABEL = r"CTF real part Re[$H$] [µm]"
CTF_IMAG_LABEL = r"CTF imaginary part Im[$H$] [µm]"

PSF_AMP_LABEL = r"PSF amplitude $|h|$ [µm$^{{-2}}$]"

DEFOCUS = r"Defocus: {} µm"

LOCATION_XY = r"($x_0$, $y_0$) = ({} µm, {} µm)"
LOCATION_X = r"$x_0$ = {} µm"
LOCATION_Y = r"$y_0$ = {} µm"
LOCATION_OPL = r"$l$ = {} µm"
LOCATION_ZRE = r"$z_\mathrm{{re}}$ = {} µm"
LOCATION_Z = r"$z_0$ = {} µm"

WAVELENGTH = r"Wavelength: {:.3f} nm"
HORIZONTAL_FREQ = r"Horizontal frequency: {:.3f} µm$^{{-1}}$"
VERTICAL_FREQ = r"Vertical frequency: {:.3f} µm$^{{-1}}$"
AXIAL_FREQ = r"Axial frequency: {:.3f} µm$^{{-1}}$"


class PSFDict(TypedDict):
    psf: NDArray
    x: NDArray
    defocus: NDArray
    opl: NDArray
    desc: str
    MODE: str


def plot_psf_xl(psf_dict: PSFDict, i=0, num=None, show_FWHM=True,
                log=False, y_i: int = None):

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

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    data = np.abs(psf[
        y_i,
        xd_r,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, ax = plt.subplots(1, 1, layout='constrained')

    pcm = ax.pcolormesh(
        ld_m,
        xd_m,
        np.log10(data) if log else data ** display_gamma,
    )
    fig.colorbar(pcm)
    if show_FWHM:
        ax.contour(
            ld_m,
            xd_m,
            data,
            [np.max(data) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT X-slice {} PSF.\n" +
                  DEFOCUS + ", " + LOCATION_Y).format(
                     MODE, desc, d[i], xd[y_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(HORIZONTAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)


def plot_psf_yl(psf_dict: PSFDict, i=0, num=None, show_FWHM=True,
                log=False, x_i: int = None):

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

    if num is None:
        xd_r = slice(None)
    else:
        xd_r = slice(xd_num // 2 - num, xd_num // 2 + num)

    data = np.abs(psf[
        xd_r,
        x_i,
        i,
        :
    ])
    ld_m = ld[:, i]
    xd_m = xd[xd_r]

    fig, ax = plt.subplots(1, 1, layout='constrained')

    pcm = ax.pcolormesh(
        ld_m,
        xd_m,
        np.log10(data) if log else data ** display_gamma,
    )
    fig.colorbar(pcm)
    if show_FWHM:
        ax.contour(
            ld_m,
            xd_m,
            data,
            [np.max(data) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT Y-slice {} PSF.\n" +
                  DEFOCUS + ", " + LOCATION_X).format(
                     MODE, desc, d[i], xd[x_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(VERTICAL_LABEL)

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)


def plot_psf_xy(psf_dict: PSFDict, i=0, num=None, l_i_s=0, show_FWHM=True, ref=None,
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

    fig, ax = plt.subplots(1, 1, layout='constrained')

    pcm = ax.pcolormesh(
        xd_m, xd_m,
        np.log10(data) if log else data ** display_gamma,
    )
    fig.colorbar(pcm)
    if show_FWHM:
        ax.contour(
            xd_m, xd_m,
            data,
            [np.max(data) / 2],
            colors=['red']
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT en-face {} PSF.\n" +
                  DEFOCUS + ", " +
                  (LOCATION_ZRE if ZR else LOCATION_OPL)
                 # "\nMax at: ({}, {}) µm."
                 ).format(
                     MODE, desc, d[i],
                     ld[ld_num // 2 + l_i_s, i],
                     # xd_m[xy_max[0]], xd_m[xy_max[1]]
                 ) + msg)

    return max_val


def plot_psf_xy_3d(psf_dict: PSFDict, i=0, num=None, l_i_s=0, show_FWHM=False, ref=None,
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
        np.log10(data) if log else data ** display_gamma,
        cmap='viridis',
    )
    fig.colorbar(pcm)
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
                    MODE, desc, d[i],
                    ld[ld_num // 2 + l_i_s, i],
                    # xd_m[xy_max[0]], xd_m[xy_max[1]]
                 ) + msg)

    return max_val


def plot_axial_psf(psf_dict: PSFDict, i=None, NORM=True,
                   x_i: int = None, y_i: int = None):
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

    def extract(i):
        x = ld[:, i]
        y = np.abs(psf[
            y_i,
            x_i,
            i,
            :
        ])
        return x, y

    LOCATION = LOCATION_XY.format(
        xd[x_i], xd[y_i]
    )

    if i is not None:
        x, y = extract(i)
        plt.plot(
            x, y / np.max(y) if NORM else y,
        )
        plt.title(("{}-OCT {} axial PSF.\n" + DEFOCUS + LOCATION).format(
            MODE, desc, d[i]
            ))
    else:
        for i in range(d.size):
            x, y = extract(i)
            plt.plot(
                x, y / np.max(y) if NORM else y,
                label=DEFOCUS.format(d[i])
            )
        plt.title("{}-OCT {} axial PSFs.\n".format(MODE, desc) + LOCATION)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if ZR:
        plt.xlabel(RECONST_AXIAL_LABEL)
    else:
        plt.xlabel(OPL_LABEL)
    plt.ylabel("Normalized amplitude [a.u.]" if NORM else "Amplitude [a.u.]")


def plot_psfs_xl(psf_dict: PSFDict, y_i: int = None):
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

    fig, ax = plt.subplots(1, 1, layout='constrained')

    for i in range(d.size):
        pcm = ax.pcolormesh(
            ld[:, i], xd,
            np.abs(psf[
                y_i, :, i, :
            ]),
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT X-slice {} PSFs\n" + LOCATION_Y).format(
        MODE, desc, xd[y_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(HORIZONTAL_LABEL)
    ax.set_facecolor(colors[0])
    cbar = fig.colorbar(pcm)
    cbar.set_ticks([])

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)


def plot_psfs_yl(psf_dict: PSFDict, x_i: int = None):
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

    fig, ax = plt.subplots(1, 1, layout='constrained')

    for i in range(d.size):
        pcm = ax.pcolormesh(
            ld[:, i], xd,
            np.abs(psf[
                :, x_i, i, :
            ]),
        )
    ax.set_aspect('equal')
    ax.set_title(("{}-OCT Y-slice {} PSFs\n" + LOCATION_X).format(
        MODE, desc, xd[x_i]))
    if ZR:
        ax.set_xlabel(RECONST_AXIAL_LABEL)
    else:
        ax.set_xlabel(OPL_LABEL)
    ax.set_ylabel(VERTICAL_LABEL)
    ax.set_facecolor(colors[0])
    cbar = fig.colorbar(pcm)
    cbar.set_ticks([])

    fig.draw_without_rendering()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.set_size_inches(tb.width, tb.height)


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
        ) * (xd[1] - xd[0]) ** 2 * (ld[1, 0] - ld[0, 0])

    plt.semilogy(d, powers, 'x')
    # plt.ylim(0, None)
    plt.title("{}-OCT {} PSF intensities".format(MODE, desc))
    plt.xlabel("Defocus [µm]")


def plot_2Dpupil(pupil: Pupil3D, σx: NDArray, σy: NDArray,
                 NAME: str = ''):
    P = pupil.pupil2D(σx, σy)

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


def plot_3Dpupil(pupil: Pupil3D, σx: NDArray, σy: NDArray, NAME: str = ''):
    P = pupil.pupil2D(σx, σy)
    pupil.calc_sigma_z(σx, σy)

    # σz = - np.sqrt(1 - σx ** 2 - σy ** 2)
    σz = - pupil.σ[2]
    σz[P == 0.0] = np.nan

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
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

    ax.view_init(elev=-15, azim=0, roll=90)

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

    fig.show()


def plot_wavefronterror(pupil: AberratedPupil3D, σx: NDArray, σy: NDArray,
                        NAME: str = ''):
    pupil.set_wavefront_error(σx, σy)
    W_ill = pupil.we

    plt.pcolormesh(
        σx, σy,
        W_ill,
    )
    plt.gca().set_aspect('equal')
    plt.ylabel(VERTICAL_COS_LABEL)
    plt.colorbar(label='Error [µm]')
    plt.title("Wavefront error of the {0:} pupil.".format(NAME))
    plt.xlabel(HORIZONTAL_COS_LABEL)


def plot_ctf_xz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                νy_i: int, λ: float, νz12: List = None,
                Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    plt.pcolormesh(
        νz, νx,
        (H.real[νy_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.xlim(νz12[0], νz12[1])
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_REAL_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(HORIZONTAL_FREQ_LABEL)
    plt.title(
        ("Real part of Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i])
    )
    plt.show()

    plt.pcolormesh(
        νz, νx,
        (H.imag[νy_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.xlim(νz12[0], νz12[1])
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_IMAG_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(HORIZONTAL_FREQ_LABEL)
    plt.title(
        ("Imaginary part of Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i]
        )
    )
    plt.show()


def plot_ctf_amp_xz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                    νy_i: int, λ: float, Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    plt.pcolormesh(
        νz, νx,
        np.abs(H[νy_i]),
        vmin=0, vmax=Hmax,
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_AMP_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(HORIZONTAL_FREQ_LABEL)
    plt.title(
        ("Magnitude of Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i]
        )
    )


def plot_ctf_phase_xz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                      νy_i: int, λ: float, νz12: List = None):

    plt.pcolormesh(
        νz, νx,
        np.angle(H[νy_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='hsv'
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_PHASE_LABEL)
    if νz12 is not None:
        plt.xlim(νz12[0], νz12[1])
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(HORIZONTAL_FREQ_LABEL)
    plt.title(
        ("Phase of Coherent transfer function\n" +
         WAVELENGTH + ", " + VERTICAL_FREQ).format(
            λ, νy[νy_i]
        )
    )


def plot_ctf_yz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                νx_i: int, λ: float, νz12: List = None,
                Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    plt.pcolormesh(
        νz, νy,
        (H.real[:, νx_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.xlim(νz12[0], νz12[1])
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_REAL_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Real part of Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )
    plt.show()

    plt.pcolormesh(
        νz, νy,
        (H.imag[:, νx_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.xlim(νz12[0], νz12[1])
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_IMAG_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Imaginary part of Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )
    plt.show()


def plot_ctf_amp_yz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                    νx_i: int, λ: float, Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H))

    plt.pcolormesh(
        νz, νy,
        np.abs(H[:, νx_i]),
        vmin=0, vmax=Hmax,
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_AMP_LABEL)
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Magnitude of Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )


def plot_ctf_phase_yz(H: NDArray, νx: NDArray, νy: NDArray, νz: NDArray,
                      νx_i: int, λ: float, νz12: List = None):

    plt.pcolormesh(
        νz, νy,
        np.angle(H[:, νx_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='hsv'
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_PHASE_LABEL)
    if νz12 is not None:
        plt.xlim(νz12[0], νz12[1])
    plt.xlabel(AXIAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Phase of Coherent transfer function\n" +
         WAVELENGTH + ", " + HORIZONTAL_FREQ).format(
            λ, νx[νx_i]
        )
    )


def plot_ctf_xy(H: NDArray, νx: NDArray, νy: NDArray,
                νz: NDArray,
                νz_i: int, λ: float,
                Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H[..., νz_i]))

    plt.pcolormesh(
        νx, νy,
        (H.real[..., νz_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_REAL_LABEL)
    plt.xlabel(HORIZONTAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Real part of Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )
    plt.show()

    plt.pcolormesh(
        νx, νy,
        (H.imag[..., νz_i]),
        vmin=-Hmax, vmax=Hmax,
        cmap='PiYG'
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_IMAG_LABEL)
    plt.xlabel(HORIZONTAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Imaginary part of Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )
    plt.show()


def plot_ctf_amp_xy(H: NDArray, νx: NDArray, νy: NDArray,
                    νz: NDArray,
                    νz_i: int, λ: float, Hmax: float = None):

    if Hmax is None:
        Hmax = np.max(np.abs(H[..., νz_i]))

    plt.pcolormesh(
        νx, νy,
        np.abs(H[..., νz_i]),
        vmin=0, vmax=Hmax,
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_AMP_LABEL)
    plt.xlabel(HORIZONTAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Magnitude of Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )


def plot_ctf_phase_xy(H: NDArray, νx: NDArray, νy: NDArray,
                      νz: NDArray,
                      νz_i: int, λ: float):

    plt.pcolormesh(
        νx, νy,
        np.angle(H[..., νz_i]),
        vmin=-np.pi, vmax=np.pi,
        cmap='hsv'
    )
    plt.gca().set_aspect('equal')
    plt.colorbar(label=CTF_PHASE_LABEL)
    plt.xlabel(HORIZONTAL_FREQ_LABEL)
    plt.ylabel(VERTICAL_FREQ_LABEL)
    plt.title(
        ("Phase of Coherent transfer function\n" +
         WAVELENGTH + ", " + AXIAL_FREQ).format(
            λ, νz[νz_i]
        )
    )
