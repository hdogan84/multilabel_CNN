import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_color_spectorgram(
    I,
    times,
    sfrqs,
    brightness=1.5,
    contrast=1.2,
    pltw=5,
    plth=3.0,
    block=False,
    axs=None,
):
    # plot colorspect

    lbmin = 0.5 - brightness
    lbmax = lbmin + 1.6 / contrast
    L = (np.log(I) - lbmin) / (lbmax - lbmin) + 1.0
    L = np.minimum(np.maximum(L, 0), 1)
    # cfg.axs[0].cla()
    if axs is None:
        figs, axs = plt.subplots(1, 1, figsize=(pltw, plth))
    axs.imshow(
        L,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], sfrqs[0], sfrqs[-1]),
    )
    plt.show(block=block)

    return axs, lbmin, lbmax


def plot_clusters(B, S, nplot, times, sfrqs, wts_out, direction,block=False):
    npl = np.minimum(nplot, len(S))
    pltw = 6
    plth = 0.9 * npl
    figs, axs = plt.subplots(npl, 1, figsize=(pltw, plth))
    # plot the spectrograms
    n, nseg = B[0].shape
    for i in range(npl):
        # plot colorspect
        # plt.get_current_fig_manager().window.setGeometry(60+i*20,40+i*40,1000,300)
        plt.sca(axs[i])
        plt.rcParams.update({"font.size": 8})
        plt.rcParams.update({"xtick.labelsize": 6})
        plt.rcParams.update({"ytick.labelsize": 6})
        plt.cla()
        axs[i].imshow(
            np.power(S[i], 0.5),
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], sfrqs[0], sfrqs[-1]),
        )
        # axs[i].title.set_text('Clus '+str(i+1)+', Dir='+str(cfg.direction[i])+' deg, wt=' +str(np.round(cfg.wts_out[i]*100))+'%' ,size=6)
        axs[i].set_title(
            "Clus "
            + str(i + 1)
            + ", Dir="
            + str(direction[i])
            + " deg, wt="
            + str(np.round(wts_out[i] * 100))
            + "%",
            size=6,
        )
    plt.show(block=block)
