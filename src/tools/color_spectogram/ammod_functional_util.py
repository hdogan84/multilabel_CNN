import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import scipy.io.wavfile
import scipy.io as scio
import librosa
import soundfile
from pathlib import Path
from skimage import color
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb


def read_audio_segment(
    filepath: str, start: int, stop: int, sample_frequency: int,
):
    audio_data = []
    filepath = Path(filepath)
    if filepath.exists() == False:
        raise Exception("File does not exsts")

    audio_data, sr = librosa.load(
        filepath, offset=start, duration=stop - start, sr=sample_frequency, mono=False
    )

    return audio_data


def write_time_series_to_file(time_series, sample_frequency):
    for index, y in enumerate(time_series, start=1):
        soundfile.write("cluster-{}.wav".format(index), y, sample_frequency)


def calc_color_spectogram(
    data,
    sample_frequency=0,
    channels=0,
    fmin=1200.0,
    angular_resolution=None,  # former ang_res in (degrees)
    start_time=None,  # former tstart
    end_time=None,  # former tend
    mic_distance=None,  # former d
    mic_c=None,  # former c
    spectogram_nperseg=None,  # former N
    cstyp=0,
):

    N = spectogram_nperseg
    fs = sample_frequency

    # print("Computing CSPEC with c=", mic_c, "d=", mic_distance)

    spec_avg = False
    shift = N // 3
    d = mic_distance

    # microphone positions (m) and pointing angles (radians)
    pi = np.pi
    if channels == 1:
        pos = (0, 0)
        pos = np.asarray(pos).reshape((channels, 2))
        am = np.asarray((0,))
    elif channels == 4:
        pos = (-d / 2, d / 2, d / 2, d / 2, d / 2, -d / 2, -d / 2, -d / 2)
        pos = np.asarray(pos).reshape((channels, 2))
        am = np.asarray((-pi / 4, pi / 4, 3 * pi / 4, -3 * pi / 4))
    else:
        pos = (d / 2, 0, -d / 2, 0)
        pos = np.asarray(pos).reshape((channels, 2))
        am = np.asarray((-pi / 2, pi / 2))

    if channels == 1:
        # beamformer beam angles
        angs = np.asarray(range(0, 360, angular_resolution)) * (pi / 180)
        maxang = 360
    elif channels == 4:
        # beamformer beam angles
        angs = np.asarray(range(0, 360, angular_resolution)) * (pi / 180)
        maxang = 360
    else:
        angs = np.asarray(range(0, 180, angular_resolution)) * (pi / 180)
        maxang = 180

    na = len(angs)

    #
    # dst(i,j) is distance from microphone i to a distant point at angle j
    #  dctr is distance to center of array
    dst = np.zeros((channels, na))
    r = 1000  # target at this range
    for ia in range(na):
        ry = np.cos(angs[ia]) * r
        rx = np.sin(angs[ia]) * r
        dctr = np.sqrt(np.square(rx) + np.square(ry))
        for i in range(channels):
            dst[i, ia] = (
                np.sqrt(np.square(rx - pos[i, 0]) + np.square(ry - pos[i, 1])) - dctr
            )

    # create hsv color map of size equal to number of angles
    mp = matplotlib.cm.get_cmap("hsv")
    co = np.zeros((3, na))
    for ia in range(na):
        ct = mp(1.0 * ia / na)
        co[:, ia] = ct[0:3]

    # high-pass filtering
    Bf, Af = scipy.signal.butter(2, fmin / fs * 2, btype="high")
    w, h = frsp = scipy.signal.freqz(Bf, Af, worN=N, whole=True, plot=None)
    frsp = np.square(np.abs(h))

    # set up steering vectors
    n = N // 2 + 1
    sfrqs = np.asarray(range(n)) / N * fs
    s = [None] * channels
    for i in range(channels):
        # ft = (f)requency times delay (t)time
        ft = np.outer(sfrqs, dst[i, :] / mic_c)
        if channels == 4:
            # microphone cardioid pattern
            mr = (1 + np.cos(angs - am[i])) / 2
            # mr=np.exp( -np.power((angs-am[i])/(np.pi),4))
        else:
            # omnidirec
            mr = np.ones((na,))
        # steering vector
        s[i] = np.exp(1j * 2 * pi * ft) * mr

    B = [None] * channels
    T = start_time - end_time

    bavg = 0

    for ch in range(channels):
        x = data[ch]
        f, t, B[ch] = scipy.signal.spectrogram(
            x,
            fs=fs,
            window="hanning",
            nperseg=N,
            noverlap=N - shift,
            detrend=False,
            scaling="spectrum",
            axis=-1,
            mode="complex",
        )
        B[ch] = B[ch] / 2  # divide by 2 to have same scaling as MATLAB

        # implement filtering

        B[ch] = B[ch] * np.tile(frsp[0:n].reshape((n, 1)), (1, B[ch].shape[1]))

        if spec_avg:
            b2 = np.square(np.real(B[ch])) + np.square(np.real(B[ch]))
            b2 = np.mean(b2, axis=1)
            bavg = bavg + b2 / channels
        else:
            B[ch] *= N

    [n, NSEG] = B[0].shape
    # print('n=',n,'nt=',nt)
    times = start_time + (N / 2 + np.asarray(range(NSEG)) * shift) / fs

    if spec_avg:
        bavg = np.sqrt(bavg) * 10
        for ch in range(channels):
            B[ch] = (B[ch].T / bavg).T

    # compute colorspect
    I = get_cspect(B, s, co, cstyp)
    I = np.maximum(I, 1e-6)

    return I, B, times, sfrqs


def get_cspect(B, s, co, cstyp):
    M = len(s)
    [nc, na] = co.shape
    [n, NSEG] = B[0].shape
    I = np.zeros((n, NSEG, 3))
    for ismp in range(NSEG):
        xbf = 0
        for i in range(M):
            bt = B[i][:, ismp]
            xbf = xbf + bt * s[i].T
        b = np.abs(xbf)
        # fast way to compute b.^bpwr
        bp = np.square(b)
        bpwr = 2
        for i in range(3):
            bp = np.square(bp)
            bpwr = bpwr * 2

        # amplitude averaged over direction
        bs = np.power(np.mean(bp, axis=0), 1 / bpwr)

        # normalized beam pattern to compute color
        bps = np.sum(bp, axis=0)
        iz = np.where(bps == 0)[0]
        bps[iz] = 1
        bn = bp / bps
        if cstyp == 0:
            # compute color
            bc = np.dot(bn.T, co.T)
            I[:, ismp, :] = (bc.T * bs).T
        else:
            ic = np.argmax(bn, axis=0)
            ht = (ic / (na - 1)).reshape((n, 1))
            st = np.ones((n, 1))
            vt = bs.reshape((n, 1))
            hsv = np.concatenate([ht, st, vt], axis=1)
            rgb = hsv2rgb(hsv.reshape((n, 1, 3)))
            I[:, ismp, :] = rgb.reshape((n, 3))

    return I


# clustering section
def cluster_dist(cmean, cvar, gwts):
    [dim, P, M] = cmean.shape
    Cd = np.zeros((M, M))
    for i in range(M):
        cn = cmean[:, :, i].reshape((dim, P)).T  # shape=(P,dim)
        for j in range(M):
            # measure how well cluster i's mean values fit to cluster j
            Lp = cluster_eval(cn, cmean, cvar, j)
            # lm=max(Lp,[],2);
            lm = np.amax(Lp, axis=1).reshape((P,))
            p = np.dot(gwts[j, :].reshape((1, P)), np.exp(Lp.T - lm))
            lp = np.log(p.reshape((P,))) + lm
            lm = np.amax(lp)
            p = np.sum(np.exp(lp - lm) * gwts[i, :].reshape((P,)))
            Cd[i, j] = np.log(p) + lm
    for i in range(M):
        Cd[i, :] = Cd[i, :] - Cd[i, i]
        return Cd


def rgb2feat(Cn):
    n, m = Cn.shape
    out = color.rgb2hsv(Cn.reshape((n, 1, m)))
    out = out[:, :, 0].reshape((n, 1))
    return out


def mixture_pdf(L, wts):
    [n, m] = L.shape
    ii = np.where(wts <= 0)[0]
    if len(ii) > 0:
        wts[ii] = 1
    lw = np.log(wts)
    if len(ii) > 0:
        lw[ii] = -np.inf
    w = L + lw
    mx = np.amax(w, axis=1)
    p = np.exp(w.T - mx).T
    nor = np.sum(p, axis=1)
    W = (p.T / nor).T
    lp = np.log(nor) + mx
    return lp, W


def cluster_eval(Cn, cmean, cvar, i):
    [dim, P, M] = cmean.shape
    [nc, dim2] = Cn.shape
    if dim != dim2:
        print("dim ~= dim2")
        quit()
    Lp = np.zeros((nc, P))
    for p in range(P):
        mu = cmean[:, p, i]
        t = Cn - np.tile(mu.reshape((1, dim)), (nc, 1))
        t = np.fmod(t + 1.5, 1) - 0.5
        R = cvar[:, p, i].reshape((dim, dim))
        lp = -0.5 * dim * np.log(2 * np.pi) - 0.5 * np.square(t) / R - 0.5 * np.log(R)
        Lp[:, p] = lp.reshape((nc,))
    return Lp


def cluster_color_spectorgram(
    I,  # in this function here name C
    thr,
    nclus,  # in this function here name M
    iplt,
    times,
    sfrqs,
    expfac=0.5,  # for dwts (make larger to give more weight to higher signal strength),
    lfac=1.0,  # (make larger to give more weight to color (direction) )
    dfac=30.0,  # 1 or 2. make larger to  give more weight to spatial effects)
    minwt=1e-3,  # mixture weight assigned to noise
    radius=6,  # search radius for neighborhoods (pixels)
    P=2,  # mixture components per cluster
    nit_gmm=90,  # number of GMM iterations
    nit_spatial=10,  # number of spatial iterations
    wts_exp=0.01,  # if greater than zero, makes weak clusters disappear
    minvar=0.00015,
    merge_thresh=-0.2,
    nplot=5,
):
    C = I
    M = nclus
    nhbd_type = (2,)  # 1 = r^2/d^2   2 = exp( - d^2/r^2  )
    use_kmeans = (True,)
    # parameters, same as in cluster_C.m
    # expfac = expfac
    # lfac = lfac
    # dfac = dfac
    # nplot = cfg.nplot  # 5          # how many of the largest clusters to display
    # wts_exp = wts_exp

    [n, nseg, dim] = C.shape

    # reshape as a matrix
    C = C.reshape((n * nseg, dim))

    # compute amplitude
    Cs = np.sqrt(np.sum(np.square(C), axis=1))

    # determine location indices that can be used to index into C
    ipos = np.tile(np.arange(nseg).reshape((1, nseg)), [n, 1]).reshape((n * nseg,))
    jpos = np.tile(np.arange(n).reshape((n, 1)), [1, nseg]).reshape((n * nseg,))
    ijpos = jpos * nseg + ipos
    if False:
        outfile = "Cn.mat"
        mdict = {"ipos": ipos, "jpos": jpos}
        print("saving Cn ")
        scio.savemat(
            outfile,
            mdict,
            appendmat=True,
            format="5",
            long_field_names=False,
            do_compression=False,
            oned_as="row",
        )
        quit()

    # create normalized colors for clustering
    cs = np.sum(C, axis=1)
    ct = np.isinf(cs) | np.isnan(cs)
    ii = np.where(ct)[0]
    C[ii, :] = 1e-3
    cs[ii] = 1
    Cn = C / np.tile(cs.reshape((n * nseg, 1)), (1, 3))
    Cn = rgb2feat(Cn)
    dim = Cn.shape[1]

    if thr > 0.0:
        # thresholding
        ii = np.where(Cs > thr)[0]
        Cs = Cs[ii]
        C = C[ii, :]
        Cn = Cn[ii, :]
        ipos = ipos[ii]
        jpos = jpos[ii]
        ijpos = ijpos[ii]
        nc = len(ii)
    else:
        nc = n * nseg

    # print("Got %d points after threshold" % nc)

    # When computing GMM clustering, the data weights 'dwts'
    # are the effective number of data samples represented by a single pixel.
    # this should be higher for higher amplitude, but apparently
    # weighting by 'Cs' is too much, so I use Cs.^.25.
    dwts = np.power(Cs, expfac)
    if False:
        outfile = "Cn.mat"
        mdict = {"Cn": Cn, "dwts": dwts, "Cs": Cs}
        print("saving Cn , dwts, Cs")
        scio.savemat(
            outfile,
            mdict,
            appendmat=True,
            format="5",
            long_field_names=False,
            do_compression=False,
            oned_as="row",
        )
        quit()

    # initial k-means clustering of colors
    # print("Initial Clustering...")
    if use_kmeans:

        kmeans = KMeans(n_clusters=nclus).fit(Cn)
        cluster_labels = kmeans.labels_
    else:
        cluster_labels = np.floor(np.random.uniform(low=0.0, high=M, size=(nc,)))

    # initialize Gaussian mixture from clusters
    cmean = np.zeros((dim, P, M,))
    cvar = np.zeros((dim * dim, P, M))
    gwts = np.ones((M, P,)) / P
    for i in range(M):
        ii = np.where(cluster_labels == i)[0]
        ni = len(ii)
        if ni == 0:
            continue
        mu = np.mean(C[ii, :], axis=0)
        mu = rgb2feat(mu.reshape((1, 3)))
        s = 0.05
        for p in range(P):
            cmean[:, p, i] = mu + s * (P - 1) / 2 - (p - 1) * s * (P - 1) * 2 / P
            t = Cn[ii, :] - cmean[:, p, i]
            t = np.fmod(t + 1.5, 1) - 0.5
            # R=np.dot(t.T,t)/ni
            R = np.eye(dim) * np.square(s)
            cvar[:, p, i] = R.reshape((dim * dim,))

    # print("Solving for spatial indexes...")
    D_i = [None] * nc
    D_w = [None] * nc
    r2 = np.square(radius)
    if nhbd_type == 1:
        for i in range(nc):
            d2 = np.square(ipos[i] - ipos * 1.0) + np.square(jpos[i] - jpos * 1.0)
            D_i[i] = np.where((d2 <= 2.0 * r2) & (d2 > 0))[0]
            D_w[i] = r2 / d2[D_i[i]]
    else:
        for i in range(nc):
            d2 = np.square(ipos[i] - ipos) + np.square(jpos[i] - jpos)
            D_i[i] = np.where((d2 <= 10 * r2) & (d2 > 0))[0]
            D_w[i] = np.exp(-d2[D_i[i]] / r2)

    if False:
        H = np.zeros((n, nseg)).reshape((n * nseg,))
        for ip in range(7):
            i = (ip + 1) * 2200
            ii = D_i[i]
            H[ijpos[ii]] = D_w[i]
        H = H.reshape((n, nseg))
        plt.cla()
        axs.imshow(H, origin="lower", aspect="auto", cmap="hot")
        plt.show(block=True)
        plt.pause(0.1)

    # GMM estimation iterations
    ql = 0
    wts = np.ones((M + 1,)) / (M + 1)
    W = np.ones((nc, M + 1)) / (M + 1)
    # figs, axs = plt.subplots(1, 1, figsize=(5, 3))
    for iter in range(nit_gmm + nit_spatial):

        if iter < nit_gmm:
            Lcp = np.zeros((nc, P, M))
            for i in range(M):
                Lcp[:, :, i] = cluster_eval(Cn, cmean, cvar, i)

            if False:
                outfile = "Cn.mat"
                mdict = {"Lcp": Lcp, "Cn": Cn, "dwts": dwts, "Cs": Cs}
                print("saving Cn , dwts, Cs")
                scio.savemat(
                    outfile,
                    mdict,
                    appendmat=True,
                    format="5",
                    long_field_names=False,
                    do_compression=False,
                    oned_as="row",
                )
                quit()
            wts2 = wts[0:M] / np.sum(wts[0:M])
            wts2 = np.tile(wts2.reshape((M, 1)), (1, P)) * gwts
            wts2 = wts2.reshape((M, P)).T
            wts2 = wts2.reshape((M * P,))

            lp, Wcp = mixture_pdf(
                Lcp.reshape((nc, P * M)) * lfac, wts2.reshape((P * M,))
            )

            gwts = np.dot(dwts, Wcp).reshape((P, M)).T
            Wcp = Wcp.reshape((nc, P, M))
            gs = np.sum(gwts, axis=1)
            iz = np.where(gs == 0)[0]
            gs[iz] = 1
            gwts = gwts / np.tile(gs.reshape((M, 1)), (1, P))

        if iter < nit_gmm and iter > 10 and iter % 1 == 0:
            # detect 'same' clusters
            Cd = cluster_dist(cmean, cvar, gwts)
            # print(np.mean(cmean,axis=1).reshape((dim,M)))
            # print(np.round(Cd*10)/10.0)
            # quit()
            isort = np.argsort(-wts[0:M])
            keep = np.ones((M,))
            idel = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if idel < 1 and Cd[isort[i], isort[j]] > merge_thresh:
                        keep[isort[j]] = 0
                        idel += 1
            nkeep = np.sum(keep)
            if nkeep < M:
                ii = np.where(keep == 1)[0]
                iip = np.where(keep == 0)[0]
                # print("Pruning clusters:", iip)
                Lcp = Lcp[:, :, ii]
                Wc = Wc[:, ii]
                W = np.concatenate([W[:, ii], W[:, -1].reshape((nc, 1))], axis=1)
                Wcp = Wcp[:, :, ii]
                cmean = cmean[:, :, ii]
                cvar = cvar[:, :, ii]
                gwts = gwts[ii, :]
                M = int(nkeep)
                wts = np.concatenate(
                    [wts[ii].reshape((M,)), wts[-1].reshape((1,))], axis=0
                ).reshape((M + 1,))

        Wcp = Wcp.reshape((nc, P, M))
        Wc = np.sum(Wcp, axis=1)
        Wc = Wc.reshape((nc, M))
        ls = np.sum(Wc, axis=1)
        wpow = pow(wts, wts_exp)
        # if iter>50 and (iter%5==0 or  iter >= nit_gmm) :
        if iter >= nit_gmm:
            Wd = np.power(W, dfac) * np.tile(dwts.reshape((nc, 1)), (1, M + 1))
            Dw = np.zeros((nc, M + 1))
            for i in range(nc):
                Dw[i, :] = np.dot(D_w[i], Wd[D_i[i], :])
            ds = np.sum(Dw, axis=1)
            iz = np.where(ds == 0)[0]
            ds[iz] = 1
            Dw = Dw / np.tile(ds.reshape((nc, 1)), (1, M + 1))
            W[:, 0:M] = Dw[:, 0:M] * Wc * np.tile(wpow[0:M].reshape((1, M)), (nc, 1))
        else:
            W[:, 0:M] = Wc * np.tile(wpow[0:M].reshape((1, M)), (nc, 1))
        W[:, M] = minwt
        ws = np.sum(W, axis=1)
        W = W / np.tile(ws.reshape((nc, 1)), (1, M + 1))
        wts = np.dot(dwts, W)
        wts = wts / np.sum(wts)
        # if iplt > 0 and (iter >= nit_gmm or iter % 10 == 0):
        #     H = np.zeros((n, nseg)).reshape((n * nseg,))
        #     ic = np.argmax(W, axis=1)
        #     H[ijpos] = ic + 4
        #     H = H.reshape((n, nseg))
        #     # plt.cla()

        #     axs.cla()
        #     # print('plotting')
        #     axs.imshow(
        #         H,
        #         origin="lower",
        #         aspect="auto",
        #         cmap="jet",
        #         extent=(times[0], times[-1], sfrqs[0], sfrqs[-1]),
        #     )
        #     plt.show(block=False)
        #     plt.pause(0.1)

        Q = np.mean(lp)
        dl = Q - ql
        # print("Q(%d)=" % iter, "%f" % Q, ",del=%f" % dl, "nclus=", M)
        # if(iter>10 and dl < .01):
        #   break
        ql = Q

        if iter >= nit_gmm:
            continue

        # Re-estimate GMM
        for i in range(M):
            if wts[i] < 0.0001:
                continue
            w = W[:, i] * dwts
            wpt = np.sum(Wcp[:, :, i], axis=1)
            iz = np.where(wpt == 0)[0]
            wpt[iz] = 1
            pp = Wcp[:, :, i] / np.tile(wpt.reshape((nc, 1)), (1, P))  # shape (nc,P)

            for p in range(P):
                wp = w * pp[:, p]
                sw = np.sum(wp)
                if sw <= 1e-19:
                    # print('wts(',i,',',p,')=0')
                    continue

                if False:
                    mu = np.dot(wp, C) / sw
                    mu = rgb2feat(mu.reshape((1, 3)))
                else:
                    mu0 = cmean[:, p, i]
                    t = Cn - np.tile(mu0.reshape((1, dim)), (nc, 1))
                    t = np.fmod(t + 1.5, 1) - 0.5
                    dmu = np.dot(t.T, wp) / sw
                    mu = mu0 + dmu
                cmean[:, p, i] = mu
                t = Cn - np.tile(mu.reshape((1, dim)), (nc, 1))
                t = np.fmod(t + 1.5, 1) - 0.5
                if False:
                    tw = t * wp.reshape(t.shape)
                    R = np.dot(t.T, tw) / sw
                    R = R + np.diag(np.ones((dim, 1)) * minvar)
                else:
                    R = np.sum(np.square(t) * wp.reshape(t.shape)) / sw + minvar
                cvar[:, p, i] = R.reshape((dim * dim,))

    isort = np.argsort(-wts[0:M])
    npl = np.minimum(nplot, len(isort))
    S_out = [None] * npl
    direction = np.zeros((npl,))
    wts_out = np.zeros((npl,))
    i_out = 0
    for i in range(npl):
        H = np.zeros((n, nseg, 3)).reshape((n * nseg, 3))
        imod = isort[i]
        if wts[imod] < 0.0001:
            continue
        ampl = C * np.tile(W[:, imod].reshape((nc, 1)), (1, 3))
        H[ijpos, :] = ampl
        H = H.reshape((n, nseg, 3))
        S_out[i_out] = H
        hsv_mean = rgb2hsv(np.mean(ampl.reshape((nc, 3)), axis=0).reshape((1, 1, 3)))
        direction[i_out] = np.round(hsv_mean[0, 0, 0] * 3600) / 10
        wts_out[i_out] = wts[imod]
        i_out += 1
    # remove empty clusters
    S_out = S_out[0:i_out]
    wts_out = wts_out[0:i_out]
    direction = direction[0:i_out]

    return S_out, direction, wts_out


def recon3(b):
    n, m = b.shape
    NFFT = (n - 1) * 2
    ntot = NFFT + (m - 1) * NFFT // 3
    x = np.zeros((ntot,))
    for i in range(m):
        xf = b[:, i]
        xht = np.fft.irfft(xf)
        xt = np.real(xht) * (2.0 / 3)
        i0 = int(i * NFFT // 3)
        x[i0 : i0 + NFFT] = x[i0 : i0 + NFFT] + xt
    return x


def spectogram_to_time_series(B, directions, S, fs, d, nmic, c, N, fmin):

    m = len(S)
    # high-pass filtering
    Bf, Af = scipy.signal.butter(2, fmin / fs * 2, btype="high")
    w, h = frsp = scipy.signal.freqz(Bf, Af, worN=N, whole=True, plot=None)
    frsp = np.square(np.abs(h))

    # microphone positions (m) and pointing angles (radians)
    pi = np.pi
    if nmic == 4:
        pos = (-d / 2, d / 2, d / 2, d / 2, d / 2, -d / 2, -d / 2, -d / 2)
        pos = np.asarray(pos).reshape((nmic, 2))
        am = np.asarray((-pi / 4, pi / 4, 3 * pi / 4, -3 * pi / 4))
    else:
        pos = (d / 2, 0, -d / 2, 0)
        pos = np.asarray(pos).reshape((nmic, 2))
        am = np.asarray((-pi / 2, pi / 2))

    shift = N // 3  # for 2/3 overlap
    wts = (
        1 + np.asarray(range(N), dtype="float64") * 2 * np.pi
    ) * 0.5  # hanning weights

    #
    # dst(i,j) is distance from microphone i to a distant point at angle j
    #  dctr is distance to center of array
    dst = np.zeros((nmic, m))
    r = 1000  # target at this range
    angs = directions * np.pi / 180
    for ia in range(m):
        ry = np.cos(angs[ia]) * r
        rx = np.sin(angs[ia]) * r
        dctr = np.sqrt(np.square(rx) + np.square(ry))
        for i in range(nmic):
            dst[i, ia] = (
                np.sqrt(np.square(rx - pos[i, 0]) + np.square(ry - pos[i, 1])) - dctr
            )

    n = N // 2 + 1
    sfrqs = np.arange(0, n) / N * fs
    # set up steering vectors
    s = [None] * nmic
    for i in range(nmic):
        # ft = (f)requency times delay (t)time
        ft = np.outer(sfrqs, dst[i, :] / c)
        if nmic == 4:
            # microphone cardioid pattern
            mr = (1 + np.cos(angs - am[i])) / 2
            # mr=np.exp( -np.power((angs-am[i])/(np.pi),4))
        else:
            # omnidirec
            mr = np.ones((na,))
        # steering vector
        s[i] = np.exp(1j * 2 * pi * ft) * mr

    n, NSEG = B[0].shape
    Bs = [None] * m
    for i in range(m):
        Bs[i] = np.zeros((n, NSEG)) + np.zeros((n, NSEG)) * 1j

    # [nc,na]=co.shape
    [n, NSEG] = B[0].shape
    I = np.zeros((n, NSEG, 3))
    for ismp in range(NSEG):
        xbf = 0
        for i in range(nmic):
            bt = B[i][:, ismp].reshape((n, 1))
            xbf = xbf + np.tile(bt, (1, m)) * (s[i]).reshape((n, m))
        for i in range(m):
            Bs[i][:, ismp] = xbf[:, i]
    Ts = [None] * m
    for i in range(m):
        amp = np.sum(S[i], axis=2).reshape((n, NSEG))
        if True:
            # source separation and beamforming
            x = recon3(Bs[i] * np.power(amp, 0.2))
            # print('mean=',np.mean(amp))
        else:
            # beamforming only
            x = recon3(Bs[i])
        Ts[i] = x
    return Ts

