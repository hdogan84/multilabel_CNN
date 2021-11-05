from os import read
from tkinter.constants import X
import numpy
from numpy.core.fromnumeric import transpose
from tools.color_spectogram.ammod_functional_util import (
    calc_color_spectogram,
    read_audio_segment,
    cluster_color_spectorgram,
    spectogram_to_time_series,
    write_time_series_to_file,
)
from tools.color_spectogram.ammod_plots import plot_color_spectorgram, plot_clusters


def separate_sources(data, sample_frequency, plot=False, transpose=True):
    if transpose:
        data = numpy.transpose(data, (1, 0))
    thr = 0.350000
    nplot = 9
    mic_distance = 0.335  # former d
    mic_c = 347  # former c
    spectogram_nperseg = 384  # former N
    fmin = 1200.0  # high pass filter frequency
    angular_resolution = 3  # former ang_res in (degrees)
    fs = sample_frequency
    channels = data.shape[0]
    I, B, times, sfrqs = calc_color_spectogram(
        data,
        sample_frequency=fs,
        channels=channels,
        angular_resolution=angular_resolution,  # former ang_res in (degrees)
        start_time=0,  # former tstart
        end_time=5,  # former tend
        mic_distance=mic_distance,  # former d
        mic_c=mic_c,  # former c
        spectogram_nperseg=spectogram_nperseg,  # former N
        fmin=fmin,  # high pass filter frequency
    )
    if plot:
        axs, lbmin, lbmax = plot_color_spectorgram(
            I, times, sfrqs, brightness=1.5, contrast=1.2, block=False
        )
    # print("Calculate Clusters")
    S_out, direction, wts_out = cluster_color_spectorgram(
        I,
        thr,
        nplot,
        1,
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
    )
    if plot:
        plot_clusters(B, S_out, nplot, times, sfrqs, wts_out, direction, block=True)

    time_series = spectogram_to_time_series(
        B, direction, S_out, fs, mic_distance, channels, mic_c, spectogram_nperseg, fmin
    )
    if transpose:
        time_series = numpy.transpose(time_series, (1, 0))
    return time_series


if __name__ == "__main__":
    audio_sample_frequency = 32000
    audio_data = read_audio_segment(
        "./Schoenow_20070325_052813_S01075000E01125000_d2.wav",
        0,
        5,
        audio_sample_frequency,
    )
    time_series = separate_sources(audio_data, audio_sample_frequency, plot=True)
    write_time_series_to_file(time_series, audio_sample_frequency)

