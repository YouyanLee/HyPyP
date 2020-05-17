#!/usr/bin/env python
# coding=utf-8
"""
PSD, intra- and inter-brain measures functions
| Option | Description |
| ------ | ----------- |
| title           | analyses.py |
| authors         | Phoebe Chen, Florence Brun, Guillaume Dumas |
| date            | 2020-03-18 |
"""


from collections import namedtuple
import copy
import numpy as np
import scipy.signal as signal
from astropy.stats import circmean
import mne
from mne.time_frequency import psd_welch
from mne.io.constants import FIFF


def PSD(epochs, fmin, fmax, time_resolved):
    """
    Computes the Power Spectral Density (PSD) on Epochs for a condition.

    Arguments:
        epochs: Epochs for a condition, for a subject (can result from the
          concatenation of epochs from different occurences of the condition
          across experiments).
                Epochs are MNE objects (data are stored in arrays of shape
          (n_epochs, n_channels, n_times) and info are into a dictionnary.
        fmin, fmax: minimum and maximum frequencies-of-interest for power
          spectral density calculation, floats in Hz.
        time_resolved: whether to collapse the time course, boolean.
          If False, PSD won't be averaged over epochs the time
          course is maintained.
          If True, PSD values are averaged over epochs.

    Note:
        The function can be iterated on the group and/or on conditions:
      for epochs in epochs['epochs_%s_%s_%s' % (subj, group, cond_name)], you
      can then visualize PSD distribution on the group with the toolbox
      vizualisation to check normality for statistics for example.

    Returns:
        freqs_mean: list of frequencies in frequency-band-of-interest actually
          used for power spectral density calculation.
        PSD_welch: PSD value in epochs for each channel and each frequency,
          ndarray (n_epochs, n_channels, n_frequencies).
          Note that if time_resolved == True, PSD values are averaged
          across epochs.
    """
    # dropping EOG channels (incompatible with connectivity map model in stats)
    for ch in epochs.info['chs']:
        if ch['kind'] == 202:  # FIFFV_EOG_CH
            epochs.drop_channels([ch['ch_name']])

    # computing power spectral density on epochs signal
    # average in the 1second window around event (mean but can choose 'median')
    kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=1)
    psds_welch, freqs_mean = psd_welch(
        epochs, **kwargs, average='mean', picks='all')  # or median

    if time_resolved is True:
        # averaging power across epochs for each ch and each f
        PSD_welch = np.mean(psds_welch, axis=0)
    else:
        PSD_welch = psds_welch

    PSDTuple = namedtuple('PSD', ['freqs_mean', 'PSD_welch'])

    return PSDTuple(freqs_mean=freqs_mean,
                    PSD_welch=PSD_welch)


def indexes_connectivity_intrabrain(epochs):
    """
    Computes indexes for connectivity analysis between all EEG
    sensors for one subject. Can be used instead of
    (n_channels, n_channels) that takes into account intra electrode
    connectivity.

    Arguments:
        epochs: one subject Epochs object to get channels info, Epochs
          are MNE objects.

    Returns:
        electrodes: electrodes pairs for which connectivity indices will be
          computed, list of tuples with channels indexes.
    """
    names = copy.deepcopy(epochs.info['ch_names'])
    for ch in epochs.info['chs']:
            if ch['kind'] == FIFF.FIFFV_EOG_CH:
                names.remove(ch['ch_name'])

    n = len(names)
    # n = 64
    bin = 0
    idx = []
    electrodes = []
    for e1 in range(n):
        for e2 in range(n):
            if e2 > e1:
                idx.append(bin)
                electrodes.append((e1, e2))
            bin = bin + 1

    return electrodes


def indexes_connectivity_interbrains(epoch_hyper):
    """
    Computes indexes for interbrains connectivity analyses between all EEG
    sensors for 2 subjects (merge data).

    Arguments:
        epoch_hyper: one dyad Epochs object to get channels info, Epochs
          are MNE objects.

    Note:
        Only interbrains connectivity will be computed.

    Returns:
        electrodes: electrodes pairs for which connectivity indices will be
          computed, list of tuples with channels indexes.
    """
    electrodes = []
    names = copy.deepcopy(epoch_hyper.info['ch_names'])
    for ch in epoch_hyper.info['chs']:
            if ch['kind'] == FIFF.FIFFV_EOG_CH:
                names.remove(ch['ch_name'])

    l = list(range(0, int(len(names)/2)))
    # l = list(range(0,62))
    L = []
    M = len(l)*list(range(len(l), len(l)*2))
    for i in range(0, len(l)):
        for p in range(0, len(l)):
            L.append(l[i])
    for i in range(0, len(L)):
        electrodes.append((L[i], M[i]))

    return electrodes


def simple_corr(data, frequencies, mode, epoch_wise=True, time_resolved=True):
    """
    Computes frequency- and time-frequency-domain connectivity measures.

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          The data from which to compute connectivity between two subjects
        frequencies : dict | list
          frequencies of interest to compute connectivity with.
          If a dict, different frequency bands are used.
          e.g. {'alpha':[8,12],'beta':[12,20]}
          If a list, every integer frequency within the range is used.
          e.g. [5,30]
        mode: string
          Connectivity measure to compute.
          'envelope': envelope correlation
          'power': power correlation
          'plv': phase locking value
          'ccorr': circular correlation coefficient
          'coh': coherence
          'imagcoh': imaginary coherence
          'proj': projected power correlation
        time_resolved: boolean
          whether to collapse the time course, only effective when
          epoch_wise==True,
          if False, synchrony won't be averaged over epochs, and the time
          course is maintained.
          if True, synchrony is averaged over epochs.

    Note:
        Connectivity is computed for all possible electrode pairs between
        the dyad, but doesn't include intrabrain synchrony.

    Returns:
        result: array
          Computed connectivity measure(s). The shape of each array is either
          (n_freq, n_epochs, n_channels, n_channels) if epoch_wise is True
          and time_resolved is False, or (n_freq, n_channels, n_channels)
          in other conditions.
    """
    # Data consists of two lists of np.array (n_epochs, n_channels, epoch_size)
    assert data[0].shape[0] == data[1].shape[0], "Two streams much have the same lengths."

    # compute correlation coefficient for all symmetrical channel pairs
    if type(frequencies) == list:
        values = compute_single_freq(data, frequencies)
    # generate a list of per-epoch end values
    elif type(frequencies) == dict:
        values = compute_freq_bands(data, frequencies)

    result = compute_sync(values, mode, time_resolved)

    return result


def _multiply_conjugate(real, imag, transpose_axes):
    formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
           np.einsum(formula, imag, imag.transpose(transpose_axes)) + 1j * \
           (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
            np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def compute_sync(complex_signal, mode, time_resolved=True):
    """
      (improved) Computes synchrony from analytic signals.

    """
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
        complex_signal.shape[3], complex_signal.shape[4]

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
    transpose_axes = (0, 1, 3, 2)
    if mode.lower() is 'plv':
        phase = complex_signal / np.abs(complex_signal)
        c = np.real(phase)
        s = np.imag(phase)
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = abs(dphi) / n_samp

    elif mode.lower() is 'envelope':
        env = np.abs(complex_signal)
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
               np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() is 'powercorr':
        env = np.abs(complex_signal)**2
        mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2 * n_ch, 1)
        env = env - mu_env
        con = np.einsum('nilm,nimk->nilk', env, env.transpose(transpose_axes)) / \
               np.sqrt(np.einsum('nil,nik->nilk', np.sum(env ** 2, axis=3), np.sum(env ** 2, axis=3)))

    elif mode.lower() is 'coh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(dphi) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                    np.nansum(amp, axis=3)))

    elif mode.lower() is 'imagcoh':
        c = np.real(complex_signal)
        s = np.imag(complex_signal)
        amp = np.abs(complex_signal) ** 2
        dphi = _multiply_conjugate(c, s, transpose_axes=transpose_axes)
        con = np.abs(np.imag(dphi)) / np.sqrt(np.einsum('nil,nik->nilk', np.nansum(amp, axis=3),
                                                    np.nansum(amp, axis=3)))
    # elif mode.lower() is 'projpowercorr':
    #     c = np.real(complex_signal)
    #     s = np.imag(complex_signal)
    #     env = np.abs(complex_signal)
    #     c_phase = np.real(complex_signal / env)
    #     s_phase = np.imag(complex_signal / env)
    #
    #     formula = 'nilm,nimk->nilk'
    #     productX = np.imag(np.einsum(formula, c, c_phase.transpose(transpose_axes)) + \
    #                np.einsum(formula, s, s_phase.transpose(transpose_axes)) + 1j * \
    #                (np.einsum(formula, c, s_phase.transpose(transpose_axes)) - \
    #                 np.einsum(formula, s, c_phase.transpose(transpose_axes))))
    #     productY = np.imag(np.einsum(formula, c_phase, c.transpose(transpose_axes)) +\
    #                np.einsum(formula, s_phase, s.transpose(transpose_axes)) + 1j *\
    #                (np.einsum(formula, c_phase, s.transpose(transpose_axes)) -\
    #                 np.einsum(formula, s_phase, c.transpose(transpose_axes))))
    #
    #     con = (productX+productY)/2

    elif mode.lower() is 'ccorr':
        angle = np.angle(complex_signal)
        mu_angle = circmean(angle, axis=3).reshape(n_epoch, n_freq, 2*n_ch, 1)
        angle = np.sin(angle - mu_angle)

        formula = 'nilm,nimk->nilk'
        con = np.einsum(formula, angle, angle.transpose(transpose_axes)) / \
                np.sqrt(np.einsum('nil,nik->nilk', np.sum(angle ** 2, axis=3), np.sum(angle ** 2, axis=3)))

    else:
        ValueError('Metric type not supported.')

    if time_resolved:
        con = np.nanmean(con, axis=0)

    return con


def compute_single_freq(data, freq_range):
    """
    Computes analytic signal per frequency bin using a multitaper method
    implemented in MNE.

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          real-valued data to compute analytic signal from.
        freq_range: list
          a list of two specifying the frequency range

    Returns:
        complex_signal: array, shape is
          (2, n_epochs, n_channels, n_frequencies, n_times)
    """
    n_samp = data[0].shape[2]

    complex_signal = np.array([mne.time_frequency.tfr_array_multitaper(data[subject], sfreq=n_samp,
                                                                       freqs=np.arange(freq_range[0], freq_range[1], 1),
                                                                       n_cycles=4,
                                                                       zero_mean=False, use_fft=True, decim=1, output='complex')
                               for subject in range(2)])

    return complex_signal


def compute_freq_bands(data, freq_bands):
    """
    Computes analytic signal per frequency band using filtering
    and hilbert transform

    Arguments:
        data: array-like, shape is (2, n_epochs, n_channels, n_times)
          real-valued data to compute analytic signal from.
        freq_bands: dict
          a dict specifying names and corresponding frequency ranges

    Returns:
        complex_signal: array, shape is
          (2, n_epochs, n_channels, n_freq_bands, n_times)
    """
    assert data[0].shape[0] == data[1].shape[0]
    n_epoch = data[0].shape[0]
    n_ch = data[0].shape[1]
    n_samp = data[0].shape[2]
    data = np.array(data)

    # filtering and hilbert transform
    complex_signal = []
    for freq_band in freq_bands.values():
        filtered = np.array([mne.filter.filter_data(data[subject], n_samp, freq_band[0], freq_band[1], verbose=False)
                             for subject in range(2)  # for each subject
                             ])
        hilb = signal.hilbert(filtered)
        complex_signal.append(hilb)

    complex_signal = np.moveaxis(np.array(complex_signal), [0], [3])
    assert complex_signal.shape == (2, n_epoch, n_ch, len(freq_bands), n_samp)

    return complex_signal


# TODO remove
def _proj_power_corr(X, Y, axis):
    # compute power proj corr using two complex signals
    # adapted from Georgios Michalareas' MATLAB script
    X_abs = np.abs(X)
    Y_abs = np.abs(Y)

    X_phase = X / X_abs
    Y_phase = Y / Y_abs

    X_abs_norm = (X_abs - np.nanmean(X_abs, axis)[:,None]) / np.nanstd(X_abs, axis)[:,None]
    Y_abs_norm = (Y_abs - np.nanmean(Y_abs, axis)[:,None]) / np.nanstd(Y_abs, axis)[:,None]

    X_ = X_abs / np.nanstd(X_abs, axis)[:,None]
    Y_ = Y_abs / np.nanstd(Y_abs, axis)[:,None]

    X_z = X_ * X_phase
    Y_z = Y_ * Y_phase
    projX = np.imag(X_z * np.conjugate(Y_phase))
    projY = np.imag(Y_z * np.conjugate(X_phase))

    projX_norm = (projX - np.nanmean(projX, axis)[:,None]) / np.nanstd(projX, axis)[:,None]
    projY_norm = (projY - np.nanmean(projY, axis)[:,None]) / np.nanstd(projY, axis)[:,None]

    proj_corr = (np.nanmean(projX_norm * Y_abs_norm, axis) +
                 np.nanmean(projY_norm * X_abs_norm, axis)) / 2

    return proj_corr

