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
        epoch_wise: boolean
          whether to compute epoch-to-epoch synchrony. default is True.
          if False, complex values from epochs will be concatenated before
          computing synchrony
          if True, synchrony is computed from matched epochs
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

    result = compute_sync(values, mode, epoch_wise, time_resolved)

    return result


def _multiply_conjugate(real, imag, time_resolved, transpose_axes):
    if time_resolved:
        formula = 'ilm,imk->ilk'
    else:
        formula = 'jilm,jimk->jilk'
    product = np.einsum(formula, real, real.transpose(transpose_axes)) + \
           np.einsum(formula, imag, imag.transpose(transpose_axes)) + 1j * \
           (np.einsum(formula, real, imag.transpose(transpose_axes)) - \
            np.einsum(formula, imag, real.transpose(transpose_axes)))

    return product


def compute_sync_new(complex_signal, mode, time_resolved=True):
    """
      (improved) Computes synchrony from analytic signals.

    """
    from tqdm import trange
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
        complex_signal.shape[3], complex_signal.shape[4]

    if time_resolved:
        # creating indices
        idx1 = np.repeat(np.arange(0, n_ch, 1), n_ch)
        idx2 = np.tile(np.arange(0, n_ch, 1), n_ch)

        con = np.zeros((len(idx1), n_freq))
        con_idx = con.shape[0]

        if mode is 'envelope':
            values = np.abs(complex_signal)
            for this_epoch in trange(n_epoch):
                this_con = np.array([_corr2_coeff_rowwise2(values[0][this_epoch, idx1[i], :, :],
                                                           values[1][this_epoch, idx2[i], :, :])
                                     for i in range(con_idx)])
                con += this_con

        elif mode is 'power':
            values = np.abs(complex_signal) ** 2
            for this_epoch in trange(n_epoch):
                this_con = np.array([_corr2_coeff_rowwise2(values[0][this_epoch, idx1[i], :, :],
                                                           values[1][this_epoch, idx2[i], :, :])
                                     for i in range(con_idx)])
                con += this_con


        elif mode is 'plv':
            values = complex_signal / np.abs(complex_signal)
            for this_epoch in trange(n_epoch):
                this_con = np.array([_plv(values[0][this_epoch, idx1[i], :, :],
                                          values[1][this_epoch, idx2[i], :, :], axis=1)
                                     for i in range(con_idx)])
                con += this_con

        elif mode is 'ccorr':
            values = np.angle(complex_signal)
            for this_epoch in trange(n_epoch):
                this_con = np.array([_circcorrcoef(values[0][this_epoch, idx1[i], :, :],
                                                   values[1][this_epoch, idx2[i], :, :], axis=1)
                                     for i in range(con_idx)])
                con += this_con

        elif mode is 'proj':
            values = complex_signal
            for this_epoch in trange(n_epoch):
                this_con = np.array([_proj_power_corr_simplified(values[0][this_epoch, idx1[i], :, :],
                                                      values[1][this_epoch, idx2[i], :, :], axis=1)
                                     for i in range(con_idx)])
                con += this_con

        elif mode is 'imagcoh':
            values = complex_signal
            for this_epoch in trange(n_epoch):
                this_con = np.array([_icoh(values[0][this_epoch, idx1[i], :, :],
                                           values[1][this_epoch, idx2[i], :, :], axis=1)
                                     for i in range(con_idx)])
                con += this_con
        elif mode is 'coh':
            values = complex_signal
            for this_epoch in trange(n_epoch):
                this_con = np.array([_coh(values[0][this_epoch, idx1[i], :, :],
                                          values[1][this_epoch, idx2[i], :, :], axis=1)
                                     for i in range(con_idx)])
                con += this_con

        # transform con to a matrix
        con = con.reshape((n_ch, n_ch, -1))
        con = con/n_epoch

    return con

def compute_sync_test(complex_signal, mode, time_resolved=True):
    """
      (improved) Computes synchrony from analytic signals.

    """
    from tqdm import trange
    n_epoch, n_ch, n_freq, n_samp = complex_signal.shape[1], complex_signal.shape[2], \
        complex_signal.shape[3], complex_signal.shape[4]

    if time_resolved:
        con = np.zeros((n_freq, 2 * n_ch, 2 * n_ch))
        complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)

        if mode is 'envelope':
            env = np.abs(complex_signal)
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2*n_ch, 1)
            env = env - mu_env
            formula = 'ilm,imk->ilk'
            for this_epoch in trange(n_epoch):
                this_env = env[this_epoch]
                corr = np.einsum(formula, this_env, this_env.transpose((0, 2, 1))) / \
                        np.sqrt(np.einsum('il,ik->ilk', np.sum(this_env ** 2, axis=2), np.sum(this_env ** 2, axis=2)))
                con+=corr

        elif mode is 'power':
            env = np.abs(complex_signal)**2
            mu_env = np.mean(env, axis=3).reshape(n_epoch, n_freq, 2*n_ch, 1)
            env = env - mu_env
            formula = 'ilm,imk->ilk'
            for this_epoch in trange(n_epoch):
                this_env = env[this_epoch]
                corr = np.einsum(formula, this_env, this_env.transpose((0, 2, 1))) / \
                       np.sqrt(np.einsum('il,ik->ilk', np.sum(this_env ** 2, axis=2), np.sum(this_env ** 2, axis=2)))
                con+=corr

        elif mode is 'plv':
            phase = complex_signal / np.abs(complex_signal)
            c = np.real(phase)
            s = np.imag(phase)
            for this_epoch in trange(n_epoch):
                this_c = c[this_epoch]
                this_s = s[this_epoch]
                dphi = _multiply_conjugate(this_c, this_s, time_resolved=time_resolved, transpose_axes=(0,2,1))
                this_con = abs(dphi) / n_samp
                con += this_con

        elif mode is 'coh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            for this_epoch in trange(n_epoch):
                this_c = c[this_epoch]
                this_s = s[this_epoch]
                this_amp = amp[this_epoch]
                dphi = _multiply_conjugate(this_c, this_s, time_resolved=time_resolved, transpose_axes=(0,2,1))
                this_con = np.abs(dphi) / np.sqrt(np.einsum('ilm,imk->ilk', this_amp, this_amp.transpose((0, 2, 1))))
                con += this_con

        elif mode is 'imagcoh':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            amp = np.abs(complex_signal) ** 2
            for this_epoch in trange(n_epoch):
                this_c = c[this_epoch]
                this_s = s[this_epoch]
                this_amp = amp[this_epoch]
                dphi = _multiply_conjugate(this_c, this_s, time_resolved=time_resolved, transpose_axes=(0, 2, 1))
                this_con = np.imag(dphi) / np.sqrt(np.einsum('ilm,imk->ilk', this_amp, this_amp.transpose((0, 2, 1))))
                con += this_con

        elif mode is 'proj':
            c = np.real(complex_signal)
            s = np.imag(complex_signal)
            env = np.abs(complex_signal)
            c_phase = np.real(complex_signal / env)
            s_phase = np.imag(complex_signal / env)

            con = np.zeros((n_freq, 2 * n_ch, 2 * n_ch))

            for this_epoch in trange(n_epoch):
                this_c, this_s, this_c_phase, this_s_phase = c[this_epoch],s[this_epoch],c_phase[this_epoch],s_phase[this_epoch]
                formula = 'ilm,imk->ilk'
                transpose_axes = (0, 2, 1)
                productX = np.imag(np.einsum(formula, this_c, this_c_phase.transpose(transpose_axes)) + \
                           np.einsum(formula, this_s, this_s_phase.transpose(transpose_axes)) + 1j * \
                           (np.einsum(formula, this_c, this_s_phase.transpose(transpose_axes)) - \
                            np.einsum(formula, this_s, this_c_phase.transpose(transpose_axes))))
                productY = np.imag(np.einsum(formula, this_c_phase, this_c.transpose(transpose_axes)) +\
                           np.einsum(formula, this_s_phase, this_s.transpose(transpose_axes)) + 1j *\
                           (np.einsum(formula, this_c_phase, this_s.transpose(transpose_axes)) -\
                            np.einsum(formula, this_s_phase, this_c.transpose(transpose_axes))))

                con += (productX+productY)/2

        elif mode is 'ccorr':
            angle = np.angle(complex_signal)
            mu_angle = circmean(angle, axis=3).reshape(n_epoch, n_freq, 2*n_ch, 1)
            angle = np.sin(angle - mu_angle)

            formula = 'ilm,imk->ilk'
            for this_epoch in trange(n_epoch):
                this_angle = angle[this_epoch]
                ccorr = np.einsum(formula, this_angle, this_angle.transpose((0, 2, 1))) / \
                        np.sqrt(np.einsum('il,ik->ilk', np.sum(this_angle ** 2, axis=2), np.sum(this_angle ** 2, axis=2)))
                con+=ccorr

        con = con / n_epoch  # average over number of epochs

    # calculate all epochs at once, the only downside is that the disk may not have enough space
    # TODO this can be easily implemented after we discuss this?
    else:
        if mode is 'plv':
            complex_signal = complex_signal.transpose((1, 3, 0, 2, 4)).reshape(n_epoch, n_freq, 2 * n_ch, n_samp)
            phase = complex_signal / np.abs(complex_signal)
            c = np.real(phase)
            s = np.imag(phase)
            dphi = _multiply_conjugate(c, s, time_resolved=time_resolved, transpose_axes=(0, 1, 3, 2))
            con = abs(dphi) / n_samp



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


#  Synchrony metrics


def _plv(X, Y, axis):
    """
    Phase Locking Value

    Takes two vectors (phase) and compute their plv

    adapted for 2D arrays
    """
    return np.abs(np.sum(np.exp(1j * (np.angle(X) - np.angle(Y))), axis)) / X.shape[axis]


def _coh(X, Y, axis):
    """
    Coherence

    Instantaneous coherence computed from hilbert transformed signal,
    then averaged across time points

            |A1·A2·e^(i*delta_phase)|
    Coh = -----------------------------
               sqrt(A1^2 * A2^2)

    A1: envelope of X
    A2: envelope of Y
    reference: Kida, Tetsuo, Emi Tanaka, and Ryusuke Kakigi.
    “Multi-Dimensional Dynamics of Human Electromagnetic Brain Activity.”
    Frontiers in Human Neuroscience 9 (January 19, 2016).
    https://doi.org/10.3389/fnhum.2015.00713.
    """
    X_phase = X/np.abs(X)
    Y_phase = Y/np.abs(Y)

    Sxy = np.nansum(np.abs(X) * np.abs(Y) * np.exp(1j * (np.angle(X_phase) - np.angle(Y_phase))), axis)/X.shape[axis]
    Sxx = np.nansum(np.abs(X)**2, axis)/X.shape[axis]
    Syy = np.nansum(np.abs(Y)**2, axis)/X.shape[axis]

    coh = np.abs(Sxy)/(np.sqrt(Sxx*Syy))
    return coh


def _icoh(X, Y, axis):
    """
    Coherence

    Instantaneous imaginary coherence computed from hilbert transformed signal,
    then averaged across time points

            |A1·A2·sin(delta_phase)|
    iCoh = -----------------------------
               sqrt(A1^2 * A2^2)
    """
    X_phase = X/np.abs(X)
    Y_phase = Y/np.abs(Y)

    iSxy = np.imag(np.nansum(np.abs(X) * np.abs(Y) * np.exp(1j * (np.angle(X_phase) - np.angle(Y_phase))), axis))/X.shape[axis]
    Sxx = np.nansum(np.abs(X)**2, axis)/X.shape[axis]
    Syy = np.nansum(np.abs(Y)**2, axis)/X.shape[axis]

    icoh = np.abs(iSxy)/(np.sqrt(Sxx*Syy))

    return icoh


def _corr2_coeff_rowwise2(A,B):
    """
    compute row-wise correlation for 2D arrays
    """
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = np.einsum('ij,ij->i',A_mA,A_mA)
    ssB = np.einsum('ij,ij->i',B_mB,B_mB)
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)


def _corr_test(A,B):
    """
    compute row-wise correlation for 2D arrays
    """
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    ssA = np.einsum('ij,ij->i',A_mA,A_mA)
    ssB = np.einsum('ij,ij->i',B_mB,B_mB)
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)


# this function is modified from astropy in order to support 2D array operation
def _circcorrcoef(alpha, beta, axis=None):
    """ Computes the circular correlation coefficient between two array of
    circular data.

    Parameters
    ----------
    alpha : numpy.ndarray or Quantity
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    beta : numpy.ndarray or Quantity
        Array of circular (directional) data, which is assumed to be in
        radians whenever ``data`` is ``numpy.ndarray``.
    axis : int, optional
        Axis along which circular correlation coefficients are computed.
        The default is the compute the circular correlation coefficient of the
        flattened array.
    weights_alpha : numpy.ndarray, optional
        In case of grouped data, the i-th element of ``weights_alpha``
        represents a weighting factor for each group such that
        ``sum(weights_alpha, axis)`` equals the number of observations.
        See [1]_, remark 1.4, page 22, for detailed explanation.
    weights_beta : numpy.ndarray, optional
        See description of ``weights_alpha``.

    Returns
    -------
    rho : numpy.ndarray or dimensionless Quantity
        Circular correlation coefficient.

    References
    ----------
    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".
       Series on Multivariate Analysis, Vol. 5, 2001.
    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from 'Topics in
       Circular Statistics (2001)'". 2015.
       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>
    """
    if(np.size(alpha, axis) != np.size(beta, axis)):
        raise ValueError("alpha and beta must be arrays of the same size")

    mu_a = circmean(alpha, axis)
    mu_b = circmean(beta, axis)

    sin_a = np.sin(alpha - mu_a[:, None])
    sin_b = np.sin(beta - mu_b[:, None])
    rho = np.sum(sin_a*sin_b, axis)/np.sqrt(np.sum(sin_a*sin_a, axis)*np.sum(sin_b*sin_b, axis))

    return rho


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

def _proj_power_corr_simplified(X, Y, axis):
    # compute power proj corr using two complex signals
    # adapted from Georgios Michalareas' MATLAB script
    X_abs = np.abs(X)
    Y_abs = np.abs(Y)

    X_phase = X / X_abs
    Y_phase = Y / Y_abs

    projX = np.imag(X * np.conjugate(Y_phase))
    projY = np.imag(Y * np.conjugate(X_phase))

    proj_corr = np.nanmean(projX + projY, axis) / 2

    return proj_corr

