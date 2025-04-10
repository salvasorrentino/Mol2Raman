import torch
import pandas as pd
import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
import math
from numpy import dot
from numpy.linalg import norm


class EarlyStopping:
    
    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):

        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:  self.early_stop = True
        else: self.counter = 0


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def count_matched_peaks(arr_a, arr_b, tolerance=6):
    count = 0
    for peak_a in arr_a:
        for peak_b in arr_b:
            if abs(peak_a - peak_b) <= tolerance and peak_a != 0 and peak_b != 0:
                count += 1
                arr_b = list(filter(lambda x: x != peak_b, arr_b))
                break  # Found a match, move to the next peak in arr_a
    return count


def count_num_peaks(lst):
    count = 0
    for i in lst:
        if i>0:
            count+=1
    return count


def apply_mask(lst, msk):
    new_intensity_values = np.zeros_like(lst)
    for i in msk:
        new_intensity_values[i] = lst[i]

    return new_intensity_values


def keep_peaks_prom(data, n):
    peaks, _ = scipy.signal.find_peaks(data)
    prominences = scipy.signal.peak_prominences(data, peaks)[0]

    top_n_indices = np.argsort(prominences)[-n:]
    top_n_peaks = peaks[top_n_indices]

    result = np.zeros_like(data)
    result[top_n_peaks] = data[top_n_peaks]

    return result


def f1_score_mod(arr_true, arr_pred, prominence=0, tolerance=5):
    mask, _ = scipy.signal.find_peaks(arr_pred, prominence=prominence)
    mask = mask.tolist()

    mask_true, _ = scipy.signal.find_peaks(arr_true, prominence=prominence)
    mask_true = mask_true.tolist()

    tp = count_matched_peaks(mask_true, mask, tolerance)
    fp = len(mask) - tp
    fn = len(mask_true) - tp

    if (tp+fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (fn+tp) != 0:
        recall = tp / (fn + tp)
    else:
        recall = 0

    if tp != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1


def precision_score_mod(arr_true, arr_pred, prominence=0, tolerance=5):
    mask, _ = scipy.signal.find_peaks(arr_pred, prominence=prominence)
    mask = mask.tolist()

    mask_true, _ = scipy.signal.find_peaks(arr_true, prominence=prominence)
    mask_true = mask_true.tolist()

    tp = count_matched_peaks(mask_true, mask, tolerance)
    fp = len(mask) - tp
    fn = len(mask_true) - tp

    if (tp+fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    return precision


def recall_score_mod(arr_true, arr_pred, prominence=0, tolerance=5):
    mask, _ = scipy.signal.find_peaks(arr_pred, prominence=prominence)
    mask = mask.tolist()

    mask_true, _ = scipy.signal.find_peaks(arr_true, prominence=prominence)
    mask_true = mask_true.tolist()

    tp = count_matched_peaks(mask_true, mask, tolerance)
    fp = len(mask) - tp
    fn = len(mask_true) - tp

    if (fn+tp) != 0:
        recall = tp / (fn + tp)
    else:
        recall = 0

    return recall


def fnr_score_mod(arr_true, arr_pred, prominence=0, tolerance=5):
    mask, _ = scipy.signal.find_peaks(arr_pred, prominence=prominence)
    mask = mask.tolist()

    mask_true, _ = scipy.signal.find_peaks(arr_true, prominence=prominence)
    mask_true = mask_true.tolist()

    tp = count_matched_peaks(mask_true, mask, tolerance)
    fp = len(mask) - tp
    fn = len(mask_true) - tp

    if (fn+tp) != 0:
        fnr = fn / (fn + tp)
    else:
        fnr = 0

    return fnr


def fpr_score_mod(arr_true, arr_pred, prominence=0, tolerance=5):
    mask, _ = scipy.signal.find_peaks(arr_pred, prominence=prominence)
    mask = mask.tolist()

    mask_true, _ = scipy.signal.find_peaks(arr_true, prominence=prominence)
    mask_true = mask_true.tolist()

    tp = count_matched_peaks(mask_true, mask, tolerance)
    fp = len(mask) - tp
    fn = len(mask_true) - tp
    tn = len(arr_pred)-tp-fp-fn

    if (fn+tp) != 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0

    return fpr


def calc_cos_sim(row_true, row_pred):

    cos_sim = dot(row_true, row_pred) / (norm(row_pred) * norm(row_true))

    return cos_sim


def nomi_file_in_cartella(cartella):
    # Verifica se la cartella esiste
    if not os.path.isdir(cartella):
        print(f"La cartella {cartella} non esiste.")
        return ()

    nomi_file = os.listdir(cartella)
    tupla_nomi_file = tuple(nomi_file)
    return tupla_nomi_file


def keep_peaks_int(arr, n):
    sorted_arr = sorted(arr, reverse=True)
    nth_highest_value = sorted_arr[n - 1]
    count = 0

    for i in range(len(arr)):
        if arr[i] < nth_highest_value:
            arr[i] = 0
    return arr


def rescale(arr_true, arr_pred):
    return np.interp(np.linspace(0, 1, len(arr_true)), np.linspace(0, 1, len(arr_pred)), arr_pred)


def only_peaks(yy_pred, prominence=0):
    mask, _ = scipy.signal.find_peaks(yy_pred, prominence=prominence)
    mask = mask.tolist()
    raman_pred_mask_int = apply_mask(yy_pred, mask)
    return raman_pred_mask_int


@retry(stop=stop_after_attempt(50), wait=wait_fixed(1))
def featurize_with_retry(featurizer, mol):
    return featurizer._featurize(mol)


def rescale_intensity(intensity):
    return intensity / np.nansum(intensity)


def lorentzian_kernel(x, gammas):
    return (gammas**2) / (x**2 + gammas**2)


def generate_lorentzian_kernel(kernel_size, gamma):
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    kernel = lorentzian_kernel(x, gamma)
    return kernel / np.sum(kernel)


def convolve_with_lorentzian(predicted_intensity, lorentzian_kernels):
    return scipy.signal.convolve(predicted_intensity, lorentzian_kernels, mode='same')


def lorentz(x, x0, amp, fwhm = 10):
    f = amp * (1 / math.pi) * (fwhm / 2) / ((x - x0) ** 2 + (fwhm / 2) ** 2)
    return f


def lorentz_conv(arr_prova, int_len=900, int_fwhm=5):
    lst_p, _ = scipy.signal.find_peaks(arr_prova)
    arr_tmp = np.empty((len(lst_p), int_len))
    for k, peaks in enumerate(lst_p):
        for i in range(len(arr_prova)):
            arr_tmp[k, i] = lorentz(i, peaks, arr_prova[peaks], int_fwhm)

    arr_tmp = arr_tmp.sum(axis=0)
    return arr_tmp


def spectral_information_similarity(spectrum1, spectrum2, conv_matrix, frequencies=list(range(800,3500,2)), threshold=1e-8):
    length = len(spectrum1)
    nan_mask = np.isnan(spectrum1)+np.isnan(spectrum2)
    # print(length,conv_matrix.shape,spectrum1.shape,spectrum2.shape)
    assert length == len(spectrum2), "compared spectra are of different lengths"
    assert length == len(frequencies), "compared spectra are a different length than the frequencies list, which can be specified"
    spectrum1[spectrum1 < threshold] = threshold
    spectrum2[spectrum2 < threshold] = threshold
    spectrum1[nan_mask] = 0
    spectrum2[nan_mask] = 0

    spectrum1=np.expand_dims(spectrum1, axis=0)
    spectrum2=np.expand_dims(spectrum2, axis=0)

    conv1 = np.matmul(spectrum1, conv_matrix)

    conv2 = np.matmul(spectrum2, conv_matrix)
    conv1[0, nan_mask] = np.nan
    conv2[0, nan_mask] = np.nan

    sum1 = np.nansum(conv1)
    sum2 = np.nansum(conv2)
    norm1 = conv1/sum1
    norm2 = conv2/sum2
    distance = norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
    sim = 1/(1+np.nansum(distance))
    return sim


def make_conv_matrix(frequencies=list(range(500, 2100, 2)), std_dev=10):
    length = len(frequencies)
    gaussian = [(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2))
                for i in range(length)]
    conv_matrix = np.empty([length, length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i, j] = gaussian[abs(i-j)]
    return conv_matrix


def sim(arr_true, arr_pred, arr_peaks, conv, freq, lor_kernel):
    arr_resc = rescale(arr_true, np.array(arr_pred))

    lore_pred_resc = convolve_with_lorentzian(keep_peaks_prom(arr_resc, arr_peaks),
                                              lor_kernel)
    lore_true_resc = convolve_with_lorentzian(np.array(arr_true), lor_kernel)

    sim_resc = spectral_information_similarity(lore_true_resc, lore_pred_resc, conv, frequencies=freq)
    return sim_resc


def sim_dir(arr_true, arr_pred, conv, freq, lor_kernel):
    lore_pred_resc = convolve_with_lorentzian(arr_pred,
                                              lor_kernel)
    lore_true_resc = convolve_with_lorentzian(np.array(arr_true), lor_kernel)

    sim_resc = spectral_information_similarity(lore_true_resc, lore_pred_resc, conv, frequencies=freq)
    return sim_resc


def common_smile(dtf1, dtf2, smile1, smile2):
    lst_use = list(set(dtf1[smile1]) & set(dtf2[smile2]))
    return lst_use

def lst_metrics_mean(dtf):
    lst_metrics = [dtf['precision_10cm^-1'].mean(), dtf['precision_15cm^-1'].mean(), dtf['precision_20cm^-1'].mean(),
        dtf['recall_10cm^-1'].mean(), dtf['recall_15cm^-1'].mean(), dtf['recall_20cm^-1'].mean(),
        dtf['fnr_10cm^-1'].mean(), dtf['fnr_15cm^-1'].mean(), dtf['fnr_20cm^-1'].mean(),
        dtf['F1_10cm^-1'].mean(), dtf['F1_15cm^-1'].mean(), dtf['F1_20cm^-1'].mean(),
        dtf['sis'].mean(), dtf['cos_sim'].mean(),
        dtf['MAE'].mean(), dtf['RMSE'].mean(), dtf['R2'].mean()]
    return lst_metrics