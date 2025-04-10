import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error
import scipy
from numpy import dot
from numpy.linalg import norm
import math
from Scripts.utils_model.utils_data_processing import convolve_with_lorentzian, rescale

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


def metrics_raman_peaks(result_df, prom=0, true='RAMAN_SPECTRUM', pred='raman_pred'):
    result_df['precision_10cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row[true], row[pred], prominence=prom), axis=1)
    result_df['precision_15cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row[true], row[pred], prominence=prom,
                                                                         tolerance=7.5), axis=1)
    result_df['precision_20cm^-1'] = result_df.apply(lambda row:
                                                     precision_score_mod(row[true], row[pred], prominence=prom,
                                                                         tolerance=10), axis=1)
    result_df['recall_10cm^-1'] = result_df.apply(lambda row: recall_score_mod(row[true], row[pred], prominence=prom),
                                                  axis=1)
    result_df['recall_15cm^-1'] = result_df.apply(lambda row: recall_score_mod(row[true], row[pred], prominence=prom,
                                                                               tolerance=7.5), axis=1)
    result_df['recall_20cm^-1'] = result_df.apply(lambda row: recall_score_mod(row[true], row[pred], prominence=prom,
                                                                               tolerance=10), axis=1)
    result_df['fnr_10cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row[true], row[pred], prominence=prom), axis=1)
    result_df['fnr_15cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row[true], row[pred], prominence=prom,
                                                                         tolerance=7.5), axis=1)
    result_df['fnr_20cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row[true], row[pred], prominence=prom,
                                                                         tolerance=10), axis=1)
    result_df['F1_10cm^-1'] = result_df.apply(lambda row: f1_score_mod(row[true], row[pred], prominence=prom), axis=1)
    result_df['F1_15cm^-1'] = result_df.apply(lambda row: f1_score_mod(row[true], row[pred], prominence=prom,
                                                                       tolerance=7.5), axis=1)
    result_df['F1_20cm^-1'] = result_df.apply(lambda row: f1_score_mod(row[true], row[pred], prominence=prom,
                                                                       tolerance=10), axis=1)
    return result_df



def metrics_spectra(result_df, conv, leng, true_col='RAMAN_SPECTRUM_CONV', pred_col='raman_pred_conv'):
    result_df['sis'] = result_df.apply(lambda row: spectral_information_similarity(pd.array(row[true_col]),
                                                                                   pd.array(row[pred_col]), conv,
                                                                                   frequencies=leng), axis=1)
    result_df['R2'] = result_df.apply(lambda row: r2_score(row[true_col], row[pred_col]), axis=1)
    result_df['MSE'] = result_df.apply(lambda row: mean_squared_error(row[true_col], row[pred_col]), axis=1)
    result_df['RMSE'] = result_df.apply(lambda row: root_mean_squared_error(row[true_col], row[pred_col]), axis=1)
    result_df['cos_sim'] = result_df.apply(lambda row: calc_cos_sim(row[pred_col], row[true_col]), axis=1)
    result_df['MAE'] = result_df.apply(lambda row: np.mean(np.abs(row[pred_col] - row[true_col])), axis=1)
    return result_df


def excel_data_raman(result_df):
    excel = [result_df['precision_10cm^-1'].mean(), result_df['precision_15cm^-1'].mean(), result_df['precision_20cm^-1'].mean(),
            result_df['recall_10cm^-1'].mean(), result_df['recall_15cm^-1'].mean(), result_df['recall_20cm^-1'].mean(),
            result_df['fnr_10cm^-1'].mean(), result_df['fnr_15cm^-1'].mean(), result_df['fnr_20cm^-1'].mean(),
            result_df['F1_10cm^-1'].mean(), result_df['F1_15cm^-1'].mean(), result_df['F1_20cm^-1'].mean(),
            result_df['sis'].mean(), result_df['cos_sim'].mean(),
            result_df['MAE'].mean(), result_df['RMSE'].mean(), result_df['R2'].mean()]
    return excel

