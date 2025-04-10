import numpy as np
import scipy


def keep_peaks_prom(data, n):
    peaks, _ = scipy.signal.find_peaks(data)
    prominences = scipy.signal.peak_prominences(data, peaks)[0]

    top_n_indices = np.argsort(prominences)[-n:]
    top_n_peaks = peaks[top_n_indices]

    result = np.zeros_like(data)
    result[top_n_peaks] = data[top_n_peaks]

    return result


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


def rescale(arr_true, arr_pred):
    return np.interp(np.linspace(0, 1, len(arr_true)), np.linspace(0, 1, len(arr_pred)), arr_pred)


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


def post_processing_pred(dtf_pred, gamma=2.5, kernel_size=600, true='RAMAN_SPECTRUM_1', pred='raman_pred'):

    lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)

    # Rescale
    dtf_pred[pred] = dtf_pred.apply(lambda row: rescale(row[true], row[pred]), axis=1)

    # Keep peaks based on prominence and predicted number of peaks
    dtf_pred[pred] = dtf_pred.apply(lambda row: keep_peaks_prom(row[pred], round(row.pred_num_peak)),
                                              axis=1)

    # Convolve RAMAN_SPECTRUM and prediction with a lorentzian
    dtf_pred['raman_pred_conv']=dtf_pred.apply(lambda row:  convolve_with_lorentzian(row[pred], lorentzian_kernel), axis=1)
    dtf_pred['RAMAN_SPECTRUM_CONV']=dtf_pred.apply(lambda row:  convolve_with_lorentzian(row[true], lorentzian_kernel), axis=1)

    return dtf_pred