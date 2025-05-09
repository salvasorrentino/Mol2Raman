import pandas as pd
import numpy as np
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel,  rescale, \
    convolve_with_lorentzian
from Scripts.utils_model.utils_metrics import make_conv_matrix, metrics_spectra, metrics_raman_peaks, keep_peaks_prom


def process_arrays(df1, df2, col1, col2, new_column):
    # Ensure the columns are numpy arrays of the correct length
    assert all(df1[col1].apply(lambda x: len(x) == 800)), "All elements in col1 should be arrays of length 900"
    assert all(df2[col2].apply(lambda x: len(x) == 800)), "All elements in col2 should be arrays of length 900"

    def create_combined_array(arr1, arr2):
        first_700 = arr1[:700]
        last_700 = arr2[-700:]

        last_200_of_450 = arr1[-100:]
        first_200_of_900 = arr2[:100]

        mean_200 = (last_200_of_450 + first_200_of_900) / 2

        combined_array = np.concatenate([first_700, mean_200, last_700])

        return combined_array

    df1[new_column] = [create_combined_array(arr1, arr2) for arr1, arr2 in zip(df1[col1], df2[col2])]

    return df1


gamma = 2.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(501, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

# Load Dataframe number of peaks predicted and raman spectrum predicted, for lower and higher spectra
df_down = pd.read_parquet(r'data/predictions/pred_spectra_predictions_fingerprint_500_2100.parquet')
df_up = pd.read_parquet(r'data/predictions/pred_spectra_predictions_ch_1900_3500.parquet')

n_peak_down = pd.read_pickle(r'data/predictions/pred_pred_num_peak_down.pickle')
n_peak_down = n_peak_down['test'].rename(columns={'TRUE_NUM_PEAK': 'raman_true_num_peak_down'})
n_peak_down = n_peak_down.rename(columns={'PRED_NUM_PEAK': 'raman_pred_num_peak_down'})

n_peak_up = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')
n_peak_up = n_peak_up['test'].rename(columns={'TRUE_NUM_PEAK': 'raman_true_num_peak_up'})
n_peak_up = n_peak_up.rename(columns={'PRED_NUM_PEAK': 'raman_pred_num_peak_up'})

# Add number of peaks predicted to use keep_peaks_prom
df_down = pd.merge(df_down, n_peak_down[['SMILE', 'raman_pred_num_peak_down']], left_on='smile', right_on='SMILE',
                   how='left')
df_up = pd.merge(df_up, n_peak_up[['SMILE', 'raman_pred_num_peak_up']], left_on='smile', right_on='SMILE', how='left')
df_down = df_down.drop(['SMILE', 'pred_num_peak'], axis=1)
df_up = df_up.drop(['SMILE', 'pred_num_peak'], axis=1)

# Rescale raman_pred and raman_true
df_down['raman_pred'] = df_down['raman_pred'].apply(lambda row: rescale(range(0, 800), row))
df_up['raman_pred'] = df_up['raman_pred'].apply(lambda row: rescale(range(0, 800), row))

# Apply keep_peaks_prom
df_down['raman_pred'] = df_down.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.raman_pred_num_peak_down)),
                                      axis=1)
df_up['raman_pred'] = df_up.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.raman_pred_num_peak_up)),
                                  axis=1)

# Merge into a unique Dataframe for all Spectrum, first 700 down + 200 mean up&down + last 700
result_df = process_arrays(df_down, df_up, 'raman_pred', 'raman_pred', 'raman_pred')

result_df['raman_pred_conv'] = result_df.apply(lambda row:
                                               convolve_with_lorentzian(row['raman_pred'], lorentzian_kernel), axis=1)

final_output = result_df.copy()
