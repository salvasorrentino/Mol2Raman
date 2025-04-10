import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Scripts.utils import generate_lorentzian_kernel, make_conv_matrix, sim, rescale, f1_score_mod, \
    convolve_with_lorentzian, keep_peaks_prom, lorentz_conv, spectral_information_similarity, \
    sim_dir, precision_score_mod, recall_score_mod,  calc_cos_sim, fnr_score_mod
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
from Scripts.plot_spectra_function import plot_spectrum


# Load Dataframe number of peaks predicted and raman spectrum predicted
df_predictions_sp = pd.read_parquet(r'data/predictions/pred_IR_spectra_predictions_fingerprint_300_3500_feat_numpeak_morgan_pooling5_loss15381.parquet')
df_reaL_sp = pd.read_pickle(r'data/raw/test_dtf_data_IR_smile_no_dup_no_conv.pickle')
result_df = pd.merge(df_predictions_sp, df_reaL_sp[['SMILE', 'RAMAN_SPECTRUM']], left_on='smile', right_on='SMILE', how='inner')
result_df = result_df.drop('SMILE', axis=1)

# Rescale raman_pred and raman_true
result_df['raman_pred'] = result_df['raman_pred'].apply(lambda row: rescale(range(0, 1600), row))
result_df['raman_true'] = result_df['raman_true'].apply(lambda row: rescale(range(0, 1600), row))

# Apply keep_peaks_prom
result_df['raman_pred'] = result_df.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.pred_num_peak)), axis=1)

# Parameters for Metrics calculations
gamma = 5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(301, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

# Metrics F1
result_df['precision_30cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
result_df['precision_20cm^-1'] = result_df.apply(lambda row: precision_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

result_df['recall_30cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
result_df['recall_20cm^-1'] = result_df.apply(lambda row: recall_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

result_df['fnr_30cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
result_df['fnr_20cm^-1'] = result_df.apply(lambda row: fnr_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

result_df['F1_30cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=15), axis=1)
result_df['F1_20cm^-1'] = result_df.apply(lambda row: f1_score_mod(row['RAMAN_SPECTRUM'], row['raman_pred'], tolerance=10), axis=1)

# Metrics
result_df['sis'] = result_df.apply(lambda row: sim_dir(row['RAMAN_SPECTRUM'], row['raman_pred'], conv, freq=leng,
                                                       lor_kernel=lorentzian_kernel), axis=1)
# Convolve spectra with lorentzian curves
result_df['raman_pred'] = result_df.apply(lambda row: convolve_with_lorentzian(row['raman_pred'], lorentzian_kernel), axis=1)
result_df['RAMAN_SPECTRUM_CONV'] = result_df.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel), axis=1)

# Other Metrics: R^2, MAE, RMSE, Cosine Sim
result_df['R2'] = result_df.apply(lambda row: r2_score(row['RAMAN_SPECTRUM_CONV'], row['raman_pred']), axis=1)
result_df['MAE'] = result_df.apply(lambda row: mean_squared_error(row['RAMAN_SPECTRUM_CONV'], row['raman_pred']), axis=1)
result_df['RMSE'] = result_df.apply(lambda row: root_mean_squared_error(row['RAMAN_SPECTRUM_CONV'], row['raman_pred']), axis=1)
result_df['cos_sim'] = result_df.apply(lambda row: calc_cos_sim(row['raman_pred'], row['RAMAN_SPECTRUM_CONV']), axis=1)

# Save the IR dataframe
# result_df.to_pickle(r'data/results/pred_spectra_mol2raman_IR.pickle')

excel = [result_df['precision_20cm^-1'].mean(),  result_df['precision_30cm^-1'].mean(),
        result_df['recall_20cm^-1'].mean(), result_df['recall_30cm^-1'].mean(),
        result_df['fnr_20cm^-1'].mean(), result_df['fnr_30cm^-1'].mean(),
        result_df['F1_20cm^-1'].mean(), result_df['F1_30cm^-1'].mean(),
        result_df['sis'].mean(), result_df['cos_sim'].mean(),
        result_df['MAE'].mean(), result_df['RMSE'].mean(), result_df['R2'].mean()]

print('Mean F1_20cm^-1 score value:', result_df['F1_20cm^-1'].mean())
print('Mean F1_30cm^-1 score value:', result_df['F1_30cm^-1'].mean())
print('Mean sis value:', result_df.sis.mean())
print('Mean cosine sim value:', result_df.cos_sim.mean())

# # Sort Dataframe by SIS value and select a particular molecule
# result_df.sort_values('sis', ascending=False, inplace=True)
# str_use = result_df.smile.iloc[300]
#
# # Plot particular molecule
# plot_spectrum(result_df[result_df['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]/
#             sum(result_df[result_df['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]),
#             result_df[result_df['smile'] == str_use]['raman_pred'].iloc[0] /
#             sum(result_df[result_df['smile'] == str_use]['raman_pred'].iloc[0]), 501, 3500, fill=False)
#
# # Mean Spectrum and Plot
# trasposta = list(zip(*result_df['RAMAN_SPECTRUM']))
# media_liste = [sum(t) / len(t) for t in trasposta]
# plt.figure()
# plt.plot(media_liste)
# plt.show()
