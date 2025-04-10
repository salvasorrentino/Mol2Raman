import pandas as pd
from Scripts.utils_model.utils_metrics import make_conv_matrix, metrics_spectra, metrics_raman_peaks
from Scripts.utils_model.utils_data_processing import post_processing_pred


# Dataframe for computed spectra
raman_spectrum = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')

# result_df = pd.read_parquet(r'data/predictions/pred_spectra_predictions_fingerprint_500_2100_feat_numpeak_loss_8651.parquet')
# raman_spectrum['RAMAN_SPECTRUM_1'] = raman_spectrum.apply(lambda row: row.RAMAN_SPECTRUM[100:900], axis=1)
# leng = list(range(501, 2100, 2))

# Smile2Raman prediction
model = 'spectra_predictions_ch_1900_3500'
result_df = pd.read_parquet(rf'data/predictions/pred_{model}.parquet')
raman_spectrum['RAMAN_SPECTRUM_1'] = raman_spectrum.apply(lambda row: row.RAMAN_SPECTRUM[800:1600], axis=1)
leng = list(range(1901, 3500, 2))

# Generate a dataframe with prediction a computed spectra
result_df = pd.merge(result_df, raman_spectrum, left_on='smile', right_on='SMILE', how='inner')
result_df = result_df.drop('SMILE', axis=1)

conv = make_conv_matrix(std_dev=10, frequencies=leng)

result_df = post_processing_pred(result_df)

result_df = metrics_raman_peaks(result_df)

result_df = metrics_spectra(result_df, conv, leng)

# save in the data\results\ directory
# result_df.to_pickle(rf'data\results\res_{model}.pickle')