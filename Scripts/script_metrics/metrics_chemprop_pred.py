import pandas as pd
from Scripts.utils_model.utils_metrics import make_conv_matrix, \
    metrics_raman_peaks, metrics_spectra, excel_data_raman


chemprop_prediction = pd.read_pickle(r'data/results/chemprop_predictions/chemprop_raman_10cm_conv_pred.pickle')
our_prediction = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')

result_df = chemprop_prediction

leng = list(range(400, 4001, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

result_df = metrics_raman_peaks(result_df)

result_df = metrics_spectra(result_df, conv, leng, true_col='RAMAN_SPECTRUM')

# result_df.to_pickle(r'data/results/chemprop_ir_their_model_their_computed_mol.pickle')





