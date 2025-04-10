import pandas as pd
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel, make_conv_matrix, \
    convolve_with_lorentzian
import numpy as np


gamma = 7.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(301, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)


df = pd.read_csv('data/results/chemprop_raw_results/chemprop_ir_30cm_conv_pred_300_3500.csv')

dtf_spectrum = pd.read_pickle(r'data/raw/test_dtf_data_smile_no_dup_no_conv.pickle')

dtf_spectrum['RAMAN_SPECTRUM'] = dtf_spectrum['RAMAN_SPECTRUM'].apply(lambda row: row[100:])
dtf_spectrum['RAMAN_SPECTRUM_CONV'] = dtf_spectrum.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel), axis=1)


smiles_column = df.columns[0]
intensity_columns = df.columns[1:]

# Create the new DataFrame
new_df = pd.DataFrame({
    'SMILES': df[smiles_column],
    'raman_pred': df[intensity_columns].apply(lambda row: np.array(row.tolist()), axis=1)
})
new_df = new_df.merge(dtf_spectrum[['SMILE', 'RAMAN_SPECTRUM', 'RAMAN_SPECTRUM_CONV']], left_on='SMILES', right_on='SMILE', how='inner')
new_df = new_df.drop('SMILE', axis=1)
new_df.reset_index(drop=True, inplace=True)

new_df.to_pickle(r'data/results/chemprop_predictions/chemprop_ir_30cm_conv_pred_300_3500.pickle')