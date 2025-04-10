import pandas as pd
import numpy as np

dtf_num_peak_down = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')

dtf_full = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')

dtf_test = pd.read_pickle(r'data/raw/test_dtf_data_smile_no_dup_no_conv.pickle')

true_values_up = dtf_test['RAMAN_PEAK_NUM_UP'].astype(int)
predicted_values_up = dtf_test['PRED_NUM_PEAK_UP'].astype(int)
rmse_up = np.sqrt(mean_squared_error(true_values_up, predicted_values_up))
print("RMSE:", rmse_up)

r2_up = r2_score(true_values_up, predicted_values_up)
print("R^2:", r2_up)


true_values_up = dtf_num_peak_down['test']['TRUE_NUM_PEAK'].astype(int)
predicted_values_up = dtf_num_peak_down['test']['PRED_NUM_PEAK'].astype(int)
rmse_up = np.sqrt(mean_squared_error(true_values_up, predicted_values_up))
print("RMSE:", rmse_up)

accuracy = accuracy_score(true_values_up, predicted_values_up)
print("accuracy:", accuracy)
