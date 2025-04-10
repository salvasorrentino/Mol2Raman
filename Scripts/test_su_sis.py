import pandas as pd
from utils import sim_dir, generate_lorentzian_kernel, make_conv_matrix, f1_score_mod
import matplotlib.pyplot as plt

gamma = 2.5
kernel_size = 200
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(301, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

dtf_tan = pd.read_parquet('data/mol_tanimoto_new_with_spectrum.parquet')
our_pred = pd.read_pickle(r'data/raw/dtf_full_spectra_predicted.pickle')

lst_use = list(set(our_pred['smile']) & set(dtf_tan['SMILE']))
# str_smile = lst_use[3000]

# # int_plot = 10
# plt.figure()
# plt.plot(dtf_tan.loc[dtf_tan['SMILE']==str_smile, 'tanimoto_spectra_5'].iloc[0])
# # plt.plot(dtf_tan['tanimoto_spectra_5'].iloc[int_plot])
# plt.plot(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0])
# plt.plot(our_pred[our_pred['SMILE']==str_smile]['raman_pred'].iloc[0])

# plt.legend(['tan_10', 'raman true', 'raman_pred'])
# plt.show()
for i, str_smile in enumerate(lst_use):
    # tan_sim_5 = sim_dir(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0], dtf_tan.loc[dtf_tan['SMILE']==str_smile, 'tanimoto_spectra_5'].iloc[0], conv, leng, lorentzian_kernel)
    # tan_f1_5 = f1_score_mod(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0],
    #                       dtf_tan.loc[dtf_tan['SMILE']==str_smile, 'tanimoto_spectra_5'].iloc[0])
    # dtf_tan.loc[dtf_tan[dtf_tan['SMILE'] == str_smile].index, 'sis_calc_5'] = tan_sim_5
    # dtf_tan.loc[dtf_tan[dtf_tan['SMILE'] == str_smile].index, 'tan_f1_5'] = tan_f1_5
    # tan_sim_10 = sim_dir(dtf_tan[dtf_tan['SMILE'] == str_smile]['RAMAN_SPECTRUM'].iloc[0],
    #                     dtf_tan.loc[dtf_tan['SMILE'] == str_smile, 'tanimoto_spectra_10'].iloc[0], conv, leng,
    #                     lorentzian_kernel)
    # tan_f1_10 = f1_score_mod(dtf_tan[dtf_tan['SMILE'] == str_smile]['RAMAN_SPECTRUM'].iloc[0],
    #                         dtf_tan.loc[dtf_tan['SMILE'] == str_smile, 'tanimoto_spectra_10'].iloc[0])
    # dtf_tan.loc[dtf_tan[dtf_tan['SMILE'] == str_smile].index, 'sis_calc_10'] = tan_sim_10
    # dtf_tan.loc[dtf_tan[dtf_tan['SMILE'] == str_smile].index, 'tan_f1_10'] = tan_f1_10
    our_sis = sim_dir(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0],
                      our_pred.loc[our_pred['SMILE']==str_smile, 'raman_pred'].iloc[0], conv, leng, lorentzian_kernel)
    our_f1 = f1_score_mod(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0],
                          our_pred.loc[our_pred['SMILE']==str_smile, 'raman_pred'].iloc[0])
    our_pred.loc[our_pred[our_pred['SMILE'] == str_smile].index, 'our_f1'] = our_f1
    our_pred.loc[our_pred[our_pred['SMILE'] == str_smile].index, 'our_sis'] = our_sis

    print(i)

    # pred_sim = sim_dir(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0],
    #                    our_pred[our_pred['SMILE']==str_smile]['raman_pred'].iloc[0], conv, leng, lorentzian_kernel)
    # pred_f1 = f1_score_mod(dtf_tan[dtf_tan['SMILE']==str_smile]['RAMAN_SPECTRUM'].iloc[0],
    #                        our_pred[our_pred['SMILE']==str_smile]['raman_pred'].iloc[0])
# print(f'tan sim: {tan_sim}')
# print(f'pred_sim: {pred_sim}')
# print(f'tan_f1: {tan_f1}')
# print(f'pred_f1: {pred_f1}')

our_pred.sort_values('our_sis', ascending=False, inplace=True)

from dtaidistance import dtw, similarity
sim = similarity.distance_to_similarity(dtw.distance_matrix((dtf_tan['tanimoto_spectra_10'].iloc[1000], dtf_tan['RAMAN_SPECTRUM'].iloc[1000])),
                                        method='reciprocal')
from numpy import dot
from numpy.linalg import norm
cos_sim = dot(dtf_tan['tanimoto_spectra_10'].iloc[1000], dtf_tan['RAMAN_SPECTRUM'].iloc[1000])/(norm(dtf_tan['RAMAN_SPECTRUM'].iloc[1000])*norm(dtf_tan['tanimoto_spectra_10'].iloc[1000]))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dtf_tan = pd.read_parquet('data/mol_tanimoto_new_with_spectrum.parquet')
dtf_data = pd.read_pickle('data/raw/dtf_data_smile_no_dup_no_conv.pickle')

dct_val = dtf_tan.iloc[10, :].to_dict()

lst_best = []
for str_key, val in dct_val.items():
    if 'top' in str_key:
        plt.figure()
        plt.plot(dtf_data[dtf_data['SMILE']==val]['RAMAN_SPECTRUM'].iloc[0])
        plt.title(f"{str_key}-{val}")
        plt.show()
        lst_best.append(dtf_data[dtf_data['SMILE']==val]['RAMAN_SPECTRUM'].iloc[0].tolist())
    if 'SMILE' in str_key:
        plt.figure()
        plt.plot(dtf_data[dtf_data['SMILE'] == val]['RAMAN_SPECTRUM'].iloc[0])
        plt.title(f"{str_key}-{val}")
        plt.show()


a = np.array(lst_best)
b = np.sum(a, axis=0)

plt.figure()
plt.plot(b)
plt.show()

plt.figure()
plt.plot(dct_val['tanimoto_spectra_10'])
plt.show()