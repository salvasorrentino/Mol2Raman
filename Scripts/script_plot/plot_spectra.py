import pandas as pd
import matplotlib.pyplot as plt
from Scripts.utils_model.utils_plot import plot_two_spectrum, plot_three_spectrum
from Scripts.utils_model.utils_metrics import keep_peaks_prom
import numpy as np

import matplotlib
matplotlib.use('QtAgg')

# Dataframe with the results for Chemprop, Mol2Raman and Tanimoto Similarity
dtf_chemprop_spectra = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\Results_'
            r'dtf\res_raman_chemprop_trained_on_our_raman_spectra_conv.pickle')
dtf_smile2ramam_spectra = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\Results_'
            r'dtf\res_raman_mol2raman_full_spectrum.pickle')
dtf_tanimoto = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\Results_'
            r'dtf\res_raman_tanimoto_weigthed_spectra.pickle')

# Dataframe with the ground truth spectra
dtf_data = pd.read_pickle('data/raw/dtf_data_smile.pickle')

# Dataframe of Mol2Raman prediction for Fingerprint and CH regions
dtf_smile2ramam_ch_spectra = pd.read_parquet(r'data/predictions/pred_spectra_predictions_ch_1900_3500_feat_numpeak_'
                                             r'daylight_morgan_loss_8651.parquet')
dtf_smile2ramam_fingerprint_spectra = pd.read_parquet(r'data/predictions/pred_spectra_predictions_fingerprint_'
                                                      r'500_2100_feat_numpeak_daylight_morgan_loss_8651.parquet')

# Add a column for further analysis of spectra post-processing via promininence reduction
dtf_smile2ramam_fingerprint_spectra['raman_pred_peak_prom'] = (
    dtf_smile2ramam_fingerprint_spectra.apply(lambda row:
                                              keep_peaks_prom(row.raman_pred, round(row.pred_num_peak)), axis=1))
dtf_smile2ramam_ch_spectra['raman_pred_peak_prom'] = (
    dtf_smile2ramam_ch_spectra.apply(lambda row: keep_peaks_prom(row.raman_pred, round(row.pred_num_peak)), axis=1))


# Check if the test SMILEs are the same
lst_use = list(set(dtf_smile2ramam_spectra['smile']) & set(dtf_chemprop_spectra['SMILES']) & set(dtf_tanimoto['SMILE']))


# Sort the detaframe by metric value
dtf_smile2ramam_spectra.sort_values('F1_10cm^-1', ascending=False, inplace=True)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex']      = False
# plt.rcParams['font.family']      = 'serif'
# plt.rcParams['font.serif']       = 'cm'
plt.rcParams['lines.markersize'] = 4
plt.rcParams['font.size']        = 15
plt.rcParams['figure.constrained_layout.use'] = True

# Distributions confrontation
plt.figure(figsize=(10, 7))
plt.hist(dtf_smile2ramam_spectra['F1_15cm^-1'], bins=65, color='tab:red', alpha=1, label='Mol2Raman')
plt.hist(dtf_tanimoto['F1_15cm^-1'], bins=65, color='tab:blue', alpha=0.8, label='Tanimoto \nbenchmark')
plt.xlabel('F1 (15 $cm^{-1}$)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc='best', fontsize=fontsize)
plt.savefig(dpi=300, fname=r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\SS_PHD_dissertation_material\PhD '
                           r'Thesis Images\f1_15_full_histogram_2.png')
plt.show()

tan_filt = pd.read_pickle(r'data/tanimoto_filterd_list_daylight.pkl')

lst_smile = list(tan_filt.keys())

# plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(10, 7))
plt.hist(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'].isin(lst_smile)]['F1_15cm^-1'], bins=34,
         color='tab:red', alpha=1, label='Mol2Raman')
plt.hist(dtf_tanimoto[dtf_tanimoto['SMILE'].isin(lst_smile)]['F1_15cm^-1'], bins=34,
         color='tab:blue', alpha=0.8, label='Tanimoto \nbenchmark')
plt.xlabel('F1 (15 $cm^{-1}$)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc='best', fontsize=fontsize)
plt.savefig(dpi=300, fname=r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\SS_PHD_dissertation_'
                           r'material\PhD Thesis Images\mol2raman-tanimoto_comparison_filtered_2.png')
plt.show()

fontsize = 20
plt.figure(figsize=(10, 7))
plt.hist(dtf_smile2ramam_spectra['F1_15cm^-1'], bins=65,
         color='tab:red', alpha=1, label='Mol2Raman')
plt.hist(dtf_chemprop_spectra['F1_15cm^-1'], bins=65,
         color='tab:cyan', alpha=0.8, label='Chemprop \nbenchmark')
plt.xlabel('F1 (15 $cm^{-1}$)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc='best', fontsize=fontsize)
plt.savefig(dpi=300, fname=r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\SS_PHD_dissertation_'
                           r'material\PhD Thesis Images\mol2raman-chemprop_comparison_2.png')
plt.show()

fig, ax = plt.subplots(1, 1, dpi=240)
pred = dtf_smile2ramam_spectra.iloc[155]['raman_pred_conv']
x = np.linspace(501, 3500, len(pred))
plt.plot(x, pred/pred.sum())

# Decorations
ax.set_title('Raman Spectrum (DFT-Calculated vs Predicted)', fontsize=18/3.75)
ax.set_xlabel('Raman shift ($cm^{-1}$)', fontsize=25/3.75)
ax.set_ylabel('Intensity (a.u.)', fontsize=25/3.75)
# ax.legend(loc='best', fontsize=18/3.75)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
plt.xticks(fontsize=25/3.75, horizontalalignment='center')
plt.yticks(fontsize=25/3.75)
plt.xlim(501, 3500)
plt.ylim(bottom=0)
plt.savefig(dpi=300, fname=r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\SS_PHD_dissertation_'
                           r'material\PhD Thesis Images\mol2raman_res_example.png')


from Scripts.utils_model.utils_metrics import metrics_spectra
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel,  rescale, \
    convolve_with_lorentzian
from Scripts.utils_model.utils_metrics import make_conv_matrix, metrics_spectra, metrics_raman_peaks, keep_peaks_prom

gamma = 2.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(501, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

prova = metrics_spectra(dtf_smile2ramam_spectra, conv=conv, leng=leng)
prova_chem = metrics_spectra(dtf_chemprop_spectra, true_col='RAMAN_SPECTRUM_CONV',
                             pred_col='raman_pred',conv=conv, leng=leng)


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex'] = False
# plt.rcParams['font.family']      = 'serif'
# plt.rcParams['font.serif']       = 'cm'
plt.rcParams['lines.markersize'] = 4
plt.rcParams['font.size'] = 20
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.fontsize'] = 'large'

# colors = ['yellow', 'magenta']
# cmap = LinearSegmentedColormap.from_list('CustomMap', colors, N=2)
# plot of a prediction of Mol2Raman vs ground truth
dtf_smile2ramam_spectra.sort_values("F1_15cm^-1", inplace=True, ascending=False)
str_use = dtf_smile2ramam_spectra.SMILE.loc[2452]
print(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['F1_15cm^-1'])
# print(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['smile'].iloc[0])
plot_two_spectrum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]/
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM'].iloc[0]),
              dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0] /
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred'].iloc[0]), 501, 3500, fill=False,
              rescale=2)


# plot of a prediction of Mol2Raman vs ground truth (convolved)
plot_two_spectrum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]/
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]),
              dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred_conv'].iloc[0] /
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['raman_pred_conv'].iloc[0]),
                  501, 3500, fill=False,
              rescale=2, line_width=8, fontsize=40)

print(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['F1_15cm^-1'])
plt.savefig(dpi=300, fname=r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\SS_PHD_dissertation_'
                           r'material\PhD Thesis Images\mol2raman_result_40_large.png')


# plot comparison with Tanimoto Benchmark
plot_two_spectrum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]/
              sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]),
              dtf_tanimoto[dtf_tanimoto['SMILE'] == str_use]['tanimoto_weigthed_spectra_10_conv'].iloc[0] /
              sum(dtf_tanimoto[dtf_tanimoto['SMILE'] == str_use]['tanimoto_weigthed_spectra_10_conv'].iloc[0]),
                  501, 3500, fill=False,
              rescale=1)


# plot comparison with Chemprop IR Benchmark
plot_two_spectrum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['RAMAN_SPECTRUM_CONV'].iloc[0]),
            np.array(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0])/
            sum(dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'] == str_use]['raman_pred'].iloc[0]), 501,
                  3500, fill=False)


# plot comparison of peaks reduction via prominence (CH and Fingerprint region separately)
i = 7
print(dtf_smile2ramam_fingerprint_spectra.iloc[i].smile)
print(dtf_smile2ramam_fingerprint_spectra.iloc[i].pred_num_peak)
plt.figure(figsize=(16, 9), dpi=150)
x = np.linspace(501, 2100, len(dtf_smile2ramam_fingerprint_spectra.iloc[i].raman_pred))
plt.plot(x, dtf_smile2ramam_fingerprint_spectra.iloc[i].raman_pred, label='Predicted Spectrum')
plt.plot(x, dtf_smile2ramam_fingerprint_spectra.iloc[i].raman_pred_peak_prom, label='Predicted Spectrum with post-processing')
plt.xlim(501, 2100)
plt.ylim(bottom=0)
plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=45 / 3.75)
plt.ylabel('Intensity (a.u.)', fontsize=45 / 3.75)
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
plt.title('Fingerprint Region - Raman Spectrum', fontsize=60 / 3.75)
plt.legend()
plt.show()

print(dtf_smile2ramam_ch_spectra.iloc[i].smile)
print(dtf_smile2ramam_ch_spectra.iloc[i].pred_num_peak)

plt.figure(figsize=(16, 9), dpi=150)
x = np.linspace(501, 2100, len(dtf_smile2ramam_ch_spectra.iloc[i].raman_pred))
plt.plot(x, dtf_smile2ramam_ch_spectra.iloc[i].raman_pred,  label='Predicted Spectrum')
plt.plot(x, dtf_smile2ramam_ch_spectra.iloc[i].raman_pred_peak_prom, label='Predicted Spectrum with post-processing')
plt.xlim(501, 2100)
plt.ylim(bottom=0)
plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=45 / 3.75)
plt.ylabel('Intensity (a.u.)', fontsize=45 / 3.75)
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
plt.title('CH Region - Raman Spectrum', fontsize=60 / 3.75)
plt.legend()
plt.show()


from scipy.stats import mannwhitneyu

mannwhitneyu(dtf_smile2ramam_spectra['F1_15cm^-1'], dtf_chemprop_spectra['F1_15cm^-1'])