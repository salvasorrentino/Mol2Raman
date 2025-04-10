import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel, convolve_with_lorentzian
from Scripts.utils_model.utils_metrics import make_conv_matrix, metrics_spectra, metrics_raman_peaks
import pickle
import scipy
from Scripts.utils_model.utils_plot import plot_two_spectrum


def plot_two_spectrum(true_s, pred_s, start, stop, rescale=3, line_width=3, title="Raman Spectrum (DFT-Calculated vs Predicted)"):
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    true = true_s
    pred = pred_s
    x = np.linspace(start, stop, len(true))

    peaks_true, _ = scipy.signal.find_peaks(true)
    peaks_pred, _ = scipy.signal.find_peaks(pred)

    # Rescale the intensity of the Fingerprint Region
    true[:750] = true[:750] * rescale
    pred[:750] = pred[:750] * rescale

    # Draw Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=100)

    ax.plot(x, true/sum(true), label='DFT-Calculated Spectrum', color=mycolors[1], linewidth=line_width)
    ax.plot(x, pred/sum(pred), label='Predicted Raman Spectrum', color=mycolors[0], linewidth=line_width)

    # Decorations
    ax.set_title(title, fontsize=90/3.75)
    ax.set_xlabel('Raman shift ($cm^{-1}$)', fontsize=75/3.75)
    ax.set_ylabel('Intensity (a.u.)', fontsize=75/3.75)
    ax.legend(loc='best', fontsize=50/3.75)
    # # ax.tick_params(axis='x', labelsize=12)
    # # ax.tick_params(axis='y', labelsize=12)
    plt.xticks(fontsize=50/3.75, horizontalalignment='center')
    plt.yticks(fontsize=50/3.75)
    plt.xlim(start, stop)
    plt.ylim(bottom=0)

    # # Thickness of the plot corner lines
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['right'].set_linewidth(1.5)

    # Draw Tick lines
    # for y in np.arange(0, max(max(true), max(pred)), step=0.1):
    #     plt.hlines(y, xmin=start, xmax=stop, colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


# Tanimoto analysis code
with open(r'data/tanimoto_filterd_list_daylight.pkl', 'rb') as file:
    dayligth_dict = pickle.load(file)

with open(r'data/tanimoto_filterd_list_morgan.pkl', 'rb') as file:
    morgan_dict = pickle.load(file)

model = 'spectra_predictions_ch_1900_3500_prova'
result_df = pd.read_pickle(rf'data/results/pred_spectra_mol2raman.pickle')

# Load a dataframe with the 10 most similar molecule (Tanimoto similiraty for Daylight or Morgan Fingerprint)
dtf_tan = pd.read_parquet('data/mol_tanimoto_daylight_fingerprint_with_spectrum.parquet')
# dtf_tan = pd.read_parquet('data/mol_tanimoto_morgan_fingerprint_with_spectrum.parquet')

## Load dictionaries with the test SMILEs as keys and the top 10 most similar molecules with the relative
## tanimoto similarity value
# dct_tan = pd.read_pickle('data/tanimoto_dictionary_morgan.pkl')
dct_tan = pd.read_pickle('data/tanimoto_dictionary_daylight.pkl')

## Filtered Dataframe with molecules which have values of Tanimoto similarity under a certain treshold
# result_df = result_df[result_df['smile'].isin(morgan_dict)]
# dtf_tan = dtf_tan[dtf_tan['SMILE'].isin(morgan_dict)]

# dtf_tan Metrics
gamma = 5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)
leng = list(range(501, 3500, 2))
conv = make_conv_matrix(std_dev=10, frequencies=leng)

dtf_tan['tanimoto_spectra_10'] = dtf_tan.apply(lambda row: row['tanimoto_spectra_10'][100:1600], axis=1)
dtf_tan['RAMAN_SPECTRUM'] = dtf_tan.apply(lambda row: row['RAMAN_SPECTRUM'][100:1600], axis=1)

dtf_tan['tanimoto_spectra_10_conv'] = dtf_tan.apply(lambda row: convolve_with_lorentzian(row['tanimoto_spectra_10'], lorentzian_kernel), axis=1)
dtf_tan['RAMAN_SPECTRUM_CONV'] = dtf_tan.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel),
                                                 axis=1)

dtf_tan = metrics_raman_peaks(dtf_tan, true='RAMAN_SPECTRUM', pred='tanimoto_spectra_10')
dtf_tan = metrics_spectra(dtf_tan, conv=conv, leng=leng, true_col='RAMAN_SPECTRUM_CONV', pred_col='tanimoto_spectra_10_conv')

# Save Tanimoto Dataframe
# dtf_tan.to_pickle(r'data/results/tanimoto_spectra.pickle')


# Plot of Tanimoto and prediciton
lst_use = list(set(result_df['smile']) & set(dtf_tan['SMILE']))
# str_use = lst_use[2000]
result_df.sort_values('F1_10cm^-1', ascending=False, inplace=True)
str_use = result_df.SMILE.iloc[100]
# str_use = r'CCC[C@@H](O)C(F)(F)F'


# Plot of True Spectrum and Tanimoto Spectrum
plt.figure()
plt.plot(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]/
         sum(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]))
plt.plot(dtf_tan[dtf_tan['SMILE']==str_use]['raman_pred_tan'].iloc[0]/
         sum(dtf_tan[dtf_tan['SMILE']==str_use]['raman_pred_tan'].iloc[0]))
plt.legend(['raman_true', 'raman_tan'])


fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=100)
x = np.linspace(500, 3500, len(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]))
ax.plot(x, result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]/
         sum(result_df[result_df['smile']==str_use]['RAMAN_SPECTRUM'].iloc[0]), label='DFT-Calculated Spectrum', color='tab:blue', linewidth=3)
ax.plot(x, dtf_tan[dtf_tan['SMILE']==str_use]['tanimoto_spectra_10_conv'].iloc[0]/
         sum(dtf_tan[dtf_tan['SMILE']==str_use]['tanimoto_spectra_10_conv'].iloc[0]), label='Tanimoto Similarity Spectrum', color='tab:green', linewidth=3)
ax.set_title('Raman Spectrum (DFT-Calculated vs Tanimoto)', fontsize=90/3.75)
ax.set_xlabel('Raman shift ($cm^{-1}$)', fontsize=75/3.75)
ax.set_ylabel('Intensity (a.u.)', fontsize=75/3.75)
ax.legend(loc='best', fontsize=50/3.75)
plt.xticks(fontsize=50/3.75, horizontalalignment='center')
plt.yticks(fontsize=50/3.75)
plt.xlim(500, 3500)
plt.ylim(bottom=0)
plt.gca().spines["top"].set_alpha(0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(0)
plt.gca().spines["left"].set_alpha(.3)
plt.show()

dct_tan_tmp = {k: list(v.values())[0] for k, v in dct_tan.items()}
# dtf_tan_tmp = pd.DataFrame.from_dict(dct_tan_tmp, orient='index')
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(list(range(len(dct_tan_tmp))), dtf_tan_tmp[0])
#
#
# fig, ax = plt.subplots(1, 1, figsize=(25, 15), dpi=300)
plt.figure()
plt.hist(result_df['F1_15cm^-1'], bins=34, alpha=0.85, color='blue')
plt.hist(dtf_tan['F1_15cm^-1'], bins=34, alpha=0.85, color='red')
plt.xlabel('F1 [15 $cm^{-1}$]')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions', 'Average on Tanimoto similarity'])
#
plt.figure()
plt.hist(result_df['F1_15cm^-1'], bins=34, alpha=0.85, color='blue')
plt.xlabel('F1 [15 $cm^{-1}$]')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions'])
plt.show()

plt.figure()
plt.hist(result_df[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='blue')
plt.hist(dtf_tan[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='red')
plt.xlabel('F1 [15 $cm^{-1}$]')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions', 'Average on Tanimoto similarity'])
plt.show()

dtf_chemprop_spectra = pd.read_pickle(r'data/results/pred_our_spectra_chemprop_raman_conv.pickle')
plt.figure()
plt.hist(result_df['F1_15cm^-1'], bins=34, alpha=0.85, color='blue')
plt.hist(dtf_chemprop_spectra['F1_15cm^-1'], bins=34, alpha=0.85, color='orange')
plt.xlabel('F1 [15 $cm^{-1}$]')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions', 'Chemprop_IR'])
plt.show()

plt.figure()
plt.hist(result_df[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='blue')
plt.hist(dtf_chemprop_spectra[result_df['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'], bins=34, alpha=0.85, color='orange')
plt.xlabel('F1 [15 $cm^{-1}$]')
plt.ylabel('Number of molecules')
plt.legend(['Model Predictions', 'Chemprop_IR'])
plt.show()
# dtf_data = pd.read_pickle('data/raw/dtf_data_smile_no_conv.pickle')
# plt.figure(dpi=200)
# # str_use = dtf_data.SMILE.iloc[9517]
# str_use = 'C[C@H]1OCCC(=C1)C'
# plt.plot(np.linspace(501, 3500, 1500),
#          dtf_data[dtf_data['SMILE'] == str_use]['RAMAN_SPECTRUM'].iloc[0][100:], color='tab:blue')
# (dtf_data[dtf_data['SMILE'] == str_use]['RAMAN_SPECTRUM'].iloc[0][100:]!=0).sum()
# plt.ylabel('Intensity (a.u.)')
# plt.xlabel('Raman shift ($cm^{-1}$)')

print('mol2raman F1_15 cm^-1:', result_df['F1_15cm^-1'].mean())
print('reduced mol2raman F1_15 cm^-1:', result_df[result_df['smile'].isin(dayligth_dict.keys())]['F1_15cm^-1'].mean())

print('tanimoto F1_15 cm^-1:', dtf_tan['F1_15cm^-1'].mean())
print('reduced tanimoto F1_15 cm^-1:', dtf_tan[dtf_tan['SMILE'].isin(dayligth_dict.keys())]['F1_15cm^-1'].mean())

print('chemprop_IR F1_15 cm^-1:', dtf_chemprop_spectra['F1_15cm^-1'].mean())
print('reduced chemprop_IR F1_15 cm^-1:', dtf_chemprop_spectra[dtf_chemprop_spectra['SMILES'].isin(dayligth_dict.keys())]['F1_15cm^-1'].mean())