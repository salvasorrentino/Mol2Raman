import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def number_of_peaks(spectrum):
    peaks,_ = find_peaks(spectrum)
    return round(len(peaks))


def n_atoms(smile):
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()
    return num_atoms


def n_bond(smile):
    mol = Chem.MolFromSmiles(smile)
    num_bonds = mol.GetNumBonds()
    return num_bonds


dtf_results = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')

dtf_full_dataset = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')
dtf_finger_peak = pd.read_pickle(r'data/predictions/pred_pred_num_peak_down.pickle')
dtf_ch_peak = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')

dtf_results['n_atoms'] = dtf_results.apply(lambda row: n_atoms(row['smile']), axis=1)
dtf_results['n_bond'] = dtf_results.apply(lambda row: n_bond(row['smile']), axis=1)
dtf_full_dataset['n_peaks'] = dtf_full_dataset.apply(lambda row: number_of_peaks(row['RAMAN_SPECTRUM']), axis=1)

plt.figure(figsize=(16, 9), dpi=150)
plt.hist(dtf_full_dataset['n_peaks'], bins=58)
plt.title('Distribution of molecule by number of peaks of the Raman spectrum', fontsize=60 / 3.75)
plt.xlabel('Number of peaks', fontsize=45 / 3.75)
plt.show()


plt.figure()
plt.scatter(dtf_results['n_atoms'], dtf_results['sis'], color='blue', label='Predizioni')
plt.xlabel('Number of Atoms')
plt.ylabel('Sis Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(dtf_results['n_bond'], dtf_results['sis'], color='blue', label='Predizioni')
plt.xlabel('Number of Bond')
plt.ylabel('Sis Value')
plt.legend()
plt.grid(True)
plt.show()