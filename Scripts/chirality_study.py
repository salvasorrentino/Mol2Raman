import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pybel  # Open Babel
import avogadro


dtf_test = pd.read_pickle(r'data/raw/test_dtf_data_smile_no_dup_no_conv.pickle')
dtf_data_full = pd.read_pickle(r'data/raw/dtf_data_smile_no_dup_no_conv.pickle')

# Filtra le molecole che contengono almeno un "@"
molecules_with_chirality = dtf_test[dtf_test['SMILE'].str.contains('@')]

# Liste per memorizzare le molecole originali e modificate
original_molecules = []
modified_molecules = []

def modify_chirality(smile):
    # Sostituisci "@@" con un segnaposto temporaneo
    smile = smile.replace('@@', 'TEMP')
    # Sostituisci "@" con "@@"
    smile = smile.replace('@', '@@')
    # Sostituisci il segnaposto temporaneo con "@"
    smile = smile.replace('TEMP', '@')
    return smile

# Itera sulle molecole e modifica la chiralità
for smile in molecules_with_chirality['SMILE']:
    original_molecules.append(smile)
    modified_molecules.append(modify_chirality(smile))

lst_mod = dtf_data_full[dtf_data_full['SMILE'].isin(modified_molecules)]['SMILE'].to_list()


def smiles_to_avogadro(smiles):
    """Convert a SMILES string to an Avogadro Molecule"""
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)  # Add hydrogens

    # Generate 3D coordinates
    AllChem.EmbedMolecule(rdkit_mol, AllChem.ETKDG())

    # Convert to Open Babel molecule
    ob_mol = pybel.Molecule(rdkit_mol)

    # Convert to Avogadro molecule
    avo_mol = avogadro.Molecule()
    avo_mol.fromOBMol(ob_mol.OBMol)

    return avo_mol


import subprocess
import os
import tempfile

AVOGADRO_PATH = r"C:\Program Files (x86)\Avogadro\bin\avogadro.exe"


def smiles_to_orca(smiles_list, output_folder="orca_inputs"):
    """Convert a batch of SMILES strings to ORCA input files using Avogadro CLI."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, smiles in enumerate(smiles_list):
        # Creare un file temporaneo per il SMILES
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".smi") as temp_file:
            temp_file.write(smiles + "\n")  # Scrive il SMILES nel file
            temp_smiles_file = temp_file.name  # Salva il percorso del file temporaneo

        # Nome del file ORCA da generare
        output_file = os.path.join(output_folder, f"molecule_{i + 1}.inp")

        # Costruire il comando corretto
        command = [
            AVOGADRO_PATH,
            "--input", temp_smiles_file,  # Passa il file SMILES a Avogadro
            "--optimize",
            "--export", "orca",
            "-o", output_file
        ]

        try:
            subprocess.run(command, check=True)
            print(f"✔ Molecola {i + 1} salvata in {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore nella conversione di {smiles}: {e}")

        # Rimuovere il file temporaneo dopo l'uso
        os.remove(temp_smiles_file)


# Lista di SMILES da convertire
smiles_list = [
    "CCO",  # Etanolo
    "O=C[C@@]12CO[C@@H]1C(=O)N2",  # Molecola complessa
    "CC(=O)OC1=CC=CC=C1C(=O)O"  # Acido ftalico
]

# Esegui la conversione batch
smiles_to_orca(smiles_list)

# Example
smiles_to_orca("O=C[C@@]12CO[C@@H]1C(=O)N2", "prova_orca.inp")


# Example usage
mol = smiles_to_avogadro("CCO")  # Ethanol
print(f"Number of Atoms: {mol.numAtoms()}")


def generate_orca_input(mol, filename="orca_input.inp", method="B3LYP", basis_set="6-31G(d)"):
    """Generate an ORCA input file from an Avogadro Molecule"""
    xyz_data = mol.toXYZ()

    with open(filename, "w") as f:
        f.write(f"! {method} {basis_set}\n")
        f.write("* xyz 0 1\n")  # Assuming neutral molecule, singlet state
        f.write(xyz_data)
        f.write("*\n")

    print(f"ORCA input file saved as {filename}")


# Example usage
generate_orca_input(mol, "ethanol_orca.inp")

import os

str_path = r"D:\Documents\Orca_software\calculate_molecules"

for i in range(1, 100):
    str_folder = f"geometry_{i}"
    os.makedirs(os.path.join(str_path, str_folder), exist_ok=True)

import os

# List of SMILES strings
smiles_list = [
    "CCO",  # Example SMILES
    "O=C[C@@]12CO[C@@H]1C(=O)N2",
    "CC(=O)OC1=CC=CC=C1C(=O)O"
]

# Base path for the folders
base_path = r"D:\Documents\Orca_software\calculate_molecules"
smiles_list = pd.read_pickle(r'data/raw/chiral_modified_molecules.pickle')

# Iterate over the SMILES list and corresponding folder index
for i, smile in enumerate(smiles_list):
    folder_name = f"geometry_{i}"
    folder_path = os.path.join(base_path, folder_name)

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    pd.to_pickle(smile, os.path.join(folder_path, 'smile.pickle'))

dtf_smile2ramam_spectra = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\Results_'
            r'dtf\res_raman_mol2raman_full_spectrum.pickle')

dtf_smile2ramam_spectra_use = dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['SMILE'].isin(original_molecules[:50])]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
plt.figure()
plt.hist(dtf_smile2ramam_spectra_use['F1_15cm^-1'], bins=50, color='tab:red', alpha=0.8, label='Model Predictions',
         density=True)
plt.hist(dtf_smile2ramam_spectra['F1_15cm^-1'], bins=50, color='tab:blue', alpha=0.8, label='Model Predictions All',
         density=True)
plt.show()


import pandas as pd
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel, convolve_with_lorentzian


gamma = 2.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)

type_data = 'raman'     # 'raman' or 'IR'
train = pd.read_pickle(rf'C:\Users\Utente\OneDrive - Politecnico di Milano\script_AI'
                       rf'_Raman\AI_Chemistry\data\raw\dtf_data_raman_smile_no_conv_chiral.pickle')
dtf_smile2ramam_spectra = pd.read_pickle(r'C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\Results_'
            r'dtf\res_raman_mol2raman_full_spectrum.pickle')

train['WAVELENGTH'] = train['WAVELENGTH'].apply(lambda row: row[100:])
train['RAMAN_SPECTRUM'] = train['RAMAN_SPECTRUM'].apply(lambda row: row[100:])
train['RAMAN_SPECTRUM_CONV'] = train.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel), axis=1)

import matplotlib.pyplot as plt

import matplotlib
import numpy as np
matplotlib.use('QtAgg')
int_use = 50
smile_use = train.iloc[int_use]['SMILE']
print(smile_use)
smile_mod = modify_chirality(smile_use)

plt.figure()
# plt.plot(train.iloc[int_use]['WAVELENGTH'],
#          dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']==smile_mod]['raman_pred_conv'].iloc[0]/
#          np.sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']==smile_mod]['raman_pred_conv'].iloc[0]), label='Model Predictions')
# plt.plot(train.iloc[int_use]['WAVELENGTH'],
#          dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']==smile_mod]['raman_pred_conv'].iloc[0], label='Model Predictions')
plt.plot(train.iloc[int_use]['WAVELENGTH'],
         dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']==smile_mod]['RAMAN_SPECTRUM_CONV'].iloc[0]/
         np.sum(dtf_smile2ramam_spectra[dtf_smile2ramam_spectra['smile']==smile_mod]['RAMAN_SPECTRUM_CONV'].iloc[0]), label='True Spectrum')
plt.plot(train.iloc[int_use]['WAVELENGTH'], train.iloc[int_use]['RAMAN_SPECTRUM_CONV']/np.sum(train.iloc[int_use]['RAMAN_SPECTRUM_CONV']),
         label="True chiral")
plt.legend()
plt.show()


## handling chiral spectra

import os
str_path = r"D:\Documents\Orca_software\calculate_molecules"
dct_spectra = {}
for i in range(50):
    str_folder = f"geometry_{i}"
    str_file = os.path.join(str_path, str_folder, 'raman_spectrum.csv')
    dtf_spectra = pd.read_csv(str_file, sep=';', decimal=',')
    dtf_spectra.set_index('smile', inplace=True)
    dtf_spectra = dtf_spectra[:].astype(float)
    key = dtf_spectra.index[0].replace("\n", "")
    dct_spectra.update({f'{key}': dtf_spectra.iloc[0].tolist()})
    # lst_spectra.append(dtf_spectra)


dtf_final = pd.DataFrame(columns=['SMILE', 'RAMAN_PRED_CONV'])
dtf_final['SMILE'] = dct_spectra.keys()
dtf_final['RAMAN_PRED_CONV'] = dct_spectra.values()

train_all = pd.merge(train, dtf_final, left_on='SMILE', right_on='SMILE', how='inner')

set(list(train.SMILE))-set(list(dtf_final.SMILE))

set(list(dtf_final.SMILE)) - set(list(train.SMILE))

train_all['SMILE_OR'] = train_all['SMILE'].apply(lambda row: modify_chirality(row))

train_all['RAMAN_PRED_CONV_OR'] = (
    dtf_smile2ramam_spectra[['smile', 'raman_pred_conv']].merge(train_all, left_on='smile',
                                                                right_on='SMILE_OR', how='inner'))['raman_pred_conv']
dtf_smile2ramam_spectra['RAMAN_SPECTRUM_CONV_2'] = dtf_smile2ramam_spectra['RAMAN_SPECTRUM_CONV']
train_all['RAMAN_SPECTRUM_CONV_OR'] = (
    dtf_smile2ramam_spectra[['smile', 'RAMAN_SPECTRUM_CONV_2']].merge(train_all, left_on='smile',
                                                                right_on='SMILE_OR', how='inner'))['RAMAN_SPECTRUM_CONV_2']

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

train_all['cos_sim_pred'] = train_all.apply(lambda row: cosine_similarity([row['RAMAN_PRED_CONV']],
                                                                          [row['RAMAN_PRED_CONV_OR']])[0][0], axis=1)
train_all['cos_sim'] = train_all.apply(lambda row: cosine_similarity([row['RAMAN_SPECTRUM_CONV']],
                                                                          [row['RAMAN_SPECTRUM_CONV_OR']])[0][0], axis=1)

res = r2_score(train_all['cos_sim'], train_all['cos_sim_pred'])

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('QtAgg')

plt.figure()
plt.scatter(train_all['cos_sim'], train_all['cos_sim_pred'], s=80)
plt.xlabel('Cosine similarity \nDFT enantiomer')
plt.ylabel('Cosine similarity \nPrediction enantiomer')
plt.show()

pearsonr(train_all['cos_sim'], train_all['cos_sim_pred'])
spearmanr(train_all['cos_sim'], train_all['cos_sim_pred'])

plot_two_spectrum(train_all.loc[32, 'RAMAN_SPECTRUM_CONV']/
              sum(train_all.loc[32, 'RAMAN_SPECTRUM_CONV']),
              train_all.loc[32, 'RAMAN_SPECTRUM_CONV_OR'] /
              sum(train_all.loc[32, 'RAMAN_SPECTRUM_CONV_OR']),
                  501, 3500, fill=False,
              rescale=2, line_width=8, fontsize=40)

plot_two_spectrum(np.array(train_all.loc[32, 'RAMAN_PRED_CONV'])/sum(train_all.loc[32, 'RAMAN_PRED_CONV']),
              train_all.loc[32, 'RAMAN_PRED_CONV_OR']/sum(train_all.loc[32, 'RAMAN_PRED_CONV_OR']),
                  501, 3500, fill=False,
              rescale=2, line_width=8, fontsize=40)