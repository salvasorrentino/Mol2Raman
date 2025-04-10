from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator

fingerprint = 'morgan'

# read and Conconate the csv's
df_1 = pd.read_pickle('data/raw/dtf_data_smile_no_dup_no_conv.pickle')
df_1 = df_1.drop_duplicates(['SMILE'])
dtf_test = pd.read_parquet('data/predictions/pred_spectra_predictions_ch_1900_3500_feat_numpeak_daylight_morgan'
                            '_loss_8651.parquet')
lst_smile_test = dtf_test.smile.tolist()

# proof and make a list of SMILES
ser_smiles = df_1['SMILE']
lst_smiles = list(set(ser_smiles))

if fingerprint == 'daylight':
    fpgen = AllChem.GetRDKitFPGenerator()
    dct_fingerprint = {k: fpgen.GetFingerprint(Chem.MolFromSmiles(k)) for k in lst_smiles}
elif fingerprint == 'morgan':
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    dct_fingerprint = {k: gen.GetFingerprint(Chem.MolFromSmiles(k))
                       for k in lst_smiles}

top_similarities = {}

for i, str_smile in enumerate(dct_fingerprint.keys()):
    if str_smile not in lst_smile_test:
        continue
    similarities = DataStructs.BulkTanimotoSimilarity(dct_fingerprint[str_smile],
                                                      list(dct_fingerprint.values()))
    top_similarities[str_smile] = list(zip(dct_fingerprint.keys(), similarities))
    if i % 100 == 0:
        print(i)

# Find the top 10 similar SMILES for each SMILE
for i, smile in enumerate(top_similarities):
    top_similarities[smile].sort(key=lambda x: x[1], reverse=True)
    top_similarities[smile] = top_similarities[smile][:11]
    print(i)
    try:
        top_similarities[smile].remove((smile, 1))
    except ValueError as e:
        print('empty list for smile:', smile)
        continue

# Create a DataFrame
columns = ['SMILE'] + [f'top_{i + 1}' for i in range(10)]
data = []

for smile, sims in top_similarities.items():
    row = [smile] + [sim[0] for sim in sims]
    data.append(row)

df_top_similarities = pd.DataFrame(data, columns=columns)

# save
# df_top_similarities.to_parquet(rf'data/mol_tanimoto_{fingerprint}_fingerprint.parquet', index=False)


def tan_spectra_mean(target, datas, int_comparison):
    spectra = []
    for smile in target[1:int_comparison+1]:
        sp = datas[datas.SMILE == smile].RAMAN_SPECTRUM
        if sp.empty:
            print(f"No match found for SMILE: {smile}")
        else:
            # print(f"Match found for SMILE: {smile}, Spectrum: {sp.values[0]}")
            spectra.append(sp.values[0])
    if not spectra:
        return []
    full_sp = [sum(elements) for elements in zip(*spectra)]
    full_sp_mean = [x / len(spectra) for x in full_sp]
    return full_sp_mean


df = pd.read_parquet(rf'data/mol_tanimoto_{fingerprint}_fingerprint.parquet')
data = pd.read_pickle('data/raw/dtf_data_smile_no_dup_no_conv.pickle')

df['tanimoto_spectra_10'] = df.apply(lambda row: tan_spectra_mean(row, data, 10), axis=1)
df['tanimoto_spectra_5'] = df.apply(lambda row: tan_spectra_mean(row, data, 5), axis=1)

df = df.merge(data[['SMILE', 'RAMAN_SPECTRUM']], how='left', on='SMILE', validate='one_to_one')

df.to_parquet(rf'data/mol_tanimoto_{fingerprint}_fingerprint_with_spectrum.parquet')

