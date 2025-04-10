import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pickle
from rdkit.Chem import rdFingerprintGenerator


def filter_smiles_by_similarity_threshold(similarity_dict, threshold):
    filtered_dict = {}

    for main_smile, similarities in similarity_dict.items():
        if all(first_similarity <= threshold for first_smile, first_similarity in similarities.items()):
            filtered_dict[main_smile] = similarities

    return filtered_dict


def smiles_to_fp(smile, fingerprint):
    if fingerprint == 'daylight':
        fpgen = AllChem.GetRDKitFPGenerator()
        fp = fpgen.GetFingerprint(Chem.MolFromSmiles(smile))
    elif fingerprint == 'morgan':
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smile))
    return fp


fingerprint = ['daylight', 'morgan']
treshold = [0.6, 0.4]

for fg, th in zip(fingerprint, treshold):
    dtf_tan = pd.read_parquet(rf'data/mol_tanimoto_{fg}_fingerprint_with_spectrum.parquet')
    dtf_tan = dtf_tan.drop(['tanimoto_spectra_10', 'tanimoto_spectra_5', 'RAMAN_SPECTRUM'], axis=1)

    result_dict = {}

    for index, row in dtf_tan.iterrows():
        main_smile = row['SMILE']
        main_fp = smiles_to_fp(main_smile, fg)

        top_smiles = row[1:11]
        top_fps = [smiles_to_fp(smile, fg) for smile in top_smiles]

        similarities = DataStructs.BulkTanimotoSimilarity(main_fp, top_fps)

        inner_dict = {smile: similarity for smile, similarity in zip(top_smiles, similarities)}
        result_dict[main_smile] = inner_dict

    with open(fr'data/tanimoto_dictionary_{fg}.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    # Filter data based on Tanimoto Similarity and a treshold
    filter_data = filter_smiles_by_similarity_threshold(result_dict, th)

    with open(fr'data/tanimoto_filterd_list_{fg}.pkl', 'wb') as f:
        data = pickle.dump(filter_data, f)

    print(f'Done dictionary for {fg} fingerprint')