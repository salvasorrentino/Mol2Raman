import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


def add_external_feature(dtf_in, dtf_feat, str_feat, str_rename):

    dtf_out = dtf_in.copy()
    dtf_out = dtf_out.merge(dtf_feat[['SMILE', str_feat]], on='SMILE', how='inner')
    dtf_out.rename({str_feat: str_rename}, axis=1, inplace=True)

    return dtf_out


def add_daylight_fingerprint(dtf_in):

    dtf_out = dtf_in.copy()
    lst_smiles = dtf_out.SMILE.tolist()

    fpgen = AllChem.GetRDKitFPGenerator(fpSize=2048)
    dct_fingerprint = {k: [list(fpgen.GetFingerprint(Chem.MolFromSmiles(k)))]
                       for k in lst_smiles}

    dtf_tmp = pd.DataFrame.from_dict(dct_fingerprint, orient='index', columns=['MOLECULAR_FINGERPRINT'])
    dtf_out = dtf_out.merge(dtf_tmp, how='left', left_on='SMILE', right_index=True)

    return dtf_out


def add_morgan_fingerprint(dtf_in):

    dtf_out = dtf_in.copy()
    lst_smiles = dtf_out.SMILE.tolist()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    dct_fingerprint = {k: [list(gen.GetFingerprint(Chem.MolFromSmiles(k)))]
                       for k in lst_smiles}

    dtf_tmp = pd.DataFrame.from_dict(dct_fingerprint, orient='index', columns=['MOLECULAR_FINGERPRINT_MORGAN'])
    dtf_out = dtf_out.merge(dtf_tmp, how='left', left_on='SMILE', right_index=True)

    return dtf_out



if __name__ == '__main__':

    dct_pred_num_peak_down = pd.read_pickle(r'data/predictions/pred_pred_num_peak_down.pickle')
    dct_pred_num_peak_up = pd.read_pickle(r'data/predictions/pred_pred_num_peak_up.pickle')

    dtf_test = pd.read_pickle(r'data/raw/test_dtf_data_smile_no_dup_no_conv.pickle')
    dtf_train = pd.read_pickle(r'data/raw/train_dtf_data_smile_no_dup_no_conv.pickle')
    dtf_validation = pd.read_pickle(r'data/raw/validation_dtf_data_smile_no_dup_no_conv.pickle')

    dtf_test = add_external_feature(dtf_test, dct_pred_num_peak_down['test'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_DOWN')
    dtf_test = add_external_feature(dtf_test, dct_pred_num_peak_up['test'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_UP')

    dtf_validation = add_external_feature(dtf_validation, dct_pred_num_peak_down['validation'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_DOWN')
    dtf_validation = add_external_feature(dtf_validation, dct_pred_num_peak_up['validation'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_UP')

    dtf_train = add_external_feature(dtf_train, dct_pred_num_peak_down['train'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_DOWN')
    dtf_train = add_external_feature(dtf_train, dct_pred_num_peak_up['train'],
                                    'PRED_NUM_PEAK', 'PRED_NUM_PEAK_UP')

    dtf_train = add_morgan_fingerprint(dtf_train)
    dtf_test = add_morgan_fingerprint(dtf_test)
    dtf_validation = add_morgan_fingerprint(dtf_validation)

    dtf_train = add_daylight_fingerprint(dtf_train)
    dtf_test = add_daylight_fingerprint(dtf_test)
    dtf_validation = add_daylight_fingerprint(dtf_validation)

    pd.to_pickle(dtf_test, r'data/raw/test_dtf_data_smile_no_dup_no_conv.pickle')
    pd.to_pickle(dtf_train, r'data/raw/train_dtf_data_smile_no_dup_no_conv.pickle')
    pd.to_pickle(dtf_validation, r'data/raw/validation_dtf_data_smile_no_dup_no_conv.pickle')
    print('External Global Features added.')
    print('Datasets Saved')