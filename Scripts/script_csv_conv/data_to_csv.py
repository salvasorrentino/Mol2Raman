import pandas as pd
from Scripts.utils_model.utils_data_processing import generate_lorentzian_kernel, convolve_with_lorentzian


gamma = 7.5
kernel_size = 600
lorentzian_kernel = generate_lorentzian_kernel(kernel_size, gamma)

type_data = 'IR'     # 'raman' or 'IR'
train = pd.read_pickle(rf'data/raw/train_dtf_data_{type_data}_smile_no_dup_no_conv.pickle')
valid = pd.read_pickle(rf'data/raw/validation_dtf_data_{type_data}_smile_no_dup_no_conv.pickle')

train = pd.concat([train, valid], ignore_index=True)
train['WAVELENGTH'] = train['WAVELENGTH'].apply(lambda row: row[100:])
train['RAMAN_SPECTRUM'] = train['RAMAN_SPECTRUM'].apply(lambda row: row[100:])
train['RAMAN_SPECTRUM'] = train.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel), axis=1)
lst_down = train.WAVELENGTH.iloc[0]
new_def = train[['SMILE', 'RAMAN_SPECTRUM']]


df2 = pd.DataFrame(new_def.RAMAN_SPECTRUM.tolist(), index=new_def.SMILE)
df2.columns = [int(c) for c in lst_down]
df2 = df2.reset_index().rename({'SMILE': 'smiles'}, axis=1)

df2.to_csv(rf'C:\Users\user\Documents\Gioele_Pasotti\ChemProp_IR\chemprop-IR\train\train_{type_data}_30cm_conv_500_3500.csv', sep=',', decimal='.', index=False)


test = pd.read_pickle(r'data/raw/test_dtf_data_IR_smile_no_dup_no_conv.pickle')
test['WAVELENGTH'] = test['WAVELENGTH'].apply(lambda row: row[100:])
test['RAMAN_SPECTRUM'] = test['RAMAN_SPECTRUM'].apply(lambda row: row[100:])
test['RAMAN_SPECTRUM'] = test.apply(lambda row: convolve_with_lorentzian(row['RAMAN_SPECTRUM'], lorentzian_kernel), axis=1)
lst_down_t = test.WAVELENGTH.iloc[0]
new_def_t = test[['SMILE', 'RAMAN_SPECTRUM']]


df2_t = pd.DataFrame(new_def_t.RAMAN_SPECTRUM.tolist(), index=new_def_t.SMILE)
df2_t.columns = [int(c) for c in lst_down_t]
df2_t = df2_t.reset_index().rename({'SMILE': 'smiles'}, axis=1)

df2_t.to_csv(rf'C:\Users\user\Documents\Gioele_Pasotti\ChemProp_IR\chemprop-IR\test\test_{type_data}_30cm_conv_500_3500.csv', sep=',', decimal='.', index=False)
