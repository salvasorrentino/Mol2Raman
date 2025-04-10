import pandas as pd
import matplotlib.pyplot as plt


def get_first_element(smile, smile_dict):
    if smile in smile_dict:
        return next(iter(smile_dict[smile].values()))
    return None
 

dtf_our_pred = pd.read_pickle(r'data/results/pred_spectra_mol2raman.pickle')
dct_tanimoto = pd.read_pickle(r'data/tanimoto_dictionary_daylight.pkl')

dtf_our_pred["tan_sim_best"] = dtf_our_pred.apply(lambda row: get_first_element(row['smile'], dct_tanimoto), axis=1)


plt.figure()
plt.scatter(dtf_our_pred['tan_sim_best'], dtf_our_pred['sis'], color='blue', label='Predizioni')
plt.xlabel('best tanimoto sim')
plt.ylabel('Sis Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(dtf_our_pred['tan_sim_best'], dtf_our_pred['F1_15cm^-1'], color='blue', label='Predizioni')
plt.xlabel('best tanimoto sim')
plt.ylabel('F1_15cm^-1 Value')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(dtf_our_pred['sis'], dtf_our_pred['F1_15cm^-1'], color='blue', label='Predizioni')
plt.xlabel('sis Value')
plt.ylabel('F1_15cm^-1 Value')
plt.legend()
plt.grid(True)
plt.show()
