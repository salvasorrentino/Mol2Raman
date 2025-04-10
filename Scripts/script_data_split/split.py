import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
import json


def max_pooling(row, spectrum, interval=5):
    value = row[spectrum]
    arr_wave = np.array(value)
    arr_pool = block_reduce(arr_wave, interval, np.max)

    return arr_pool


def run_split(full_df, config, int_inter):

    train_df, test_df = train_test_split(full_df, test_size=config['split']['test_size'],
                                         random_state=42)
    dct_region = config['wl_region']

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_df['WAVELENGTH_DOWN'] = train_df['WAVELENGTH'].apply(lambda x: x[dct_region['finger']['low']:
                                                                           dct_region['finger']['up']])
    train_df['WAVELENGTH_UP'] = train_df['WAVELENGTH'].apply(lambda x: x[dct_region['ch']['low']:
                                                                         dct_region['ch']['up']])

    train_df['RAMAN_SPECTRUM_DOWN'] = \
        train_df['RAMAN_SPECTRUM'].apply(lambda x: x[dct_region['finger']['low']:
                                                     dct_region['finger']['up']])

    train_df['RAMAN_SPECTRUM_UP'] = \
        train_df['RAMAN_SPECTRUM'].apply(lambda x:  x[dct_region['ch']['low']:
                                                      dct_region['ch']['up']])

    test_df['WAVELENGTH_DOWN'] = test_df['WAVELENGTH'].apply(lambda x: x[dct_region['finger']['low']:
                                                                         dct_region['finger']['up']])
    test_df['WAVELENGTH_UP'] = test_df['WAVELENGTH'].apply(lambda x:  x[dct_region['ch']['low']:
                                                                        dct_region['ch']['up']])

    test_df['RAMAN_SPECTRUM_DOWN'] =\
        test_df['RAMAN_SPECTRUM'].apply(lambda x: x[dct_region['finger']['low']:
                                                    dct_region['finger']['up']])
    test_df['RAMAN_SPECTRUM_UP'] = \
        test_df['RAMAN_SPECTRUM'].apply(lambda x:  x[dct_region['ch']['low']:
                                                     dct_region['ch']['up']])

    print("Raman spectra splitted")

    train_df['RAMAN_SPECTRUM_POOLED'] = train_df.apply(max_pooling, spectrum="RAMAN_SPECTRUM",
                                                       interval=int_inter, axis=1)
    print("Full spectrum pooled")

    train_df['RAMAN_SPECTRUM_POOLED_DOWN'] = train_df.apply(max_pooling, spectrum='RAMAN_SPECTRUM_DOWN',
                                                            interval=int_inter, axis=1)
    train_df['RAMAN_SPECTRUM_POOLED_UP'] = train_df.apply(max_pooling, spectrum='RAMAN_SPECTRUM_UP',
                                                          interval=int_inter, axis=1)

    print("Train ended")

    test_df['RAMAN_SPECTRUM_POOLED'] = test_df.apply(max_pooling, spectrum="RAMAN_SPECTRUM",
                                                     interval=int_inter, axis=1)
    test_df['RAMAN_SPECTRUM_POOLED_DOWN'] = test_df.apply(max_pooling, spectrum='RAMAN_SPECTRUM_DOWN',
                                                          interval=int_inter, axis=1)
    test_df['RAMAN_SPECTRUM_POOLED_UP'] = test_df.apply(max_pooling, spectrum='RAMAN_SPECTRUM_UP',
                                                        interval=int_inter, axis=1)
    print("Test ended")
    return train_df, test_df


def add_peak_number(dtf_in, type_pred, **kwargs):
    dtf_out = dtf_in.copy()


    dtf_out['RAMAN_PEAK_NUM_DOWN'] = dtf_out['RAMAN_SPECTRUM_DOWN'].apply(lambda x: len([c for c in x if c > 1e-6]))
    dtf_out['RAMAN_PEAK_NUM_UP'] = dtf_out['RAMAN_SPECTRUM_UP'].apply(lambda x: len([c for c in x if c > 1e-6]))

    return dtf_out


if __name__ == '__main__':
    with open("config/config_no_conv_inter_6_down.json") as file:
        config = json.load(file)

    full_df = pd.read_pickle(fr"data\raw\{config['starting_dtf']}.pickle")

    int_inter = config.get('split_f', 1)

    train_df, test_df = run_split(full_df, config, int_inter)
    train_df = add_peak_number(train_df, config['type_pred'], **config.get('source_peak_data', {}))
    test_df = add_peak_number(test_df, config['type_pred'], **config.get('source_peak_data', {}))

    # Split in validation and test dataset
    validation_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    pd.to_pickle(train_df, rf'data\raw\train_{config["starting_dtf"]}.pickle')
    pd.to_pickle(validation_df, rf'data\raw\validation_{config["starting_dtf"]}.pickle')
    pd.to_pickle(test_df, rf'data\raw\test_{config["starting_dtf"]}.pickle')

    print('Dataset Split Saved!')