import torch
from Scripts.dataset import MoleculeDataset
from torch_geometric.loader import DataLoader
from Scripts.script_model.model import GNN, GINE, ClassModel, ModelPredPos, ModelPredNumPeak, GINEGLOBAL, ClassModelGlobal
from Scripts.utils_model.utils_model import resume, count_parameters, enable_dropout
import pandas as pd
from tqdm import tqdm
import os


def predict(config, config_model, mc_sam):

    y_true = []
    y_pred = []
    smiles = []
    num_peaks = []
    str_dir = r"C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\other_models"

    model_name = config['model']
    str_version = config['dir_name']
    path_proc = 'processed_' + config['type_pred']
    params = config_model[model_name]['params']
    arch_params = config_model[model_name]['arch_params']
    interval = config['type_pred']

    true_dataset = pd.read_pickle(f"data/raw/test_{config['starting_dtf']}.pickle")
    true_dataset.rename(columns={'raman_pred': 'raman_pred_num_peak'}, inplace=True)
    test_dataset = MoleculeDataset(root="data", filename=f"test_{config['starting_dtf']}.pickle",
                                   target_col=config['target_col'], path_proc=path_proc, test='test',
                                   global_feature_col=config.get('global_feature_list', []),
                                   remove_bad_mol=False)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    if config['type_pred'] == 'pred_num_peak' or config['type_pred'] == 'pred_num_peak_train':
        n_data_points = 1
    else:
        n_data_points = len(test_dataset.data[config['target_col']].iloc[0])

    # lst_file = [model for model in os.listdir(f'models/{str_version}') if model.endswith('.pth')]
    lst_file = [model for model in os.listdir(f'{str_dir}/{str_version}') if model.endswith('.pth')]
    lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    best_model_ckpt = lst_file[-1]

    params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = eval(model_name)(node_feature_size=test_dataset[0].x.shape[1],
                             edge_feature_size=test_dataset[0].edge_attr.shape[1],
                             n_data_points=n_data_points, **arch_params)
    print("Number of params: ", count_parameters(model))
    model.to(device)

    # resume(model, os.path.join(f'models/{str_version}', best_model_ckpt))
    resume(model, os.path.join(fr'{str_dir}\{str_version}', best_model_ckpt))
    model.eval()
    enable_dropout(model)

    for batch in tqdm(test_loader):
        lst_pred = []
        batch.to(device)
        if len(batch.smiles) < 32: continue

        for i in range(mc_sam):
            pred = model(batch.x.float(),
                         batch.graph_level_feats,
                         batch.edge_attr.float(),
                         batch.edge_index,
                         batch.batch)
            lst_pred.append(pred)

        pred = torch.mean(torch.stack(lst_pred, dim=2), dim=2)
        y_pred_batch = torch.squeeze(pred).cpu().detach().numpy()
        y_true_batch = batch.y.reshape(len(batch.smiles), -1).float().cpu().detach().numpy()

        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)
        num_peaks.extend(batch.graph_level_feats.reshape(len(batch.smiles), -1)[:, 0].cpu().detach().numpy().tolist())
        smiles.extend(batch.smiles)

    df = pd.DataFrame({
        "smile": smiles,
        "raman_true": y_true,
        "raman_pred": y_pred,
        "pred_num_peak": num_peaks
    })

    df.to_parquet(rf"data\predictions\pred_{str_version}.parquet")
    print("Done")


if __name__ == '__main__':

    import json
    import time

    # str_dir = "spectra_predictions_ch_1900_3500_prova"
    # with open(rf"models/{str_dir}/config/config_model.json") as file:
    #     config_model = json.load(file)
    # with open(rf"models/{str_dir}/config/config.json") as file:
    #     config = json.load(file)

    str_dir = r"C:\Users\Utente\OneDrive - Politecnico di Milano\PhD Polli\Smile2Raman\other_models"
    for str_fold in os.listdir(str_dir):
        str_use = os.path.join(str_dir, str_fold)
        with open(rf"{str_use}\config\config_model.json") as file:
            config_model = json.load(file)
        with open(rf"{str_use}\config\config.json") as file:
            config = json.load(file)

        t1 = time.time()
        predict(config, config_model, mc_sam=10)
        print(time.time()-t1)
