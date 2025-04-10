import torch
from Scripts.dataset import MoleculeDataset
from torch_geometric.loader import DataLoader
from Scripts.script_model.model import GNN, GINE, ClassModel, ModelPredPos, ModelPredNumPeak, GINEGLOBAL, ClassModelGlobal
from Scripts.utils_model.utils_model import resume, count_parameters, enable_dropout
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

def predict(config, config_model, mc_sam):

    model_name = config['model']
    str_version = config['dir_name']
    path_proc = 'processed_' + config['type_pred']
    params = config_model[model_name]['params']
    arch_params = config_model[model_name]['arch_params']
    interval = config['type_pred']
    n_data_points = 1

    dct_predictions = {}
    for str_dataset in ['train', 'test', 'validation']:
        y_true = []
        y_pred = []
        smiles = []
        lst_f1 = []

        true_dataset = pd.read_pickle(f"data/raw/{str_dataset}_{config['starting_dtf']}.pickle")
        true_dataset.rename(columns={'raman_pred': 'raman_pred_num_peak'}, inplace=True)
        dataset = MoleculeDataset(root="data", filename=f"{str_dataset}_{config['starting_dtf']}.pickle",
                                       target_col=config['target_col'], path_proc=path_proc, test=str_dataset,
                                  global_feature_col=config.get('global_feature_list', []),
                                  remove_bad_mol=False)
        loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

        lst_file = [model for model in os.listdir(f'models/{str_version}') if model.endswith('.pth')]
        lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        best_model_ckpt = lst_file[-1]

        params["model_edge_dim"] = dataset[0].edge_attr.shape[1]

        device = "cuda" if torch.cuda.is_available else "cpu"
        model = eval(model_name)(node_feature_size=dataset[0].x.shape[1],
                                 edge_feature_size=dataset[0].edge_attr.shape[1],
                                 n_data_points=n_data_points, **arch_params)
        print("Number of params: ", count_parameters(model))
        model.to(device)

        resume(model, os.path.join(f'models/{str_version}', best_model_ckpt))
        model.eval()
        enable_dropout(model)

        for batch in tqdm(loader):
            lst_pred = []
            batch.to(device)
            if len(batch.smiles) < 32: continue

            for i in range(mc_sam):
                pred = model(batch.x.float(),
                             # None,
                             batch.graph_level_feats,
                             batch.edge_attr.float(),
                             batch.edge_index,
                             batch.batch)
                lst_pred.append(pred)

            pred = torch.mean(torch.stack(lst_pred, dim=2), dim=2)
            y_pred_batch = np.round(torch.squeeze(pred).cpu().detach().numpy())
            y_true_batch = batch.y.reshape(32, -1).float().cpu().detach().numpy().squeeze()

            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)

            smiles.extend(batch.smiles)
        dct_predictions[str_dataset] = pd.DataFrame({'SMILE': smiles, 'TRUE_NUM_PEAK': y_true, 'PRED_NUM_PEAK': y_pred})

    pd.to_pickle(dct_predictions, rf"data\predictions\pred_{str_version}.pickle")
    print("Done")


if __name__ == '__main__':

    import json
    t1 = time.time()
    str_dir = "pred_num_peak_up"
    if "peak" not in str_dir:
        raise NameError("Prediction should be related to predictions of number of peaks!")
    with open(rf"models/{str_dir}/config/config_model.json") as file:
        config_model = json.load(file)
    with open(rf"models/{str_dir}/config/config.json") as file:
        config = json.load(file)

    predict(config, config_model, mc_sam=10)
    print(time.time()-t1)