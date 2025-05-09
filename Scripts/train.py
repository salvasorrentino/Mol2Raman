import torch
import shutil
import time
from torch_geometric.loader import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from Scripts.dataset import MoleculeDataset
from Scripts.script_model.model import GNN, GINE, ClassModel, ModelPredPos, ModelPredNumPeak, GINEGLOBAL, ClassModelGlobal
from Scripts.utils_model.utils_model import resume, checkpoint, EarlyStopping, count_parameters
from Scripts.loss import (RMSELoss, PeakAwareLoss, BinaryWeightedRMSELoss, WeightedRMSELoss, FocalLoss,
                  HuberLoss, SID, CosineSimilarityLoss, ModifiedRMSELossPeakEmphasis, RMSE, F1Loss)
import os


def compute_metrics(y_pred, y_true, epoch, state):
    rmse_loss = RMSELoss(compute=False)
    peak_rmse_loss = PeakAwareLoss()
    rmse_loss = rmse_loss(torch.tensor(y_pred), torch.tensor(y_true))
    peak_rmse_loss = peak_rmse_loss(torch.tensor(y_pred), torch.tensor(y_true))
    print(f"RMSE {state} rmse at {epoch} epoch: {rmse_loss}")
    print(f"Peak RMSE {state} rmse at {epoch} epoch: {peak_rmse_loss}")


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def train_one_epoch(epoch, model, device, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(train_loader):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                     # None,
                     batch.graph_level_feats,
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)

        if torch.isnan(pred).any():
            print("NaN in predictions")
            print(pred)

        if torch.isnan(batch.y).any():
            print("NaN in labels")
            print(batch.y)

        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred).reshape(-1), batch.y.float())

        if torch.isnan(loss).any():
            print("NaN in loss")
            print(loss)

        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss.item()
        step += 1

        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    #compute_metrics(all_preds, all_labels, epoch, "train")

    return running_loss / step


def test_one_epoch(epoch, model, device, valid_loader, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in valid_loader:
        batch.to(device)
        pred = model(batch.x.float(),
                     # None,
                     batch.graph_level_feats,
                     batch.edge_attr.float(),
                     batch.edge_index,
                     batch.batch)
        loss = loss_fn(torch.squeeze(pred).reshape(-1), batch.y.float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    compute_metrics(all_preds, all_labels, epoch, "valid")

    return running_loss / step


def main(config, config_model, interval):

    print("Train started")

    if os.path.exists(f"models/{config['dir_name']}") and os.path.isdir(f"models/{config['dir_name']}"):
        shutil.rmtree(f"models/{config['dir_name']}")
    os.makedirs(f"models/{config['dir_name']}")

    if os.path.exists(f"models/{config['dir_name']}/config") and os.path.isdir(f"models/{config['dir_name']}/config"):
        shutil.rmtree(f"models/{config['dir_name']}/config")
    os.makedirs(f"models/{config['dir_name']}/config")

    with open(f"models/{config['dir_name']}/config/config_model.json", 'w') as f:
        json.dump(config_model, f)
    with open(f"models/{config['dir_name']}/config/config.json", 'w') as f:
        json.dump(config, f)

    filename = fr"models/{config['dir_name']}/first_run_epoch_"
    model_name = config['model']
    path_proc = 'processed_' + config['type_pred']
    loss_fn_name = config_model[model_name]['loss']['name']
    params = config_model[model_name]['params']
    arch_params = config_model[model_name]['arch_params']

    train_dataset = MoleculeDataset(root="data", filename=f"train_{config['starting_dtf']}.pickle",
                                    target_col=config['target_col'], path_proc=path_proc,
                                    test=None, featuriz=config['featurizer'],
                                    global_feature_col=config.get('global_feature_list', []),
                                    remove_bad_mol=False)
    valid_dataset = MoleculeDataset(root="data", filename=f"validation_{config['starting_dtf']}.pickle",
                                    target_col=config['target_col'], path_proc=path_proc,
                                    test='validation', featuriz=config['featurizer'],
                                    global_feature_col=config.get('global_feature_list', []),
                                    remove_bad_mol=False)

    if model_name == 'ModelPredNumPeak':
        n_data_points = 1
    else:
        n_data_points = len(train_dataset.data[config['target_col']].iloc[0])

    print("Dataset created")

    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False)

    model = eval(model_name)(node_feature_size=train_dataset[0].x.shape[1],
                             edge_feature_size=train_dataset[0].edge_attr.shape[1],
                             n_data_points=n_data_points,
                             **arch_params)

    device = "cuda" if torch.cuda.is_available else "cpu"
    model.to(device)

    print("Number of params: ", count_parameters(model))

    dct_loss_params = config_model[model_name]['loss']['params']

    if loss_fn_name == 'SID':
        dct_loss_params['n_data_points'] = n_data_points

    loss_fn = eval(config_model[model_name]['loss']['name'])(**dct_loss_params)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=params["learning_rate"],
    #                              weight_decay=params["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

    best_loss = np.inf

    print("Training Started")

    tlosses = []
    vlosses = []

    # Model test and train loss annotation
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%S")
    model_log_dir = os.path.join('log_dir', config['dir_name'])
    writer = SummaryWriter(model_log_dir)
    start_time = time.time()
    print(datetime.datetime.now().strftime("%Y%m%d-%H%S"))

    for epoch in range(1500):

        model.train()
        train_loss = train_one_epoch(epoch, model, device, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {train_loss}")

        model.eval()
        if epoch % 2 == 0:
            valid_loss = test_one_epoch(epoch, model, device, valid_loader, loss_fn)
            print(f"Epoch {epoch} | Valid Loss {valid_loss}")

            if (valid_loss < best_loss) & (epoch >= 50):
                checkpoint(model, filename + str(epoch) + ".pth")
                best_loss = valid_loss

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)

        scheduler.step()

        tlosses.append(train_loss)
        vlosses.append(valid_loss)

    end_time = time.time()

    # Time
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(datetime.datetime.now().strftime("%Y%m%d-%H%S"))
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:05.2f}"

    # Stamp time
    print(f"Tempo di esecuzione: {formatted_time}")
    print(f"Finishing training with best valid loss: {best_loss}")

    return [best_loss]


if __name__ == '__main__':

    import json

    # use the split.py script to split the dataset, add the target information and save it in the data folder

    with open(r"config/config_model.json") as file:
        config_model = json.load(file)
    with open(r"config/config_no_conv_inter_6_up.json") as file:
        config = json.load(file)

    type_pred = config['type_pred']

    best_loss = main(config, config_model, type_pred)

    print("Best loss: ", best_loss[0])
