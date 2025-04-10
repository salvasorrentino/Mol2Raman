import numpy as np
import pandas as pd
import torch
import os

cartella= r'data/processed_inter_16_feature'
nomi_file = os.listdir(cartella)
lst_morph_node = []
for string_files in nomi_file[:-2]:
    string_path = os.path.join(cartella, string_files)
    mol_F = torch.load(string_path)
    lst_morph_node.append(mol_F.x.cpu().numpy().tolist())

arr_node = np.vstack(lst_morph_node)
arr_sum = arr_node.sum(axis=0)

lst_morph_edge_atr = []
for string_files in nomi_file[:-2]:
    string_path = os.path.join(cartella, string_files)
    mol_F = torch.load(string_path)
    lst_morph_edge_atr.append(mol_F.edge_attr.cpu().numpy().tolist())

arr_edge_atr = np.vstack(lst_morph_edge_atr)
arr_sum_edge_atr = arr_edge_atr.sum(axis=0)
