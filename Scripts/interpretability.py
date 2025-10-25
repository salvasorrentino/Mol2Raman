# explain.py
import torch
import pandas as pd
from captum.attr import Saliency, IntegratedGradients
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import numpy as np

from rdkit import Chem

import json
import time
import os

from Scripts.utils_model.utils_model import resume, count_parameters, enable_dropout

from Scripts.script_model.model import (GNN, GINE, ClassModel, ModelPredPos, ModelPredNumPeak, GINEGLOBAL, ClassModelGlobal,
                                        SOAPOnlyMLP, GATv2GLOBAL)

from Scripts.dataset import MoleculeDataset

import debug_attach

str_fold = "spectra_predictions_ch_1900_3500_feat_numpeak_daylight_morgan_loss_8651"
str_dir = r"models"
with open(rf"{str_dir}\{str_fold}\config\config_model.json") as file:
    config_model = json.load(file)
with open(rf"{str_dir}\{str_fold}\config\config.json") as file:
    config = json.load(file)

@torch.no_grad()
def topk_peak_indices(y_pred, k=3):
    # y_pred: (N, 267) or (267,)
    y = y_pred.detach().cpu().numpy().reshape(-1)
    return np.argsort(y)[-k:][::-1]

def _forward_for_attr(model, edge_attr, edge_index, batch_index):
    # returns a function that only takes (x, graph_feats) as inputs for Captum
    def f(x, graph_feats):
        return model(x, graph_feats, edge_attr, edge_index, batch_index)
    return f

def atom_attributions(model, data, peak_idx, method="ig", steps=32, device="cuda"):
    model.eval()
    x = data.x.float().to(device).requires_grad_(True)                   # (N_nodes, F)
    gf = data.graph_level_feats.float().to(device).requires_grad_(True)  # (1, G)
    edge_attr  = data.edge_attr.float().to(device)
    edge_index = data.edge_index.to(device)
    batch_index= data.batch.to(device)

    def f(x_in, gf_in):
        return model(x_in, gf_in, edge_attr, edge_index, batch_index)    # (1, P)

    if method == "saliency":
        explainer = Saliency(f)
        attr_x, attr_g = explainer.attribute(inputs=(x, gf), target=int(peak_idx))
    else:
        explainer = IntegratedGradients(f)
        bx = torch.zeros_like(x)
        bg = torch.zeros_like(gf)
        attr_x, attr_g = explainer.attribute(
            inputs=(x, gf),
            baselines=(bx, bg),
            target=int(peak_idx),
            n_steps=steps,
            internal_batch_size=1,   # <<< prevents (steps * N_nodes) stacking
        )

    atom_scores = attr_x.abs().sum(dim=1).detach().cpu().numpy()   # (N_nodes,)
    global_scores = attr_g.abs().sum().item()
    return atom_scores, global_scores


from torch_geometric.data import Batch

def pick_single_graph(batched_data, graph_idx: int, device="cpu"):
    # Convert Batch -> list[Data], pick one, then re-batch it to keep the same model signature
    g = batched_data.to_data_list()[graph_idx]
    single = Batch.from_data_list([g]).to(device)
    return single

@torch.no_grad()
def occlusion_drop(model, data, atom_indices, device="cuda"):
    data = data.to(device)
    y0 = model(data.x.float(), data.graph_level_feats.float(),
               data.edge_attr.float(), data.edge_index, data.batch)   # (1, P)
    x_masked = data.x.float().clone()
    x_masked[atom_indices] = 0.0
    y1 = model(x_masked, data.graph_level_feats.float(),
               data.edge_attr.float(), data.edge_index, data.batch)   # (1, P)
    return (y0 - y1).squeeze(0).detach().cpu().numpy()


str_dir = r"."

model_name = config['model']
str_version = config['dir_name']
path_proc = 'processed_' + config['type_pred']
params = config_model[model_name]['params']
arch_params = config_model[model_name]['arch_params']
interval = config['type_pred']

true_dataset = pd.read_pickle(rf"{str_dir}\data\raw\test_{config['starting_dtf']}.pickle")
true_dataset.rename(columns={'raman_pred': 'raman_pred_num_peak'}, inplace=True)
test_dataset = MoleculeDataset(root="data", filename=f"test_{config['starting_dtf']}.pickle",
                               target_col=config['target_col'], path_proc=path_proc, test='test',
                               global_feature_col=config.get('global_feature_list', []),
                               extra_global_featurizers=[],
                               remove_bad_mol=False)

test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

lst_file = [model for model in os.listdir(rf'{str_dir}\models\{str_version}') if model.endswith('.pth')]
lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
best_model_ckpt = lst_file[-1]

params["model_edge_dim"] = test_dataset[0].edge_attr.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

if model_name == 'ModelPredNumPeak':
    n_data_points = 1
else:
    n_data_points = len(test_dataset.data[config['target_col']].iloc[0])

print("Dataset created")

model = eval(model_name)(node_feature_size=test_dataset[0].x.shape[1],
                         edge_feature_size=test_dataset[0].edge_attr.shape[1],
                         n_data_points=n_data_points, **arch_params)
# print("Number of params: ", count_parameters(model))
model.to(device)
resume(model, os.path.join(fr'{str_dir}\models\{str_version}', best_model_ckpt))
model.eval()

# Get one batched batch from the loader
idx = 40
for i, batch in enumerate(test_loader):
    data = batch.to(device)
    if i == idx:
        break

# Pick one graph from that batch, e.g., the 11th item (index 10)
single = pick_single_graph(data, graph_idx=22, device=device)

# Forward on that single graph
with torch.no_grad():
    y_single = model(single.x.float(),
                     single.graph_level_feats.float(),
                     single.edge_attr.float(),
                     single.edge_index,
                     single.batch)                # shape: (1, P)

# Choose top-K peak indices from the single graph
peak_ids = topk_peak_indices(y_single.squeeze(0), k=3)

# Attribution per chosen peak
# for j in peak_ids:
#     atom_scores, _ = atom_attributions(model, single, j, method="ig", steps=32, device=device)
#     # visualize atom_scores on RDKit drawing (color by score)
#
# # Occlusion: e.g., all aromatic atoms from an RDKit match
#
rdkit_mol = Chem.MolFromSmiles(single.smiles[0], sanitize=False)
Chem.SanitizeMol(rdkit_mol)
# aromatic_idx = [a.GetIdx() for a in rdkit_mol.GetAromaticAtoms()]
# delta = occlusion_drop(model, single, aromatic_idx) # change in predicted spectrum
# report delta[peak_j] for peaks of interest


import os, io
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase

def normalize_scores(s, q=0.90):
    s = np.clip(np.asarray(s, float), 0, None)
    scale = np.percentile(s, q*100)
    if scale <= 0: scale = s.max() if s.max() > 0 else 1.0
    return np.clip(s / (scale + 1e-12), 0.0, 1.0)

def score_to_rgb(t):
    t = float(np.clip(t, 0, 1))
    if t < 0.5:  # blue → yellow
        u = t/0.5
        return (u, 0.2 + 0.8*u, 1.0 - u)
    else:        # yellow → red
        u = (t-0.5)/0.5
        return (1.0, 1.0 - u, 0.0)

def make_cmap():  # matches score_to_rgb
    cdict = {
        "red":   ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)),
        "green": ((0.0, 0.2, 0.2), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)),
        "blue":  ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }
    return LinearSegmentedColormap("bYr", cdict)

def draw_atom_importance_with_colorbar(
    mol, atom_scores, outfile_png,
    legend="", q=0.90, thr=0.2, w=900, h=650,
    label_mode="element"  # 'element' | 'none'
):
    # --- RDKit drawing ---
    m = Chem.Mol(mol)
    try:
        rdMolDraw2D.PrepareMolForDrawing(m, kekulize=True)
    except Exception:
        AllChem.Compute2DCoords(m)

    norm = normalize_scores(atom_scores, q=q)
    high_atoms = [i for i, v in enumerate(norm) if v >= thr]
    atom_colors = {i: score_to_rgb(norm[i]) for i in high_atoms}

    bond_ids, bond_colors = [], {}
    for b in m.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in atom_colors and a2 in atom_colors:
            c1, c2 = atom_colors[a1], atom_colors[a2]
            bond_ids.append(b.GetIdx())
            bond_colors[b.GetIdx()] = tuple((c1[k] + c2[k]) / 2 for k in range(3))

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    dopts = drawer.drawOptions()
    dopts.legendFontSize = 18
    dopts.addAtomIndices = False  # do NOT show RDKit indices

    # Replace labels with actual atom symbols (C, N, O, ...).
    if label_mode == "element":
        for aidx in range(m.GetNumAtoms()):
            dopts.atomLabels[int(aidx)] = str(m.GetAtomWithIdx(aidx).GetSymbol())
    # else: leave default; or set 'none' and do nothing

    drawer.DrawMolecule(
        m,
        highlightAtoms=high_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=bond_ids,
        highlightBondColors=bond_colors,
        legend=legend
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # --- Compose colorbar with matplotlib ---
    img = Image.open(io.BytesIO(png_bytes))

    fig = plt.figure(figsize=(w/100.0, (h+80)/100.0), dpi=100)  # extra space for the bar
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.88])  # image area
    ax.imshow(img)
    ax.axis("off")

    cax = fig.add_axes([0.15, 0.05, 0.70, 0.03])  # colorbar area
    cmap = make_cmap()
    cb = ColorbarBase(cax, cmap=cmap, orientation="horizontal",
                      ticks=[0.0, 0.5, 1.0])
    cb.set_label("Relative attribution (0–1)", fontsize=11)

    os.makedirs(os.path.dirname(outfile_png), exist_ok=True)
    fig.savefig(outfile_png, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {outfile_png}")


assert rdkit_mol.GetNumAtoms() == single.x.size(0)

j = int(peak_ids[2])
atom_scores, _ = atom_attributions(model, single, j, method="ig", steps=32, device=device)

outdir = r"D:\Salvatore\Projects\Mol2Raman_data\explanations"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"attr_peak_{j}_with_colorbar.png")

draw_atom_importance_with_colorbar(
    rdkit_mol, atom_scores, outfile_png=outfile,
    legend=f"Attribution for peak bin {j}",
    q=0.90, thr=0.20, w=900, h=650,
    label_mode="element"   # ← element symbols instead of numbers
)

