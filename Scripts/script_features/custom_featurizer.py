import pandas as pd
import torch
import numpy as np
from deepchem import feat
import os
import logging
from typing import List, Union, Tuple
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
import Scripts.script_features.custom_featurizer_utils
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol


def custom_featuriz(config):
    if config == 'Pagtn':
        CustomFeaturizer = feat.PagtnMolGraphFeaturizer
    elif config == 'DMPNN':
        CustomFeaturizer = feat.DMPNNFeaturizer
    else:
        CustomFeaturizer = feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
    return CustomFeaturizer
