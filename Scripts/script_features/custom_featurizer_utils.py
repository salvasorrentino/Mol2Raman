import deepchem.utils.molecule_feature_utils as dpch_f
from typing import List, Tuple
import numpy as np
from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.molecule_featurizers import mol_graph_conv_featurizer as mgcf


DEFAULT_ATOM_TYPE_SET = [
    "C",
    "N",
    "O",
    "F",
]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_FORMAL_CHARGE_SET = [-2, -1, 0, 1, 2]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_RING_SIZE_SET = [3, 4, 5, 6, 7, 8]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
DEFAULT_GRAPH_DISTANCE_SET = [1, 2, 3, 4, 5, 6, 7]


def custom_construct_atom_feature(
    atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
    use_partial_charge: bool) -> np.ndarray:

    atom_type = dpch_f.get_atom_type_one_hot(atom, DEFAULT_ATOM_TYPE_SET, include_unknown_set=True)
    formal_charge = dpch_f.get_atom_formal_charge(atom)
    hybridization = dpch_f.get_atom_hybridization_one_hot(atom)
    acceptor_donor = dpch_f.get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = dpch_f.get_atom_is_in_aromatic_one_hot(atom)
    degree = dpch_f.get_atom_total_degree_one_hot(atom, DEFAULT_TOTAL_DEGREE_SET, include_unknown_set=True)
    total_num_Hs = dpch_f.get_atom_total_num_Hs_one_hot(atom, DEFAULT_TOTAL_NUM_Hs_SET, include_unknown_set=True)
    atom_feat = np.concatenate([atom_type, hybridization, acceptor_donor, aromatic, degree, total_num_Hs,
                                formal_charge])

    if use_chirality:
        chirality = dpch_f.get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, chirality])

    if use_partial_charge:
        partial_charge = dpch_f.get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, partial_charge])

    return atom_feat


def custom_construct_bond_feature(bond: RDKitBond) -> np.ndarray:

    bond_type = dpch_f.get_bond_type_one_hot(bond, DEFAULT_BOND_TYPE_SET)
    same_ring = dpch_f.get_bond_is_in_same_ring_one_hot(bond)
    conjugated = dpch_f.get_bond_is_conjugated_one_hot(bond)
    stereo = dpch_f.get_bond_stereo_one_hot(bond, DEFAULT_BOND_STEREO_SET, include_unknown_set=True)

    return np.concatenate([bond_type, same_ring, conjugated, stereo])


def custom_pagtn_atom_featurizer(atom: RDKitAtom) -> np.ndarray:
    """Calculate Atom features from RDKit atom object.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        RDKit mol object.

    Returns
    -------
    atom_feat: np.ndarray
        numpy vector of atom features.

    """
    atom_type = dpch_f.get_atom_type_one_hot(atom, DEFAULT_ATOM_TYPE_SET, True)
    formal_charge = dpch_f.get_atom_formal_charge_one_hot(
        atom, include_unknown_set=True)
    degree = dpch_f.get_atom_total_degree_one_hot(atom, list(range(11)), True)
    exp_valence = dpch_f.get_atom_explicit_valence_one_hot(atom, list(range(7)),
                                                    True)
    imp_valence = dpch_f.get_atom_implicit_valence_one_hot(atom, list(range(6)),
                                                    True)
    armoticity = dpch_f.get_atom_is_in_aromatic_one_hot(atom)
    atom_feat = np.concatenate([atom_type, formal_charge, degree, exp_valence, imp_valence, armoticity])
    return atom_feat

# mgcf.PagtnMolGraphFeaturizer._pagtn_atom_featurizer = custom_pagtn_atom_featurizer
mgcf._construct_bond_feature = custom_construct_bond_feature
mgcf._construct_atom_feature = custom_construct_atom_feature

