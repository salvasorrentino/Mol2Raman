import numpy as np

from rdkit.Chem.GraphDescriptors import *
from rdkit.Chem.Descriptors import *
from rdkit import Chem

#
def get_features_from_smile(smile: str):

    descriptors = [
        AvgIpc,
        BalabanJ,
        BertzCT,
        Chi0
        #HallKierAlpha,
        #Ipc,
        #Kappa1,
        #Kappa2,
        #Kappa3,
        #ExactMolWt,
        #NumRadicalElectrons,
        #NumValenceElectrons,
        #FpDensityMorgan1,
        #FpDensityMorgan2,
        #FpDensityMorgan3,
        #HeavyAtomMolWt,
    ]

    mol = Chem.MolFromSmiles(smile)
    features = {}
    for descriptor in descriptors:
        try:
            features[descriptor.__name__] = descriptor(mol)
        except:
            features[descriptor.__name__] = np.nan

    return features