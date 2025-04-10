import torch
import os
from tenacity import retry, stop_after_attempt, wait_fixed


class EarlyStopping:

    def __init__(self, tolerance=10, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):

        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:  self.early_stop = True
        else:
            self.counter = 0


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


@retry(stop=stop_after_attempt(50), wait=wait_fixed(1))
def featurize_with_retry(featurizer, mol):
    return featurizer._featurize(mol)


def nomi_file_in_cartella(cartella):
    # Verifica se la cartella esiste
    if not os.path.isdir(cartella):
        print(f"La cartella {cartella} non esiste.")
        return ()

    nomi_file = os.listdir(cartella)
    tupla_nomi_file = tuple(nomi_file)
    return tupla_nomi_file