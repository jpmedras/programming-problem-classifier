from torch.utils.data import random_split
from torch import Generator

def split_data(dataset, lengths, seed=42):
    assert len(lengths) == 2, "You must define the size of train dataset and test dataset"

    train_set, test_set = random_split(dataset, lengths, Generator().manual_seed(seed))

    return train_set, test_set