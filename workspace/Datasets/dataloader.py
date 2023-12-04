from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, type):
    if type == 'train':
        shuffle = True
    else:
        shuffle = False

    loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle)

    return loader