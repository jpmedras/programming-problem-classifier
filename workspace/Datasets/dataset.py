from torch.utils.data import Dataset
from torch import tensor

class ProblemDataset(Dataset):
    def __init__(self, data, inputs_encoder, labels_encoder):
        self.inputs = inputs_encoder(data['inputs'])
        self.labels = labels_encoder(data['labels'])
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]