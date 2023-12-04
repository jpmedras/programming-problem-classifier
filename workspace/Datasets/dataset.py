from torch.utils.data import Dataset

class ProblemDataset(Dataset):
    def __init__(self, data, inputs_encoder, labels_encoder):
        self.inputs = inputs_encoder(data['inputs'])
        self.labels = labels_encoder(data['labels'])
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    
def tolist(dataset):
    inputs = []
    labels = []

    for inp, lab in dataset:
        inputs.append(inp[0, :].numpy().tolist())
        labels.append(lab.numpy().tolist())
    
    labels = [item for sublist in labels for item in sublist]

    return inputs, labels