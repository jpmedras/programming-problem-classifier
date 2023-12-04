import torch
from torch import long

def calc_loss(model, dataloader, criterion):
    with torch.no_grad():
        total_loss = 0.
        for inputs, labels in dataloader:
            ids = inputs[:, 0].to(model.device, dtype=long)
            masks = inputs[:, 1].to(model.device, dtype=long)
            tti = inputs[:, 2].to(model.device, dtype=long)
            labels = labels.squeeze().to(model.device, dtype=long)

            # print(ids.shape)

            outputs = model(ids, masks, tti)

            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(dataloader)
        
        return average_loss