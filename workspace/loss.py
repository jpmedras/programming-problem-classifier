import torch
from torch import long

def calc_loss(model, dataloader, criterion, device):

    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in dataloader:
            ids = inputs[:, 0].to(device, dtype=long)
            masks = inputs[:, 1].to(device, dtype=long)
            tti = inputs[:, 2].to(device, dtype=long)
            labels = labels.squeeze().to(device, dtype=long)

            outputs = model(ids, masks, tti)

            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(dataloader)
        
        return average_loss