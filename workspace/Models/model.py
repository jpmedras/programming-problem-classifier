from torch import nn, long, argmax, optim, save
from torch import no_grad
from transformers import BertModel
from torch import cuda
from Datasets.dataloader import create_dataloader
from loss import calc_loss

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

class BERTModule(nn.Module):
    def __init__(self, epochs = 10, learning_rate = 1e-05, dropout_p = 0.3):
        super(BERTModule, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # for param in self.bert.parameters():
        #   param.requires_grad = False
        self.dropout = nn.Dropout(p = dropout_p)
        self.fc = nn.Linear(768, 3)

        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, ids, masks, ttis):
        _, pooled_output = self.bert(ids, attention_mask = masks, token_type_ids = ttis, return_dict = False)
        output_2 = self.dropout(pooled_output)
        output = self.fc(output_2)
        
        return output
    
    def fit(self, train_loader, test_loader):
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.to(self.device)

        self.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params = self.parameters(), lr = self.learning_rate)
        
        print('Begin training...')

        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            train_loss = 0.

            for inputs, labels in train_loader:
                optimizer.zero_grad()

                ids = inputs[:, 0].to(self.device, dtype=long)
                masks = inputs[:, 1].to(self.device, dtype=long)
                tti = inputs[:, 2].to(self.device, dtype=long)
                labels = labels.squeeze().to(self.device, dtype=long)

                assert ids.shape == masks.shape, 'Ids != Masks'
                assert masks.shape == tti.shape, 'Masks != Ttis'
                assert ids.shape == tti.shape, 'Ids != Ttis'

                assert ids.shape[0] == labels.shape[0], 'inputs and labels are incompatible'

                outputs = self(ids, masks, tti)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
    
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = calc_loss(self, test_loader, criterion, self.device)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            print(f'Epoch {epoch + 1}/{self.epochs} Train Loss: {avg_train_loss} Test Loss: {avg_test_loss}')

        print('Ending training...')

        model_name = 'model' + '_' + 'ep' + str(self.epochs) + '_' + 'lr' + str(self.learning_rate) + '.pth'
        save(self.state_dict(), model_name)

        return train_losses, test_losses

    def evaluate(self, dataloader):
        self.eval()

        data_labels = []
        data_outputs = []

        with no_grad():
            for inputs, labels in dataloader:
        
                ids = inputs[:, 0].to(self.device, dtype=long)
                masks = inputs[:, 1].to(self.device, dtype=long)
                tti = inputs[:, 2].to(self.device, dtype=long)
                labels = labels.squeeze().to(self.device, dtype=long)

                assert ids.shape == masks.shape, 'Ids != Masks'
                assert masks.shape == tti.shape, 'Masks != Ttis'
                assert ids.shape == tti.shape, 'Ids != Ttis'

                assert ids.shape[0] == labels.shape[0], 'inputs and labels are incompatible'

                outputs = self(ids, masks, tti)
                outputs = nn.functional.softmax(outputs, dim=1)
                outputs = argmax(outputs, dim=1)

                data_labels.extend(labels.cpu().detach().numpy().tolist())
                data_outputs.extend(outputs.cpu().detach().numpy().tolist())

    
        target_names = ['Easy', 'Medium', 'Hard']
        macro_f1 = f1_score(data_labels, data_outputs, average='macro')
        cm = confusion_matrix(data_labels, data_outputs)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Easy', 'Medium', 'Hard'])
        print(f'Macro F1: {macro_f1}')
        disp.plot()
        plt.show()


    def predict():
        return 1