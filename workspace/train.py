from torch import nn, long, optim, save

from Datasets.dataset import ProblemDataset
from Models.model import BERTModule

from Datasets.seed import define_seed
from Datasets.encoders import define_encoders
from Datasets.load import load_data
from Datasets.split import split_data
from Datasets.dataloader import create_dataloader

from show_loss import show_loss_evolution

SEED = 42
MAX_LEN = 200
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-05
DATA_PATH = '../data/leetcode.csv'

define_seed(SEED)

inputs_encoder, labels_encoder = define_encoders(MAX_LEN)
# TODO: Error in inputs_encoder, some inputs are getting dim_size greatter than MAX_LEN
# an code is done to print the first input that gets a tensor with different shape

data = load_data(data_path=DATA_PATH)

dataset = ProblemDataset(
    data=data,
    inputs_encoder=inputs_encoder,
    labels_encoder=labels_encoder
)
train_set, test_set = split_data(
    dataset=dataset,
    lengths=[0.85, 0.15],
    seed=SEED
)

train_loader = create_dataloader(dataset = train_set, batch_size = BATCH_SIZE, type='train')
test_loader = create_dataloader(dataset = test_set, batch_size = BATCH_SIZE, type='test')

model = BERTModule(epochs = EPOCHS, learning_rate = LEARNING_RATE)

train_losses, test_losses = model.fit(train_loader=train_loader, test_loader=test_loader)

model.evaluate(dataloader=train_loader)
model.evaluate(dataloader=test_loader)

show_loss_evolution(EPOCHS, train_losses, test_losses)