from Datasets.dataset import ProblemDataset
from Models.model import BERTModule

from Datasets.seed import define_seed
from Datasets.encoders import define_encoders
from Datasets.load import load_data
from Datasets.split import split_data
from Datasets.dataloader import create_dataloader

from show_loss import show_loss_evolution

from Models.classic_classifier import ClassicClassifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from Datasets.dataset import tolist

SEED = 42
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-05
DATA_PATH = '../data/leetcode.csv'

define_seed(SEED)

inputs_encoder, labels_encoder = define_encoders(MAX_LEN)

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

train_loader = create_dataloader(dataset = train_set, batch_size = TRAIN_BATCH_SIZE, type='train')
test_loader = create_dataloader(dataset = test_set, batch_size = TEST_BATCH_SIZE, type='test')

model = BERTModule(n_classes = 3)

train_losses, test_losses = model.fit(train_loader=train_loader, test_loader=test_loader, epochs = EPOCHS, learning_rate = LEARNING_RATE)

model.evaluate(dataloader=train_loader)
model.evaluate(dataloader=test_loader)

show_loss_evolution(EPOCHS, train_losses, test_losses)

X_train, y_train = tolist(train_set)
X_test, y_test = tolist(test_set)

svc = ClassicClassifier(SVC)
gb = ClassicClassifier(GradientBoostingClassifier)
rf = ClassicClassifier(RandomForestClassifier)

svc.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)

svc.evaluate(y_test, svc.predict(X_test))
gb.evaluate(y_test, gb.predict(X_test))
rf.evaluate(y_test, rf.predict(X_test))