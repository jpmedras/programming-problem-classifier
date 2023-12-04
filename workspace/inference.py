from torch import load

from Datasets.test_dataset import ProblemDataset
from Models.model import BERTModule

model_path = 'model_ep5_lr1e-05.pth'

model = BERTModule(n_classes = 3)
model.load_state_dict(load(model_path))

text = """
You want to create as many non-degenerate triangles as possible while satisfying the following requirements. Each triangle consists of 3
 distinct special points (not necessarily from different sides) as its corners. Each special point can only become the corner of at most 1
 triangle. All triangles must not intersect with each other.

Determine the maximum number of non-degenerate triangles that you can create.
"""

model.predict(text=text)