from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

class ClassicClassifier():
    def __init__(self, clf):
        self.text_clf = Pipeline([
            ('tfidf', TfidfTransformer()),
            ('clf', clf()),
        ])

    def fit(self, inputs, labels):
        self.text_clf = self.text_clf.fit(inputs, labels)

    def predict(self, inputs):
        return self.text_clf.predict(inputs)
    
    def evaluate(self, labels, predicts):
        target_names = ['Easy', 'Medium', 'Hard']
        macro_f1 = f1_score(labels, predicts, average='macro')
        cm = confusion_matrix(labels, predicts)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        print(f'Macro F1: {macro_f1}')
        disp.plot()
        plt.show()