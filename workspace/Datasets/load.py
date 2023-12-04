import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    data = {
        'inputs': df.inputs.values.tolist(),
        'labels': df.labels.values.tolist()
    }

    return data