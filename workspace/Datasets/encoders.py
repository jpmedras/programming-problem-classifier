from torch import tensor, long
from torch import unsqueeze, cat
from transformers import BertTokenizer

def define_encoders(max_len):
    def inputs_encoder(inputs):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').encode_plus
        encoded_inputs = []
        for input in inputs:
            encoding = tokenizer(
                text=input,
                add_special_tokens=True,
                padding='max_length',
                truncation='longest_first',
                max_length=max_len
            )

            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            token_type_ids = encoding['token_type_ids']

            encoded_input_ids = tensor(input_ids, dtype=long).unsqueeze(0)
            encoded_attention_mask = tensor(attention_mask, dtype=long).unsqueeze(0)
            encoded_token_type_ids = tensor(token_type_ids, dtype=long).unsqueeze(0)

            encoded_input = cat((encoded_input_ids, encoded_attention_mask, encoded_token_type_ids), dim = 0).unsqueeze(0)
            encoded_inputs.append(encoded_input)
        
        try:
            encoded = cat(encoded_inputs)
            return encoded
        except:
            print('Number of inputs:', len(encoded_inputs))
            max_dif = max([e.shape[2] for e in encoded_inputs])
            print('Max shape:', max_dif)
            for idx, e in enumerate(encoded_inputs):
                if e.shape[2] != max_len:
                    print(idx)
                    print(e.shape[2])
                    print(inputs[idx])
                    break
            return None

    def labels_encoder(labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(
                tensor([label], dtype=long).unsqueeze(0)
            )
        
        encoded = cat(encoded_labels)

        return encoded
    
    return inputs_encoder, labels_encoder