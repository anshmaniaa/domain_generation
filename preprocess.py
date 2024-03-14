import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
import torch

class DomainDataset:
    def __init__(self, dataframe, token_to_id, max_length=20):
        self.dataframe = dataframe
        self.token_to_id = token_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_domain = self.dataframe.iloc[idx]['input']
        output_domain = self.dataframe.iloc[idx]['domain']
        return (self.tokenize(input_domain), self.tokenize(output_domain))

    def tokenize(self, domain):
        tokens = [self.token_to_id.get(char, self.token_to_id['<UNK>']) for char in domain]
        return LongTensor(tokens[:self.max_length])

def create_vocabulary(dataframe):
    """
    Creates a vocabulary from the input and domain columns of the dataframe.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the input and domain columns.

    Returns:
        dict: A dictionary mapping tokens to their IDs.

    """
    unique_chars = set(''.join(dataframe['input'].tolist() + dataframe['domain'].tolist()))
    token_to_id = {char: id for id, char in enumerate(unique_chars)}
    token_to_id['<PAD>'] = len(token_to_id)  # Padding token
    token_to_id['<UNK>'] = len(token_to_id)  # Unknown token
    return token_to_id

def collate_batch(batch):
    """
    Collates a batch of input-output pairs.

    Args:
        batch (list): A list of tuples, where each tuple contains an input tensor and an output tensor.

    Returns:
        tuple: A tuple containing the input tensor and the output tensor, both padded to have the same length.

    """
    input_list, output_list = [], []
    for _input, _output in batch:
        input_list.append(_input)
        output_list.append(_output)
    input_list = pad_sequence(input_list, batch_first=True, padding_value=0)
    output_list = pad_sequence(output_list, batch_first=True, padding_value=0)
    return input_list, output_list

def get_data_loaders(csv_file, batch_size=32):
    dataframe = pd.read_csv(csv_file)
    print(dataframe.columns)
    token_to_id = create_vocabulary(dataframe)

    train, test = train_test_split(dataframe, test_size=0.2)
    train_dataset = DomainDataset(train, token_to_id)
    test_dataset = DomainDataset(test, token_to_id)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, token_to_id

if __name__ == "__main__":
    train_loader, test_loader, token_to_id = get_data_loaders('training_set.csv')
