import torch
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    data, labels = zip(*batch)
    data_lens = [len(d) for d in data]
    data_padded = pad_sequence([torch.tensor(d) for d in data], batch_first=True)
    labels_padded = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=-1)
    return data_padded, labels_padded, data_lens
