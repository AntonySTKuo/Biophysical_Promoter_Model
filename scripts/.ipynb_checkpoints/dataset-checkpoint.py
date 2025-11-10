import torch
from torch.utils.data import Dataset

class DNADataset(Dataset):
    def __init__(self, df, seq_transform, cond_transform=None, y_transform=None):
        self.df = df.reset_index(drop=True)
        self.seq_tf = seq_transform
        self.cond_tf = cond_transform or (lambda z: z)
        self.y_tf = y_transform or (lambda y: y)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        seq = self.seq_tf(self.df.loc[i, "seq"])
        cond = self.cond_tf(self.df.loc[i, "cond"])
        y = self.y_tf(self.df.loc[i, "LogGFP"])
        return seq, cond, y

    def get_all(self):
        seqs, conds, ys = zip(*(self[i] for i in range(len(self))))
        return torch.stack(seqs), torch.stack(conds), torch.stack(ys)

