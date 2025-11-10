import torch
import numpy as np

class ToOneHotDNA:
    def __init__(self, alphabet="ACGT"):
        self.idx = {b: i for i, b in enumerate(alphabet)}

    def __call__(self, seq: str):
        x = np.zeros((4, len(seq)), dtype=np.float32)
        for j, b in enumerate(seq):
            if b in self.idx:
                x[self.idx[b], j] = 1.0
        return torch.from_numpy(x)

class ToOneHotCond:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, cond):
        import torch
        x = torch.zeros(self.num_classes, dtype=torch.float32)
        x[int(cond)] = 1.0
        return x
