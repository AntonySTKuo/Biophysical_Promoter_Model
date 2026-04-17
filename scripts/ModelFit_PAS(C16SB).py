## Standard libraries ====
import random
import numpy as np
import pandas as pd

## PyTorch and Machine learning ====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

## Utilities ====
from util import getPFM, load_obj, save_obj


# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_sample(df, col, counts, seed=42):
    """Stratified sample: take 'counts[i]' rows from each of len(counts) bins of 'col'."""
    np.random.seed(seed)
    df = df.copy()
    df['grp'] = pd.cut(df[col], bins=len(counts))
    extracted = pd.concat(
        df[df['grp'] == grp].sample(cnt, replace=True, random_state=seed)
        for grp, cnt in zip(df['grp'].unique(), counts)
        )
    return extracted.drop(columns=['grp'])


def dna_one_hot(seq, flatten=False):
    """One-hot encode a DNA string (A/C/G/T/-/N). Returns (4, L) by default, flattened if requested."""
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        '-': [0, 0, 0, 0],
        'N': [1/4, 1/4, 1/4, 1/4],
        }
    one_hot = np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq.upper()]).T
    return one_hot.flatten() if flatten else one_hot


def Parti(seq):
    return 'CAGCAGCAGTCAGGTACTCCATAAATGC' + seq[:6] + 'CTGTAGCGCAGAGGCG' + seq[-6:] + 'GCACACCCC'


class DNAFunctionPredictor(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length

        # Conv1D layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=6, stride=1)
        self._initialize_conv1()

        self.conv2 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=17, stride=1)
        self._initialize_conv2()

        # Trainable parameters for energy->expression mapping
        self.param_max = nn.Parameter(torch.tensor(np.log(1000.0)), requires_grad=True)
        self.param_min = nn.Parameter(torch.tensor(np.log(10.0)), requires_grad=True)
        self.param_e0  = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Freeze spacer energy bias selectively via gradient mask
        self.register_buffer("bias_mask", torch.tensor([1.0, 0.0, 1.0]))
        self.conv2.bias.register_hook(lambda grad: grad * self.bias_mask)

    def _initialize_conv1(self):
        motif1 = getPFM(['TTGACA', 'TTGACT'],                     ratios=False).values.T  # (4, 6)
        motif2 = getPFM(['TAAAAT', 'TACCCT', 'TAGGGT', 'TATTTT'], ratios=False).values.T  # (4, 6)
        custom_weights = np.stack([motif1, motif2], axis=0)          # (2, 4, 6)
        self.conv1.weight.data = torch.tensor(custom_weights, dtype=torch.float32)
        self.conv1.weight.requires_grad = True
        self.conv1.bias.data.zero_()
        self.conv1.bias.requires_grad = False

    def _initialize_conv2(self):
        # Hand-crafted filters to read out features from conv1
        filter1 = np.zeros((2, 25)); filter1[0, 0], filter1[1, 24] = 1, 1
        filter2 = np.zeros((2, 25)); filter2[0, 1], filter2[1, 24] = 1, 1
        filter3 = np.zeros((2, 25)); filter3[0, 2], filter3[1, 24] = 1, 1
        custom_weights = np.stack([filter1, filter2, filter3], axis=0)  # (3, 2, 25)
        self.conv2.weight.data = torch.tensor(custom_weights, dtype=torch.float32)
        self.conv2.weight.requires_grad = False
        self.conv2.bias.data.zero_()
        self.conv2.bias.requires_grad = True

    def energy(self, dna_seq):
        """Conv stack + global max pooling -> feature energy."""
        x = self.conv1(dna_seq)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.max_pool1d(x, kernel_size=x.size(-1))  # global max pooling
        return x

    def energy2expression(self, x):
        """Boltzmann-weighted mapping with log transform."""
        beta = 1.0 / (0.001987 * 310.0)  # ~1.6234
        boltz = torch.exp(beta * (x + self.param_e0))
        rmax = torch.exp(self.param_max)
        rmin = torch.exp(self.param_min)
        return torch.log(rmin + rmax * boltz / (1.0 + boltz))

    def forward(self, dna_seq):
        x = self.energy(dna_seq)
        return self.energy2expression(x)


if __name__ == "__main__":

    SEED = 7777
    EPOCHS = 7
    BATCH_SIZE = 32
    TEST_SIZE = 0.2

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and sample data
    df_PL16 = pd.read_pickle("../tables/PL_C16SB.pkl")
    df_train = stratified_sample(df_PL16, 'LogGFP', [6000]*3 + [3000]*4 + [1000]*3, seed=SEED)
    df_train['Sequence'] = df_train.index.map(Parti)

    # Prepare data
    dna_seq = np.array(list(df_train['Sequence'].apply(dna_one_hot)))  # (N, 4, L)
    dna_func = df_train['LogGFP'].values * np.log(10)

    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        dna_seq, dna_func, test_size=TEST_SIZE, random_state=42
        )

    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_seq  = torch.tensor(X_test_seq,  dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)

    train_loader = DataLoader(TensorDataset(X_train_seq, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # Model / loss / optim
    seq_length = dna_seq.shape[2]
    model = DNAFunctionPredictor(seq_length=seq_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {"params": [model.conv1.weight, model.conv2.bias], "weight_decay": 0.003},
        {"params": [model.param_max, model.param_min],     "weight_decay": 0.00003},
        {"params": [model.param_e0],                       "weight_decay": 0.0},
        ], lr=0.001)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        for seq_batch, target_batch in train_loader:
            seq_batch, target_batch = seq_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            output = model(seq_batch)
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(X_train_seq)/np.log(10), y_train/np.log(10))
            test_loss  = criterion(model(X_test_seq)/np.log(10),  y_test/np.log(10))
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] - Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # Save parameters
    spacer_energy = dict(zip([18, 17, 16], -model.conv2.bias.cpu().detach().numpy()))
    spacer_energy = dict(reversed(list(spacer_energy.items())))

    rmin = model.param_min.detach().cpu().numpy().item()
    rmax = model.param_max.detach().cpu().numpy().item()

    df_logo35 = pd.DataFrame(np.array(-model.conv1.weight.data.cpu()[0]).T, columns=['A', 'C', 'G', 'T'])
    df_logo10 = pd.DataFrame(np.array(-model.conv1.weight.data.cpu()[1]).T, columns=['A', 'C', 'G', 'T'])
    pwm35 = df_logo35.sub(df_logo35.mean(axis=1), axis=0)
    pwm10 = df_logo10.sub(df_logo10.mean(axis=1), axis=0)

    e0 = model.param_e0.detach().cpu().numpy().item()
    e0 = e0 - df_logo35.mean(axis=1).sum() - df_logo10.mean(axis=1).sum()

    params = {}
    params['pwm35'] = pwm35
    params['pwm10'] = pwm10
    params['spacer'] = spacer_energy
    params['e0'] = -e0  # Δε_0'
    params['rmin'] = np.exp(rmin)
    params['rmax'] = np.exp(rmax)

    save_obj(params, "../models/Params_C16SB.pkl")
    print("Saved parameters to ../models/Params_C16SB.pkl")
