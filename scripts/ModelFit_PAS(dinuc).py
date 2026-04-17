## Standard libraries
import os
import random
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

## PyTorch and machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

## Utilities
from util import Motif2Seqs, getPFM, save_obj
from scan import MatrixScan


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mononucleotide + di-nucleotide models on PL_C17."
        )
    parser.add_argument(
        "--sampling-distribution", "-sd",
        type=str,
        default="3000,2000,2000,2000,1000,1000,1000,1000,1000,1000",
        help="Comma-separated bin counts for stratified sampling."
        )
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

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
    """One-hot encode a DNA string (A/C/G/T/-/N). Returns (4, L) by default."""
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


def PpurR(seq):
    """Embed a 12-mer (6-bp -35 + 6-bp -10) into the PurR reporter context."""
    return 'CCAC' + seq[:6] + 'TTTTCGTCAAGATCGGC' + seq[-6:] + 'TCCA'


DINUC_ORDER = [a + b for a in "ACGT" for b in "ACGT"]
BASE_IDX    = {b: i for i, b in enumerate("ACGT")}


def pairwise_dinuc_one_hot(seq, *, dtype=np.float32, invalid="zero"):
    """
    Generate 16 × C(L, 2) one-hot for all pairwise di-nucleotide combinations.

    Returns
    -------
    X : np.ndarray, shape (16, C(L, 2))
    """
    s     = seq.upper()
    L     = len(s)
    pairs = list(combinations(range(L), 2))
    X     = np.zeros((16, len(pairs)), dtype=dtype)

    for c, (i, j) in enumerate(pairs):
        a, b = s[i], s[j]
        if a in BASE_IDX and b in BASE_IDX:
            X[BASE_IDX[a] * 4 + BASE_IDX[b], c] = 1.0
        elif invalid == "raise":
            raise ValueError(f"Invalid base at pair ({i},{j}): '{a}{b}'")
    return X


# ── Model definitions ─────────────────────────────────────────────────────────

class DNAFunctionPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=6, stride=1)
        self._initialize_conv1()

        self.conv2 = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=17, stride=1)
        self._initialize_conv2()

        self.param_max = nn.Parameter(torch.tensor(np.log(1000.0)), requires_grad=True)
        self.param_min = nn.Parameter(torch.tensor(np.log(0.5)),    requires_grad=True)
        self.param_e0  = nn.Parameter(torch.tensor(0.0),            requires_grad=True)

        self.register_buffer("bias_mask", torch.tensor([1.0, 0.0, 1.0]))
        self.conv2.bias.register_hook(lambda grad: grad * self.bias_mask)

    def _initialize_conv1(self):
        motif1 = getPFM(Motif2Seqs("TTGACA"), ratios=True).values.T
        motif2 = getPFM(Motif2Seqs("TATAAT"), ratios=True).values.T
        self.conv1.weight.data = torch.tensor(
            np.stack([motif1, motif2], axis=0), dtype=torch.float32
            )
        self.conv1.weight.requires_grad = True
        self.conv1.bias.data.zero_()
        self.conv1.bias.requires_grad = False

    def _initialize_conv2(self):
        filter1 = np.zeros((2, 25)); filter1[0, 0], filter1[1, 24] = 1, 1
        filter2 = np.zeros((2, 25)); filter2[0, 1], filter2[1, 24] = 1, 1
        filter3 = np.zeros((2, 25)); filter3[0, 2], filter3[1, 24] = 1, 1
        self.conv2.weight.data = torch.tensor(
            np.stack([filter1, filter2, filter3], axis=0), dtype=torch.float32
            )
        self.conv2.weight.requires_grad = False
        self.conv2.bias.data.zero_()
        self.conv2.bias.requires_grad = True

    def energy(self, dna_seq):
        x = self.conv1(dna_seq)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.max_pool1d(x, kernel_size=x.size(-1))
        return x

    def energy2expression(self, x):
        beta  = 1.0 / (0.001987 * 310.0)
        boltz = torch.exp(beta * (x + self.param_e0))
        rmax  = torch.exp(self.param_max)
        rmin  = torch.exp(self.param_min)
        return torch.log(rmin + rmax * boltz / (1.0 + boltz))

    def forward(self, dna_seq):
        return self.energy2expression(self.energy(dna_seq))


class DiNucEstimation(nn.Module):
    """
    Estimate pairwise di-nucleotide correction to RNAP binding energy.

    rmax, rmin, e0 are fixed buffers inherited from the mononucleotide model.
    """
    def __init__(self, seq_pairs, rmax, rmin, e0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=1,
                               kernel_size=seq_pairs, stride=1)
        self._initialize_conv1()

        self.register_buffer("param_max", torch.as_tensor(float(rmax)))
        self.register_buffer("param_min", torch.as_tensor(float(rmin)))
        self.register_buffer("param_e0",  torch.as_tensor(float(e0)))

    def _initialize_conv1(self):
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.bias.zero_()
        self.conv1.weight.requires_grad = True
        self.conv1.bias.requires_grad   = False

    def energy(self, dinuc_oneHot):
        return self.conv1(dinuc_oneHot).squeeze(-1)

    def energy2expression(self, x):
        beta  = 1.0 / (0.001987 * 310.0)
        boltz = torch.exp(beta * (x + self.param_e0))
        rmax  = torch.exp(self.param_max)
        rmin  = torch.exp(self.param_min)
        return torch.log(torch.clamp(rmin + rmax * boltz / (1.0 + boltz), min=1e-30))

    def forward(self, dinuc_oneHot, pwm_energy):
        if pwm_energy.dim() == 1:
            pwm_energy = pwm_energy.unsqueeze(1)
        elif pwm_energy.dim() == 3 and pwm_energy.size(-1) == 1:
            pwm_energy = pwm_energy.squeeze(-1)
        return self.energy2expression(self.energy(dinuc_oneHot) + pwm_energy)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load & sample training data ───────────────────────────────────────────
    sampling_distribution = [int(x) for x in args.sampling_distribution.split(",")]
    df = pd.read_pickle("../tables/PL_C17.pkl")
    df_train = stratified_sample(df, 'LogGFP', sampling_distribution, seed=args.seed)
    df_train['Sequence'] = df_train.index.map(PpurR)

    # ── Section 2: Mononucleotide (PAS) model ────────────────────────────────
    dna_seq  = np.array(list(df_train['Sequence'].apply(dna_one_hot)))
    dna_func = df_train['LogGFP'].values * np.log(10)

    X_train, X_test, y_train, y_test = train_test_split(
        dna_seq, dna_func, test_size=0.2, random_state=42
        )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)

    model     = DNAFunctionPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    print("\n── Mononucleotide model ──")
    for epoch in range(args.epochs):
        model.train()
        for oneHot_batch, target_batch in train_loader:
            optimizer.zero_grad()
            criterion(model(oneHot_batch), target_batch).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tr_loss = criterion(model(X_train) / np.log(10), y_train / np.log(10))
            te_loss = criterion(model(X_test)  / np.log(10), y_test  / np.log(10))
        print(f"Epoch [{epoch + 1:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss.item():.4f}, Test Loss: {te_loss.item():.4f}")

    # Extract mononucleotide parameters
    spacer_energy = dict(zip([18, 17, 16], -model.conv2.bias.cpu().detach().numpy()))
    spacer_energy = dict(reversed(list(spacer_energy.items())))

    rmin_val = model.param_min.detach().cpu().numpy().item()
    rmax_val = model.param_max.detach().cpu().numpy().item()

    df_logo35 = pd.DataFrame(np.array(-model.conv1.weight.data.cpu()[0]).T, columns=['A', 'C', 'G', 'T'])
    df_logo10 = pd.DataFrame(np.array(-model.conv1.weight.data.cpu()[1]).T, columns=['A', 'C', 'G', 'T'])
    pwm35 = df_logo35.sub(df_logo35.mean(axis=1), axis=0)
    pwm10 = df_logo10.sub(df_logo10.mean(axis=1), axis=0)

    # e0_raw: raw model.param_e0, used as-is in DiNucEstimation (keeps energy scale)
    # e0_adj: adjusted for PWM normalization, used for display/tick labels
    e0_raw = model.param_e0.detach().cpu().numpy().item()
    e0_adj = e0_raw - df_logo35.mean(axis=1).sum() - df_logo10.mean(axis=1).sum()

    # ── Section 3: Di-nucleotide preprocessing ────────────────────────────────
    sequences    = list(df_train.index.map(PpurR))
    pwm          = pd.concat([pwm35, pwm10]).reset_index(drop=True)
    spacer_score = dict(zip([18, 17, 16], model.conv2.bias.cpu().detach().numpy()))

    ms = MatrixScan(sequences, -pwm, spacer_score=spacer_score)
    df_train[['minus35', 'minus10', 'spacer_length']] = \
        ms.result[['minus35', 'minus10', 'spacer_length']].values

    ## Compute per-sequence mononucleotide energy
    df_train['dE'] = (
        model.energy(torch.tensor(dna_seq, dtype=torch.float32).to(device))
        .flatten().cpu().detach().numpy()
        )
    df_train['dinuc_input'] = df_train['minus35'] + df_train['minus10']

    # ── Section 4: Di-nucleotide model ───────────────────────────────────────
    dinuc_oneHot = np.array(list(df_train['dinuc_input'].apply(pairwise_dinuc_one_hot)))
    pwm_energy   = df_train['dE'].values
    dna_func     = df_train['LogGFP'].values * np.log(10)

    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        dinuc_oneHot, pwm_energy, dna_func, test_size=0.2, random_state=42
        )

    X1_tr = torch.tensor(X1_tr, dtype=torch.float32).to(device)
    X1_te = torch.tensor(X1_te, dtype=torch.float32).to(device)
    X2_tr = torch.tensor(X2_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X2_te = torch.tensor(X2_te, dtype=torch.float32).unsqueeze(1).to(device)
    y_tr  = torch.tensor(y_tr,  dtype=torch.float32).unsqueeze(1).to(device)
    y_te  = torch.tensor(y_te,  dtype=torch.float32).unsqueeze(1).to(device)

    train_loader2 = DataLoader(TensorDataset(X1_tr, X2_tr, y_tr),
                               batch_size=args.batch_size, shuffle=True)

    seq_pairs = dinuc_oneHot.shape[2]
    model2    = DiNucEstimation(seq_pairs, rmax_val, rmin_val, e0_raw).to(device)
    optimizer = optim.Adam(model2.parameters(), lr=0.001, weight_decay=0.0)

    print("\n── Di-nucleotide model ──")
    for epoch in range(args.epochs):
        model2.train()
        for oh_b, en_b, tg_b in train_loader2:
            optimizer.zero_grad()
            criterion(model2(oh_b, en_b), tg_b).backward()
            optimizer.step()

        model2.eval()
        with torch.no_grad():
            tr_loss = criterion(model2(X1_tr, X2_tr) / np.log(10), y_tr / np.log(10))
            te_loss = criterion(model2(X1_te, X2_te) / np.log(10), y_te / np.log(10))
        print(f"Epoch [{epoch + 1:02d}/{args.epochs}] "
              f"Train Loss: {tr_loss.item():.4f}, Test Loss: {te_loss.item():.4f}")

    # ── Section 5: Build di-nucleotide interaction matrix ─────────────────────
    dinuc_dEterms = np.full((48, 48), np.nan)
    data_w = model2.conv1.weight.cpu().detach().numpy()[0]  # (16, C(12,2))

    for a, (i, j) in enumerate(combinations(range(12), 2)):
        dinuc_dEterms[4*i:4*(i+1), 4*j:4*(j+1)] = data_w[:, a].reshape(4, 4)

    upper    = dinuc_dEterms
    lower    = upper.T
    result   = np.nansum((upper, lower), axis=0)
    both_nan = np.isnan(upper) & np.isnan(lower)
    result[both_nan] = np.nan
    dinuc_dEterms = result

    # ── Save parameters ───────────────────────────────────────────────────────
    params_dinuc = {
        # Mononucleotide model params
        'pwm35':    pwm35,
        'pwm10':    pwm10,
        'spacer':   spacer_energy,
        'e0':       -e0_adj,        # Δε_0' for display / x-axis tick labels
        'e0_model': e0_raw,         # raw param_e0 used in model2 (for theory curve)
        'rmin':     np.exp(rmin_val),
        'rmax':     np.exp(rmax_val),
        # Di-nucleotide interaction matrix (48×48, symmetric)
        'dinuc_dEterms': dinuc_dEterms,
        }

    os.makedirs("../models", exist_ok=True)
    save_obj(params_dinuc, "../models/Params_C17(dinuc).pkl")
    print("\nSaved: ../models/Params_C17(dinuc).pkl")
