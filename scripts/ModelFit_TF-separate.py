import re
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split

from util import writeTXT

# ===== Reproducibility & Device =====
def set_seed(seed=77777):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(77777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===== Use your modules =====
from transforms import ToOneHotDNA
from dataset import DNADataset
from util import save_obj, load_obj

# ===== Utilities =====
def stratified_sample(df, col, counts, seed=42):
    np.random.seed(seed)
    df = df.copy()
    df['__grp__'] = pd.cut(df[col], bins=len(counts))
    out = []
    for grp, cnt in zip(sorted(df['__grp__'].dropna().unique(), key=lambda x: x.left), counts):
        sub = df[df['__grp__'] == grp]
        if len(sub) == 0:
            continue
        out.append(sub.sample(cnt, replace=True, random_state=seed))
    if not out:
        raise ValueError("Stratified sampling yielded empty set; adjust bins/counts or input df.")
    return pd.concat(out, axis=0).drop(columns='__grp__')

def getSubset(df, motif, mode='select'):
    prog = re.compile(motif)
    mask = np.array([bool(prog.search(s)) for s in df.index])
    if mode == 'select':
        return df[mask]
    elif mode == 'drop':
        return df[~mask]
    else:
        raise ValueError("mode must be 'select' or 'drop'")

def extract_uppercases(s):
    return ''.join([c for c in s if c.isupper()])

def save_obj(obj, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def dataset_to_numpy(ds: Dataset):
    """
    Try DNADataset.get_all(); if not available, iterate a DataLoader to stack.
    Return:
      X: (N,4,L) float32
      y: (N,1)   float32
    (cond is ignored)
    """
    # Fast path if your DNADataset implements get_all()
    if hasattr(ds, "get_all"):
        X, C, y = ds.get_all()              # expect tensors
        return X.numpy().astype(np.float32), y.numpy().astype(np.float32)

    # Fallback: iterate once
    X_list, y_list = [], []
    tmp_loader = DataLoader(ds, batch_size=4096, shuffle=False)
    for batch in tmp_loader:
        # Handle tuple or dict style
        if isinstance(batch, (list, tuple)):
            xb, cb, yb = batch
        elif isinstance(batch, dict):
            xb, cb, yb = batch["seq"], batch["cond"], batch["y"]
        else:
            raise RuntimeError("Unsupported batch type from DNADataset.")
        X_list.append(xb.cpu())
        y_list.append(yb.cpu())
    X = torch.cat(X_list, dim=0).numpy().astype(np.float32)  # (N, 4, L)
    y = torch.cat(y_list, dim=0).numpy().astype(np.float32)  # (N, 1)
    return X, y

# ===== Model (no-cond) =====
class DNAFunctionPredictor(nn.Module):
    """
    - Conv1d: in_channels=4, out_channels=1 (shared filter)
    - param_max := log(rmax); param_min := log(rmin); param_e0 := bias in energy
    - energy -> expression via Boltzmann-weighted mapping in log space
    """
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=7, stride=1, bias=False)
        self._initialize_conv1()
        
        self.param_max = nn.Parameter(torch.tensor(np.log(100.0), dtype=torch.float32), requires_grad=True)
        self.param_min = nn.Parameter(torch.tensor(np.log(4.0),   dtype=torch.float32), requires_grad=True)
        self.param_e0  = nn.Parameter(torch.tensor(0.0,           dtype=torch.float32), requires_grad=True)
        # self.param_e1  = nn.Parameter(torch.tensor(0.0,           dtype=torch.float32), requires_grad=True)  # to test parameter sensitivity

    def _initialize_conv1(self):
        with torch.no_grad():
            self.conv1.weight.zero_()

    def energy(self, dna_seq):
        # dna_seq: (B, 4, L)
        x = self.conv1(dna_seq)                      # (B, 1, L-6)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (B, 1)
        return x

    def energy2expression(self, x):
        # x: (B, 1)
        beta = 1.0 / (0.001987 * 310.0)  # ~1.6234
        bw = torch.exp(beta * (x + self.param_e0))# + self.param_e1))
        Max = torch.exp(self.param_max)
        Min = torch.exp(self.param_min)
        expr = torch.log((Min + Max * bw) / (1.0 + bw))
        return expr

    def forward(self, dna_seq):
        x = self.energy(dna_seq)
        y = self.energy2expression(x)
        return y

# ===== Data IO =====
def load_and_filter_one(lib):
    """
    Load ../tables/PL_{lib}b.pkl and ../tables/PL_{lib}i.pkl
    Add 'seq' (uppercase bases only). Return dfB, dfI.
    Apply your motif-drop rules per library.
    """
    dfB = pd.read_pickle(f"../tables/PL_{lib}b.pkl")
    dfI = pd.read_pickle(f"../tables/PL_{lib}i.pkl")

    if lib == 'TetR':
        for motif in ['...acaAa..Tt', 'AA.aca.a...t']:
            dfB = getSubset(dfB, motif, mode='drop'); dfI = getSubset(dfI, motif, mode='drop')
    elif lib == 'LuxR':
        for motif in ['tTGac.GaTA.t']:
            dfB = getSubset(dfB, motif, mode='drop'); dfI = getSubset(dfI, motif, mode='drop')
    elif lib == 'CueR':
        for motif in ['TTGaccAa..Tt']:
            dfB = getSubset(dfB, motif, mode='drop'); dfI = getSubset(dfI, motif, mode='drop')

    dfB = dfB.copy(); dfI = dfI.copy()
    dfB['seq'] = dfB.index.map(extract_uppercases)
    dfI['seq'] = dfI.index.map(extract_uppercases)
    return dfB, dfI

def make_dataset_from_df(df, cond_value, seq_tf):
    """
    DNADataset needs columns: seq, cond, LogGFP
    - convert log10 to loge
    """
    df_use = df[['seq', 'LogGFP']].copy()
    df_use['cond'] = cond_value

    # log10 → loge
    df_use['LogGFP'] = df_use['LogGFP'].apply(lambda v: v * np.log(10))

    ds = DNADataset(
        df_use,
        seq_transform=seq_tf,
        cond_transform=lambda z: torch.tensor([float(z)], dtype=torch.float32),
        y_transform=lambda y: torch.tensor([float(y)], dtype=torch.float32),
        )
    return ds


# ===== Train/Eval =====
def fit_one_dataset(lib, cond_name, df, special_cfg=None, test_size=0.2,batch_size=64, lr=1e-3, wd=1e-5, seed=42):
    
    print(f"\n[INFO] ==== Training {lib}-{cond_name} ====")

    if lib == 'LuxR':
        df_train = stratified_sample(df, 'LogGFP', [3000] + [1000]*7, seed=seed)[['LogGFP', 'seq']].reset_index(drop=True)
    else:
        df_train = stratified_sample(df, 'LogGFP', [1000]*8, seed=seed)[['LogGFP', 'seq']].reset_index(drop=True)

    seq_tf = ToOneHotDNA()
    cond_val = 0 if cond_name == 'basal' else 1
    
    df_tr, df_te = train_test_split(df_train, test_size=test_size, random_state=seed)
    ds_tr = make_dataset_from_df(df_tr, cond_val, seq_tf)
    ds_te = make_dataset_from_df(df_te, cond_val, seq_tf)

    X_tr, y_tr = dataset_to_numpy(ds_tr)   # (N,4,L), (N,1)
    X_te, y_te = dataset_to_numpy(ds_te)

    # Tensors to device
    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    # Model
    seq_length = X_tr_t.shape[-1]
    model = DNAFunctionPredictor(seq_length=seq_length).to(device)

    # Special handling for LuxR induced (LuxRi)
    EPOCHS = 60
    if special_cfg is not None:
        if 'param_max_init' in special_cfg:
            with torch.no_grad():
                model.param_max.copy_(torch.tensor(float(special_cfg['param_max_init']), device=device))
        if 'EPOCHS' in special_cfg:
            EPOCHS = int(special_cfg['EPOCHS'])

    # Optim & loss
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # DataLoaders
    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

    # Train loop
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_mse = criterion(model(X_tr_t), y_tr_t).item()
            te_mse = criterion(model(X_te_t), y_te_t).item()
        print(f"[{lib}-{cond_name}] Epoch {ep:02d}/{EPOCHS} | train MSE: {tr_mse:.4f} | test MSE: {te_mse:.4f}")

    # ===== Save learned parameters for this dataset =====
    with torch.no_grad():
        rmin = model.param_min.detach().cpu().numpy().item()
        rmax = model.param_max.detach().cpu().numpy().item()
        e0   = model.param_e0.detach().cpu().numpy().item()  # -Δε_0
        # e1   = model.param_e1.detach().cpu().numpy().item()  # to test parameter sensitivity
        # print(f"dE1 = {e1}")
        # writeTXT("../models/document.txt", [f"{lib}-{cond_name}: dE1 = {e1}"], 'a')

        # conv1 weight: shape (1, 4, k). Take [0] -> (4, k) and transpose to (k, 4)
        pwm = pd.DataFrame(
            np.array(-model.conv1.weight.data.cpu()[0]).T,
            columns=['A', 'C', 'G', 'T']
            )

        # PWM normalization & e0 adjustment
        e0 = e0 - pwm.mean(axis=1).sum()          # -Δε_0 adjustment for pwm normalization
        pwm = pwm.sub(pwm.mean(axis=1), axis=0)   # Normalize each nucleotide position

        params = {
            'pwm': pwm,           # pandas DataFrame (positions x nucleotides)
            'e0': -e0,            # Δε_0'
            'rmin': np.exp(rmin), # back to linear scale
            'rmax': np.exp(rmax),
            }

        fname = f"../models/Params_{lib}{'b' if cond_name=='basal' else 'i'}.pkl"
        save_obj(params, fname)
        print(f"[INFO] Saved parameters to {fname}")

    return {
        'lib': lib,
        'cond': cond_name,
        'model': model,
        'train_mse': tr_mse,
        'test_mse': te_mse,
        'seq_length': seq_length,
        }

# ===== Main: loop over 6 datasets =====
if __name__ == "__main__":
    results = []
    for lib in ['TetR', 'LuxR', 'CueR']:
        dfB, dfI = load_and_filter_one(lib)
        combos = [
            (lib, 'basal',   dfB, None),
            # LuxR induced has custom EPOCHS and initial rmax
            (lib, 'induced', dfI, {'EPOCHS': 30, 'param_max_init': np.log(500.0)}) if lib=='LuxR' else (lib, 'induced', dfI, None),
            ]
        for lib_, cond_name, df_, special in combos:
            res = fit_one_dataset(
                lib_, cond_name, df_,
                special_cfg=special,
                test_size=0.2, batch_size=64, lr=1e-3, wd=1e-5, seed=42
                )
            results.append(res)

    print("\n[SUMMARY]")
    for r in results:
        print(f"- {r['lib']}-{r['cond']}: seqL={r['seq_length']}, trainMSE={r['train_mse']:.4f}, testMSE={r['test_mse']:.4f}")

    # Parameter correction
    beta = 1.0 / (0.001987 * 310.0)  # ~1.6234
    for lib in ['TetR', 'LuxR', 'CueR']:
        paramsB = load_obj(f"../models/Params_{lib}b.pkl")
        paramsI = load_obj(f"../models/Params_{lib}i.pkl")
        if lib == 'TetR':
            paramsI['e0'] -= np.log(1.2)/beta
            paramsI['rmax'] /= 1.2
            paramsB['e0'] += np.log(paramsI['rmax']/paramsB['rmax'])/beta
            paramsB['rmax'] = paramsI['rmax']
        elif lib == 'LuxR':
            paramsB['e0'] += np.log(paramsI['rmax']/paramsB['rmax'])/beta
            paramsB['rmax'] = paramsI['rmax']
        elif lib == 'CueR':
            paramsB['e0'] += np.log(paramsI['rmax']/paramsB['rmax'])/beta
            paramsB['rmax'] = paramsI['rmax']
        save_obj(paramsB, f"../models/Params_{lib}b.pkl")
        save_obj(paramsI, f"../models/Params_{lib}i.pkl")
