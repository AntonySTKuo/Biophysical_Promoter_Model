# ModelFit_TF_cond.py
# ------------------------------------------------------------
# Train 3 models (TetR, LuxR, CueR), each on basal+induced together with "cond".
# Uses your transforms.py (ToOneHotDNA) and dataset.py (DNADataset).
# Special handling:
#   - TetR uses cond in {-1, 0} and has rmin-correction applied only for TetR.
#   - LuxR, CueR use cond in {0, 1}.
# ------------------------------------------------------------

import re
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ========= Reproducibility & device =========
def set_seed(seed=77777):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(77777)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ========= Your modules =========
from transforms import ToOneHotDNA
from dataset import DNADataset
from util import save_obj, load_obj

# ========= Utilities =========
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

# ========= Model (cond-aware) =========
class DNAFunctionPredictor(nn.Module):
    """
    - Conv1d: in_channels=4, out_channels=1 (shared filter)
    - Global parameters:
        param_rmax, param_rmin in loge-space, param_E0, param_Eextra
        param_rmin_correct (only applied for TetR)
    - forward(dna_seq, conds):
        dna_seq: (N,4,L)
        conds  : (N,) or (N,1), values can be -1/0/1 (TetR uses -1/0; LuxR/CueR use 0/1)
    """
    def __init__(self, seq_length: int, tf_name: str):
        super().__init__()
        self.seq_length = seq_length
        self.tf_name = tf_name

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=7, stride=1, bias=True)
        self._initialize_conv1()

        self.param_rmax = nn.Parameter(torch.tensor(np.log(100.0), dtype=torch.float32))
        self.param_rmin = nn.Parameter(torch.tensor(np.log(0.5),   dtype=torch.float32))
        self.param_E0   = nn.Parameter(torch.tensor(0.0,           dtype=torch.float32))
        self.param_Eextra = nn.Parameter(torch.tensor(0.0,         dtype=torch.float32))

        # Only meaningful for TetR (will be ignored for others)
        self.param_rmin_correct = nn.Parameter(torch.tensor(4.0,   dtype=torch.float32))

    def _initialize_conv1(self):
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.conv1.weight.requires_grad = True
        self.conv1.bias.requires_grad = False  # keep disabled as you did

    @torch.no_grad()
    def _check_inputs(self, dna_seq, conds):
        assert dna_seq.dim() == 3 and dna_seq.size(1) == 4, "dna_seq must be (N,4,L)"
        assert conds.dim() in (1,2), "conds must be (N,) or (N,1)"
        if conds.dim() == 2:
            assert conds.size(1) == 1

    def seq2energy(self, dna_seq: torch.Tensor) -> torch.Tensor:
        x = self.conv1(dna_seq)                      # (N,1,L-6)
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (N,1)
        return x

    def energy2expression(self, energy: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        beta = 1.0 / (0.001987 * 310.0)  # ~1.6234

        if conds.dim() == 1:
            conds = conds.unsqueeze(-1)  # (N,1)

        # E_act only when cond > 0 (for TetR cond could be -1 or 0; for LuxR/CueR cond is 0/1)
        boltz = torch.exp(beta * (energy + self.param_E0 + self.param_Eextra * conds))

        Rmax = torch.exp(self.param_rmax)
        Rmin = torch.exp(self.param_rmin)

        # Apply the special rmin correction ONLY for TetR
        if self.tf_name == "TetR":
            Rmin = torch.exp(self.param_rmin) - self.param_rmin_correct * conds

        expression = torch.log((Rmin + Rmax * boltz) / (1.0 + boltz))  # (N,1), natural log
        return expression

    def forward(self, dna_seq: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        # self._check_inputs(dna_seq, conds)
        energy = self.seq2energy(dna_seq)               # (N,1)
        output = self.energy2expression(energy, conds)  # (N,1)
        return output

# ========= Data building per TF =========
def build_df_samples_for_tf(tf: str):
    if tf == 'TetR':
        lib = 'TetR'
        dfB = pd.read_pickle(f"../tables/PL_{lib}b.pkl")
        dfI = pd.read_pickle(f"../tables/PL_{lib}i.pkl")

        dfB = getSubset(dfB, '...acaAa..Tt', mode='drop'); dfI = getSubset(dfI, '...acaAa..Tt', mode='drop')
        dfB = getSubset(dfB, 'AA.aca.a...t', mode='drop'); dfI = getSubset(dfI, 'AA.aca.a...t', mode='drop')

        dfB['seq'] = dfB.index.map(extract_uppercases)
        dfI['seq'] = dfI.index.map(extract_uppercases)
        dfB['cond'] = -1
        dfI['cond'] =  0

        df_sample = pd.concat([
            stratified_sample(dfB, 'LogGFP', [500]*4 + [100]*4)[['seq', 'cond', 'LogGFP']],
            stratified_sample(dfI, 'LogGFP', [1500]*8)[['seq', 'cond', 'LogGFP']],
            ]).reset_index(drop=True)
        return df_sample

    if tf == 'LuxR':
        lib = 'LuxR'
        dfB = pd.read_pickle(f"../tables/PL_{lib}b.pkl")
        dfI = pd.read_pickle(f"../tables/PL_{lib}i.pkl")

        dfB = getSubset(dfB, 'tTGac.GaTA.t', mode='drop'); dfI = getSubset(dfI, 'tTGac.GaTA.t', mode='drop')

        dfB['seq'] = dfB.index.map(extract_uppercases)
        dfI['seq'] = dfI.index.map(extract_uppercases)
        dfB['cond'] = 0
        dfI['cond'] = 1

        df_sample = pd.concat([
            stratified_sample(dfB, 'LogGFP', [500]*8)[['seq', 'cond', 'LogGFP']],
            stratified_sample(dfI, 'LogGFP', [1000]*4 + [2000]*4)[['seq', 'cond', 'LogGFP']],
            ]).reset_index(drop=True)
        return df_sample

    if tf == 'CueR':
        lib = 'CueR'
        dfB = pd.read_pickle(f"../tables/PL_{lib}b.pkl")
        dfI = pd.read_pickle(f"../tables/PL_{lib}i.pkl")

        dfB = getSubset(dfB, 'TTGaccAa..Tt', mode='drop'); dfI = getSubset(dfI, 'TTGaccAa..Tt', mode='drop')

        dfB['seq'] = dfB.index.map(extract_uppercases)
        dfI['seq'] = dfI.index.map(extract_uppercases)
        dfB['cond'] = 0
        dfI['cond'] = 1

        df_sample = pd.concat([
            stratified_sample(dfB, 'LogGFP', [100]*4 + [100]*4)[['seq', 'cond', 'LogGFP']],
            stratified_sample(dfI, 'LogGFP', [3000] + [1000]*7)[['seq', 'cond', 'LogGFP']],
            ]).reset_index(drop=True)
        return df_sample

    raise ValueError("tf must be one of {'TetR','LuxR','CueR'}")

# ========= Build dataset (loge conversion inside) =========
def make_dataset_from_df(df, seq_tf):
    # Convert log10 to natural log (loge) so that y matches model's output space
    df_use = df[['seq', 'cond', 'LogGFP']].copy()
    df_use['LogGFP'] = df_use['LogGFP'] * np.log(10)

    ds = DNADataset(
        df_use,
        seq_transform=seq_tf,
        cond_transform=lambda z: torch.tensor([float(z)], dtype=torch.float32),
        y_transform=lambda y: torch.tensor([float(y)], dtype=torch.float32),
        )
    return ds

def dataset_to_tensors(ds):
    # Iterate once to stack tensors (agnostic to DNADataset internals)
    X_list, C_list, y_list = [], [], []
    loader = DataLoader(ds, batch_size=4096, shuffle=False)
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            xb, cb, yb = batch
        elif isinstance(batch, dict):
            xb, cb, yb = batch["seq"], batch["cond"], batch["y"]
        else:
            raise RuntimeError("Unsupported batch type from DNADataset.")
        X_list.append(xb); C_list.append(cb); y_list.append(yb)
    X = torch.cat(X_list, 0).to(device)   # (N,4,L)
    C = torch.cat(C_list, 0).to(device)   # (N,1)
    y = torch.cat(y_list, 0).to(device)   # (N,1)
    return X, C, y

# ========= Train one TF model =========
def train_one_tf(tf_name: str, epochs=30, lr=1e-3, wd=1e-5, batch_size=128, seed=42,
                 init_rmax_loge=None):
    print(f"\n[INFO] ==== Training model for {tf_name} (basal + induced) ====")
    df_sample = build_df_samples_for_tf(tf_name)

    # Split
    df_tr, df_te = train_test_split(df_sample, test_size=0.2, random_state=seed)

    # Dataset / tensors
    seq_tf = ToOneHotDNA()
    ds_tr = make_dataset_from_df(df_tr, seq_tf)
    ds_te = make_dataset_from_df(df_te, seq_tf)

    X_tr, C_tr, y_tr = dataset_to_tensors(ds_tr)
    X_te, C_te, y_te = dataset_to_tensors(ds_te)

    # Model
    seq_length = X_tr.shape[-1]
    model = DNAFunctionPredictor(seq_length=seq_length, tf_name=tf_name).to(device)

    # Optional init for specific TF (e.g., LuxR larger rmax)
    if init_rmax_loge is not None:
        with torch.no_grad():
            model.param_rmax.copy_(torch.tensor(float(init_rmax_loge), device=device))

    # Optim & loss
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # DataLoaders
    tr_loader = DataLoader(TensorDataset(X_tr, C_tr, y_tr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, C_te, y_te), batch_size=batch_size, shuffle=False)

    # Train loop
    for ep in range(1, epochs+1):
        model.train()
        for xb, cb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb, cb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_mse = criterion(model(X_tr, C_tr), y_tr).item()
            te_mse = criterion(model(X_te, C_te), y_te).item()
        print(f"[{tf_name}] Epoch {ep:02d}/{epochs} | train MSE: {tr_mse:.4f} | test MSE: {te_mse:.4f}")
        # print(float(np.exp(model.param_rmax.detach().cpu().numpy().item())))

    # ===== Save learned parameters =====
    with torch.no_grad():
        rmin_loge = model.param_rmin.detach().cpu().numpy().item()
        rmax_loge = model.param_rmax.detach().cpu().numpy().item()
        E0        = model.param_E0.detach().cpu().numpy().item()          # -Δε_0 (pre-adjust)
        Eextra    = model.param_Eextra.detach().cpu().numpy().item()
        rmin_corr = model.param_rmin_correct.detach().cpu().numpy().item()

        pwm = pd.DataFrame(
            np.array(-model.conv1.weight.data.detach().cpu()[0]).T,
            columns=['A', 'C', 'G', 'T']
            )
        # Normalize PWM per position and adjust E0 accordingly
        E0 = E0 - pwm.mean(axis=1).sum()     # adjust -Δε_0 for normalization
        pwm = pwm.sub(pwm.mean(axis=1), axis=0)

        params = {
            'tf': tf_name,
            'pwm': pwm,                        # position x {A,C,G,T}
            'Delta_e0_prime': -E0,             # Δε_0' (after sign flip)
            'Eextra': Eextra,
            'rmin': float(np.exp(rmin_loge)),  # back to linear scale
            'rmax': float(np.exp(rmax_loge)),
            'rmin_correction_only_for_TetR': rmin_corr if tf_name == 'TetR' else 0.0,
            'target_log_base': 'e',            # training target space
            'cond_encoding': {'TetR': 'basal=-1, induced=0', 'LuxR':'0/1', 'CueR':'0/1'}[tf_name],
            }

        out_params = f"../models/Params_{tf_name}.pkl"
        save_obj(params, out_params)
        print(f"[INFO] Saved parameters to {out_params}")

        # out_weights = f"../models/Model_{tf_name}.pt"
        # Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
        # torch.save({'state_dict': model.state_dict(),
        #             'seq_length': seq_length,
        #             'tf': tf_name}, out_weights)
        # print(f"[INFO] Saved model weights to {out_weights}")

    return dict(tf=tf_name, train_mse=tr_mse, test_mse=te_mse, seq_length=seq_length)

# ========= Main: train three TFs =========
if __name__ == "__main__":
    results = []

    # You can tweak per-TF settings here
    per_tf_cfg = {
        'TetR': dict(epochs=150, init_rmax_loge=None),
        'LuxR': dict(epochs=150, init_rmax_loge=math.log(500.0)),  # optional: larger initial rmax
        'CueR': dict(epochs=150, init_rmax_loge=None),
        }

    for tf in ['TetR', 'LuxR', 'CueR']:
        cfg = per_tf_cfg.get(tf, {})
        res = train_one_tf(
            tf_name=tf,
            epochs=cfg.get('epochs', 30),
            lr=1e-3, wd=1e-5, batch_size=128, seed=42,
            init_rmax_loge=cfg.get('init_rmax_loge', None)
            )
        results.append(res)

    print("\n[SUMMARY]")
    for r in results:
        print(f"- {r['tf']}: seqL={r['seq_length']}, trainMSE={r['train_mse']:.4f}, testMSE={r['test_mse']:.4f}")

    # Parameter correction
    beta = 1.0 / (0.001987 * 310.0)  # ~1.6234
    for lib in ['TetR', 'LuxR', 'CueR']:
        params = load_obj(f"../models/Params_{lib}.pkl")
        if lib == 'TetR':
            params['Delta_e0_prime'] -= np.log(2.4)/beta
            params['rmax'] /= 2.4
        # elif lib == 'LuxR':
        # elif lib == 'CueR':
        save_obj(params, f"../models/Params_{lib}.pkl")
