# #only reproducible
# import os, json, math, time, random, argparse
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# try:
#     from rdkit import Chem
#     from rdkit.Chem import MolToSmiles
# except Exception:
#     Chem = None
#     MolToSmiles = None

# from sample2 import CondTransformer
# from tokenizer import SMILESTokenizer, save_tokens, load_tokens


# # ---------------------------
# # Reproducibility (optional)
# # ---------------------------
# def set_seed(seed: int = 42):    
#     random.seed(seed) 
#     np.random.seed(seed) 
#     torch.manual_seed(seed) 
#     torch.cuda.manual_seed_all(seed) 
#     torch.backends.cudnn.deterministic = True 
#     torch.backends.cudnn.benchmark = False


# # ---------------------------
# # Dataset
# # ---------------------------
# class PolymerGenDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, tokenizer: SMILESTokenizer, max_len: int):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.smiles = df['SMILES'].astype(str).values
#         self.targets = df[['Tg', 'MAC']].values.astype(np.float32)

#     def __len__(self):
#         return len(self.smiles)

#     def __getitem__(self, idx):
#         s = self.smiles[idx]
#         y = torch.tensor(self.tokenizer.encode(s, self.max_len), dtype=torch.long)
#         cond = torch.tensor(self.targets[idx], dtype=torch.float32)
#         return cond, y


# # ---------------------------
# # Collate (dynamic trim to max non-pad per batch)
# # ---------------------------
# def make_collate_fn(pad_id: int):
#     def collate(batch):
#         conds, ys = zip(*batch)
#         lens = [int((y != pad_id).sum().item()) for y in ys]
#         T = max(lens)
#         ys = torch.stack([y[:T] for y in ys], dim=0)
#         conds = torch.stack(conds, dim=0)
#         return conds, ys
#     return collate


# # ---------------------------
# # Train / Eval helpers
# # ---------------------------
# @torch.no_grad()
# def evaluate(model, loader, loss_fn, device):
#     model.eval()
#     total, n = 0.0, 0
#     for cond, y in loader:
#         cond = cond.to(device)
#         y = y.to(device)
#         logits = model(y[:, :-1], cond)
#         loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
#         total += loss.item()
#         n += 1
#     return total / max(n, 1)


# def train(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     scheduler,
#     loss_fn,
#     device,
#     epochs=20,
#     grad_clip=1.0,
# ):
#     scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
#     global_step = 0
#     for epoch in range(1, epochs + 1):
#         model.train()
#         t0 = time.time()
#         running = 0.0
#         for step, (cond, y) in enumerate(train_loader, start=1):
#             cond = cond.to(device)
#             y = y.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
#                 logits = model(y[:, :-1], cond)
#                 loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
#             scaler.scale(loss).backward()
#             if grad_clip is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()
#             running += loss.item()
#             global_step += 1
#             if step % 50 == 0:
#                 lr = scheduler.get_last_lr()[0]
#                 print(f"Epoch {epoch:02d} | step {step:04d} | lr {lr:.2e} | train loss {running/50:.4f}")
#                 running = 0.0
#         val_loss = evaluate(model, val_loader, loss_fn, device)
#         dt = time.time() - t0
#         print(f"Epoch {epoch:02d} done in {dt:.1f}s | val loss {val_loss:.4f}")


# # ---------------------------
# # Sampling utils
# # ---------------------------
# @torch.no_grad()
# def sample_once(model, tokenizer, cond_vec, device, max_len=128, temperature=1.0, top_k=50, top_p=None):
#     model.eval()
#     y = torch.tensor([[tokenizer.stoi['<bos>']]], device=device, dtype=torch.long)
#     cond_vec = cond_vec.unsqueeze(0).to(device)
#     for _ in range(max_len):
#         logits = model(y, cond_vec)[:, -1, :] / max(temperature, 1e-6)
#         if top_p is not None:
#             # nucleus sampling (batch size 1)
#             sorted_logits, sorted_idx = torch.sort(logits, descending=True)
#             probs  = F.softmax(sorted_logits, dim=-1)
#             csum   = torch.cumsum(probs, dim=-1)
#             k      = int((csum <= top_p).sum().item())
#             k      = max(k, 1)
#             logits_trim = sorted_logits[:, :k]
#             idx_trim    = sorted_idx[:, :k]
#             probs_trim  = F.softmax(logits_trim, dim=-1)
#             choice      = torch.multinomial(probs_trim, 1)
#             next_token  = idx_trim.gather(1, choice)
#         else:
#             v, idx = torch.topk(logits, min(top_k, logits.size(-1)))
#             probs   = F.softmax(v, dim=-1)
#             next_token = idx.gather(1, torch.multinomial(probs, 1))
#         y = torch.cat([y, next_token], dim=1)
#         if next_token.item() == tokenizer.stoi['<eos>']:
#             break
#     return tokenizer.decode(y[0].tolist())


# def canonicalize(smiles: str) -> str:
#     if Chem is None:
#         return smiles
#     try:
#         mol = Chem.MolFromSmiles(smiles, sanitize=True)
#         if mol is None:
#             return ''
#         return MolToSmiles(mol, canonical=True)
#     except Exception:
#         return ''


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_csv', default='/csl/users/2026nnandaku/cluster/PolymerDesign/1Dataset/PI1M_Tg_MAC.csv')
#     parser.add_argument('--out_dir',  default='Transformer')
#     parser.add_argument('--max_len',  type=int, default=128)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs',   type=int, default=30)
#     parser.add_argument('--lr',       type=float, default=6e-4)  # higher base lr w/ schedule
#     parser.add_argument('--seed',     type=int, default=42)
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)
#     set_seed(args.seed)

#     # 1) Load data
#     df = pd.read_csv(args.data_csv)
#     df = df[df['SMILES'].notna()]

#     # Save canonical training set for novelty checks
#     if Chem is not None:
#         can_train = []
#         for s in df['SMILES'].astype(str).values:
#             cs = canonicalize(s)
#             if cs:
#                 can_train.append(cs)
#         can_train = set(can_train)
#     else:
#         can_train = set(df['SMILES'].astype(str).values)

#     # 2) Scale conditions (and persist scaler)
#     scaler = MinMaxScaler()
#     df[['Tg', 'MAC']] = scaler.fit_transform(df[['Tg', 'MAC']])
#     joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))

#     # 3) Build tokenizer from THIS dataset and persist tokens
#     tokenizer = SMILESTokenizer.from_smiles_list(df['SMILES'].astype(str).tolist())
#     save_tokens(tokenizer.tokens, os.path.join(args.out_dir, 'vocab.json'))

#     # 4) Datasets + loaders (dynamic trim)
#     dataset = PolymerGenDataset(df, tokenizer, args.max_len)
#     n_train = int(0.9 * len(dataset))
#     n_val   = len(dataset) - n_train
#     train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

#     pad_id = tokenizer.stoi['<pad>']
#     collate_fn = make_collate_fn(pad_id)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
#                               num_workers=4, pin_memory=True, collate_fn=collate_fn)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
#                               num_workers=4, pin_memory=True, collate_fn=collate_fn)

#     # 5) Model / optimizer / loss / schedule
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cfg = dict(vocab_size=len(tokenizer.tokens), d_model=256, nhead=4, num_layers=6,
#                dim_feedforward=1024, cond_dim=2, max_len=args.max_len, pad_id=pad_id)
#     model = CondTransformer(
#     vocab_size=len(tokenizer.tokens),
#     d_model=256,
#     nhead=4,
#     num_layers=6,
#     dim_feedforward=1024,
#     cond_dim=2,
#     max_len=args.max_len,
#     pad_id=pad_id,
#     dropout=0.2,
#     ).to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

#     steps_per_epoch = len(train_loader)
#     total_steps = steps_per_epoch * args.epochs
#     warmup = 1000  # <- fixed small warmup
#     def lr_lambda(step):
#         if step < warmup:
#             return float(step + 1) / float(warmup)
#         prog = (step - warmup) / max(1, total_steps - warmup)
#         return 0.5 * (1.0 + math.cos(math.pi * prog))
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

#     # 6) Train
#     print("Training begins…")
#     train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device,
#       epochs=args.epochs, grad_clip=0.5)

#     # 7) Save bundle (state_dict + config)
#     torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_state.pt'))
#     with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
#         json.dump(cfg, f)
#     print("Saved: model_state.pt, config.json, vocab.json, scaler.pkl")

#     # 8) Sampling demo + validity/novelty stats
#     print("\nGenerating samples…")
#     n_samples = 200
#     valid = []
#     unique_can = set()

#     # sample conditions from empirical (scaled) training distribution
#     cond_mat = df[['Tg', 'MAC']].values
#     for _ in range(n_samples):
#         i = np.random.randint(0, cond_mat.shape[0])
#         cond = torch.tensor(cond_mat[i], dtype=torch.float32, device=device)
#         s = sample_once(model, tokenizer, cond, device, max_len=args.max_len, temperature=0.9, top_k=50)
#         cs = canonicalize(s)
#         if cs:
#             valid.append(cs)
#             unique_can.add(cs)

#     n_valid = len(valid)
#     n_unique = len(unique_can)
#     n_new = len([c for c in unique_can if c not in can_train])
#     print(f"Valid: {n_valid}/{n_samples} | Unique valid: {n_unique} | Unique NEW valid: {n_new}")

#     out_csv = os.path.join(args.out_dir, 'samples.csv')
#     pd.DataFrame({'SMILES': list(unique_can)}).to_csv(out_csv, index=False)
#     print(f"Saved sample SMILES to {out_csv}")


# if __name__ == '__main__':
#     main()


#Reproducible + variability
import os, json, math, time, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import joblib

try:
    from rdkit import Chem
    from rdkit.Chem import MolToSmiles
except Exception:
    Chem = None
    MolToSmiles = None

from sample2 import CondTransformer
from tokenizer import SMILESTokenizer, save_tokens, load_tokens


# ---------------------------
# Reproducibility (flexible)
# ---------------------------
def set_seed(seed: int | None = None):
    """Seed all RNGs. If seed is None, use a time-based seed."""
    if seed is None:
        seed = int(time.time())
    print(f"[seed] Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


# ---------------------------
# Dataset
# ---------------------------
class PolymerGenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SMILESTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.smiles = df['SMILES'].astype(str).values
        self.targets = df[['Tg', 'MAC']].values.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        y = torch.tensor(self.tokenizer.encode(s, self.max_len), dtype=torch.long)
        cond = torch.tensor(self.targets[idx], dtype=torch.float32)
        return cond, y


# ---------------------------
# Collate (dynamic trim to max non-pad per batch)
# ---------------------------
def make_collate_fn(pad_id: int):
    def collate(batch):
        conds, ys = zip(*batch)
        lens = [int((y != pad_id).sum().item()) for y in ys]
        T = max(lens)
        ys = torch.stack([y[:T] for y in ys], dim=0)
        conds = torch.stack(conds, dim=0)
        return conds, ys
    return collate


# ---------------------------
# Train / Eval helpers
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for cond, y in loader:
        cond = cond.to(device)
        y = y.to(device)
        logits = model(y[:, :-1], cond)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs=20,
    grad_clip=1.0,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (cond, y) in enumerate(train_loader, start=1):
            cond = cond.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(y[:, :-1], cond)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            global_step += 1
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:02d} | step {step:04d} | lr {lr:.2e} | train loss {running/50:.4f}")
                running = 0.0
        val_loss = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} done in {dt:.1f}s | val loss {val_loss:.4f}")


# ---------------------------
# Sampling utils
# ---------------------------
@torch.no_grad()
def sample_once(model, tokenizer, cond_vec, device, max_len=128, temperature=1.0, top_k=50, top_p=None):
    model.eval()
    y = torch.tensor([[tokenizer.stoi['<bos>']]], device=device, dtype=torch.long)
    cond_vec = cond_vec.unsqueeze(0).to(device)
    for _ in range(max_len):
        logits = model(y, cond_vec)[:, -1, :] / max(temperature, 1e-6)
        if top_p is not None:
            # nucleus sampling (batch size 1)
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs  = F.softmax(sorted_logits, dim=-1)
            csum   = torch.cumsum(probs, dim=-1)
            k      = int((csum <= top_p).sum().item())
            k      = max(k, 1)
            logits_trim = sorted_logits[:, :k]
            idx_trim    = sorted_idx[:, :k]
            probs_trim  = F.softmax(logits_trim, dim=-1)
            choice      = torch.multinomial(probs_trim, 1)
            next_token  = idx_trim.gather(1, choice)
        else:
            v, idx = torch.topk(logits, min(top_k, logits.size(-1)))
            probs   = F.softmax(v, dim=-1)
            next_token = idx.gather(1, torch.multinomial(probs, 1))
        y = torch.cat([y, next_token], dim=1)
        if next_token.item() == tokenizer.stoi['<eos>']:
            break
    return tokenizer.decode(y[0].tolist())


def canonicalize(smiles: str) -> str:
    if Chem is None:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return ''
        return MolToSmiles(mol, canonical=True)
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', default='/csl/users/2026nnandaku/cluster/PolymerDesign/1Dataset/PI1M_Tg_MAC.csv')
    parser.add_argument('--out_dir',  default='Transformer')
    parser.add_argument('--max_len',  type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs',   type=int, default=30)
    parser.add_argument('--lr',       type=float, default=6e-4)  # higher base lr w/ schedule
    # Training stays reproducible by default
    parser.add_argument('--seed',     type=int, default=42, help='Training seed (reproducible)')
    # Sampling varies by default (None = time-based)
    parser.add_argument('--sample_seed', type=int, default=None, help='Sampling seed; None = time-based per run')
    # Diversity knobs for generation
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=None, help='Nucleus sampling cutoff; e.g., 0.9')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Training seed (keep reproducible)
    set_seed(args.seed)

    # 2) Load data
    df = pd.read_csv(args.data_csv)
    df = df[df['SMILES'].notna()]

    # Save canonical training set for novelty checks
    if Chem is not None:
        can_train = []
        for s in df['SMILES'].astype(str).values:
            cs = canonicalize(s)
            if cs:
                can_train.append(cs)
        can_train = set(can_train)
    else:
        can_train = set(df['SMILES'].astype(str).values)

    # 3) Scale conditions (and persist scaler)
    scaler = MinMaxScaler()
    df[['Tg', 'MAC']] = scaler.fit_transform(df[['Tg', 'MAC']])
    joblib.dump(scaler, os.path.join(args.out_dir, 'scaler.pkl'))

    # 4) Tokenizer
    tokenizer = SMILESTokenizer.from_smiles_list(df['SMILES'].astype(str).tolist())
    save_tokens(tokenizer.tokens, os.path.join(args.out_dir, 'vocab.json'))

    # 5) Datasets + loaders
    dataset = PolymerGenDataset(df, tokenizer, args.max_len)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    pad_id = tokenizer.stoi['<pad>']
    collate_fn = make_collate_fn(pad_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 6) Model / optimizer / schedule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(vocab_size=len(tokenizer.tokens), d_model=256, nhead=4, num_layers=6,
               dim_feedforward=1024, cond_dim=2, max_len=args.max_len, pad_id=pad_id)
    model = CondTransformer(
        vocab_size=len(tokenizer.tokens),
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_feedforward=1024,
        cond_dim=2,
        max_len=args.max_len,
        pad_id=pad_id,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = 1000  # fixed small warmup
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    # 7) Train
    print("Training begins…")
    train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device,
          epochs=args.epochs, grad_clip=0.5)

    # 8) Save bundle (state_dict + config)
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_state.pt'))
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)
    print("Saved: model_state.pt, config.json, vocab.json, scaler.pkl")

    # 9) Sampling — re-seed (None → time-based = different every run)
    set_seed(args.sample_seed)

    print("\nGenerating samples…")
    n_samples = args.n_samples
    valid = []
    unique_can = set()

    # sample conditions from empirical (scaled) training distribution
    cond_mat = df[['Tg', 'MAC']].values
    for _ in range(n_samples):
        i = np.random.randint(0, cond_mat.shape[0])
        cond = torch.tensor(cond_mat[i], dtype=torch.float32, device=device)
        s = sample_once(
            model, tokenizer, cond, device,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        cs = canonicalize(s)
        if cs:
            valid.append(cs)
            unique_can.add(cs)

    n_valid = len(valid)
    n_unique = len(unique_can)
    n_new = len([c for c in unique_can if c not in can_train])
    print(f"Valid: {n_valid}/{n_samples} | Unique valid: {n_unique} | Unique NEW valid: {n_new}")

    out_csv = os.path.join(args.out_dir, 'samples.csv')
    pd.DataFrame({'SMILES': list(unique_can)}).to_csv(out_csv, index=False)
    print(f"Saved sample SMILES to {out_csv}")


if __name__ == '__main__':
    main()
