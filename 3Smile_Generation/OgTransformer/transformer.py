import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import random
import math
import time
import os

# ---------------------------
# Tokenizer
# ---------------------------
class SMILESTokenizer:
    def __init__(self, smiles_list):
        charset = set("".join(smiles_list))
        self.tokens = ['<pad>', '<bos>', '<eos>', '<unk>'] + sorted(charset)
        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s, max_len):
        s = ['<bos>'] + list(s)[:max_len - 2] + ['<eos>']
        ids = [self.stoi.get(ch, self.stoi['<unk>']) for ch in s]
        return ids + [self.stoi['<pad>']] * (max_len - len(ids))

    def decode(self, ids):
        chars = []
        for i in ids:
            ch = self.itos[i]
            if ch == '<eos>': break
            if ch not in ('<pad>', '<bos>', '<unk>'):
                chars.append(ch)
        return ''.join(chars)

# ---------------------------
# Dataset
# ---------------------------
class PolymerGenDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.smiles = df['SMILES'].values
        self.targets = df[['Tg', 'MAC']].values.astype(np.float32)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        y = torch.tensor(self.tokenizer.encode(self.smiles[idx], self.max_len))
        cond = torch.tensor(self.targets[idx])
        return cond, y

# ---------------------------
# Model
# ---------------------------
class CondTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024, cond_dim=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, y, cond):
        B, T = y.shape
        token_embed = self.token_emb(y) * math.sqrt(self.d_model)
        pos_ids = torch.arange(T, device=y.device).unsqueeze(0)
        pos_embed = self.pos_emb(pos_ids)
        cond_embed = self.cond_proj(cond).unsqueeze(1).repeat(1, T, 1)
        x = token_embed + pos_embed + cond_embed

        memory = self.encoder(x)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(y.device)
        out = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

# ---------------------------
# Training
# ---------------------------
def train_model(model, loader, optimizer, loss_fn, device, epochs=50):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        start_time = time.time()
        for batch_idx, (cond, y) in enumerate(loader):
            cond, y = cond.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(y[:, :-1], cond)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 25 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.6f}")
        print(f"Epoch {epoch:02d} | Avg Loss: {total_loss / len(loader):.6f} | Time: {time.time() - start_time:.2f}s")

# ---------------------------
# Sampling
# ---------------------------
def sample(model, tokenizer, cond_vec, device, max_len=128, temperature=1.0, top_k=20):
    model.eval()
    y = torch.tensor([[tokenizer.stoi['<bos>']]], device=device)
    cond_vec = cond_vec.unsqueeze(0).to(device)
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(y, cond_vec)
            logits = logits[:, -1, :] / temperature
            top_k_vals, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_vals, dim=-1)
            next_token = top_k_indices.gather(1, torch.multinomial(probs, 1))
        y = torch.cat([y, next_token], dim=1)
        if next_token.item() == tokenizer.stoi['<eos>']:
            break
    return tokenizer.decode(y[0].tolist())

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Data loading
    df = pd.read_csv('/csl/users/2026nnandaku/cluster/PolymerDesign/Dataset/PI1M_Tg_MAC.csv')
    df = df[df['SMILES'].notna()].copy()
    scaler = MinMaxScaler()
    df[['Tg', 'MAC']] = scaler.fit_transform(df[['Tg', 'MAC']])

    # Tokenizer
    tokenizer = SMILESTokenizer(df['SMILES'])
    max_len = 128
    dataset = PolymerGenDataset(df, tokenizer, max_len)

    # Split
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    device = torch.device("cuda:0")
    model = CondTransformer(vocab_size=len(tokenizer.tokens), max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi['<pad>'])

    # Train
    print("Training begins...")
    train_model(model, train_loader, optimizer, loss_fn, device, epochs=50)

    # Save
    outpath = "/csl/users/2026nnandaku/cluster/PolymerDesign/transformer_polymer_gen.pt"
    torch.save(model.state_dict(), outpath)
    print(f"Model saved to {outpath}")

    # Generate samples
    print("Generating examples...")
    for i in range(5):
        cond = torch.tensor([random.uniform(0.4, 0.9), random.uniform(0.4, 0.9)], dtype=torch.float32).to(device)
        gen = sample(model, tokenizer, cond, device, temperature=0.8)
        print(f"[{i}] Tg={cond[0]:.2f} | MAC={cond[1]:.2f} | SMILES: {gen}")
