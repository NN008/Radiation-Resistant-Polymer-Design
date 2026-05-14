import torch
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import MinMaxScaler
from transformer import CondTransformer, SMILESTokenizer, sample  # your model + tokenizer + sample()

# ---------------------------
# Load dataset + scaler
# ---------------------------
df = pd.read_csv('/csl/users/2026nnandaku/cluster/PolymerDesign/Dataset/PI1M_Tg_MAC.csv')
df = df[df['SMILES'].notna()].copy()

scaler = MinMaxScaler()
df[['Tg', 'MAC']] = scaler.fit_transform(df[['Tg', 'MAC']])

# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = SMILESTokenizer(df['SMILES'])
max_len = 128

# ---------------------------
# Load model
# ---------------------------
device = torch.device("cuda:0")
model = CondTransformer(vocab_size=len(tokenizer.tokens), max_len=max_len).to(device)
model.load_state_dict(torch.load('/csl/users/2026nnandaku/cluster/PolymerDesign/transformer_polymer_gen.pt'))
model.eval()

# ---------------------------
# Generate and inverse transform
# ---------------------------
print("Generating examples with real Tg and MAC values...\n")

for i in range(5):
    # Sample random scaled values
    cond_scaled = torch.tensor([random.uniform(0.4, 0.9), random.uniform(0.4, 0.9)], dtype=torch.float32).to(device)

    # Decode SMILES
    gen = sample(model, tokenizer, cond_scaled, device, temperature=0.8)

    # Inverse transform to get real Tg/MAC
    tg_real, mac_real = scaler.inverse_transform(cond_scaled.cpu().numpy().reshape(1, -1))[0]

    print(f"[{i}] Tg = {tg_real:.2f}°C | MAC = {mac_real:.4f} cm²/g | SMILES: {gen}")
