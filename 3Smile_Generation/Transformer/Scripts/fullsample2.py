"""
Exports a clean, reproducible bundle (directory) and also an optional single .pt file.
Best practice: use the directory bundle.
"""
import os, json, joblib, torch
from sample2 import CondTransformer
from tokenizer import load_tokens

BASE = "Transformer"

state_path = os.path.join(BASE, 'model_state.pt')
config_path = os.path.join(BASE, 'config.json')
vocab_path  = os.path.join(BASE, 'vocab.json')
scaler_path = os.path.join(BASE, 'scaler.pkl')

bundle_dir = os.path.join(BASE, 'bundle')
os.makedirs(bundle_dir, exist_ok=True)

# Copy/confirm the 4 artifacts exist
assert os.path.exists(state_path)
assert os.path.exists(config_path)
assert os.path.exists(vocab_path)
assert os.path.exists(scaler_path)

# 1) Directory bundle (recommended)
for src in [state_path, config_path, vocab_path, scaler_path]:
    dst = os.path.join(bundle_dir, os.path.basename(src))
    if src != dst:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())
print(f"Wrote directory bundle at {bundle_dir}")

# 2) Single-file package (optional, pickle-based)
with open(config_path) as f:
    cfg = json.load(f)

tokens = load_tokens(vocab_path)
state = torch.load(state_path, map_location='cpu')
scaler = joblib.load(scaler_path)

package = {
    'state_dict': state,
    'config': cfg,
    'tokens': tokens,
    'scaler': scaler,
}
mono_path = os.path.join(BASE, 'transformer_polymer_gen_FULL.pt')
torch.save(package, mono_path)
print(f"Also wrote monolithic package: {mono_path}")