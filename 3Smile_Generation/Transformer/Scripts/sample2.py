import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CondTransformer(nn.Module):
    """
    Decoder-style Transformer for conditional SMILES generation.
    Encoder sees ONLY the condition vector (no target tokens) → no leakage.
    Pads are masked in decoder self-attention.
    """
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_feedforward=1024,
        cond_dim=2,
        max_len=128,
        pad_id=0,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size, bias=True)

        if tie_weights:
            self.fc_out.weight = self.token_emb.weight  # weight tying

        # light init
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.01)
        nn.init.normal_(self.fc_out.weight,    mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, y, cond):
        """
        y:    (B, T) int64 token ids (teacher-forcing input)
        cond: (B, 2) scaled [Tg, MAC]
        returns logits: (B, T, vocab_size)
        """
        B, T = y.shape

        tok = self.token_emb(y) * math.sqrt(self.d_model)  # (B,T,D)
        pos = self.pos_emb(torch.arange(T, device=y.device).unsqueeze(0))  # (1,T,D)
        tgt = tok + pos

        # condition-only memory (length 1)
        cond_vec = self.cond_proj(cond).unsqueeze(1)        # (B,1,D)
        memory   = self.encoder(cond_vec)                   # (B,1,D)

        # mask future positions and ignore PADs
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(y.device)
        tgt_key_padding_mask = y.eq(self.pad_id)            # (B,T) True where PAD

        out = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(out)

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        cond_vec,
        device=None,
        max_len=None,
        temperature=1.0,
        top_k=50,
        top_p=None,
    ):
        """Convenience sampler (batch=1)."""
        if device is None:
            device = next(self.parameters()).device
        if max_len is None:
            max_len = self.max_len

        self.eval()
        y = torch.tensor([[tokenizer.stoi['<bos>']]], device=device, dtype=torch.long)
        cond_vec = cond_vec.unsqueeze(0).to(device)

        for _ in range(max_len):
            logits = self(y, cond_vec)[:, -1, :] / max(temperature, 1e-6)
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
