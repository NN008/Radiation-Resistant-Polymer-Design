import json

SPECIALS = ['<pad>', '<bos>', '<eos>', '<unk>']

class SMILESTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens
        self.stoi = {ch: i for i, ch in enumerate(tokens)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_smiles_list(cls, smiles_list):
        charset = set()
        for s in smiles_list:
            charset.update(list(str(s)))
        tokens = SPECIALS + sorted(charset)
        return cls(tokens)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, s, max_len):
        # build sequence: <bos> + chars + <eos>, then pad
        seq = ['<bos>']
        s = str(s)
        for ch in s:
            if len(seq) >= max_len - 1:
                break
            seq.append(ch if ch in self.stoi else '<unk>')
        if len(seq) < max_len:
            seq.append('<eos>')
        ids = [self.stoi.get(ch, self.stoi['<unk>']) for ch in seq]
        while len(ids) < max_len:
            ids.append(self.stoi['<pad>'])
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            ch = self.itos.get(int(i), '<unk>')
            if ch == '<eos>':
                break
            if ch not in ('<pad>', '<bos>', '<unk>'):
                chars.append(ch)
        return ''.join(chars)


def save_tokens(tokens, path):
    with open(path, 'w') as f:
        json.dump(tokens, f)


def load_tokens(path):
    with open(path) as f:
        return json.load(f)