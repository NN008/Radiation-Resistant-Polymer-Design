import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
from collections import Counter, defaultdict

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# === Accurate MAC values at 1.25 MeV (Total Attenuation Without Coherent Scattering, cm²/g) ===
MAC_1MEV = {
    'H': 0.1129, 'C': 0.05687, 'N': 0.05690, 'O': 0.05693, 'F': 0.05394,
    'P': 0.05150, 'S': 0.05683, 'Cl': 0.05463, 'Br': 0.05049, 'I': 0.05021,
    'Si': 0.05677, 'B': 0.05265, 'As': 0.05065, 'Ca': 0.0569, 'Cd': 0.05052,
    'Co': 0.0524, 'Fe': 0.05322, 'Ge': 0.05062, 'K': 0.05539, 'Na': 0.05447,
    'Ni': 0.05461, 'Pb': 0.05682, 'Se': 0.04957, 'Sn': 0.05013, 'Te': 0.04822,
    'Zn': 0.0526,
}

# === Atomic weights (g/mol) ===
ATOMIC_MASS = {
    'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.999, 'F': 18.998,
    'P': 30.9738, 'S': 32.065, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904,
    'Si': 28.085, 'B': 10.811, 'As': 74.9216, 'Ca': 40.078, 'Cd': 112.411,
    'Co': 58.9332, 'Fe': 55.845, 'Ge': 72.63, 'K': 39.0983, 'Na': 22.9897,
    'Ni': 58.6934, 'Pb': 207.2, 'Se': 78.971, 'Sn': 118.710, 'Te': 127.60,
    'Zn': 65.38,
}

# === Rejection reason counters ===
rejection_reasons = defaultdict(int)

def get_atom_counts(smiles):
    # Replace wildcard with a placeholder atom (H or C)
    smiles_clean = smiles.replace('*', '[H]')
    try:
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None or mol.GetNumAtoms() == 0:
            rejection_reasons['RDKit parse failed'] += 1
            return None
        Chem.SanitizeMol(mol)
        return Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    except:
        rejection_reasons['RDKit sanitize failed'] += 1
        return None

def compute_mac(smiles):
    atom_counts = get_atom_counts(smiles)
    if atom_counts is None:
        return None

    for atom in atom_counts:
        if atom not in MAC_1MEV:
            rejection_reasons[f"Unsupported MAC atom: {atom}"] += 1
            return None
        if atom not in ATOMIC_MASS:
            rejection_reasons[f"Unsupported mass atom: {atom}"] += 1
            return None

    total_mass = sum(count * ATOMIC_MASS[atom] for atom, count in atom_counts.items())
    if total_mass == 0:
        rejection_reasons["Zero molecular mass"] += 1
        return None

    mac = sum(
        (count * ATOMIC_MASS[atom]) / total_mass * MAC_1MEV[atom]
        for atom, count in atom_counts.items()
    )
    return mac

# === Load dataset ===
df = pd.read_csv("/csl/users/2026nnandaku/cluster/PolymerDesign/PI1M_with_Tg_Final.csv")
df['SMILES'] = df['SMILES'].astype(str).str.strip()
print(f"Loaded dataset with {len(df)} entries.")

# === Print unique atoms in the dataset
all_atoms = Counter()
for smiles in df['SMILES']:
    mol = Chem.MolFromSmiles(smiles.replace('*', ''))
    if mol:
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        all_atoms.update(atoms)

print("\nUnique atoms found:")
for atom, count in sorted(all_atoms.items()):
    print(f"  {atom:<3}: {count} occurrences")

# === Show example failed SMILES (first 20)
bad_smiles = []
for smiles in df['SMILES']:
    try:
        mol = Chem.MolFromSmiles(smiles.replace('*', ''))
        Chem.SanitizeMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            bad_smiles.append(smiles)
        if len(bad_smiles) >= 20:
            break
    except:
        bad_smiles.append(smiles)
        if len(bad_smiles) >= 20:
            break

print("\nExample of unparseable SMILES (first 20):")
for s in bad_smiles:
    print(s)

# === Compute MAC
df['MAC'] = df['SMILES'].apply(compute_mac)

# === Summary
total = len(df)
successful = df['MAC'].notnull().sum()
skipped = total - successful

print(f"\nSuccessfully computed MAC for {successful} polymers.")
print(f"Skipped {skipped} entries.")

print("\nBreakdown of skipped entries:")
for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason:<35}: {count}")

# === Save output
df_clean = df[df['MAC'].notnull()]
df_clean.to_csv("/csl/users/2026nnandaku/cluster/PolymerDesign/PI1M_with_Tg_and_MAC.csv", index=False)
print(f"\nSaved cleaned dataset to: PI1M_with_Tg_and_MAC.csv")
