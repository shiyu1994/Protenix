"""Verify max_nuc_tokens with controlled random seed for fair comparison."""
import logging, os, sys, random, numpy as np, torch
os.environ["PROTENIX_ROOT_DIR"] = "/vepfs-mlp2/project-abd/zhujianwei/public_databases/protenix"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from protenix.config.config import parse_configs
from protenix.data.pipeline.dataset import BaseSingleDataset, get_msa_featurizer, get_template_featurizer

SEED = 42

def reset_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def run_one(ds, idx, max_nuc):
    ds.cropping_configs = {**ds.cropping_configs, "max_nuc_tokens": max_nuc}
    reset_seed(SEED + idx)
    data = ds.process_one(idx, return_atom_token_array=True)
    centre_idx = np.array(data["cropped_token_array"].get_annotation("centre_atom_index"))
    mol_types = data["cropped_atom_array"].mol_type[centre_idx]
    return (int(np.sum(mol_types == "protein")),
            int(np.sum((mol_types == "dna") | (mol_types == "rna"))),
            len(data["cropped_token_array"]))

arg_str = "--model_name protenix-v2"
configs = {**configs_base, **{"data": data_configs}}
configs = parse_configs(configs, arg_str=arg_str, fill_required_with_null=True)
ds_cfg = configs.data["weightedPDB_before2109_wopb_nometalc_0925"].to_dict()
kwargs = {
    "name": "wPDB", **ds_cfg["base_info"],
    "cropping_configs": {**ds_cfg["cropping_configs"], "max_nuc_tokens": -1},
    "msa_featurizer": None, "template_featurizer": None,
    "random_sample_if_failed": True, "limits": 200000,
    "constraint": {}, "error_dir": None, "remove_pure_nuc": False,
}
ds = BaseSingleDataset(**kwargs)
df = ds.indices_list

# Find candidates: nuc>128 (mixed samples preferred)
nuc_mask = (df["mol_1_type"] == "nuc") | (df["mol_2_type"] == "nuc")
nuc_idxs = df[nuc_mask].index.tolist()
print(f"nuc samples: {len(nuc_idxs)} / {len(df)}")

# Scan to find candidates with nuc>128
candidates = []
reset_seed(SEED)
for idx in nuc_idxs:
    try:
        reset_seed(SEED + idx)
        data = ds.process_one(idx, return_atom_token_array=True)
    except Exception:
        continue
    centre_idx = np.array(data["cropped_token_array"].get_annotation("centre_atom_index"))
    mol_types = data["cropped_atom_array"].mol_type[centre_idx]
    n_nuc = int(np.sum((mol_types == "dna") | (mol_types == "rna")))
    n_prot = int(np.sum(mol_types == "protein"))
    if n_nuc > 128:
        candidates.append((idx, data["basic"]["pdb_id"], n_prot, n_nuc))
    if len(candidates) >= 15:
        break
    if len(candidates) % 5 == 0 and len(candidates) > 0:
        print(f"  found {len(candidates)} candidates so far...")

print(f"\nTesting {min(10, len(candidates))} candidates with matched random seeds...\n")
passes = fails = 0
for i, (idx, pdb, _, _) in enumerate(candidates[:100]):
    p1, n1, t1 = run_one(ds, idx, -1)     # no limit
    p2, n2, t2 = run_one(ds, idx, 128)    # with limit

    prot_ok = (p1 == p2)
    nuc_ok = (n2 <= 128)
    ok = prot_ok and nuc_ok
    if ok: passes += 1
    else: fails += 1

    print(f"  [{i+1:2d}] {pdb:6s} | prot={p1:4d}→{p2:4d} [{'OK' if prot_ok else 'FAIL'}] | "
          f"nuc={n1:3d}→{n2:3d} Δ={n1-n2:3d} [{'OK' if nuc_ok else 'FAIL'}] | "
          f"total={t1:4d}→{t2:4d} | {'OK' if ok else 'FAIL'}")

print(f"\n{'='*45}")
print(f"  {passes} pass / {fails} fail")
print(f"{'='*45}")
