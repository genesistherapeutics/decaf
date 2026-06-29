"""Run PoseBusters quality checks on Boltz predictions.

Uses the PoseBusters library in redock_no_strain mode (redock config minus energy_ratio).
The redock mode compares predicted ligand against a ground truth ligand SDF, enabling
identity checks (molecular_formula, molecular_bonds, double_bond_stereochemistry).
The pbcheck_rmsd_≤_2å check is deleted before counting failures.

Usage:
    python scripts/eval/run_posebusters.py \
        --predictions /path/to/predictions \
        --queries /path/to/queries \
        --ref /path/to/reference/cifs \
        --num-samples 5 \
        --output posebusters_results.csv

    # Multiple prediction dirs (e.g. different tools):
    python scripts/eval/run_posebusters.py \
        --tools boltz1:/path/to/boltz1/predictions decaf:/path/to/decaf/predictions \
        --queries /path/to/queries \
        --ref /path/to/reference/cifs \
        --num-samples 5 \
        --output posebusters_results.csv
"""

from __future__ import annotations

import argparse
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import posebusters
import yaml
from rdkit import Chem
from tqdm import tqdm
from yaml import safe_load


# ---------------------------------------------------------------------------
# PoseBusters config: redock_no_strain (redock minus energy_ratio)
# ---------------------------------------------------------------------------
def _get_redock_no_strain_config() -> dict[str, Any]:
    cfg_path = Path(posebusters.__file__).parent / "config" / "redock.yml"
    with open(cfg_path, encoding="utf-8") as f:
        config = safe_load(f)
    config["modules"] = [
        m for m in config["modules"] if m["function"] != "energy_ratio"
    ]
    return config


def _get_dock_no_strain_config() -> dict[str, Any]:
    """Fallback when no reference structures are available."""
    cfg_path = Path(posebusters.__file__).parent / "config" / "dock.yml"
    with open(cfg_path, encoding="utf-8") as f:
        config = safe_load(f)
    config["modules"] = [
        m for m in config["modules"] if m["function"] != "energy_ratio"
    ]
    return config


# ---------------------------------------------------------------------------
# QUALITY_CHECKS: the set of check names that contribute to pb_valid.
# (tetrahedral_chirality intentionally excluded — known PoseBusters issue
# with symmetric ligands.)
# ---------------------------------------------------------------------------
QUALITY_CHECKS = {
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
    "rmsd_≤_2å",
}


# ---------------------------------------------------------------------------
# CIF → PDB (protein only) + SDF (ligand only) extraction
# ---------------------------------------------------------------------------
def _resolve_primary_ligand_ccd(queries_dir: Path, target_name: str) -> str | None:
    """Get the CCD code for the primary (SOI) ligand from the query YAML."""
    query_file = queries_dir / f"{target_name}.yaml"
    if not query_file.exists():
        return None
    with query_file.open() as f:
        query = yaml.safe_load(f)
    # Target name convention: {pdb_id}_{ccd_code_lower}
    parts = target_name.split("_", 1)
    if len(parts) < 2:
        return None
    target_ccd = parts[1].upper()
    # Verify it's in the query
    for seq in query.get("sequences", []):
        if "ligand" not in seq:
            continue
        if seq["ligand"].get("ccd", "").upper() == target_ccd:
            return target_ccd
    return None


def _resolve_primary_ligand_chain(queries_dir: Path, target_name: str) -> str | None:
    """Get the chain ID for the primary ligand from the query YAML."""
    query_file = queries_dir / f"{target_name}.yaml"
    if not query_file.exists():
        return None
    with query_file.open() as f:
        query = yaml.safe_load(f)
    parts = target_name.split("_", 1)
    if len(parts) < 2:
        return None
    target_ccd = parts[1].upper()
    for seq in query.get("sequences", []):
        if "ligand" not in seq:
            continue
        if seq["ligand"].get("ccd", "").upper() == target_ccd:
            chain_id = seq["ligand"].get("id")
            if isinstance(chain_id, list):
                return chain_id[0]
            return chain_id
    return None


def extract_protein_pdb_and_ligand_sdf(
    cif_path: Path,
    ligand_chain_id: str | None,
    ligand_ccd: str | None,
) -> tuple[str | None, str | None]:
    """Extract protein as PDB string and primary ligand as SDF string from a boltz CIF.

    Handles boltz multi-character chain IDs (A1, G1, ...) by remapping to single chars.
    Returns (protein_pdb, ligand_sdf) or (None, None) on failure.
    """
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("pred", str(cif_path))
    if len(list(structure.get_models())) == 0:
        return None, None

    model = structure[0]

    # Identify ligand chain
    ligand_chain_found = None
    for chain in model:
        cid = chain.get_id()
        for residue in chain:
            hetflag = residue.get_id()[0]
            resname = residue.get_resname().strip()
            if hetflag.startswith("H_"):
                if ligand_chain_id and cid == ligand_chain_id:
                    ligand_chain_found = cid
                    break
                elif ligand_ccd and resname == ligand_ccd:
                    ligand_chain_found = cid
                    break
        if ligand_chain_found:
            break

    if not ligand_chain_found:
        return None, None

    # --- Write protein PDB manually (handles multi-char chain IDs) ---
    # Remap chain IDs to single characters
    chain_ids = [c.get_id() for c in model]
    single_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    id_map = {cid: single_chars[i % len(single_chars)] for i, cid in enumerate(chain_ids)}

    pdb_lines = []
    atom_idx = 1
    for chain in model:
        cid = chain.get_id()
        pdb_chain = id_map[cid]
        for residue in chain:
            hetflag = residue.get_id()[0]
            # Skip the primary ligand (it goes into SDF)
            if cid == ligand_chain_found and hetflag.startswith("H_"):
                continue
            resname = residue.get_resname().strip()
            resseq = residue.get_id()[1]
            icode = residue.get_id()[2].strip()
            record = "HETATM" if hetflag.startswith("H_") else "ATOM"
            for atom in residue:
                name = atom.get_name()
                x, y, z = atom.get_vector()
                element = atom.element.strip() if atom.element else ""
                name_field = name if len(name) == 4 else f" {name}"
                line = (
                    f"{record:<6}{atom_idx:>5} {name_field:<4} "
                    f"{resname:>3} {pdb_chain}{resseq:>4}{icode:>1}   "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                    f"{'1.00':>6}{'0.00':>6}          "
                    f"{element:>2}"
                )
                pdb_lines.append(line)
                atom_idx += 1

    pdb_lines.append("END")
    protein_pdb = "\n".join(pdb_lines) + "\n"

    # --- Extract ligand SDF from CCD template + predicted coords ---
    ligand_sdf = _extract_ligand_sdf(model, ligand_chain_found, ligand_ccd)

    return protein_pdb, ligand_sdf


def _extract_ligand_sdf(
    model,
    chain_id: str,
    ccd_code: str | None,
) -> str | None:
    """Extract ligand atoms from Biopython model and produce SDF via CCD template."""
    try:
        chain = model[chain_id]
        ligand_atoms = []
        for residue in chain:
            if not residue.get_id()[0].startswith("H_"):
                continue
            for atom in residue:
                ligand_atoms.append(
                    (atom.get_name().strip(), atom.get_vector().get_array(), atom.element.strip())
                )

        if not ligand_atoms:
            return None

        # Try CCD template for proper bond orders
        ccd_mol = _get_ccd_template(ccd_code) if ccd_code else None

        if ccd_mol is not None:
            mol = Chem.RWMol(ccd_mol)
            conf = mol.GetConformer()
            atom_name_to_idx = {}
            for a in mol.GetAtoms():
                name = a.GetProp("name") if a.HasProp("name") else a.GetSymbol()
                atom_name_to_idx[name] = a.GetIdx()

            matched = 0
            for name, coords, _ in ligand_atoms:
                if name in atom_name_to_idx:
                    idx = atom_name_to_idx[name]
                    conf.SetAtomPosition(idx, coords.tolist())
                    matched += 1

            if matched > 0:
                return Chem.MolToMolBlock(mol)

        # Fallback: build mol from atoms (no bond info)
        rwmol = Chem.RWMol()
        conf = Chem.Conformer(len(ligand_atoms))
        for i, (name, coords, element) in enumerate(ligand_atoms):
            atom = Chem.Atom(element if element else "C")
            rwmol.AddAtom(atom)
            conf.SetAtomPosition(i, coords.tolist())
        rwmol.AddConformer(conf, assignId=True)
        return Chem.MolToMolBlock(rwmol.GetMol())

    except Exception:
        traceback.print_exc()
        return None


_ccd_cache: dict[str, Chem.Mol | None] = {}


def _get_ccd_template(ccd_code: str) -> Chem.Mol | None:
    """Load CCD template mol from boltz's cached ccd.pkl."""
    if ccd_code in _ccd_cache:
        return _ccd_cache[ccd_code]
    import pickle

    ccd_path = Path.home() / ".boltz" / "ccd.pkl"
    if not ccd_path.exists():
        _ccd_cache[ccd_code] = None
        return None

    if "_ccd_dict" not in _get_ccd_template.__dict__:
        with open(ccd_path, "rb") as f:
            _get_ccd_template._ccd_dict = pickle.load(f)

    mol = _get_ccd_template._ccd_dict.get(ccd_code)
    if mol is not None:
        mol = Chem.RWMol(mol)
        # Ensure it has a conformer
        if mol.GetNumConformers() == 0:
            from rdkit.Chem import AllChem

            AllChem.EmbedMolecule(mol, randomSeed=42)
        mol = mol.GetMol()
    _ccd_cache[ccd_code] = mol
    return mol


# ---------------------------------------------------------------------------
# GT ligand extraction from reference CIF
# ---------------------------------------------------------------------------
def extract_gt_ligand_sdf(
    ref_cif_path: Path,
    ligand_ccd: str | None,
) -> str | None:
    """Extract the ground truth ligand as SDF from a reference CIF file.

    Uses the CCD template with coordinates from the reference structure.
    """
    if ligand_ccd is None:
        return None

    try:
        import gzip
        from Bio.PDB import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        open_fn = gzip.open if str(ref_cif_path).endswith(".gz") else open
        with open_fn(ref_cif_path, "rt") as f:
            structure = parser.get_structure("ref", f)

        if len(list(structure.get_models())) == 0:
            return None

        model = structure[0]

        # Find the ligand chain with matching CCD code
        for chain in model:
            for residue in chain:
                hetflag = residue.get_id()[0]
                resname = residue.get_resname().strip()
                if hetflag.startswith("H_") and resname == ligand_ccd:
                    ligand_atoms = []
                    for atom in residue:
                        ligand_atoms.append(
                            (atom.get_name().strip(), atom.get_vector().get_array(), atom.element.strip())
                        )
                    if ligand_atoms:
                        # Use CCD template with GT coords
                        ccd_mol = _get_ccd_template(ligand_ccd)
                        if ccd_mol is not None:
                            mol = Chem.RWMol(ccd_mol)
                            conf = mol.GetConformer()
                            atom_name_to_idx = {}
                            for a in mol.GetAtoms():
                                name = a.GetProp("name") if a.HasProp("name") else a.GetSymbol()
                                atom_name_to_idx[name] = a.GetIdx()
                            matched = 0
                            for name, coords, _ in ligand_atoms:
                                if name in atom_name_to_idx:
                                    idx = atom_name_to_idx[name]
                                    conf.SetAtomPosition(idx, coords.tolist())
                                    matched += 1
                            if matched > 0:
                                return Chem.MolToMolBlock(mol)
                    break  # use first matching chain

        return None
    except Exception:
        traceback.print_exc()
        return None


def _find_ref_cif(ref_dir: Path, target_name: str) -> Path | None:
    """Find reference CIF for a target (pdb_id from target name)."""
    pdb_id = target_name.split("_")[0].lower()
    for suffix in [".cif.gz", ".cif"]:
        p = ref_dir / f"{pdb_id}{suffix}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# PoseBusters runner
# ---------------------------------------------------------------------------
def run_posebusters_on_pair(
    protein_pdb_path: str,
    ligand_sdf_path: str,
    config: dict[str, Any],
    gt_ligand_sdf_path: str | None = None,
) -> dict[str, Any]:
    """Run PoseBusters on a single protein-ligand pair.

    If gt_ligand_sdf_path is provided, runs in redock mode (identity checks enabled).
    Returns dict with pbcheck_* keys and num_failed_checks / pb_valid.
    """
    pb = posebusters.PoseBusters(config=config)
    input_row = {"mol_pred": ligand_sdf_path, "mol_cond": protein_pdb_path}
    if gt_ligand_sdf_path is not None:
        input_row["mol_true"] = gt_ligand_sdf_path
    input_df = pd.DataFrame([input_row])
    result_df = pb.bust_table(input_df, full_report=True)

    if len(result_df) == 0:
        return {"pb_valid": None, "num_failed_checks": None}

    row = result_df.iloc[0].to_dict()

    # Prefix check names with pbcheck_
    record = {}
    for k, v in row.items():
        if k in ("file", "molecule"):
            continue
        if k in QUALITY_CHECKS:
            record[f"pbcheck_{k}"] = v
        else:
            record[f"pbextra_{k}"] = v

    # Delete rmsd check before counting failures
    # (RMSD is computed separately via OST with symmetry handling)
    record.pop("pbcheck_rmsd_≤_2å", None)

    # Compute num_failed_checks and pb_valid:
    # count failed pbcheck_* (excluding mol_true_loaded)
    num_failed = sum(
        not v
        for k, v in record.items()
        if k.startswith("pbcheck_")
        and k != "pbcheck_mol_true_loaded"
        and v is not None
        and v is not pd.NA
    )
    pb_valid = float(num_failed == 0)

    record["num_failed_checks"] = num_failed
    record["pb_valid"] = pb_valid

    return record


# ---------------------------------------------------------------------------
# Process one prediction
# ---------------------------------------------------------------------------
def _find_benchmark_files(benchmark_dir: Path, target_name: str) -> tuple[Path | None, Path | None]:
    """Find protein PDB and ligand SDF from the PoseBusters benchmark set.

    Benchmark layout: {PDBID_CCD}/{PDBID_CCD}_protein.pdb, {PDBID_CCD}_ligand.sdf
    Target names are lowercase (e.g. 7twc_cxs) but benchmark dirs are uppercase (7TWC_CXS).
    """
    name_upper = target_name.upper()
    bench_dir = benchmark_dir / name_upper
    if not bench_dir.exists():
        return None, None
    protein = bench_dir / f"{name_upper}_protein.pdb"
    ligand = bench_dir / f"{name_upper}_ligand.sdf"
    if protein.exists() and ligand.exists():
        return protein, ligand
    return None, None


def process_one(
    tool_name: str,
    target_name: str,
    model_idx: int,
    cif_path: Path,
    queries_dir: Path,
    config: dict[str, Any],
    ref_dir: Path | None = None,
    benchmark_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Process a single CIF prediction file through PoseBusters.

    If benchmark_dir is provided, uses the pre-split protein PDB and GT ligand SDF
    from the PoseBusters benchmark set. Otherwise extracts
    protein/ligand from the prediction CIF.
    """
    try:
        ligand_chain = _resolve_primary_ligand_chain(queries_dir, target_name)
        ligand_ccd = _resolve_primary_ligand_ccd(queries_dir, target_name)

        # Extract predicted ligand SDF from CIF
        protein_pdb, ligand_sdf = extract_protein_pdb_and_ligand_sdf(
            cif_path, ligand_chain, ligand_ccd
        )

        if ligand_sdf is None:
            return None

        with tempfile.NamedTemporaryFile(
            suffix=".sdf", mode="w", delete=False
        ) as sdf_f:
            sdf_f.write(ligand_sdf)
            sdf_path = sdf_f.name

        # Use benchmark protein PDB + GT ligand SDF if available
        gt_sdf_path = None
        if benchmark_dir is not None:
            bench_protein, bench_ligand = _find_benchmark_files(benchmark_dir, target_name)
            if bench_protein is not None:
                pdb_path = str(bench_protein)
                gt_sdf_path = str(bench_ligand)
            elif protein_pdb is not None:
                # Fallback to extracted protein
                with tempfile.NamedTemporaryFile(
                    suffix=".pdb", mode="w", delete=False
                ) as pdb_f:
                    pdb_f.write(protein_pdb)
                    pdb_path = pdb_f.name
            else:
                return None
        elif protein_pdb is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".pdb", mode="w", delete=False
            ) as pdb_f:
                pdb_f.write(protein_pdb)
                pdb_path = pdb_f.name

            # Extract GT ligand SDF for redock mode from ref CIF
            if ref_dir is not None:
                ref_cif = _find_ref_cif(ref_dir, target_name)
                if ref_cif is not None:
                    gt_sdf = extract_gt_ligand_sdf(ref_cif, ligand_ccd)
                    if gt_sdf is not None:
                        with tempfile.NamedTemporaryFile(
                            suffix=".sdf", mode="w", delete=False
                        ) as gt_f:
                            gt_f.write(gt_sdf)
                            gt_sdf_path = gt_f.name
        else:
            return None

        record = run_posebusters_on_pair(pdb_path, sdf_path, config, gt_sdf_path)
        record["tool"] = tool_name
        record["pdb_id"] = target_name
        record["model_idx"] = model_idx
        return record

    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def find_predictions(pred_dir: Path) -> list[Path]:
    """Find prediction directories, handling both flat and shard layouts."""
    # Flat layout: predictions/{target_name}/{target_name}_model_0.cif
    if any(d.is_dir() and (d / f"{d.name}_model_0.cif").exists() for d in pred_dir.iterdir()):
        return [pred_dir]

    # Shard layout: shard_*/boltz_results_shard_*/predictions/
    shard_dirs = []
    for shard in sorted(pred_dir.glob("shard_*/*/predictions")):
        shard_dirs.append(shard)
    if shard_dirs:
        return shard_dirs

    # Try one level deeper
    for shard in sorted(pred_dir.glob("*/predictions")):
        shard_dirs.append(shard)
    return shard_dirs if shard_dirs else [pred_dir]


def main():
    parser = argparse.ArgumentParser(
        description="Run PoseBusters checks on Boltz predictions"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions",
        type=str,
        help="Single predictions directory",
    )
    group.add_argument(
        "--tools",
        nargs="+",
        help="tool_name:predictions_dir pairs",
    )
    parser.add_argument("--queries", type=str, required=True, help="Queries YAML directory")
    parser.add_argument("--ref", type=str, default=None, help="Reference CIF directory (enables redock_no_strain mode)")
    parser.add_argument("--benchmark", type=str, default=None, help="PoseBusters benchmark dir with {PDBID_CCD}/{PDBID_CCD}_protein.pdb + _ligand.sdf")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=1, help="Sequential by default (PB is heavy)")
    args = parser.parse_args()

    queries_dir = Path(args.queries)
    ref_dir = Path(args.ref) if args.ref else None
    benchmark_dir = Path(args.benchmark) if args.benchmark else None

    if benchmark_dir is not None or ref_dir is not None:
        config = _get_redock_no_strain_config()
        if benchmark_dir:
            print(f"Mode: redock_no_strain (benchmark dir: {benchmark_dir})")
        else:
            print(f"Mode: redock_no_strain (ref dir: {ref_dir})")
    else:
        config = _get_dock_no_strain_config()
        print("Mode: dock_no_strain (no reference structures)")

    # Build tool configs
    tool_configs: dict[str, Path] = {}
    if args.predictions:
        tool_configs["default"] = Path(args.predictions)
    else:
        for spec in args.tools:
            name, path = spec.split(":", 1)
            tool_configs[name] = Path(path)

    # Collect all (tool, target, model_idx, cif_path) tuples
    tasks = []
    for tool_name, base_dir in tool_configs.items():
        pred_dirs = find_predictions(base_dir)
        targets_seen = set()
        for pred_dir in pred_dirs:
            for target_dir in sorted(pred_dir.iterdir()):
                if not target_dir.is_dir():
                    continue
                target_name = target_dir.name
                if target_name in targets_seen:
                    continue
                targets_seen.add(target_name)
                for model_idx in range(args.num_samples):
                    cif_path = target_dir / f"{target_name}_model_{model_idx}.cif"
                    if cif_path.exists():
                        tasks.append((tool_name, target_name, model_idx, cif_path))

        print(f"{tool_name}: {len(targets_seen)} targets, {sum(1 for t in tasks if t[0] == tool_name)} CIF files")

    print(f"Total files to process: {len(tasks)}")

    # Process sequentially (PoseBusters is CPU-heavy and not picklable)
    records = []
    for tool_name, target_name, model_idx, cif_path in tqdm(tasks, desc="PoseBusters"):
        record = process_one(
            tool_name, target_name, model_idx, cif_path, queries_dir, config, ref_dir, benchmark_dir
        )
        if record is not None:
            records.append(record)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} records to {args.output}")

    # Summary
    if "pb_valid" in df.columns:
        for tool_name in df["tool"].unique():
            tool_df = df[df["tool"] == tool_name]
            n_targets = tool_df["pdb_id"].nunique()
            valid_top1 = tool_df[tool_df["model_idx"] == 0]["pb_valid"].mean()
            valid_oracle = tool_df.groupby("pdb_id")["pb_valid"].max().mean()
            print(f"\n{tool_name} ({n_targets} targets):")
            print(f"  pb_valid (top-1):  {valid_top1:.4f}")
            print(f"  pb_valid (oracle): {valid_oracle:.4f}")

            # Per-check failure rates (top-1)
            top1 = tool_df[tool_df["model_idx"] == 0]
            pb_cols = [c for c in top1.columns if c.startswith("pbcheck_")]
            if pb_cols:
                print("  Per-check pass rates (top-1):")
                for col in sorted(pb_cols):
                    vals = top1[col].dropna()
                    if len(vals) > 0:
                        print(f"    {col:50s} {vals.mean():.4f}")


if __name__ == "__main__":
    main()
