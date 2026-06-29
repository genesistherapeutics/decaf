"""Run OpenStructure evaluations using local ost binary (no Docker)."""

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path

from tqdm import tqdm


def evaluate_structure(name: str, pred: Path, ref: Path, outdir: Path) -> None:
    out_lig = outdir / f"{name}_ligand.json"
    if not out_lig.exists():
        subprocess.run(
            [
                os.environ.get("OST_BIN", "ost"), "compare-ligand-structures",
                "-m", str(pred), "-r", str(ref),
                "--fault-tolerant", "--lddt-pli", "--rmsd", "--substructure-match",
                "-o", str(out_lig),
            ],
            check=False, capture_output=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Predictions directory (one subdir per target)")
    parser.add_argument("pdb", type=Path, help="Reference structures directory")
    parser.add_argument("outdir", type=Path, help="Output eval JSONs directory")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    names = {f.name: f for f in args.data.iterdir() if f.is_dir()}
    print(f"Found {len(names)} targets, {args.num_samples} samples each")

    # Build case-insensitive lookup for reference files
    ref_files = {}
    for f in args.pdb.iterdir():
        ref_files[f.name.lower()] = f

    tasks = []
    for name, folder in sorted(names.items()):
        name_lower = name.lower()
        # Try full name first, then PDB ID prefix (before first underscore)
        pdb_id = name_lower.split("_")[0]
        ref_path = (
            ref_files.get(f"{name_lower}.cif.gz") or ref_files.get(f"{name_lower}.cif")
            or ref_files.get(f"{pdb_id}.cif.gz") or ref_files.get(f"{pdb_id}.cif")
        )
        if ref_path is None:
            print(f"Skipping {name}: no reference found")
            continue
        for model_id in range(args.num_samples):
            pred_path = folder / f"{name}_model_{model_id}.cif"
            if not pred_path.exists():
                continue
            tasks.append((f"{name}_model_{model_id}", pred_path, ref_path))

    print(f"Total evaluations: {len(tasks)}")

    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_structure, task_name, pred, ref, args.outdir)
            for task_name, pred, ref in tasks
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print("Done!")


if __name__ == "__main__":
    main()
