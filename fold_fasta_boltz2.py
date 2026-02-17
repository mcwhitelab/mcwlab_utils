import os
import csv
import time
from pathlib import Path
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse

import requests

API_URL = "https://biolm.ai/api/v3/boltz2/predict/"


def _load_ligands(ligand_path):
    """Load ligands from a CSV file with columns: id, smiles."""
    ligands = []
    with open(ligand_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ligands.append({"id": row["id"], "smiles": row["smiles"]})
    return ligands


def _get_pdb_name(record, pad_width):
    """Get the zero-padded PDB name for a record."""
    step_match = re.search(r'step(\d+)', record.id)
    if step_match:
        step_num = step_match.group(1)
        padded_step = str(int(step_num)).zfill(pad_width)
        return record.id.replace(f"step{step_num}", f"step{padded_step}")
    return record.id


def _fold_one(job_name, molecules, output_dir, api_key):
    """Submit a single folding job to the BioLM Boltz2 API. Returns (name, status)."""
    cif_path = output_dir / f"{job_name}.cif"
    if cif_path.exists():
        return (job_name, "skipped")

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"items": [{"molecules": molecules}]}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as e:
        return (job_name, f"error: {e}")

    # Normalize response
    if isinstance(results, dict):
        results = results.get("results", results.get("items", [results]))
    if not isinstance(results, list):
        results = [results]

    entry = results[0] if results else None
    if entry is None:
        return (job_name, "error: no result returned")

    if isinstance(entry, dict):
        cif_text = entry.get("cif", "")
        confidence = entry.get("confidence", {})
    else:
        cif_text = ""
        confidence = {}

    if cif_text.strip():
        with open(cif_path, 'w') as f:
            f.write(cif_text)
        score = confidence.get("confidence_score", "?")
        ptm = confidence.get("ptm", "?")
        return (job_name, f"ok (confidence={score}, ptm={ptm})")
    else:
        return (job_name, f"error: empty cif in response: {str(entry)[:200]}")


def _build_jobs(records, pad_width, truncate=None, homodimer=False, ligands=None):
    """Expand records (and optional ligands) into a list of (job_name, molecules) tuples."""
    jobs = []
    for rec in records:
        pdb_name = _get_pdb_name(rec, pad_width)

        sequence = str(rec.seq).replace(" ", "")
        if truncate and len(sequence) > truncate:
            print(f"Truncating {pdb_name} from {len(sequence)} to {truncate} residues")
            sequence = sequence[:truncate]

        base_molecules = [
            {"id": "A", "type": "protein", "sequence": sequence}
        ]
        if homodimer:
            base_molecules.append(
                {"id": "B", "type": "protein", "sequence": sequence}
            )

        if ligands:
            for lig in ligands:
                job_name = f"{pdb_name}__{lig['id']}"
                molecules = base_molecules + [
                    {"id": "L", "type": "ligand", "smiles": lig["smiles"]}
                ]
                jobs.append((job_name, molecules))
        else:
            jobs.append((pdb_name, base_molecules))

    return jobs


def fold_sequences_with_boltz2(fasta_path, api_key, max_workers=4, truncate=None, homodimer=False, ligand_file=None):
    """
    Fold all sequences in a FASTA file using BioLM Boltz2 API.
    Each job is submitted as a single API call (batch_size=1).
    Creates a directory named after the FASTA file to store CIF outputs.
    """
    fasta_path = Path(fasta_path)
    fasta_name = fasta_path.stem
    suffix = "_dimer" if homodimer else ""
    if ligand_file:
        suffix += "_ligands"
    output_dir = fasta_path.parent / f"boltz2_pdbs_{fasta_name}{suffix}"
    output_dir.mkdir(exist_ok=True)

    ligands = _load_ligands(ligand_file) if ligand_file else None
    if ligands:
        print(f"Loaded {len(ligands)} ligands: {', '.join(l['id'] for l in ligands)}")

    # First pass to determine max step number for padding
    max_step = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        step_match = re.search(r'step(\d+)', record.id)
        if step_match:
            max_step = max(max_step, int(step_match.group(1)))
    pad_width = len(str(max_step))

    # Collect records and build all jobs
    records = list(SeqIO.parse(fasta_path, "fasta"))
    jobs = _build_jobs(records, pad_width, truncate, homodimer, ligands)
    print(f"Folding {len(jobs)} jobs ({len(records)} sequences"
          f"{f' x {len(ligands)} ligands' if ligands else ''}, "
          f"workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fold_one, job_name, molecules, output_dir, api_key): job_name
            for job_name, molecules in jobs
        }
        for future in as_completed(futures):
            name, status = future.result()
            if status == "skipped":
                print(f"Skipping {name} - already exists")
            elif status.startswith("ok"):
                print(f"Successfully saved {name}.cif  {status}")
            else:
                print(f"Error folding {name}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fold sequences using BioLM Boltz2 API')
    parser.add_argument('fasta_file', type=str,
                        help='Path to input FASTA file')
    parser.add_argument('--api-key', type=str, required=True,
                        help='BioLM API key')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel requests (default: 4)')
    parser.add_argument('--truncate', type=int, default=None,
                        help='Truncate sequences longer than this many residues')
    parser.add_argument('--homodimer', action='store_true',
                        help='Add a second copy of each protein as molecule B')
    parser.add_argument('--ligands', type=str, default=None,
                        help='CSV file with ligands (columns: id, smiles)')

    args = parser.parse_args()

    fold_sequences_with_boltz2(
        args.fasta_file,
        api_key=args.api_key,
        max_workers=args.workers,
        truncate=args.truncate,
        homodimer=args.homodimer,
        ligand_file=args.ligands,
    )
