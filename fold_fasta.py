import os
import time
from pathlib import Path
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse

import requests

API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def _get_pdb_name(record, pad_width):
    """Get the zero-padded PDB name for a record."""
    step_match = re.search(r'step(\d+)', record.id)
    if step_match:
        step_num = step_match.group(1)
        padded_step = str(int(step_num)).zfill(pad_width)
        return record.id.replace(f"step{step_num}", f"step{padded_step}")
    return record.id


def _fold_one(record, output_dir, pad_width, truncate=None):
    """Fold a single sequence and write its PDB file. Returns (name, status)."""
    pdb_name = _get_pdb_name(record, pad_width)
    pdb_path = output_dir / f"{pdb_name}.pdb"

    if pdb_path.exists():
        return pdb_name, "skipped"

    sequence = str(record.seq).replace(" ", "")
    if truncate and len(sequence) > truncate:
        print(f"Truncating {pdb_name} from {len(sequence)} to {truncate} residues")
        sequence = sequence[:truncate]

    try:
        response = requests.post(
            API_URL,
            data=sequence,
            headers={"Content-Type": "text/plain"},
        )
        response.raise_for_status()

        with open(pdb_path, 'w') as f:
            f.write(response.text)

        return pdb_name, "ok"

    except requests.HTTPError as e:
        return pdb_name, f"error: {e}"
    except requests.RequestException as e:
        return pdb_name, f"error: {e}"


def fold_sequences_with_esmfold(fasta_path, max_workers=4, truncate=None):
    """
    Fold all sequences in a FASTA file using ESMFold API.
    Creates a directory named after the FASTA file to store PDB outputs.
    """
    fasta_path = Path(fasta_path)
    fasta_name = fasta_path.stem
    output_dir = fasta_path.parent / f"esmfold_pdbs_{fasta_name}"
    output_dir.mkdir(exist_ok=True)

    # First pass to determine max step number for padding
    max_step = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        step_match = re.search(r'step(\d+)', record.id)
        if step_match:
            max_step = max(max_step, int(step_match.group(1)))
    pad_width = len(str(max_step))

    # Collect records so we can fan them out
    records = list(SeqIO.parse(fasta_path, "fasta"))
    print(f"Folding {len(records)} sequences with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fold_one, rec, output_dir, pad_width, truncate): rec
            for rec in records
        }
        for future in as_completed(futures):
            name, status = future.result()
            if status == "skipped":
                print(f"Skipping {name} - already exists")
            elif status == "ok":
                print(f"Successfully saved {name}.pdb")
            else:
                print(f"Error folding {name}: {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fold sequences using ESMFold API')
    parser.add_argument('fasta_file', type=str,
                        help='Path to input FASTA file')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel requests (default: 4)')
    parser.add_argument('--truncate', type=int, default=None,
                        help='Truncate sequences longer than this many residues')

    args = parser.parse_args()

    fold_sequences_with_esmfold(args.fasta_file, max_workers=args.workers, truncate=args.truncate)
