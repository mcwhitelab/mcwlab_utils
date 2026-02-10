import os
import time
from pathlib import Path
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import argparse

import requests

API_URL = "https://biolm.ai/api/v3/boltz2/predict/"


def _get_pdb_name(record, pad_width):
    """Get the zero-padded PDB name for a record."""
    step_match = re.search(r'step(\d+)', record.id)
    if step_match:
        step_num = step_match.group(1)
        padded_step = str(int(step_num)).zfill(pad_width)
        return record.id.replace(f"step{step_num}", f"step{padded_step}")
    return record.id


def _fold_batch(records, output_dir, pad_width, api_key, truncate=None, homodimer=False):
    """Fold a batch of sequences via BioLM Boltz2 API. Returns list of (name, status)."""
    names = []
    items = []
    for rec in records:
        pdb_name = _get_pdb_name(rec, pad_width)
        cif_path = output_dir / f"{pdb_name}.cif"

        if cif_path.exists():
            names.append((pdb_name, "skipped"))
            continue

        sequence = str(rec.seq).replace(" ", "")
        if truncate and len(sequence) > truncate:
            print(f"Truncating {pdb_name} from {len(sequence)} to {truncate} residues")
            sequence = sequence[:truncate]

        names.append((pdb_name, None))
        molecules = [
            {
                "id": "A",
                "type": "protein",
                "sequence": sequence,
            }
        ]
        if homodimer:
            molecules.append({
                "id": "B",
                "type": "protein",
                "sequence": sequence,
            })
        items.append({"molecules": molecules})

    # Nothing to submit (all skipped)
    if not items:
        return names

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"items": items}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as e:
        return [
            (n, s if s == "skipped" else f"error: {e}")
            for n, s in names
        ]

    # Normalize response
    if isinstance(results, dict):
        results = results.get("results", results.get("items", [results]))
    if not isinstance(results, list):
        results = [results]

    # Match results back to the non-skipped entries
    result_iter = iter(results)
    final = []
    for pdb_name, status in names:
        if status == "skipped":
            final.append((pdb_name, "skipped"))
            continue

        entry = next(result_iter, None)
        if entry is None:
            final.append((pdb_name, "error: no result returned for this sequence"))
            continue

        if isinstance(entry, dict):
            cif_text = entry.get("cif", "")
            confidence = entry.get("confidence", {})
        else:
            cif_text = ""
            confidence = {}

        if cif_text.strip():
            cif_path = output_dir / f"{pdb_name}.cif"
            with open(cif_path, 'w') as f:
                f.write(cif_text)
            score = confidence.get("confidence_score", "?")
            ptm = confidence.get("ptm", "?")
            final.append((pdb_name, f"ok (confidence={score}, ptm={ptm})"))
        else:
            final.append((pdb_name, f"error: empty cif in response: {str(entry)[:200]}"))

    return final


def fold_sequences_with_boltz2(fasta_path, api_key, max_workers=4, batch_size=2, truncate=None, homodimer=False):
    """
    Fold all sequences in a FASTA file using BioLM Boltz2 API.
    Creates a directory named after the FASTA file to store PDB outputs.
    """
    fasta_path = Path(fasta_path)
    fasta_name = fasta_path.stem
    suffix = "_dimer" if homodimer else ""
    output_dir = fasta_path.parent / f"boltz2_pdbs_{fasta_name}{suffix}"
    output_dir.mkdir(exist_ok=True)

    # First pass to determine max step number for padding
    max_step = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        step_match = re.search(r'step(\d+)', record.id)
        if step_match:
            max_step = max(max_step, int(step_match.group(1)))
    pad_width = len(str(max_step))

    # Collect records and split into batches
    records = list(SeqIO.parse(fasta_path, "fasta"))
    batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
    print(f"Folding {len(records)} sequences in {len(batches)} batches "
          f"(batch_size={batch_size}, workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fold_batch, batch, output_dir, pad_width, api_key, truncate, homodimer): batch
            for batch in batches
        }
        for future in as_completed(futures):
            for name, status in future.result():
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
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of sequences per API call (default: 2)')
    parser.add_argument('--truncate', type=int, default=None,
                        help='Truncate sequences longer than this many residues')
    parser.add_argument('--homodimer', action='store_true',
                        help='Add a second copy of each protein as molecule B')

    args = parser.parse_args()

    fold_sequences_with_boltz2(
        args.fasta_file,
        api_key=args.api_key,
        max_workers=args.workers,
        batch_size=args.batch_size,
        truncate=args.truncate,
        homodimer=args.homodimer,
    )
