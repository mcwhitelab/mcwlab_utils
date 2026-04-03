#!/usr/bin/env python3
"""
Convert FASTA sequences to AlphaFold Server JSON job format.

Samples n sequences from a FASTA file and outputs a single JSON file with multiple jobs for AlphaFold Server.
"""

import argparse
import json
import random
from pathlib import Path
from Bio import SeqIO


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample sequences from FASTA and convert to AlphaFold Server JSON format"
    )
    parser.add_argument("-i", "--input", dest="input_fasta", type=str, required=True,
                        help="Input FASTA file")
    parser.add_argument("-n", "--num_samples", dest="n", type=int, required=True,
                        help="Number of sequences to sample")
    parser.add_argument("-o", "--output", dest="output_file", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("-c", "--count", dest="count", type=int, default=1,
                        help="Number of copies of each protein chain (default: 1)")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--all", dest="use_all", action="store_true",
                        help="Use all sequences instead of sampling (ignores -n)")

    return parser.parse_args()


def fasta_to_alphafold_json(name, sequence, count=1):
    """
    Convert a single sequence to AlphaFold Server JSON format.

    Args:
        name: Sequence name (from FASTA header)
        sequence: Protein sequence string
        count: Number of copies of the protein chain

    Returns:
        dict: AlphaFold Server job format
    """
    job = {
        "name": name,
        "modelSeeds": [],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": str(sequence),
                    "count": count
                }
            }
        ],
        "dialect": "alphafoldserver",
        "version": 1
    }
    return job


def main():
    args = get_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Create output directory if needed
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read all sequences from FASTA
    records = list(SeqIO.parse(args.input_fasta, "fasta"))

    if len(records) == 0:
        print(f"ERROR: No sequences found in {args.input_fasta}")
        return

    print(f"Loaded {len(records)} sequences from {args.input_fasta}")

    # Sample or use all
    if args.use_all:
        selected = records
        print(f"Using all {len(selected)} sequences")
    else:
        if args.n > len(records):
            print(f"WARNING: Requested {args.n} samples but only {len(records)} sequences available")
            selected = records
        else:
            selected = random.sample(records, args.n)
        print(f"Sampled {len(selected)} sequences")

    # Convert all selected sequences to jobs
    all_jobs = []
    for i, record in enumerate(selected, 1):
        name = record.id
        sequence = str(record.seq)

        # Create job
        job = fasta_to_alphafold_json(name, sequence, args.count)
        all_jobs.append(job)
        print(f"  [{i}/{len(selected)}] Added: {name}")

    # Save all jobs to a single JSON file
    with open(output_file, 'w') as f:
        json.dump(all_jobs, f, indent=2)

    print(f"\nDone! {len(selected)} jobs saved to {output_file}")


if __name__ == "__main__":
    main()
