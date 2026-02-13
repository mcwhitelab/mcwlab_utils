#!/usr/bin/env python3
"""
Domain Swap Embedding Analysis with Fixed Region Comparisons

This script creates chimeric proteins by swapping domains between two proteins
and analyzes how embeddings change in the fusion context vs original context.

Key difference from domain_swap_embed_analysis.py:
- Compares FIXED regions that are the same length across all fusion constructs
- This provides fair cosine similarity comparisons (same number of residues)
- User specifies explicit ranges to compare from each parent protein

Example:
    python domain_swap_embed_analysis_fixedregions.py \
        -m /path/to/esm2_model \
        -p1 protein1.fasta \
        -p2 protein2.fasta \
        -o output_prefix \
        --nterm_scan 540 570 \
        --cterm_scan 140 200 \
        --p1_compare_regions 1 400 \
        --p2_compare_regions 250 350 400 500

This would compare:
- P1 residues 1-400 (in original vs fusion context)
- P2 residues 250-350 (in original vs fusion context)
- P2 residues 400-500 (in original vs fusion context)
"""

from transformers import AutoTokenizer, AutoModel, AutoConfig, T5Tokenizer, T5EncoderModel, AutoModelForMaskedLM, EsmTokenizer
import torch
import torch.nn as nn
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pickle
import argparse
import numpy as np
import tempfile
import os
from pathlib import Path
import pandas as pd

# Import from existing embedding script
from hf_embed_new import load_model, get_embeddings, parse_fasta_for_embed
from model.architectures import SWE_Pooling

np.random.seed(42)


def get_args():
    parser = argparse.ArgumentParser(
        description="Domain swap embedding analysis with fixed region comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python domain_swap_embed_analysis_fixedregions.py \\
      -m /path/to/esm2_model \\
      -p1 protein1.fasta -p2 protein2.fasta \\
      -o output_prefix \\
      --nterm_scan 540 570 --cterm_scan 140 200 \\
      --p1_compare_regions 1 400 \\
      --p2_compare_regions 250 350 400 500

Region specification:
  Regions are specified as pairs of start end positions (1-indexed, inclusive).
  Multiple regions can be specified as: start1 end1 start2 end2 ...

  For --p1_compare_regions: These residues come from Protein 1 (N-term contributor)
      and will be present in all fusions (as long as p1_cut >= end of region)

  For --p2_compare_regions: These residues come from Protein 2 (C-term contributor)
      and will be present in all fusions (as long as p2_cut <= start of region)
"""
    )
    parser.add_argument("-m", "--model", dest="model_path", type=str, required=True,
                        help="Model directory path (e.g., /path/to/esm2_model)")
    parser.add_argument("-p1", "--protein1", dest="protein1", type=str, required=True,
                        help="Protein 1 (N-term contributor) sequence or FASTA file")
    parser.add_argument("-p2", "--protein2", dest="protein2", type=str, required=True,
                        help="Protein 2 (C-term contributor) sequence or FASTA file")

    # Scan ranges for junction points
    parser.add_argument("--nterm_scan", dest="nterm_scan", type=int, nargs=2,
                        default=[529, 545],
                        help="Scan range for Protein 1 cut positions (start, end). Default: 529 545")
    parser.add_argument("--cterm_scan", dest="cterm_scan", type=int, nargs=2,
                        default=[133, 157],
                        help="Scan range for Protein 2 cut positions (start, end). Default: 133 157")
    parser.add_argument("--step_size", dest="step_size", type=int, default=1,
                        help="Step size for scanning (default: 1 = test every position)")

    # Fixed comparison regions - the key new feature
    parser.add_argument("--p1_compare_regions", dest="p1_compare_regions", type=int, nargs='+',
                        default=None,
                        help="Fixed regions from Protein 1 to compare (pairs of start end). "
                             "E.g., --p1_compare_regions 1 400 means compare residues 1-400. "
                             "Multiple regions: --p1_compare_regions 1 200 300 400")
    parser.add_argument("--p2_compare_regions", dest="p2_compare_regions", type=int, nargs='+',
                        default=None,
                        help="Fixed regions from Protein 2 to compare (pairs of start end). "
                             "E.g., --p2_compare_regions 250 350 400 500 means compare "
                             "residues 250-350 and 400-500 separately.")

    # Output
    parser.add_argument("-o", "--output", dest="output_prefix", type=str, required=True,
                        help="Output prefix for results")

    # Model parameters
    parser.add_argument("-ss", "--strategy", dest="strat", type=str, nargs="+",
                        default=["mean"], choices=['swe', 'mean', 'meansig'],
                        help="Embedding strategies (default: mean)")
    parser.add_argument("-co", "--cpu_only", dest="cpu_only", action="store_true",
                        help="Use CPU only")
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--layers", dest="layers", nargs="+", type=int, default=[-1],
                        help="Layers to use (default: -1, last layer)")
    parser.add_argument("--all_layers", dest="all_layers", action="store_true",
                        help="Use all layers")

    return parser.parse_args()


def parse_region_args(region_list):
    """
    Parse region arguments into list of (start, end) tuples.

    Args:
        region_list: List of integers [start1, end1, start2, end2, ...]

    Returns:
        List of (start, end) tuples, or empty list if None
    """
    if region_list is None:
        return []

    if len(region_list) % 2 != 0:
        raise ValueError(f"Region list must have even number of values (pairs of start end): {region_list}")

    regions = []
    for i in range(0, len(region_list), 2):
        start, end = region_list[i], region_list[i+1]
        if start > end:
            raise ValueError(f"Region start ({start}) must be <= end ({end})")
        if start < 1:
            raise ValueError(f"Region start ({start}) must be >= 1 (1-indexed)")
        regions.append((start, end))

    return regions


def validate_regions(p1_regions, p2_regions, p1_len, p2_len, nterm_scan, cterm_scan):
    """
    Validate that comparison regions make sense given the scan ranges.

    P1 regions should be fully included in all fusions (end <= min p1_cut)
    P2 regions should be fully included in all fusions (start >= max p2_cut)
    """
    min_p1_cut = nterm_scan[0]  # Minimum amount of P1 kept
    max_p2_cut = cterm_scan[1]  # Maximum starting position in P2

    warnings = []

    for i, (start, end) in enumerate(p1_regions):
        if end > p1_len:
            raise ValueError(f"P1 region {i+1} end ({end}) exceeds protein length ({p1_len})")
        if end > min_p1_cut:
            warnings.append(f"P1 region {i+1} ({start}-{end}) may be partially cut off "
                          f"when p1_cut < {end}. Min p1_cut is {min_p1_cut}.")

    for i, (start, end) in enumerate(p2_regions):
        if end > p2_len:
            raise ValueError(f"P2 region {i+1} end ({end}) exceeds protein length ({p2_len})")
        if start < max_p2_cut:
            warnings.append(f"P2 region {i+1} ({start}-{end}) may be partially cut off "
                          f"when p2_cut > {start}. Max p2_cut is {max_p2_cut}.")

    return warnings


def load_sequence(input_str):
    """Load protein sequence from string or FASTA file."""
    if os.path.isfile(input_str):
        records = list(SeqIO.parse(input_str, "fasta"))
        if not records:
            raise ValueError(f"No sequences in FASTA: {input_str}")
        if len(records) > 1:
            print(f"Warning: Multiple sequences in {input_str}. Using first.")
        return records[0].id, str(records[0].seq)
    else:
        seq = input_str.strip().replace(" ", "").upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        if not all(aa in valid_aa for aa in seq):
            raise ValueError(f"Invalid amino acid sequence: {input_str}")
        return "input_seq", seq


def create_fusion_grid(seq1, seq2, nterm_scan_range, cterm_scan_range, step_size,
                       id1="protein1", id2="protein2"):
    """
    Create grid of fusion constructs.

    Returns:
        List of (fusion_id, fusion_seq, p1_cut, p2_cut, p1_end, p2_start) tuples
    """
    constructs = []

    nterm_start, nterm_end = nterm_scan_range
    cterm_start, cterm_end = cterm_scan_range

    for p1_cut in range(nterm_start, nterm_end + 1, step_size):
        for p2_cut in range(cterm_start, cterm_end + 1, step_size):
            fusion_id = f"{id1}_{id2}_p1cut{p1_cut}_p2cut{p2_cut}"

            # Fusion: Protein1[1:p1_cut] + Protein2[p2_cut:end]
            p1_fragment = seq1[:p1_cut]
            p2_fragment = seq2[p2_cut-1:]  # p2_cut is 1-indexed

            fusion_seq = p1_fragment + p2_fragment

            p1_end_pos = len(p1_fragment)
            p2_start_pos = p1_end_pos + 1

            constructs.append((fusion_id, fusion_seq, p1_cut, p2_cut, p1_end_pos, p2_start_pos))

    return constructs


def create_temp_fasta(sequences_dict):
    """Create temporary FASTA file."""
    records = []
    for seq_id, seq in sequences_dict.items():
        records.append(SeqRecord(Seq(seq), id=seq_id, description=""))

    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    SeqIO.write(records, temp_fasta, "fasta")
    temp_fasta.close()

    return temp_fasta.name


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_swe_similarity(original_aa_embed, fusion_aa_embed, swe_pooler):
    """
    Compute SWE-based similarity between original and fusion regions.
    """
    device = next(swe_pooler.parameters()).device

    original_tensor = torch.from_numpy(original_aa_embed).unsqueeze(0).float().to(device)
    fusion_tensor = torch.from_numpy(fusion_aa_embed).unsqueeze(0).float().to(device)

    with torch.no_grad():
        original_swe = swe_pooler(original_tensor).cpu().numpy().squeeze()
        fusion_swe = swe_pooler(fusion_tensor).cpu().numpy().squeeze()

    swe_cosine_sim = cosine_similarity(original_swe, fusion_swe)
    swe_euclidean_dist = np.linalg.norm(original_swe - fusion_swe)

    return {
        'original_swe': original_swe,
        'fusion_swe': fusion_swe,
        'swe_cosine_similarity': swe_cosine_sim,
        'swe_euclidean_distance': swe_euclidean_dist
    }


def get_fusion_position_for_p1_region(region_start, region_end, p1_cut):
    """
    Get the position of a P1 region in the fusion construct.

    P1 regions stay at the same position in the fusion (1-indexed).
    Returns (fusion_start, fusion_end) in 0-indexed for slicing.

    Returns None if region is cut off by p1_cut.
    """
    if region_end > p1_cut:
        return None  # Region is partially or fully cut off

    # P1 regions are at the same position in fusion (just convert to 0-indexed)
    return (region_start - 1, region_end)


def get_fusion_position_for_p2_region(region_start, region_end, p1_cut, p2_cut):
    """
    Get the position of a P2 region in the fusion construct.

    P2 regions are shifted because:
    - Original P2 position X becomes fusion position (p1_cut + X - p2_cut + 1)

    Returns (fusion_start, fusion_end) in 0-indexed for slicing.

    Returns None if region is cut off by p2_cut.
    """
    if region_start < p2_cut:
        return None  # Region is partially or fully cut off

    # Calculate offset: P2[p2_cut] becomes fusion[p1_cut + 1]
    offset = p1_cut - p2_cut + 1

    fusion_start = region_start + offset - 1  # Convert to 0-indexed
    fusion_end = region_end + offset

    return (fusion_start, fusion_end)


def analyze_fixed_regions(model, tokenizer, config_attrs, seq1, seq2, id1, id2,
                          fusion_constructs, p1_regions, p2_regions, args):
    """
    Main analysis function with fixed region comparisons.
    """
    results = {
        'protein1_id': id1,
        'protein2_id': id2,
        'protein1_seq': seq1,
        'protein2_seq': seq2,
        'protein1_length': len(seq1),
        'protein2_length': len(seq2),
        'nterm_scan_range': args.nterm_scan,
        'cterm_scan_range': args.cterm_scan,
        'p1_compare_regions': p1_regions,
        'p2_compare_regions': p2_regions,
        'num_constructs': len(fusion_constructs),
        'fusion_analyses': []
    }

    # Step 1: Embed original proteins
    print("\n" + "="*70)
    print("Step 1: Embedding original proteins")
    print("="*70)

    original_seqs = {id1: seq1, id2: seq2}
    original_fasta = create_temp_fasta(original_seqs)

    try:
        ids_orig, seqs_orig, seqs_spaced_orig = parse_fasta_for_embed(
            original_fasta,
            truncate=config_attrs.get("max_sequence_length"),
            padding=0
        )
        seqlens_orig = [len(s) for s in seqs_orig]

        original_embeddings = get_embeddings(
            model=model,
            tokenizer=tokenizer,
            config_attrs=config_attrs,
            seqs=seqs_spaced_orig,
            seqlens=seqlens_orig,
            get_sequence_embeddings=True,
            get_aa_embeddings=True,
            get_sequence_activations=False,
            get_aa_activations=False,
            padding=0,
            layers=args.layers,
            all_layers=args.all_layers,
            strat=args.strat,
            cpu_only=args.cpu_only,
            half=torch.cuda.is_available() and not args.cpu_only,
            batch_size=args.batch_size
        )

        # Store original embeddings
        original_aa_embeds = {}
        original_seq_embeds = {}
        for i, seq_id in enumerate(ids_orig):
            if 'aa_embeddings' in original_embeddings:
                actual_len = seqlens_orig[i]
                original_aa_embeds[seq_id] = original_embeddings['aa_embeddings'][i, :actual_len, :]
            if 'sequence_embeddings' in original_embeddings:
                original_seq_embeds[seq_id] = original_embeddings['sequence_embeddings'][i]

        results['original_sequence_embeddings'] = original_seq_embeds
        results['original_aa_embeddings'] = original_aa_embeds

        print(f"\nOriginal embeddings generated:")
        print(f"  {id1}: length={len(seq1)}, aa_embed_shape={original_aa_embeds[id1].shape}")
        print(f"  {id2}: length={len(seq2)}, aa_embed_shape={original_aa_embeds[id2].shape}")

    finally:
        os.unlink(original_fasta)

    # Step 2: Initialize SWE pooler
    print("\n" + "="*70)
    print("Step 2: Initializing SWE pooler")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    hidden_dim = config_attrs['hidden_size']

    swe_pooler = SWE_Pooling(
        d_in=hidden_dim,
        num_slices=hidden_dim,
        num_ref_points=100,
        freeze_swe=True
    ).to(device)
    print(f"SWE pooler initialized on {device}")

    # Pre-extract original region embeddings (these don't change)
    print("\n" + "="*70)
    print("Step 3: Extracting original region embeddings")
    print("="*70)

    original_p1_region_embeds = {}
    for i, (start, end) in enumerate(p1_regions):
        region_name = f"p1_region_{start}_{end}"
        # 0-indexed slicing: start-1 to end
        original_p1_region_embeds[region_name] = original_aa_embeds[id1][start-1:end]
        print(f"  {region_name}: {original_p1_region_embeds[region_name].shape}")

    original_p2_region_embeds = {}
    for i, (start, end) in enumerate(p2_regions):
        region_name = f"p2_region_{start}_{end}"
        original_p2_region_embeds[region_name] = original_aa_embeds[id2][start-1:end]
        print(f"  {region_name}: {original_p2_region_embeds[region_name].shape}")

    # Step 4: Process fusion constructs
    print("\n" + "="*70)
    print(f"Step 4: Processing {len(fusion_constructs)} fusion constructs")
    print("="*70)

    chunk_size = 50
    num_chunks = (len(fusion_constructs) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(fusion_constructs))
        chunk_constructs = fusion_constructs[start_idx:end_idx]

        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: Processing constructs {start_idx + 1}-{end_idx}")

        fusion_seqs = {fid: fseq for fid, fseq, _, _, _, _ in chunk_constructs}
        fusion_fasta = create_temp_fasta(fusion_seqs)

        try:
            ids_fusion, seqs_fusion, seqs_spaced_fusion = parse_fasta_for_embed(
                fusion_fasta,
                truncate=config_attrs.get("max_sequence_length"),
                padding=0
            )
            seqlens_fusion = [len(s) for s in seqs_fusion]

            fusion_embeddings = get_embeddings(
                model=model,
                tokenizer=tokenizer,
                config_attrs=config_attrs,
                seqs=seqs_spaced_fusion,
                seqlens=seqlens_fusion,
                get_sequence_embeddings=True,
                get_aa_embeddings=True,
                get_sequence_activations=False,
                get_aa_activations=False,
                padding=0,
                layers=args.layers,
                all_layers=args.all_layers,
                strat=args.strat,
                cpu_only=args.cpu_only,
                half=torch.cuda.is_available() and not args.cpu_only,
                batch_size=args.batch_size
            )

            fusion_id_to_idx = {fid: i for i, fid in enumerate(ids_fusion)}
            fusion_id_to_len = {fid: seqlens_fusion[i] for i, fid in enumerate(ids_fusion)}

        finally:
            os.unlink(fusion_fasta)

        # Analyze each construct
        for construct_idx, (fusion_id, fusion_seq, p1_cut, p2_cut, p1_end_pos, p2_start_pos) in enumerate(chunk_constructs):

            fusion_idx = fusion_id_to_idx[fusion_id]
            fusion_len = fusion_id_to_len[fusion_id]
            fusion_aa_embed = fusion_embeddings['aa_embeddings'][fusion_idx, :fusion_len, :]
            fusion_seq_embed = fusion_embeddings['sequence_embeddings'][fusion_idx]

            fusion_result = {
                'fusion_id': fusion_id,
                'p1_cut': p1_cut,
                'p2_cut': p2_cut,
                'fusion_length': len(fusion_seq),
            }

            # Compare P1 fixed regions
            for i, (start, end) in enumerate(p1_regions):
                region_name = f"p1_region_{start}_{end}"

                fusion_pos = get_fusion_position_for_p1_region(start, end, p1_cut)

                if fusion_pos is None:
                    # Region is cut off
                    fusion_result[f'{region_name}_swe_similarity'] = np.nan
                    fusion_result[f'{region_name}_included'] = False
                else:
                    fusion_start, fusion_end = fusion_pos
                    fusion_region_embed = fusion_aa_embed[fusion_start:fusion_end]
                    original_region_embed = original_p1_region_embeds[region_name]

                    # Verify shapes match
                    if fusion_region_embed.shape[0] != original_region_embed.shape[0]:
                        print(f"  Warning: Shape mismatch for {region_name} in {fusion_id}")
                        fusion_result[f'{region_name}_swe_similarity'] = np.nan
                        fusion_result[f'{region_name}_included'] = False
                    else:
                        swe_result = compute_swe_similarity(original_region_embed, fusion_region_embed, swe_pooler)
                        fusion_result[f'{region_name}_swe_similarity'] = swe_result['swe_cosine_similarity']
                        fusion_result[f'{region_name}_included'] = True

            # Compare P2 fixed regions
            for i, (start, end) in enumerate(p2_regions):
                region_name = f"p2_region_{start}_{end}"

                fusion_pos = get_fusion_position_for_p2_region(start, end, p1_cut, p2_cut)

                if fusion_pos is None:
                    # Region is cut off
                    fusion_result[f'{region_name}_swe_similarity'] = np.nan
                    fusion_result[f'{region_name}_included'] = False
                else:
                    fusion_start, fusion_end = fusion_pos

                    # Bounds check
                    if fusion_end > fusion_len:
                        print(f"  Warning: P2 region extends beyond fusion for {fusion_id}")
                        fusion_result[f'{region_name}_swe_similarity'] = np.nan
                        fusion_result[f'{region_name}_included'] = False
                    else:
                        fusion_region_embed = fusion_aa_embed[fusion_start:fusion_end]
                        original_region_embed = original_p2_region_embeds[region_name]

                        if fusion_region_embed.shape[0] != original_region_embed.shape[0]:
                            print(f"  Warning: Shape mismatch for {region_name} in {fusion_id}")
                            fusion_result[f'{region_name}_swe_similarity'] = np.nan
                            fusion_result[f'{region_name}_included'] = False
                        else:
                            swe_result = compute_swe_similarity(original_region_embed, fusion_region_embed, swe_pooler)
                            fusion_result[f'{region_name}_swe_similarity'] = swe_result['swe_cosine_similarity']
                            fusion_result[f'{region_name}_included'] = True

            # Full construct similarities
            fusion_result['fusion_to_p1_similarity'] = cosine_similarity(fusion_seq_embed, original_seq_embeds[id1])
            fusion_result['fusion_to_p2_similarity'] = cosine_similarity(fusion_seq_embed, original_seq_embeds[id2])

            results['fusion_analyses'].append(fusion_result)

        # Free memory
        del fusion_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAnalysis complete!")
    return results


def create_summary_dataframe(results):
    """Create pandas DataFrame with summary statistics."""
    rows = []
    for fusion_result in results['fusion_analyses']:
        row = {
            'fusion_id': fusion_result['fusion_id'],
            'p1_cut': fusion_result['p1_cut'],
            'p2_cut': fusion_result['p2_cut'],
            'fusion_length': fusion_result['fusion_length'],
            'fusion_to_p1_similarity': fusion_result['fusion_to_p1_similarity'],
            'fusion_to_p2_similarity': fusion_result['fusion_to_p2_similarity'],
        }

        # Add all region columns
        for key, value in fusion_result.items():
            if key.startswith('p1_region_') or key.startswith('p2_region_'):
                row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_results(results, output_prefix, fusion_constructs=None):
    """Save results to pickle, CSV, description, and FASTA files."""
    p1_start, p1_end = results['nterm_scan_range']
    p2_start, p2_end = results['cterm_scan_range']
    range_suffix = f"_p1_{p1_start}-{p1_end}_p2_{p2_start}-{p2_end}_fixedregions"

    pkl_file = f"{output_prefix}{range_suffix}.pkl"
    desc_file = f"{output_prefix}{range_suffix}.description.txt"
    csv_file = f"{output_prefix}{range_suffix}.summary.csv"
    fasta_file = f"{output_prefix}{range_suffix}.fasta"

    # Save pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults saved to: {pkl_file}")

    # Save CSV
    df = create_summary_dataframe(results)
    df.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")

    # Save FASTA
    if fusion_constructs:
        records = []
        for fusion_id, fusion_seq, p1_cut, p2_cut, p1_end_pos, p2_start_pos in fusion_constructs:
            desc = f"p1_cut={p1_cut} p2_cut={p2_cut} len={len(fusion_seq)}"
            records.append(SeqRecord(Seq(fusion_seq), id=f"p1cut{p1_cut}_p2cut{p2_cut}", description=desc))
        SeqIO.write(records, fasta_file, "fasta")
        print(f"Fusion sequences FASTA saved to: {fasta_file}")

    # Save description
    with open(desc_file, 'w') as f:
        f.write("Domain Swap Embedding Analysis - Fixed Region Comparisons\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Protein 1 (N-term): {results['protein1_id']}\n")
        f.write(f"  Length: {results['protein1_length']}\n\n")

        f.write(f"Protein 2 (C-term): {results['protein2_id']}\n")
        f.write(f"  Length: {results['protein2_length']}\n\n")

        f.write(f"Scan Parameters:\n")
        f.write(f"  Protein 1 cut range: {results['nterm_scan_range'][0]}-{results['nterm_scan_range'][1]}\n")
        f.write(f"  Protein 2 cut range: {results['cterm_scan_range'][0]}-{results['cterm_scan_range'][1]}\n")
        f.write(f"  Total constructs: {results['num_constructs']}\n\n")

        f.write("Fixed Comparison Regions:\n")
        f.write("-" * 70 + "\n")
        f.write("These regions are compared between original and fusion contexts.\n")
        f.write("Each region has the SAME number of residues across all comparisons.\n\n")

        if results['p1_compare_regions']:
            f.write("Protein 1 (N-term) regions:\n")
            for start, end in results['p1_compare_regions']:
                length = end - start + 1
                f.write(f"  - Residues {start}-{end} ({length} residues)\n")
        else:
            f.write("Protein 1 (N-term) regions: None specified\n")

        f.write("\n")

        if results['p2_compare_regions']:
            f.write("Protein 2 (C-term) regions:\n")
            for start, end in results['p2_compare_regions']:
                length = end - start + 1
                f.write(f"  - Residues {start}-{end} ({length} residues)\n")
        else:
            f.write("Protein 2 (C-term) regions: None specified\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CSV Column Descriptions:\n")
        f.write("=" * 70 + "\n\n")
        f.write("- p1_cut, p2_cut: Junction positions\n")
        f.write("- fusion_length: Total fusion construct length\n")
        f.write("- fusion_to_p1_similarity: Full fusion vs original P1 (mean-pooled cosine)\n")
        f.write("- fusion_to_p2_similarity: Full fusion vs original P2 (mean-pooled cosine)\n\n")

        f.write("For each region (e.g., p1_region_1_400):\n")
        f.write("- *_swe_similarity: SWE-based cosine similarity (distributional shape)\n")
        f.write("- *_included: Whether the region is fully present in this fusion\n")
        f.write("\nNote: Each region produces one heatmap column. Regions are specified\n")
        f.write("in original protein coordinates (not fusion coordinates).\n")

    print(f"Description saved to: {desc_file}")


def main():
    args = get_args()

    print("=" * 70)
    print("Domain Swap Embedding Analysis - Fixed Region Comparisons")
    print("=" * 70)

    # Parse comparison regions
    p1_regions = parse_region_args(args.p1_compare_regions)
    p2_regions = parse_region_args(args.p2_compare_regions)

    if not p1_regions and not p2_regions:
        print("\nWARNING: No comparison regions specified!")
        print("Use --p1_compare_regions and/or --p2_compare_regions to specify regions.")
        print("Example: --p1_compare_regions 1 400 --p2_compare_regions 250 350")
        return

    # Load sequences
    print("\nLoading protein sequences...")
    id1, seq1 = load_sequence(args.protein1)
    id2, seq2 = load_sequence(args.protein2)

    print(f"\nProtein 1 ({id1}): {len(seq1)} residues")
    print(f"  Scan range: {args.nterm_scan[0]}-{args.nterm_scan[1]}")
    if p1_regions:
        print(f"  Comparison regions: {p1_regions}")

    print(f"\nProtein 2 ({id2}): {len(seq2)} residues")
    print(f"  Scan range: {args.cterm_scan[0]}-{args.cterm_scan[1]}")
    if p2_regions:
        print(f"  Comparison regions: {p2_regions}")

    # Validate regions
    warnings = validate_regions(p1_regions, p2_regions, len(seq1), len(seq2),
                                args.nterm_scan, args.cterm_scan)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    # Create fusion grid
    print(f"\nCreating fusion grid (step size: {args.step_size})...")
    fusion_constructs = create_fusion_grid(
        seq1, seq2,
        tuple(args.nterm_scan),
        tuple(args.cterm_scan),
        args.step_size,
        id1, id2
    )

    print(f"Grid size: {len(fusion_constructs)} fusion constructs")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    half_precision = torch.cuda.is_available() and not args.cpu_only

    model, tokenizer, config_attrs = load_model(
        args.model_path,
        output_hidden_states=True,
        output_attentions=False,
        half=half_precision
    )

    print(f"\nModel configuration:")
    print(f"  Type: {config_attrs['model_type']}")
    print(f"  Hidden size: {config_attrs['hidden_size']}")

    # Run analysis
    results = analyze_fixed_regions(
        model, tokenizer, config_attrs,
        seq1, seq2, id1, id2,
        fusion_constructs, p1_regions, p2_regions, args
    )

    # Save results
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    save_results(results, args.output_prefix, fusion_constructs)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
