#!/usr/bin/env python3
"""
Domain Swap Embedding Analysis for Sensor Kinase Engineering

This script creates chimeric sensor kinases by swapping domains between two proteins
(e.g., AHK2 sensor domain + NarQ signaling domain) and analyzes how embeddings change
in the fusion context vs original context.

Strategy:
- Protein 1 (N-term, e.g., AHK2): Contributes ligand-binding sensor domain
- Protein 2 (C-term, e.g., NarQ): Contributes TM2 + HAMP + kinase signaling core
- Grid search: Test all combinations of junction points in specified scan ranges
- Evaluation: Compare fragment embeddings and full construct embeddings

Example use case:
Creating AHK2(CHASE sensor) → NarQ(TM2-HAMP-kinase) chimeras to create
cytokinin-responsive sensor kinases.
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
        description="Domain swap embedding analysis for sensor kinase engineering"
    )
    parser.add_argument("-m", "--model", dest="model_path", type=str, required=True,
                        help="Model directory path (e.g., /path/to/esm2_model)")
    parser.add_argument("-p1", "--protein1", dest="protein1", type=str, required=True,
                        help="Protein 1 (N-term contributor) sequence or FASTA file")
    parser.add_argument("-p2", "--protein2", dest="protein2", type=str, required=True,
                        help="Protein 2 (C-term contributor) sequence or FASTA file")

    # Alignment information
    parser.add_argument("--nterm_overlap", dest="nterm_overlap", type=int, nargs=2,
                        default=[546, 553],
                        help="Aligned region in Protein 1 (start, end). Default: 546 553")
    parser.add_argument("--cterm_overlap", dest="cterm_overlap", type=int, nargs=2,
                        default=[158, 165],
                        help="Aligned region in Protein 2 (start, end). Default: 158 165")

    # Scan ranges for junction points
    parser.add_argument("--nterm_scan", dest="nterm_scan", type=int, nargs=2,
                        default=[529, 545],
                        help="Scan range for Protein 1 cut positions (start, end). Default: 529 545")
    parser.add_argument("--cterm_scan", dest="cterm_scan", type=int, nargs=2,
                        default=[133, 157],
                        help="Scan range for Protein 2 cut positions (start, end). Default: 133 157")
    parser.add_argument("--step_size", dest="step_size", type=int, default=1,
                        help="Step size for scanning (default: 1 = test every position)")

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

    Args:
        seq1: Protein 1 sequence (N-term contributor)
        seq2: Protein 2 sequence (C-term contributor)
        nterm_scan_range: (start, end) positions to cut Protein 1
        cterm_scan_range: (start, end) positions to cut Protein 2
        step_size: Scanning step size
        id1, id2: Protein IDs

    Returns:
        List of (fusion_id, fusion_seq, p1_cut, p2_cut, p1_end, p2_start) tuples
    """
    constructs = []

    nterm_start, nterm_end = nterm_scan_range
    cterm_start, cterm_end = cterm_scan_range

    # Scan through Protein 1 cut positions
    for p1_cut in range(nterm_start, nterm_end + 1, step_size):
        # Scan through Protein 2 cut positions
        for p2_cut in range(cterm_start, cterm_end + 1, step_size):
            fusion_id = f"{id1}_{id2}_p1cut{p1_cut}_p2cut{p2_cut}"

            # Create fusion: Protein1[1:p1_cut] + Protein2[p2_cut:end]
            # Note: Python slicing is [start:end) so seq1[:p1_cut] gives positions 0 to p1_cut-1
            # For 1-indexed biological positions, p1_cut means "keep up to and including position p1_cut"
            # So we use seq1[:p1_cut] which in 0-indexed is [0:p1_cut), giving positions 1-p1_cut in 1-indexed
            p1_fragment = seq1[:p1_cut]
            p2_fragment = seq2[p2_cut-1:]  # p2_cut is 1-indexed, so p2_cut-1 in 0-indexed

            fusion_seq = p1_fragment + p2_fragment

            # Store 1-indexed positions for clarity
            p1_end_pos = len(p1_fragment)  # Last position of P1 in fusion (1-indexed)
            p2_start_pos = p1_end_pos + 1   # First position of P2 in fusion (1-indexed)

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

    # Handle edge case where one or both vectors have zero norm
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_swe_similarity(original_aa_embed, fusion_aa_embed, swe_pooler):
    """
    Compute SWE-based similarity between original and fusion fragments.

    Uses Sliced Wasserstein Embedding to compare the distributional shape
    of the embedding clouds.

    Args:
        original_aa_embed: Original per-residue embeddings (seq_len, hidden_dim)
        fusion_aa_embed: Fusion per-residue embeddings (seq_len, hidden_dim)
        swe_pooler: Pre-initialized SWE_Pooling module

    Returns:
        dict with SWE embeddings and similarity metrics
    """
    device = next(swe_pooler.parameters()).device

    # Convert to torch tensors and add batch dimension
    original_tensor = torch.from_numpy(original_aa_embed).unsqueeze(0).float().to(device)
    fusion_tensor = torch.from_numpy(fusion_aa_embed).unsqueeze(0).float().to(device)

    # Compute SWE embeddings
    with torch.no_grad():
        original_swe = swe_pooler(original_tensor).cpu().numpy().squeeze()
        fusion_swe = swe_pooler(fusion_tensor).cpu().numpy().squeeze()

    # Compute similarity between SWE embeddings
    swe_cosine_sim = cosine_similarity(original_swe, fusion_swe)
    swe_euclidean_dist = np.linalg.norm(original_swe - fusion_swe)

    return {
        'original_swe': original_swe,
        'fusion_swe': fusion_swe,
        'swe_cosine_similarity': swe_cosine_sim,
        'swe_euclidean_distance': swe_euclidean_dist
    }


def compute_fragment_similarity(original_aa_embed, fusion_aa_embed):
    """
    Compute cosine similarity between mean-pooled fragment embeddings.

    Takes the mean across residues for each fragment, then computes cosine similarity
    between the two mean-pooled vectors.

    Args:
        original_aa_embed: Original per-residue embeddings (seq_len, hidden_dim)
        fusion_aa_embed: Fusion per-residue embeddings (seq_len, hidden_dim)

    Returns:
        float: Cosine similarity between mean-pooled embeddings
    """
    if original_aa_embed.shape != fusion_aa_embed.shape:
        raise ValueError(f"Shape mismatch: {original_aa_embed.shape} vs {fusion_aa_embed.shape}")

    # Convert to float64 to prevent overflow during mean computation
    # fp16 can overflow when summing many large values (529 residues * large values)
    original_f64 = original_aa_embed.astype(np.float64)
    fusion_f64 = fusion_aa_embed.astype(np.float64)

    # Mean pool across residues (axis=0)
    original_mean = np.mean(original_f64, axis=0)  # Shape: (hidden_dim,)
    fusion_mean = np.mean(fusion_f64, axis=0)      # Shape: (hidden_dim,)

    # Check for inf/nan after mean
    if np.isinf(original_mean).any() or np.isinf(fusion_mean).any():
        return np.nan
    if np.isnan(original_mean).any() or np.isnan(fusion_mean).any():
        return np.nan

    # Compute cosine similarity between mean-pooled vectors
    # Convert back to float32 for cosine computation
    similarity = cosine_similarity(original_mean.astype(np.float32),
                                   fusion_mean.astype(np.float32))

    return similarity


def analyze_domain_swap_grid(model, tokenizer, config_attrs, seq1, seq2, id1, id2,
                             fusion_constructs, args):
    """
    Main analysis function for domain swap grid search.

    Returns:
        dict: Analysis results including grid of similarities
    """
    results = {
        'protein1_id': id1,
        'protein2_id': id2,
        'protein1_seq': seq1,
        'protein2_seq': seq2,
        'protein1_length': len(seq1),
        'protein2_length': len(seq2),
        'nterm_overlap': args.nterm_overlap,
        'cterm_overlap': args.cterm_overlap,
        'nterm_scan_range': args.nterm_scan,
        'cterm_scan_range': args.cterm_scan,
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

        # Debug: Check the raw embeddings shape
        print(f"\nDEBUG: Raw embeddings from get_embeddings:")
        print(f"  ids_orig: {ids_orig}")
        if 'aa_embeddings' in original_embeddings:
            print(f"  aa_embeddings shape (padded): {original_embeddings['aa_embeddings'].shape}")
        if 'sequence_embeddings' in original_embeddings:
            print(f"  sequence_embeddings shape: {original_embeddings['sequence_embeddings'].shape}")

        # Store original embeddings - TRIM TO ACTUAL SEQUENCE LENGTH
        original_aa_embeds = {}
        original_seq_embeds = {}
        for i, seq_id in enumerate(ids_orig):
            if 'aa_embeddings' in original_embeddings:
                # Trim to actual sequence length (no padding)
                actual_len = seqlens_orig[i]
                original_aa_embeds[seq_id] = original_embeddings['aa_embeddings'][i, :actual_len, :]
                print(f"  Storing {seq_id} (index {i}): trimmed from {original_embeddings['aa_embeddings'].shape[1]} to {actual_len}, final shape {original_aa_embeds[seq_id].shape}")
            if 'sequence_embeddings' in original_embeddings:
                original_seq_embeds[seq_id] = original_embeddings['sequence_embeddings'][i]

        results['original_sequence_embeddings'] = original_seq_embeds
        results['original_aa_embeddings'] = original_aa_embeds

        print(f"\nOriginal embeddings generated:")
        print(f"  {id1}: length={len(seq1)}, aa_embed_shape={original_aa_embeds[id1].shape}")
        print(f"  {id2}: length={len(seq2)}, aa_embed_shape={original_aa_embeds[id2].shape}")
        print(f"  {id1}: seq_embed_shape={original_seq_embeds[id1].shape}")
        print(f"  {id2}: seq_embed_shape={original_seq_embeds[id2].shape}")

    finally:
        os.unlink(original_fasta)

    # Step 2: Initialize SWE pooler and compute full-sequence SWE for original proteins
    print("\n" + "="*70)
    print("Step 2: Initializing SWE pooler and computing full-sequence SWE")
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

    # Compute full-sequence SWE for original proteins
    original_swe_embeds = {}
    for seq_id in [id1, id2]:
        aa_embed_torch = torch.from_numpy(original_aa_embeds[seq_id]).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            swe_embed = swe_pooler(aa_embed_torch).cpu().numpy()[0]
        original_swe_embeds[seq_id] = swe_embed
        print(f"  Computed full-sequence SWE for {seq_id}: shape {swe_embed.shape}")

    results['original_full_swe_embeddings'] = original_swe_embeds

    # Step 3: Process fusion constructs in batches (memory-efficient)
    print("\n" + "="*70)
    print(f"Step 3: Processing {len(fusion_constructs)} fusion constructs")
    print("="*70)

    # Process in smaller chunks to avoid memory issues
    chunk_size = 50  # Process 50 fusions at a time
    num_chunks = (len(fusion_constructs) + chunk_size - 1) // chunk_size

    print(f"Processing in {num_chunks} chunks of {chunk_size} constructs each")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(fusion_constructs))
        chunk_constructs = fusion_constructs[start_idx:end_idx]

        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: Processing constructs {start_idx + 1}-{end_idx}")

        # Create FASTA for this chunk
        fusion_seqs = {fid: fseq for fid, fseq, _, _, _, _ in chunk_constructs}
        fusion_fasta = create_temp_fasta(fusion_seqs)

        try:
            ids_fusion, seqs_fusion, seqs_spaced_fusion = parse_fasta_for_embed(
                fusion_fasta,
                truncate=config_attrs.get("max_sequence_length"),
                padding=0
            )
            seqlens_fusion = [len(s) for s in seqs_fusion]

            print(f"  Embedding {len(ids_fusion)} fusion constructs...")
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

            # Map fusion IDs to embedding indices and lengths for this chunk
            fusion_id_to_idx = {fid: i for i, fid in enumerate(ids_fusion)}
            fusion_id_to_len = {fid: seqlens_fusion[i] for i, fid in enumerate(ids_fusion)}

        finally:
            os.unlink(fusion_fasta)

        # Step 4: Analyze each fusion construct in this chunk
        print(f"  Computing similarity metrics...")

        for construct_idx, (fusion_id, fusion_seq, p1_cut, p2_cut, p1_end_pos, p2_start_pos) in enumerate(chunk_constructs):

            # Get fusion embeddings - TRIM TO ACTUAL SEQUENCE LENGTH
            fusion_idx = fusion_id_to_idx[fusion_id]
            fusion_len = fusion_id_to_len[fusion_id]
            fusion_aa_embed = fusion_embeddings['aa_embeddings'][fusion_idx, :fusion_len, :]
            fusion_seq_embed = fusion_embeddings['sequence_embeddings'][fusion_idx]

            # Extract protein fragments from fusion (0-indexed slicing)
            p1_fusion_fragment = fusion_aa_embed[:p1_end_pos]  # Protein 1 fragment
            p2_fusion_fragment = fusion_aa_embed[p1_end_pos:]  # Protein 2 fragment

            # Get original fragments - need to handle the actual lengths
            # p1_cut is 1-indexed, so seq1[:p1_cut] has length p1_cut
            original_p1_fragment = original_aa_embeds[id1][:p1_cut]

            # p2_cut is 1-indexed position where we start taking from seq2
            # seq2[p2_cut-1:] in 0-indexed gives us from position p2_cut onwards
            original_p2_fragment = original_aa_embeds[id2][p2_cut-1:]

            # Debug first construct in first chunk
            if chunk_idx == 0 and construct_idx == 0:
                print(f"\n  DEBUG first construct:")
                print(f"    fusion_id: {fusion_id}")
                print(f"    p1_cut={p1_cut}, p2_cut={p2_cut}, p1_end_pos={p1_end_pos}, p2_start_pos={p2_start_pos}")
                print(f"    fusion_len={fusion_len}, fusion_aa_embed.shape={fusion_aa_embed.shape}")
                print(f"    original_aa_embeds[{id1}].shape={original_aa_embeds[id1].shape}")
                print(f"    original_aa_embeds[{id2}].shape={original_aa_embeds[id2].shape}")
                print(f"    p1_fusion_fragment.shape={p1_fusion_fragment.shape}, original_p1_fragment.shape={original_p1_fragment.shape}")
                print(f"    p2_fusion_fragment.shape={p2_fusion_fragment.shape}, original_p2_fragment.shape={original_p2_fragment.shape}")

                # Check for NaN, inf, and extreme values
                print(f"\n  DEBUG: Checking fragment embedding quality:")
                print(f"    original_p1_fragment: min={np.min(original_p1_fragment):.4f}, max={np.max(original_p1_fragment):.4f}")
                print(f"    p1_fusion_fragment: min={np.min(p1_fusion_fragment):.4f}, max={np.max(p1_fusion_fragment):.4f}")
                print(f"    original_p2_fragment: min={np.min(original_p2_fragment):.4f}, max={np.max(original_p2_fragment):.4f}")
                print(f"    p2_fusion_fragment: min={np.min(p2_fusion_fragment):.4f}, max={np.max(p2_fusion_fragment):.4f}")

                print(f"    original_p1_fragment has inf: {np.isinf(original_p1_fragment).any()}")
                print(f"    p1_fusion_fragment has inf: {np.isinf(p1_fusion_fragment).any()}")
                print(f"    original_p2_fragment has inf: {np.isinf(original_p2_fragment).any()}")
                print(f"    p2_fusion_fragment has inf: {np.isinf(p2_fusion_fragment).any()}")

                # Test mean-pooled vectors
                orig_p1_clipped = np.clip(original_p1_fragment, -1e4, 1e4)
                fus_p1_clipped = np.clip(p1_fusion_fragment, -1e4, 1e4)
                orig_p1_mean = np.mean(orig_p1_clipped, axis=0)
                fus_p1_mean = np.mean(fus_p1_clipped, axis=0)
                print(f"\n  DEBUG: Mean-pooled vectors after clipping:")
                print(f"    original_p1_mean norm: {np.linalg.norm(orig_p1_mean):.4f}")
                print(f"    fusion_p1_mean norm: {np.linalg.norm(fus_p1_mean):.4f}")

            # Debug: Check if shapes match - skip if mismatch
            if original_p1_fragment.shape[0] != p1_fusion_fragment.shape[0]:
                print(f"  Warning: P1 shape mismatch for {fusion_id}: original={original_p1_fragment.shape[0]}, fusion={p1_fusion_fragment.shape[0]}")
                print(f"    p1_cut={p1_cut}, p1_end_pos={p1_end_pos}")
                continue

            if original_p2_fragment.shape[0] != p2_fusion_fragment.shape[0]:
                print(f"  Warning: P2 shape mismatch for {fusion_id}: original={original_p2_fragment.shape[0]}, fusion={p2_fusion_fragment.shape[0]}")
                print(f"    p2_cut={p2_cut}, p2_start_pos={p2_start_pos}, fusion_len={len(fusion_seq)}")
                print(f"    Expected P2 length: {len(seq2) - (p2_cut - 1)}")
                print(f"    Actual original P2 embed length: {len(original_aa_embeds[id2])}")
                continue

            # Compare fragments with original contexts (per-residue)
            p1_fragment_similarity = compute_fragment_similarity(
                original_p1_fragment,
                p1_fusion_fragment
            )

            p2_fragment_similarity = compute_fragment_similarity(
                original_p2_fragment,
                p2_fusion_fragment
            )

            # Debug first construct
            if chunk_idx == 0 and construct_idx == 0:
                print(f"\n  DEBUG: Mean-pooled cosine similarity results:")
                print(f"    p1_fragment_similarity: {p1_fragment_similarity}")
                print(f"    p2_fragment_similarity: {p2_fragment_similarity}")

            # Compare fragments using SWE (distributional shape)
            p1_swe_similarity = compute_swe_similarity(
                original_p1_fragment,
                p1_fusion_fragment,
                swe_pooler
            )

            p2_swe_similarity = compute_swe_similarity(
                original_p2_fragment,
                p2_fusion_fragment,
                swe_pooler
            )

            # Compare full fusion to original proteins (mean-pooled)
            fusion_to_p1_similarity = cosine_similarity(fusion_seq_embed, original_seq_embeds[id1])
            fusion_to_p2_similarity = cosine_similarity(fusion_seq_embed, original_seq_embeds[id2])

            # Compute full-sequence SWE for fusion and compare to original proteins
            fusion_aa_torch = torch.from_numpy(fusion_aa_embed).float().unsqueeze(0).to(device)
            with torch.inference_mode():
                fusion_full_swe = swe_pooler(fusion_aa_torch).cpu().numpy()[0]

            fusion_swe_to_p1_similarity = cosine_similarity(fusion_full_swe, original_swe_embeds[id1])
            fusion_swe_to_p2_similarity = cosine_similarity(fusion_full_swe, original_swe_embeds[id2])

            # Store results
            fusion_result = {
                'fusion_id': fusion_id,
                'p1_cut': p1_cut,
                'p2_cut': p2_cut,
                'fusion_length': len(fusion_seq),
                'p1_fragment_length': p1_end_pos,
                'p2_fragment_length': len(fusion_seq) - p1_end_pos,

                # Fragment similarities - mean-pooled cosine (original context vs fusion context)
                'p1_fragment_similarity': p1_fragment_similarity,
                'p2_fragment_similarity': p2_fragment_similarity,

                # Fragment similarities - SWE (distributional shape)
                'p1_swe_cosine_similarity': p1_swe_similarity['swe_cosine_similarity'],
                'p1_swe_euclidean_distance': p1_swe_similarity['swe_euclidean_distance'],
                'p2_swe_cosine_similarity': p2_swe_similarity['swe_cosine_similarity'],
                'p2_swe_euclidean_distance': p2_swe_similarity['swe_euclidean_distance'],

                # Full construct similarities (mean-pooled)
                'fusion_to_p1_similarity': fusion_to_p1_similarity,
                'fusion_to_p2_similarity': fusion_to_p2_similarity,

                # Full construct SWE similarities
                'fusion_swe_to_p1_similarity': fusion_swe_to_p1_similarity,
                'fusion_swe_to_p2_similarity': fusion_swe_to_p2_similarity,

                # SWE embeddings (for further analysis)
                'p1_original_swe': p1_swe_similarity['original_swe'],
                'p1_fusion_swe': p1_swe_similarity['fusion_swe'],
                'p2_original_swe': p2_swe_similarity['original_swe'],
                'p2_fusion_swe': p2_swe_similarity['fusion_swe'],
            }

            results['fusion_analyses'].append(fusion_result)

        # Free memory after processing this chunk
        del fusion_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAnalysis complete!")
    return results


def create_summary_dataframe(results):
    """Create pandas DataFrame with summary statistics for all fusions."""
    rows = []
    for fusion_result in results['fusion_analyses']:
        rows.append({
            'fusion_id': fusion_result['fusion_id'],
            'p1_cut': fusion_result['p1_cut'],
            'p2_cut': fusion_result['p2_cut'],
            'fusion_length': fusion_result['fusion_length'],

            # Mean-pooled cosine similarities
            'p1_fragment_similarity': fusion_result['p1_fragment_similarity'],
            'p2_fragment_similarity': fusion_result['p2_fragment_similarity'],

            # SWE similarities (distributional shape)
            'p1_swe_similarity': fusion_result['p1_swe_cosine_similarity'],
            'p2_swe_similarity': fusion_result['p2_swe_cosine_similarity'],
            'p1_swe_distance': fusion_result['p1_swe_euclidean_distance'],
            'p2_swe_distance': fusion_result['p2_swe_euclidean_distance'],

            # Full construct similarities (mean-pooled)
            'fusion_to_p1_similarity': fusion_result['fusion_to_p1_similarity'],
            'fusion_to_p2_similarity': fusion_result['fusion_to_p2_similarity'],

            # Full construct SWE similarities
            'fusion_swe_to_p1_similarity': fusion_result['fusion_swe_to_p1_similarity'],
            'fusion_swe_to_p2_similarity': fusion_result['fusion_swe_to_p2_similarity'],
        })

    df = pd.DataFrame(rows)
    return df


def save_results(results, output_prefix):
    """Save results to pickle and description files."""
    pkl_file = f"{output_prefix}.pkl"
    desc_file = f"{output_prefix}.description.txt"
    csv_file = f"{output_prefix}.summary.csv"

    # Save pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nResults saved to: {pkl_file}")

    # Create and save summary DataFrame
    df = create_summary_dataframe(results)
    df.to_csv(csv_file, index=False)
    print(f"Summary CSV saved to: {csv_file}")

    # Save description
    with open(desc_file, 'w') as f:
        f.write("Domain Swap Embedding Analysis Results\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Protein 1 (N-term): {results['protein1_id']}\n")
        f.write(f"  Length: {results['protein1_length']}\n")
        f.write(f"  Sequence: {results['protein1_seq'][:60]}...\n")
        f.write(f"  Aligned region: {results['nterm_overlap'][0]}-{results['nterm_overlap'][1]}\n\n")

        f.write(f"Protein 2 (C-term): {results['protein2_id']}\n")
        f.write(f"  Length: {results['protein2_length']}\n")
        f.write(f"  Sequence: {results['protein2_seq'][:60]}...\n")
        f.write(f"  Aligned region: {results['cterm_overlap'][0]}-{results['cterm_overlap'][1]}\n\n")

        f.write(f"Scan Parameters:\n")
        f.write(f"  Protein 1 cut range: {results['nterm_scan_range'][0]}-{results['nterm_scan_range'][1]}\n")
        f.write(f"  Protein 2 cut range: {results['cterm_scan_range'][0]}-{results['cterm_scan_range'][1]}\n")
        f.write(f"  Total constructs: {results['num_constructs']}\n\n")

        f.write("="*70 + "\n")
        f.write("Top 10 Constructs by Combined Fragment Similarity\n")
        f.write("="*70 + "\n\n")

        # Sort by average of both fragment similarities
        sorted_fusions = sorted(results['fusion_analyses'],
                               key=lambda x: (x['p1_fragment_similarity'] +
                                            x['p2_fragment_similarity']) / 2,
                               reverse=True)

        for i, fusion in enumerate(sorted_fusions[:10], 1):
            f.write(f"{i}. {fusion['fusion_id']}\n")
            f.write(f"   P1 cut: {fusion['p1_cut']}, P2 cut: {fusion['p2_cut']}\n")
            f.write(f"   P1 fragment similarity (mean-pooled): {fusion['p1_fragment_similarity']:.4f}\n")
            f.write(f"   P2 fragment similarity (mean-pooled): {fusion['p2_fragment_similarity']:.4f}\n")
            f.write(f"   P1 SWE similarity (distributional): {fusion['p1_swe_cosine_similarity']:.4f}\n")
            f.write(f"   P2 SWE similarity (distributional): {fusion['p2_swe_cosine_similarity']:.4f}\n")
            f.write(f"   Fusion→P1 similarity: {fusion['fusion_to_p1_similarity']:.4f}\n")
            f.write(f"   Fusion→P2 similarity: {fusion['fusion_to_p2_similarity']:.4f}\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Data Contents:\n")
        f.write("="*70 + "\n")
        f.write("- original_sequence_embeddings: Full sequence embeddings of original proteins\n")
        f.write("- original_aa_embeddings: Per-residue embeddings of original proteins\n")
        f.write("- fusion_analyses: List of results for each fusion construct\n")
        f.write("  Each contains:\n")
        f.write("    - Fragment similarity metrics (mean-pooled and SWE)\n")
        f.write("    - Full construct similarities to both parent proteins\n")
        f.write("    - SWE embeddings for further analysis\n\n")

        f.write("Summary CSV columns:\n")
        f.write("- p1_cut, p2_cut: Junction positions\n")
        f.write("- p1_fragment_similarity: Cosine similarity of mean-pooled P1 fragment (original vs fusion)\n")
        f.write("- p2_fragment_similarity: Cosine similarity of mean-pooled P2 fragment (original vs fusion)\n")
        f.write("- p1_swe_similarity: SWE-based cosine similarity of P1 (captures distributional shape)\n")
        f.write("- p2_swe_similarity: SWE-based cosine similarity of P2 (captures distributional shape)\n")
        f.write("- p1_swe_distance: SWE-based Euclidean distance of P1\n")
        f.write("- p2_swe_distance: SWE-based Euclidean distance of P2\n")
        f.write("- fusion_to_p1_similarity: Cosine similarity of full fusion to original P1\n")
        f.write("- fusion_to_p2_similarity: Cosine similarity of full fusion to original P2\n")

    print(f"Description saved to: {desc_file}")


def main():
    args = get_args()

    print("=" * 70)
    print("Domain Swap Embedding Analysis for Sensor Kinase Engineering")
    print("=" * 70)

    # Load sequences
    print("\nLoading protein sequences...")
    id1, seq1 = load_sequence(args.protein1)
    id2, seq2 = load_sequence(args.protein2)

    print(f"\nProtein 1 ({id1}): {len(seq1)} residues")
    print(f"  Aligned region: {args.nterm_overlap[0]}-{args.nterm_overlap[1]}")
    print(f"  Scan range: {args.nterm_scan[0]}-{args.nterm_scan[1]}")

    print(f"\nProtein 2 ({id2}): {len(seq2)} residues")
    print(f"  Aligned region: {args.cterm_overlap[0]}-{args.cterm_overlap[1]}")
    print(f"  Scan range: {args.cterm_scan[0]}-{args.cterm_scan[1]}")

    # Create fusion grid
    print(f"\nCreating fusion grid (step size: {args.step_size})...")
    fusion_constructs = create_fusion_grid(
        seq1, seq2,
        tuple(args.nterm_scan),
        tuple(args.cterm_scan),
        args.step_size,
        id1, id2
    )

    nterm_range = len(range(args.nterm_scan[0], args.nterm_scan[1] + 1, args.step_size))
    cterm_range = len(range(args.cterm_scan[0], args.cterm_scan[1] + 1, args.step_size))

    print(f"Grid size: {nterm_range} × {cterm_range} = {len(fusion_constructs)} fusion constructs")
    print(f"Length range: {min(len(f[1]) for f in fusion_constructs)}-{max(len(f[1]) for f in fusion_constructs)} residues")

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
    print(f"  Max sequence length: {config_attrs['max_sequence_length']}")

    # Check sequence lengths
    max_fusion_len = max(len(f[1]) for f in fusion_constructs)
    if max_fusion_len > config_attrs['max_sequence_length']:
        print(f"\nWARNING: Longest fusion ({max_fusion_len}) exceeds model max ({config_attrs['max_sequence_length']})")
        print("Sequences will be truncated!")

    # Run analysis
    results = analyze_domain_swap_grid(
        model, tokenizer, config_attrs,
        seq1, seq2, id1, id2,
        fusion_constructs, args
    )

    # Save results
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)
    save_results(results, args.output_prefix)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Load the pickle file to access full per-residue data")
    print("2. Use the CSV file for quick overview and plotting")
    print("3. Create heatmaps of fragment similarities across the grid")
    print("4. Identify optimal junction points for experimental testing")


if __name__ == "__main__":
    main()
