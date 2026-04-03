#!/usr/bin/env python3
"""
Visualization script for domain swap fixed region analysis results.

Creates heatmaps for each fixed comparison region, showing SWE similarity
across the grid of junction points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from Bio import SeqIO

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize domain swap fixed region analysis results"
    )
    parser.add_argument("-i", "--input", dest="input_csv", type=str, required=True,
                        help="Input CSV file from domain_swap_embed_analysis_fixedregions.py")
    parser.add_argument("-o", "--output", dest="output_dir", type=str, required=True,
                        help="Output directory for images")
    parser.add_argument("--protein1-fasta", dest="protein1_fasta", type=str, required=False,
                        help="Protein 1 FASTA file (for amino acid labels)")
    parser.add_argument("--protein2-fasta", dest="protein2_fasta", type=str, required=False,
                        help="Protein 2 FASTA file (for amino acid labels)")
    parser.add_argument("--dpi", dest="dpi", type=int, default=300,
                        help="DPI for output images (default: 300)")

    return parser.parse_args()


def load_sequence_from_fasta(fasta_file):
    """Load single sequence from FASTA file."""
    if not fasta_file:
        return None
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)
    return None


def format_axis_label_with_aa(position, sequence):
    """Format axis label with position and amino acid."""
    if sequence and 1 <= position <= len(sequence):
        aa = sequence[position - 1]
        return f"{position}-{aa}"
    return str(position)


def save_figure(output_file):
    """Save figure as both PNG and PDF."""
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    pdf_file = Path(output_file).with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    print(f"Saved: {pdf_file}")


def create_pivot_table(df, value_column):
    """Create pivot table for heatmap from dataframe."""
    pivot = df.pivot(index='p1_cut', columns='p2_cut', values=value_column)
    return pivot


def add_aa_labels_to_axis(ax, p1_seq=None, p2_seq=None):
    """Add amino acid labels to existing heatmap axes."""
    if p2_seq:
        x_labels = [format_axis_label_with_aa(int(label.get_text()), p2_seq)
                   for label in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if p1_seq:
        y_labels = [format_axis_label_with_aa(int(label.get_text()), p1_seq)
                   for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    else:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def detect_region_columns(df):
    """
    Detect region columns in the dataframe.

    Returns:
        dict with 'p1_regions' and 'p2_regions' lists of (region_name, start, end) tuples
    """
    p1_regions = []
    p2_regions = []

    for col in df.columns:
        if col.endswith('_swe_similarity'):
            # Extract region info from column name like "p1_region_1_400_swe_similarity"
            parts = col.replace('_swe_similarity', '').split('_')
            if len(parts) >= 4 and parts[0] in ['p1', 'p2'] and parts[1] == 'region':
                protein = parts[0]
                start = int(parts[2])
                end = int(parts[3])
                region_name = f"{protein}_region_{start}_{end}"

                if protein == 'p1':
                    p1_regions.append((region_name, start, end))
                else:
                    p2_regions.append((region_name, start, end))

    return {'p1_regions': p1_regions, 'p2_regions': p2_regions}


def plot_region_heatmap(df, region_name, start, end, protein, output_file,
                        p1_seq=None, p2_seq=None):
    """
    Plot heatmap for a single region's SWE similarity.
    """
    col_name = f"{region_name}_swe_similarity"

    if col_name not in df.columns:
        print(f"Warning: Column {col_name} not found in dataframe")
        return

    pivot = create_pivot_table(df, col_name)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(pivot, cmap='RdYlGn',
                cbar_kws={'label': 'SWE Cosine Similarity'},
                ax=ax)

    protein_label = "Protein 1 (N-term)" if protein == "p1" else "Protein 2 (C-term)"
    length = end - start + 1

    ax.set_title(f'{protein_label} Region {start}-{end}\n({length} residues, SWE similarity)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Protein 2 cut position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protein 1 cut position', fontsize=12, fontweight='bold')

    ax.invert_yaxis()

    # Add amino acid labels
    if p2_seq:
        x_labels = [format_axis_label_with_aa(int(label.get_text()), p2_seq)
                   for label in ax.get_xticklabels()]
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if p1_seq:
        y_labels = [format_axis_label_with_aa(int(label.get_text()), p1_seq)
                   for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    else:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    save_figure(output_file)


def plot_all_regions_summary(df, regions_dict, output_file, p1_seq=None, p2_seq=None):
    """
    Plot a summary figure with all regions as subplots.
    """
    p1_regions = regions_dict['p1_regions']
    p2_regions = regions_dict['p2_regions']
    all_regions = p1_regions + p2_regions

    n_regions = len(all_regions)
    if n_regions == 0:
        print("No regions to plot")
        return

    # Determine subplot layout
    if n_regions == 1:
        nrows, ncols = 1, 1
    elif n_regions == 2:
        nrows, ncols = 1, 2
    elif n_regions <= 4:
        nrows, ncols = 2, 2
    elif n_regions <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (n_regions + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 6*nrows))

    # Flatten axes for easier indexing
    if n_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (region_name, start, end) in enumerate(all_regions):
        ax = axes[i]
        col_name = f"{region_name}_swe_similarity"

        if col_name not in df.columns:
            ax.text(0.5, 0.5, f"No data for\n{region_name}",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        pivot = create_pivot_table(df, col_name)

        sns.heatmap(pivot, cmap='RdYlGn',
                    cbar_kws={'label': 'SWE Similarity'},
                    ax=ax)

        protein = region_name.split('_')[0]
        protein_label = "P1" if protein == "p1" else "P2"
        length = end - start + 1

        ax.set_title(f'{protein_label}: {start}-{end} ({length} res)',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('P2 cut', fontsize=10)
        ax.set_ylabel('P1 cut', fontsize=10)
        ax.invert_yaxis()

        # Simplify tick labels for summary view
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    # Hide unused subplots
    for i in range(n_regions, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('SWE Similarity by Fixed Region', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(output_file)


def plot_combined_regions_heatmap(df, regions_dict, output_file, p1_seq=None, p2_seq=None):
    """
    Plot heatmap of mean SWE similarity across all regions.
    """
    p1_regions = regions_dict['p1_regions']
    p2_regions = regions_dict['p2_regions']
    all_regions = p1_regions + p2_regions

    if len(all_regions) == 0:
        print("No regions to combine")
        return

    # Calculate mean across all region similarities
    sim_cols = [f"{r[0]}_swe_similarity" for r in all_regions if f"{r[0]}_swe_similarity" in df.columns]

    if len(sim_cols) == 0:
        print("No similarity columns found")
        return

    df['combined_region_swe'] = df[sim_cols].mean(axis=1)

    pivot = create_pivot_table(df, 'combined_region_swe')

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(pivot, cmap='RdYlGn',
                cbar_kws={'label': 'Mean SWE Cosine Similarity'},
                ax=ax)

    ax.set_title(f'Combined SWE Similarity (mean of {len(sim_cols)} regions)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Protein 2 cut position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Protein 1 cut position', fontsize=12, fontweight='bold')

    ax.invert_yaxis()
    add_aa_labels_to_axis(ax, p1_seq, p2_seq)

    plt.tight_layout()
    save_figure(output_file)


def create_summary_statistics(df, regions_dict, output_file):
    """Create a text file with summary statistics."""
    p1_regions = regions_dict['p1_regions']
    p2_regions = regions_dict['p2_regions']
    all_regions = p1_regions + p2_regions

    with open(output_file, 'w') as f:
        f.write("Domain Swap Fixed Region Analysis - Summary Statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total constructs analyzed: {len(df)}\n\n")

        f.write("Region Statistics:\n")
        f.write("-" * 70 + "\n\n")

        for region_name, start, end in all_regions:
            col_name = f"{region_name}_swe_similarity"
            if col_name not in df.columns:
                continue

            vals = df[col_name].dropna()
            length = end - start + 1

            f.write(f"{region_name} ({length} residues):\n")
            f.write(f"  SWE Similarity: {vals.mean():.4f} ± {vals.std():.4f}\n")
            f.write(f"  Range: {vals.min():.4f} - {vals.max():.4f}\n")
            f.write(f"  Constructs with region: {len(vals)} / {len(df)}\n\n")

        # Combined analysis
        sim_cols = [f"{r[0]}_swe_similarity" for r in all_regions if f"{r[0]}_swe_similarity" in df.columns]
        if len(sim_cols) > 1:
            df['combined_swe'] = df[sim_cols].mean(axis=1)

            f.write("\n" + "=" * 70 + "\n")
            f.write("Combined Analysis (mean of all regions)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Combined SWE: {df['combined_swe'].mean():.4f} ± {df['combined_swe'].std():.4f}\n")
            f.write(f"Range: {df['combined_swe'].min():.4f} - {df['combined_swe'].max():.4f}\n\n")

            # Top constructs
            f.write("Top 10 Constructs by Combined SWE Similarity:\n")
            f.write("-" * 70 + "\n\n")

            top_10 = df.nlargest(10, 'combined_swe')
            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"{i:2d}. P1 cut: {row['p1_cut']:4.0f} | P2 cut: {row['p2_cut']:4.0f} | ")
                f.write(f"Combined SWE: {row['combined_swe']:.4f}\n")

    print(f"Saved: {output_file}")


def main():
    args = get_args()

    # Set DPI
    plt.rcParams['figure.dpi'] = args.dpi

    # Extract basename from input CSV
    input_path = Path(args.input_csv)
    basename = input_path.name
    if basename.endswith('.summary.csv'):
        basename = basename[:-12]
    elif basename.endswith('.csv'):
        basename = basename[:-4]

    # Create output directory
    output_dir = Path(args.output_dir) / "images" / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Domain Swap Fixed Region Analysis - Visualization")
    print("=" * 70)
    print(f"\nInput CSV: {args.input_csv}")
    print(f"Basename: {basename}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}\n")

    # Load data
    print("Loading data...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} fusion constructs\n")

    # Load sequences for amino acid labels
    p1_seq = load_sequence_from_fasta(args.protein1_fasta)
    p2_seq = load_sequence_from_fasta(args.protein2_fasta)
    if p1_seq:
        print(f"Loaded Protein 1 sequence: {len(p1_seq)} aa")
    if p2_seq:
        print(f"Loaded Protein 2 sequence: {len(p2_seq)} aa")
    print()

    # Detect regions
    regions_dict = detect_region_columns(df)
    p1_regions = regions_dict['p1_regions']
    p2_regions = regions_dict['p2_regions']

    print(f"Detected {len(p1_regions)} P1 regions and {len(p2_regions)} P2 regions\n")

    if len(p1_regions) == 0 and len(p2_regions) == 0:
        print("ERROR: No region columns detected in CSV!")
        print("Expected columns like: p1_region_1_400_swe_similarity")
        return

    print("Generating visualizations...\n")

    # 1. Individual heatmaps for each region
    plot_num = 1
    for region_name, start, end in p1_regions:
        print(f"{plot_num}. Creating heatmap for {region_name}...")
        plot_region_heatmap(df, region_name, start, end, "p1",
                           output_dir / f"{basename}_{plot_num:02d}_{region_name}_heatmap.png",
                           p1_seq, p2_seq)
        plot_num += 1

    for region_name, start, end in p2_regions:
        print(f"{plot_num}. Creating heatmap for {region_name}...")
        plot_region_heatmap(df, region_name, start, end, "p2",
                           output_dir / f"{basename}_{plot_num:02d}_{region_name}_heatmap.png",
                           p1_seq, p2_seq)
        plot_num += 1

    # 2. Summary plot with all regions
    print(f"{plot_num}. Creating summary plot with all regions...")
    plot_all_regions_summary(df, regions_dict,
                            output_dir / f"{basename}_{plot_num:02d}_all_regions_summary.png",
                            p1_seq, p2_seq)
    plot_num += 1

    # 3. Combined regions heatmap
    if len(p1_regions) + len(p2_regions) > 1:
        print(f"{plot_num}. Creating combined regions heatmap...")
        plot_combined_regions_heatmap(df, regions_dict,
                                     output_dir / f"{basename}_{plot_num:02d}_combined_regions_heatmap.png",
                                     p1_seq, p2_seq)
        plot_num += 1

    # 4. Summary statistics
    print(f"{plot_num}. Creating summary statistics...")
    create_summary_statistics(df, regions_dict,
                             output_dir / f"{basename}_{plot_num:02d}_summary_statistics.txt")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nGenerated {plot_num} output files:")
    print("  - One heatmap per region (SWE similarity)")
    print("  - Summary plot with all regions")
    if len(p1_regions) + len(p2_regions) > 1:
        print("  - Combined regions heatmap (mean SWE)")
    print("  - Summary statistics text file")


if __name__ == "__main__":
    main()
