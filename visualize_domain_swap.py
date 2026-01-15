#!/usr/bin/env python3
"""
Visualization script for domain swap embedding analysis results.

Creates heatmaps and plots to analyze the 2D grid of fusion constructs and
identify optimal junction points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import pickle
from Bio import SeqIO

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize domain swap embedding analysis results"
    )
    parser.add_argument("-i", "--input", dest="input_csv", type=str, required=True,
                        help="Input CSV file from domain_swap_embed_analysis.py")
    parser.add_argument("-p", "--pickle", dest="input_pkl", type=str, required=False,
                        help="Optional: Input pickle file for additional analyses")
    parser.add_argument("-o", "--output", dest="output_dir", type=str, required=True,
                        help="Output directory for PNG files")
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
        aa = sequence[position - 1]  # Convert to 0-indexed
        return f"{position}\n{aa}"
    return str(position)


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


def create_pivot_table(df, value_column):
    """
    Create pivot table for heatmap from dataframe.

    Args:
        df: DataFrame with p1_cut, p2_cut, and value columns
        value_column: Column name to use for heatmap values

    Returns:
        Pivot table with p1_cut as rows, p2_cut as columns
    """
    pivot = df.pivot(index='p1_cut', columns='p2_cut', values=value_column)
    return pivot


def plot_heatmap(data, title, output_file, cmap='viridis', vmin=None, vmax=None,
                 xlabel='Protein 2 cut position', ylabel='Protein 1 cut position',
                 cbar_label='Similarity', p1_seq=None, p2_seq=None):
    """
    Create and save a heatmap.

    Args:
        data: 2D array or pivot table
        title: Plot title
        output_file: Output filename
        cmap: Colormap name
        vmin, vmax: Color scale limits
        xlabel, ylabel: Axis labels
        cbar_label: Colorbar label
        p1_seq: Protein 1 sequence (for amino acid labels)
        p2_seq: Protein 2 sequence (for amino acid labels)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label},
                linewidths=0.5, linecolor='gray',
                square=False, ax=ax)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add amino acid labels if sequences provided
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
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_combined_score_heatmap(df, output_file, p1_seq=None, p2_seq=None):
    """
    Plot heatmap of combined fragment similarity score (per-residue cosine).

    Combined score = mean of P1 and P2 fragment similarities.
    """
    # Calculate combined score
    df['combined_fragment_similarity'] = (
        df['p1_fragment_similarity'] + df['p2_fragment_similarity']
    ) / 2

    pivot = create_pivot_table(df, 'combined_fragment_similarity')

    plot_heatmap(
        pivot,
        title='Combined Fragment Similarity - Per-residue Cosine (P1 + P2)',
        output_file=output_file,
        cmap='RdYlGn',
        cbar_label='Mean Cosine Similarity',
        p1_seq=p1_seq,
        p2_seq=p2_seq
    )


def plot_combined_swe_score_heatmap(df, output_file, p1_seq=None, p2_seq=None):
    """
    Plot heatmap of combined SWE similarity score.

    Combined score = mean of P1 and P2 SWE similarities.
    """
    # Calculate combined SWE score
    df['combined_swe_similarity'] = (
        df['p1_swe_similarity'] + df['p2_swe_similarity']
    ) / 2

    pivot = create_pivot_table(df, 'combined_swe_similarity')

    plot_heatmap(
        pivot,
        title='Combined Fragment Similarity - SWE (P1 + P2)',
        output_file=output_file,
        cmap='RdYlGn',
        cbar_label='SWE Cosine Similarity',
        p1_seq=p1_seq,
        p2_seq=p2_seq
    )


def plot_dual_swe_heatmap(df, output_file, p1_seq=None, p2_seq=None):
    """
    Plot side-by-side heatmaps for P1 and P2 SWE similarities.
    """
    pivot_p1 = create_pivot_table(df, 'p1_swe_similarity')
    pivot_p2 = create_pivot_table(df, 'p2_swe_similarity')

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # P1 SWE similarity
    sns.heatmap(pivot_p1, cmap='RdYlGn',
                cbar_kws={'label': 'SWE Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[0])
    axes[0].set_title('Protein 1 SWE Similarity\n(Distributional shape: N-term in fusion vs original)',
                      fontsize=12, fontweight='bold', pad=15)
    axes[0].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    add_aa_labels_to_axis(axes[0], p1_seq, p2_seq)

    # P2 SWE similarity
    sns.heatmap(pivot_p2, cmap='RdYlGn',
                cbar_kws={'label': 'SWE Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[1])
    axes[1].set_title('Protein 2 SWE Similarity\n(Distributional shape: C-term in fusion vs original)',
                      fontsize=12, fontweight='bold', pad=15)
    axes[1].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    add_aa_labels_to_axis(axes[1], p1_seq, p2_seq)

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_dual_heatmap(df, output_file, p1_seq=None, p2_seq=None):
    """
    Plot side-by-side heatmaps for P1 and P2 fragment similarities (per-residue).
    """
    pivot_p1 = create_pivot_table(df, 'p1_fragment_similarity')
    pivot_p2 = create_pivot_table(df, 'p2_fragment_similarity')

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # P1 fragment similarity
    sns.heatmap(pivot_p1, cmap='RdYlGn',
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[0])
    axes[0].set_title('Protein 1 Fragment Similarity\n(N-term in fusion vs original)',
                      fontsize=12, fontweight='bold', pad=15)
    axes[0].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    add_aa_labels_to_axis(axes[0], p1_seq, p2_seq)

    # P2 fragment similarity
    sns.heatmap(pivot_p2, cmap='RdYlGn',
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[1])
    axes[1].set_title('Protein 2 Fragment Similarity\n(C-term in fusion vs original)',
                      fontsize=12, fontweight='bold', pad=15)
    axes[1].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    add_aa_labels_to_axis(axes[1], p1_seq, p2_seq)

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_full_construct_heatmaps(df, output_file):
    """
    Plot side-by-side heatmaps for fusion→P1 and fusion→P2 similarities.
    """
    pivot_fus_p1 = create_pivot_table(df, 'fusion_to_p1_similarity')
    pivot_fus_p2 = create_pivot_table(df, 'fusion_to_p2_similarity')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Fusion to P1
    sns.heatmap(pivot_fus_p1, cmap='viridis',
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[0])
    axes[0].set_title('Full Fusion Similarity to Original Protein 1',
                      fontsize=12, fontweight='bold', pad=15)
    axes[0].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Fusion to P2
    sns.heatmap(pivot_fus_p2, cmap='viridis',
                cbar_kws={'label': 'Cosine Similarity'},
                linewidths=0.5, linecolor='gray',
                ax=axes[1])
    axes[1].set_title('Full Fusion Similarity to Original Protein 2',
                      fontsize=12, fontweight='bold', pad=15)
    axes[1].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_scatter_p1_vs_p2(df, output_file, use_swe=False):
    """
    Scatter plot of P1 fragment similarity vs P2 fragment similarity.
    Color by combined score.

    Args:
        use_swe: If True, use SWE similarities; if False, use per-residue similarities
    """
    if use_swe:
        p1_col = 'p1_swe_similarity'
        p2_col = 'p2_swe_similarity'
        title_suffix = '(SWE - Distributional)'
    else:
        p1_col = 'p1_fragment_similarity'
        p2_col = 'p2_fragment_similarity'
        title_suffix = '(Per-residue Cosine)'

    df['combined_similarity'] = (df[p1_col] + df[p2_col]) / 2

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df[p1_col],
        df[p2_col],
        c=df['combined_similarity'],
        cmap='RdYlGn',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Get min/max for diagonal line
    vmin = min(df[p1_col].min(), df[p2_col].min())
    vmax = max(df[p1_col].max(), df[p2_col].max())
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.3, linewidth=1, label='Equal similarity')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Combined Similarity', fontsize=11, fontweight='bold')

    ax.set_xlabel(f'Protein 1 Fragment Similarity {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Protein 2 Fragment Similarity {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_title(f'Fragment Similarity Trade-off Analysis {title_suffix}', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_top_constructs_bar(df, output_file, n_top=20, use_swe=False):
    """
    Bar plot showing top N constructs by combined fragment similarity.

    Args:
        use_swe: If True, use SWE similarities; if False, use per-residue similarities
    """
    if use_swe:
        p1_col = 'p1_swe_similarity'
        p2_col = 'p2_swe_similarity'
        title_suffix = '(SWE - Distributional)'
    else:
        p1_col = 'p1_fragment_similarity'
        p2_col = 'p2_fragment_similarity'
        title_suffix = '(Per-residue Cosine)'

    df['combined_similarity'] = (df[p1_col] + df[p2_col]) / 2

    # Sort and get top N
    top_df = df.nlargest(n_top, 'combined_similarity').copy()
    top_df['label'] = top_df.apply(
        lambda row: f"P1:{row['p1_cut']}\nP2:{row['p2_cut']}", axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(top_df))
    bar_width = 0.35

    # Transform to dissimilarity (1 - similarity) for better log-scale visualization
    # This makes small differences in high similarity values (0.99x) more visible
    p1_dissim = 1 - top_df[p1_col]
    p2_dissim = 1 - top_df[p2_col]
    combined_dissim = 1 - top_df['combined_similarity']

    bars1 = ax.bar(x_pos - bar_width/2, p1_dissim,
                   bar_width, label='P1 fragment', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + bar_width/2, p2_dissim,
                   bar_width, label='P2 fragment', color='darkorange', alpha=0.8)

    # Add combined dissimilarity as scatter points
    ax.scatter(x_pos, combined_dissim,
               color='red', s=100, marker='D', label='Combined', zorder=5,
               edgecolors='black', linewidth=1)

    ax.set_xlabel('Fusion Construct (P1 cut : P2 cut)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dissimilarity (1 - Similarity, log scale)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_top} Fusion Constructs by Combined Fragment Similarity {title_suffix}\n(Lower is better)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_df['label'], rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, which='both')

    # Use log scale for y-axis to better visualize differences in high-similarity values
    ax.set_yscale('log')

    # Invert y-axis so lower dissimilarity (better) is at the top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_line_profiles(df, output_file):
    """
    Line plots showing how similarity changes along P1 cuts (for selected P2 cuts)
    and along P2 cuts (for selected P1 cuts).
    """
    # Get unique P1 and P2 cut positions
    p1_cuts = sorted(df['p1_cut'].unique())
    p2_cuts = sorted(df['p2_cut'].unique())

    # Select representative P2 cuts (quartiles)
    p2_selected = [p2_cuts[0], p2_cuts[len(p2_cuts)//4],
                   p2_cuts[len(p2_cuts)//2], p2_cuts[3*len(p2_cuts)//4],
                   p2_cuts[-1]]

    # Select representative P1 cuts (quartiles)
    p1_selected = [p1_cuts[0], p1_cuts[len(p1_cuts)//4],
                   p1_cuts[len(p1_cuts)//2], p1_cuts[3*len(p1_cuts)//4],
                   p1_cuts[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: P1 fragment similarity vs P1 cut (for selected P2 cuts)
    for p2 in p2_selected:
        subset = df[df['p2_cut'] == p2].sort_values('p1_cut')
        axes[0, 0].plot(subset['p1_cut'], subset['p1_fragment_similarity'],
                       marker='o', label=f'P2={p2}', linewidth=2)
    axes[0, 0].set_xlabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('P1 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('P1 Fragment Similarity vs P1 Cut Position', fontsize=12, fontweight='bold')
    axes[0, 0].legend(title='P2 cut', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # Panel 2: P2 fragment similarity vs P1 cut (for selected P2 cuts)
    for p2 in p2_selected:
        subset = df[df['p2_cut'] == p2].sort_values('p1_cut')
        axes[0, 1].plot(subset['p1_cut'], subset['p2_fragment_similarity'],
                       marker='o', label=f'P2={p2}', linewidth=2)
    axes[0, 1].set_xlabel('Protein 1 cut position', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('P2 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('P2 Fragment Similarity vs P1 Cut Position', fontsize=12, fontweight='bold')
    axes[0, 1].legend(title='P2 cut', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Panel 3: P1 fragment similarity vs P2 cut (for selected P1 cuts)
    for p1 in p1_selected:
        subset = df[df['p1_cut'] == p1].sort_values('p2_cut')
        axes[1, 0].plot(subset['p2_cut'], subset['p1_fragment_similarity'],
                       marker='o', label=f'P1={p1}', linewidth=2)
    axes[1, 0].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('P1 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('P1 Fragment Similarity vs P2 Cut Position', fontsize=12, fontweight='bold')
    axes[1, 0].legend(title='P1 cut', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Panel 4: P2 fragment similarity vs P2 cut (for selected P1 cuts)
    for p1 in p1_selected:
        subset = df[df['p1_cut'] == p1].sort_values('p2_cut')
        axes[1, 1].plot(subset['p2_cut'], subset['p2_fragment_similarity'],
                       marker='o', label=f'P1={p1}', linewidth=2)
    axes[1, 1].set_xlabel('Protein 2 cut position', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('P2 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('P2 Fragment Similarity vs P2 Cut Position', fontsize=12, fontweight='bold')
    axes[1, 1].legend(title='P1 cut', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_fusion_length_analysis(df, output_file):
    """
    Analyze how fusion construct length relates to similarity metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Combined similarity vs fusion length
    axes[0, 0].scatter(df['fusion_length'],
                      (df['p1_fragment_similarity'] + df['p2_fragment_similarity']) / 2,
                      alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Fusion Length (residues)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Combined Fragment Similarity', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Combined Similarity vs Fusion Length', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: P1 fragment similarity vs fusion length
    axes[0, 1].scatter(df['fusion_length'], df['p1_fragment_similarity'],
                      alpha=0.6, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Fusion Length (residues)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('P1 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('P1 Fragment Similarity vs Fusion Length', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: P2 fragment similarity vs fusion length
    axes[1, 0].scatter(df['fusion_length'], df['p2_fragment_similarity'],
                      alpha=0.6, s=30, c='darkorange', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Fusion Length (residues)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('P2 Fragment Similarity', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('P2 Fragment Similarity vs Fusion Length', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Distribution of fusion lengths
    axes[1, 1].hist(df['fusion_length'], bins=30, color='steelblue',
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Fusion Length (residues)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Count', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Distribution of Fusion Lengths', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def create_summary_statistics(df, output_file):
    """
    Create a text file with summary statistics.
    """
    df['combined_similarity'] = (
        df['p1_fragment_similarity'] + df['p2_fragment_similarity']
    ) / 2

    with open(output_file, 'w') as f:
        f.write("Domain Swap Embedding Analysis - Summary Statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total constructs analyzed: {len(df)}\n\n")

        f.write("Overall Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"P1 fragment similarity: {df['p1_fragment_similarity'].mean():.4f} ± {df['p1_fragment_similarity'].std():.4f}\n")
        f.write(f"  Range: {df['p1_fragment_similarity'].min():.4f} - {df['p1_fragment_similarity'].max():.4f}\n\n")

        f.write(f"P2 fragment similarity: {df['p2_fragment_similarity'].mean():.4f} ± {df['p2_fragment_similarity'].std():.4f}\n")
        f.write(f"  Range: {df['p2_fragment_similarity'].min():.4f} - {df['p2_fragment_similarity'].max():.4f}\n\n")

        f.write(f"Combined similarity: {df['combined_similarity'].mean():.4f} ± {df['combined_similarity'].std():.4f}\n")
        f.write(f"  Range: {df['combined_similarity'].min():.4f} - {df['combined_similarity'].max():.4f}\n\n")

        f.write(f"Fusion→P1 similarity: {df['fusion_to_p1_similarity'].mean():.4f} ± {df['fusion_to_p1_similarity'].std():.4f}\n")
        f.write(f"Fusion→P2 similarity: {df['fusion_to_p2_similarity'].mean():.4f} ± {df['fusion_to_p2_similarity'].std():.4f}\n\n")

        # Top constructs
        f.write("\n" + "=" * 70 + "\n")
        f.write("Top 10 Constructs by Combined Fragment Similarity:\n")
        f.write("=" * 70 + "\n\n")

        top_10 = df.nlargest(10, 'combined_similarity')
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            f.write(f"{i:2d}. P1 cut: {row['p1_cut']:4d} | P2 cut: {row['p2_cut']:4d} | "
                   f"Combined: {row['combined_similarity']:.4f} | "
                   f"P1: {row['p1_fragment_similarity']:.4f} | "
                   f"P2: {row['p2_fragment_similarity']:.4f}\n")

        # Best balanced constructs (P1 and P2 both high)
        f.write("\n" + "=" * 70 + "\n")
        f.write("Top 10 Most Balanced Constructs (minimum of P1 and P2):\n")
        f.write("=" * 70 + "\n\n")

        df['min_similarity'] = df[['p1_fragment_similarity', 'p2_fragment_similarity']].min(axis=1)
        top_balanced = df.nlargest(10, 'min_similarity')

        for i, (idx, row) in enumerate(top_balanced.iterrows(), 1):
            f.write(f"{i:2d}. P1 cut: {row['p1_cut']:4d} | P2 cut: {row['p2_cut']:4d} | "
                   f"Min: {row['min_similarity']:.4f} | "
                   f"P1: {row['p1_fragment_similarity']:.4f} | "
                   f"P2: {row['p2_fragment_similarity']:.4f}\n")

    print(f"Saved: {output_file}")


def main():
    args = get_args()

    # Set DPI
    plt.rcParams['figure.dpi'] = args.dpi

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Domain Swap Embedding Analysis - Visualization")
    print("=" * 70)
    print(f"\nInput CSV: {args.input_csv}")
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

    print("Generating visualizations...\n")

    # 1. Combined fragment similarity heatmap (per-residue cosine)
    print("1. Creating combined fragment similarity heatmap (per-residue)...")
    plot_combined_score_heatmap(df, output_dir / "01_combined_similarity_heatmap_cosine.png", p1_seq, p2_seq)

    # 1b. Combined SWE similarity heatmap (distributional shape)
    print("1b. Creating combined SWE similarity heatmap (distributional)...")
    plot_combined_swe_score_heatmap(df, output_dir / "01b_combined_similarity_heatmap_swe.png", p1_seq, p2_seq)

    # 2. Dual heatmaps (P1 and P2 side by side) - per-residue
    print("2. Creating dual fragment similarity heatmaps (per-residue)...")
    plot_dual_heatmap(df, output_dir / "02_fragment_similarity_dual_heatmap.png", p1_seq, p2_seq)

    # 2b. Dual heatmaps for SWE similarities
    print("2b. Creating dual SWE similarity heatmaps (distributional)...")
    plot_dual_swe_heatmap(df, output_dir / "02b_swe_similarity_dual_heatmap.png", p1_seq, p2_seq)

    # 3. Full construct similarity heatmaps
    print("3. Creating full construct similarity heatmaps...")
    plot_full_construct_heatmaps(df, output_dir / "03_full_construct_similarity_heatmaps.png")

    # 4. Scatter plot - per-residue
    print("4. Creating P1 vs P2 scatter plot (per-residue)...")
    plot_scatter_p1_vs_p2(df, output_dir / "04_p1_vs_p2_scatter_cosine.png", use_swe=False)

    # 4b. Scatter plot - SWE
    print("4b. Creating P1 vs P2 scatter plot (SWE)...")
    plot_scatter_p1_vs_p2(df, output_dir / "04b_p1_vs_p2_scatter_swe.png", use_swe=True)

    # 5. Top constructs bar plot - per-residue
    print("5. Creating top constructs bar plot (per-residue)...")
    plot_top_constructs_bar(df, output_dir / "05_top_constructs_bar_cosine.png", n_top=20, use_swe=False)

    # 5b. Top constructs bar plot - SWE
    print("5b. Creating top constructs bar plot (SWE)...")
    plot_top_constructs_bar(df, output_dir / "05b_top_constructs_bar_swe.png", n_top=20, use_swe=True)

    # 6. Line profiles
    print("6. Creating line profile plots...")
    plot_line_profiles(df, output_dir / "06_line_profiles.png")

    # 7. Fusion length analysis
    print("7. Creating fusion length analysis...")
    plot_fusion_length_analysis(df, output_dir / "07_fusion_length_analysis.png")

    # 8. Summary statistics
    print("8. Creating summary statistics...")
    create_summary_statistics(df, output_dir / "08_summary_statistics.txt")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  01_combined_similarity_heatmap.png - Overall best junction points")
    print("  02_fragment_similarity_dual_heatmap.png - P1 and P2 per-residue comparisons")
    print("  02b_swe_similarity_dual_heatmap.png - P1 and P2 distributional shape (SWE)")
    print("  03_full_construct_similarity_heatmaps.png - Full fusion to parent similarities")
    print("  04_p1_vs_p2_scatter.png - Trade-off between P1 and P2 preservation")
    print("  05_top_constructs_bar.png - Top 20 constructs ranked")
    print("  06_line_profiles.png - How similarity varies along cut positions")
    print("  07_fusion_length_analysis.png - Length effects on similarity")
    print("  08_summary_statistics.txt - Numerical summary and rankings")


if __name__ == "__main__":
    main()
