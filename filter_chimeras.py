import argparse
from Bio import SeqIO


def filter_chimeras(fasta_path, output_path, p1_min, p1_max, p2_min, p2_max):
    kept = 0
    total = 0
    with open(output_path, 'w') as out:
        for record in SeqIO.parse(fasta_path, "fasta"):
            total += 1
            desc = record.description
            p1_cut = None
            p2_cut = None
            for field in desc.split():
                if field.startswith("p1_cut="):
                    p1_cut = int(field.split("=")[1])
                elif field.startswith("p2_cut="):
                    p2_cut = int(field.split("=")[1])

            if p1_cut is None or p2_cut is None:
                continue

            if p1_min <= p1_cut <= p1_max and p2_min <= p2_cut <= p2_max:
                SeqIO.write(record, out, "fasta")
                kept += 1

    print(f"Kept {kept}/{total} sequences (p1_cut {p1_min}-{p1_max}, p2_cut {p2_min}-{p2_max})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter chimera FASTA by p1_cut and p2_cut ranges')
    parser.add_argument('fasta_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--p1-min', type=int, required=True)
    parser.add_argument('--p1-max', type=int, required=True)
    parser.add_argument('--p2-min', type=int, required=True)
    parser.add_argument('--p2-max', type=int, required=True)

    args = parser.parse_args()
    filter_chimeras(args.fasta_file, args.output_file, args.p1_min, args.p1_max, args.p2_min, args.p2_max)
