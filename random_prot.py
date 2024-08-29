import argparse

import random

import os

from Bio import SeqIO

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def generate_random_protein(n):

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    protein_sequence = ''.join(random.choice(amino_acids) for _ in range(n))
    return protein_sequence



def read_fasta_sequence(file_path):
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            return record



def main():

    parser = argparse.ArgumentParser(description='Generate a random protein sequence based on the length of the first sequence in a FASTA file.')

    parser.add_argument('input_fasta', help='Path to the input FASTA file')

    args = parser.parse_args()



    fasta_record = read_fasta_sequence(args.input_fasta)

    sequence_length = len(fasta_record.seq)



    random_protein_seq = generate_random_protein(sequence_length)

    print("HERE", random_protein_seq)
    random_protein_record = SeqRecord(Seq(random_protein_seq), id='Random_Protein', description='')



    output_records = [random_protein_record, fasta_record]



    input_dir, input_filename = os.path.split(args.input_fasta)

    input_name, input_ext = os.path.splitext(input_filename)

    output_filename = f'randomprot_{input_name}.fasta'

    output_path = os.path.join(input_dir, output_filename)



    with open(output_path, 'w') as output_file:

        SeqIO.write(output_records, output_file, 'fasta')



    print(f'Random protein and original protein saved to: {output_path}')



if __name__ == '__main__':

    main()
