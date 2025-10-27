#!/usr/bin/env python3
import os
import subprocess

INPUT_DIRECTORY = '/mnt/craig/pan_phylon/Enterobacter/zenodo_data_mSpectrum_revision/data/raw/mash_genomes'
OUTPUT_DIRECTORY = '/mnt/craig/pan_phylon/Enterobacter/zenodo_data_mSpectrum_revision/data/processed/mlst'


def annotate_genome(fna_file):
    print(f"Annotating: {fna_file}")
    input_path = os.path.join(INPUT_DIRECTORY, fna_file)
    output_path = os.path.join(OUTPUT_DIRECTORY, fna_file.rsplit('.', 1)[0] + '.tsv')

    # Run the mlst command and capture its output
    result = subprocess.run(
        ['mlst', input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Write the stdout (output) to file
    with open(output_path, 'w') as f:
        f.write(result.stdout)

    # Optional: handle errors or log them
    if result.stderr:
        print(f"Warning: Error while processing {fna_file}:\n{result.stderr}")


def annotate_genomes():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Get the list of all .fna files in the input directory
    fna_files = [f for f in os.listdir(INPUT_DIRECTORY)]
    
    for genome in fna_files:
        annotate_genome(genome)
                        
    print(f"All {len(fna_files)} genomes have been annotated.")


if __name__ == "__main__":
    annotate_genomes()
