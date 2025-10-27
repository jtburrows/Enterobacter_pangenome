#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from multiprocessing import Pool

INPUT_DIRECTORY = '/mnt/craig/pan_phylon/Enterobacter/zenodo_data_mSpectrum_revision/data/raw/mash_genomes'
OUTPUT_DIRECTORY = '/mnt/craig/pan_phylon/Enterobacter/zenodo_data_mSpectrum_revision/data/processed/bakta'
PATH_TO_BAKTA_DB = '/mnt/craig/DATABASES/bakta/db'
NUM_THREADS = 4

mash_scrubbed_metadata = pd.read_csv(
    '/mnt/craig/pan_phylon/Enterobacter/zenodo_data_mSpectrum_revision/data/metadata/scrubbed_species_summary.csv',
    index_col=0, dtype='object'
)


def annotate_genome(fna_file):
    print(fna_file)
    input_path = os.path.join(INPUT_DIRECTORY, fna_file)
    output_path = os.path.join(OUTPUT_DIRECTORY, fna_file.rsplit('.', 1)[0])

    # Construct the bakta command with the --cpus option
    cmd = [
        'bakta',
        '--db', PATH_TO_BAKTA_DB,
        '--output', output_path,
        '--threads', str(NUM_THREADS),
        input_path,
        '--force',
        '--skip-plot',
    ]

    # Run the command
    subprocess.run(cmd)


def annotate_genomes(NUM_CPUS):
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Get the list of all .fna files in the input directory
    fna_files = [f for f in os.listdir(INPUT_DIRECTORY)]
    # Make sure to filter for only genomes which passed Mash filtration
    filtered_fna_files = set([f.split('.fna')[0] for f in fna_files])
    filtered_fna_files = sorted(
        filtered_fna_files.intersection(mash_scrubbed_metadata.genome_id.astype('str').values)
    )
    print(mash_scrubbed_metadata.genome_id.astype('str').values)
    filtered_fna_files = [f'{f}.fna' for f in filtered_fna_files]
    print(f'There are {len(filtered_fna_files)} files to process.')
    # Use multiprocessing Pool to run the annotation in parallel
    with Pool(processes=NUM_CPUS) as pool:
        pool.map(annotate_genome, filtered_fna_files)

    print(f"All {len(fna_files)} genomes have been annotated.")


if __name__ == "__main__":
    annotate_genomes(16)
