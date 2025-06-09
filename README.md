# Enterobacter Pangenome Creation 

The Data and Notebooks contained in this repository are available for the creation of a Pangenome of *Enterobacter spp.*

The notebooks outline a workflow necessary to create the pangenome and involve the following tools:
1. [MASH](https://mash.readthedocs.io/en/latest/)
2. [Bakta](https://github.com/oschwengers/bakta)
3. [CD-HIT](https://www.bioinformatics.org/cd-hit/cd-hit-user-guide)
4. [eggNOG](https://github.com/eggnogdb/eggnog-mapper.git)
5. [MAFFT](https://mafft.cbrc.jp/alignment/software/linuxportable.html)
6. [Blast](https://www.ncbi.nlm.nih.gov/books/NBK279690/)
7. [BGCFlow](https://github.com/NBChub/bgcflow)

These tools were used for analysis of sequences and some of the outputs of these tools were included. Fasta files and Bakta annotated Gff files are not included due to their large size, however these can be downloaded using the notebooks and then annotated with Bakta by the user. 

The notebooks are meant to be gone through in order, however analysis of the pangenome initailly created can be performed by starting from section __5__, as upstream results necessary for analysis (excluding the annotated Bakta files), are available at [this link](https://zenodo.org/records/15595724?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4MmM4ZGU2LWQ0NzAtNGUwMi05MDkyLThjOWNmNzg0NDI5MiIsImRhdGEiOnt9LCJyYW5kb20iOiI1ZmNmYzIxNDFlNjlmOTM0NjE0YmU5MTdiMWZjN2IxZSJ9.XsjNiwZwxgBpbM0sE67QcUd0Vxy1bdH4UeBDpQ2vMoPgyRHbe8MD2kITZ2gWDV881wZMX921AN7w4iF2hPDhWw). 

Installation of the python packages necessary for running the notebooks can be accomplished by using the included environment.yml file. 
