Run optNMF.py using the following command in order to select the optimal rank by observing the outputs: python optNMF.py --csv ../../data/processed/CAR_genomes/df_acc_panaroo.csv --index-col 0 --seeds 0,1,42 --ranks 4:100:1 --beta-loss frobenius --max-iter 7500 --x-binarize-L kmeans --x-binarize-A kmeans --corr-threshold 0.7 --make-sankey --outdir ../../data/processed/nmf-outputs/opt_nmf_test --name ebacter --n-jobs 40 --threads-per-job 1

Then run notebook 4b using the selected optimal rank.
