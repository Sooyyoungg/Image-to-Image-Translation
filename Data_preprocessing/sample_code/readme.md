## Skull stripping for single T2 image
- T2_bet.sh : using BET
- T2_mask.sh : using mrcalc

## Sampling for old T1 image > NOT USING
- structure_sample_old.sh

## Quality Control
- structure : QC_structure.sh > qc_structure_list_8922.csv
- diffusion : QC_dwi.sh > qc_dwi_list_1436.csv
- common : QC_common.py > qc_common_list_1225.csv

# Sampling for T1, T2, dwi, gradient
- make_sample_data.sh : copy 4 kinds of data into 1 directory (/scratch/connectome/GANBERT/data/sample)
