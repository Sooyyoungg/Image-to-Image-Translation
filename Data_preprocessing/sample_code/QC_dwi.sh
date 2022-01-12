#!/bin/bash
  
# check whether subjects in fa.qc.csv have dwi
qc_subject=()
while IFS=", " read -r num subjectkey else
do
        qc_subject+=("$subjectkey")
done < fa.qc.csv

work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'
data_dir='/scratch/connectome/GANBERT/data/diffusion/DWI'

for sub in ${qc_subject[@]}
do
        sub=$(echo $sub | tr -d "\"")
        sub=$(echo $sub | tr -d "_")
        sub="sub-$sub"

        dwi="$data_dir/$sub/mr_dwi_denoised_gibbs_preproc_biasCorr_upsample125.mif.gz"

        if [ -f "$dwi" ]
        then
                echo $sub
                echo $sub >> "$work_dir/dwi_qc_list.csv"
        fi
done
