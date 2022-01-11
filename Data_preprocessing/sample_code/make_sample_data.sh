#!/bin/bash
  
qc_subject=()
while IFS=", " read -r num subjectkey
do
        qc_subject+=("$subjectkey")
done < qc_common_list_1225.csv

work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing/sample_code'

T1_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T1_brain_npy'
T2_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_masked_npy'
dwi_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_masked_npy'
grad_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_gradient'

data_dir='/scratch/connectome/GANBERT/data/sample'

count=0
for sub in ${qc_subject[@]}
do
        sub=$(echo $sub | tr -d "\"")
        sub=$(echo $sub | tr -d "_")
        sub="sub-$sub"

        T1="$T1_dir/$sub.brain.npy"
        T2="$T2_dir/$sub.brain.npy"
        dwi="$dwi_dir/$sub.brain.npy"
        grad="$grad_dir/$sub.grad.b"

        cp $T1 "$data_dir/$sub.T1.npy"
        cp $T2 "$data_dir/$sub.T2.npy"
        cp $dwi "$data_dir/$sub.dwi.npy"
        cp $grad "$data_dir/$sub.grad.npy"

        count=$((count + 1))
        echo $count
done
