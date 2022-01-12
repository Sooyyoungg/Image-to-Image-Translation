#!/bin/bash
  
# check whether subjects in demo.total.csv have both T1 and T2 images
qc_subject=()
while IFS=", " read -r subjectkey else
do
        qc_subject+=("$subjectkey")
done < demo.total.csv

work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'
data_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri'

for sub in ${qc_subject[@]}
do
        sub=$(echo $sub | tr -d "\"")
        sub=$(echo $sub | tr -d "_")
        sub="sub-$sub"

        T1="$data_dir/$sub.T1.mgz"
        T2="$data_dir/$sub.T2.mgz"

        if [ -f "$T1" ] && [ -f "$T2" ]
        then
                echo $sub
                echo $sub >> "$work_dir/structure_qc_list.csv"
        fi
