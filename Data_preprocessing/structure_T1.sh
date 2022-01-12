#!/bin/bash
# This code is used for changing T1's format: .nii.gz -> .npy
work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'

T1_prep_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri_brain'
T1s=$(find $T1_prep_dir -name "*brain.nii.gz")

T1_npy_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T1_brain_npy'

count=0
for T1 in $T1s
do
        sub_list=($(echo $T1 | tr "/" " "))
        subs=${sub_list[-1]}
        sub_dot=($(echo $subs | tr "." " "))
        sub=${sub_dot[0]}

        T1_npy="$T1_npy_dir/$sub.brain.npy"
        python "$work_dir/make_npy.py" --image_path $T1 --output_path $T1_npy

        count=$((count + 1))
        echo $count
done
