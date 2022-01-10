#!/bin/bash
work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'

# old directory
T2_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri'
T2s=$(find $T2_dir -name "*.T2.mgz")

# new directory
T2_nii_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_nii'
T2_masked_niigz_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_masked'

mask_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_mask'
T2_masked_npy_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_masked_npy'

count=0
for T2 in $T2s
do
        sub_list=($(echo $T2 | tr "/" " "))
        subs=${sub_list[-1]}
        sub_dot=($(echo $subs | tr "." " "))
        sub=${sub_dot[0]}

        # T2: .mgz -> .nii
        T2_nii="$T2_nii_dir/$sub.T2.nii"
        mri_convert $T2 $T2_nii

        # bet (skull-stripping & make mask file)
        T2_masked="$T2_masked_niigz_dir/$sub.brain.nii.gz"
        bet $T2_nii $T2_masked -m

        # skull-stripped T2: .nii.gz -> .npy
        T2_masked_npy="$T2_masked_npy_dir/$sub.brain.npy"
        python "$work_dir/make_npy.py" --image_path $T2_masked --output_path $T2_masked_npy

        # move mask file
        mask=$(find $T2_masked_niigz_dir -name "$sub.brain_mask.nii.gz")
        mv $mask $mask_dir

        count=$((count + 1))
        echo $count
done
