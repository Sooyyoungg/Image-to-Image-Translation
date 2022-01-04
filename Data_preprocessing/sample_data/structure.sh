#!/bin/bash
  
# sample data directory
data_dir='/scratch/connectome/GANBERT/swimming_sample/structure'
work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'

subs=$(find $data_dir -name "sub-*")

for sub in $subs
do
        sub_list=($(echo $sub | tr "/" " "))
        sub=${sub_list[-1]}
        cd "$data_dir/$sub"

        T2=$(find . -name "T2.mgz")
        mask=$(find . -name "brainmask.mgz")

        # make brain mask binary
        mrthreshold -comparison gt -abs 0 $mask brainmask_binary.mgz
        mask_bi=$(find . -name "brainmask_binary.mgz")

        # brain mask: .mgz -> .nii
        mrconvert $mask_bi "mask_bi.nii"
        mask_nii=$(find . -name "mask_bi.nii")

        # T2 x mask
        T2_mask="$data_dir/$sub/T2_mask.mgz"
        mrcalc $mask_nii $T2 $T2_mask

        # T2: .mgz -> .npy
        T2_mask_niigz="$data_dir/$sub/T2_nii.gz"
        mrconvert $T2_mask $T2_mask_niigz
        python "$work_dir/make_npy.py" --image_path $T2_mask_niigz --output_path "$data_dir/$sub/t2.npy"

        rm $T2_mask
        rm $T2_mask_niigz
done
