#!/bin/bash
  
data_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_tar'
subs=$(find $data_dir -name "sub-*")

gradient_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_gradient'
masked_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_masked'
masked_npy_dir='/scratch/connectome/GANBERT/data/diffusion/DWI_masked_npy'

work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'

count=0
for sub in $subs
do
        sub_list=($(echo $sub | tr "/" " "))
        sub=${sub_list[-1]}
        cd "$data_dir/$sub"

        # find dwi & mask
        dwi=$(find . -name "mr_dwi_denoised_gibbs_preproc_biasCorr_upsample125.mif.gz")
        mask=$(find . -name "mr_dwi_denoised_gibbs_preproc_biasCorr_upsample125_bet2_mask.nii.gz")

        # extract bval, bvec
        mrinfo $dwi -export_grad_mrtrix "$gradient_dir/$sub.grad.b"

        # skull stripping: dwi x mask
        dwi_prep="$masked_dir/$sub.brain.nii.gz"
        mrcalc $dwi $mask -mult $dwi_prep

        # dwi: .nii.gz -> .npy
        python "$work_dir/make_npy.py" --image_path $dwi_prep --output_path "$masked_npy_dir/$sub.brain.npy"

        count=$((count + 1))
        echo $count
done
