############################ NOT USING THIS CODE JUST FOR RECORD ############################
#!/bin/bash
# This code is for T2 structural image data.
# using T1 raw image, mask image.
# preprocessing: skull-stripping(using mask), cropping, change form of image(finally it should be .npy)

# old directory
T1_prep_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri_brain'
# new directory
T1_npy_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T1_brain_npy'

# old directory
raw_data_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri'
mask_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri_brain_mask'

# new directory
T2_masked_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_masked'
T2_masked_npy_dir='/scratch/bigdata/ABCD/freesurfer/smri/fs_T2_masked_npy'

work_dir='/home/connectome/conmaster/Projects/Image_Translation/preprocessing'

T2s=$(find $raw_data_dir -name "$sub.T2.mgz")
count=0

for T2 in $T2s
do
        sub_list=($(echo $T2 | tr "/" " "))
        sub_file=${sub_list[-1]}
        subs=($(echo $sub_file | tr "." " "))
        sub=${subs[0]}

        ############ T1 ############
        # T1: .nii.gz -> .npy
        T1="$T1_prep_dir/$sub.brain.nii.gz"
        T1_npy="$T1_npy_dir/$sub/t1.npy"
        mkdir "$T1_npy_dir/$sub"
        python "$work_dir/make_npy.py" --image_path $T1 --output_path $T1_npy

        ############ T2 ############
        # mask file
        mask=$(find $mask_dir -name "$sub.brain_mask.nii.gz")

        # T2 x mask
        T2_mask="./T2_masked.mgz"
        mrcalc $mask $T2 -mult $T2_mask

        # T2: .mgz -> .nii.gz
        mkdir "$T2_masked_dir/$sub"
        T2_mask_niigz="$T2_masked_dir/$sub/T2_masked.nii.gz"
        mrconvert $T2_mask $T2_mask_niigz

        # T2: .nii.gz -> .npy
        mkdir "$T2_masked_npy_dir/$sub"
        T2_mask_npy="$T2_masked_npy_dir/$sub/t2.npy"
        python "$work_dir/make_npy.py" --image_path $T2_mask_niigz --output_path $T2_mask_npy

        rm $T2_mask

        count=$((count + 1))
        echo $count
done
