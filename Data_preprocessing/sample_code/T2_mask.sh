#!/bin/bash
  
t2='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri/sub-NDARINVZZLZCKAY.T2.mgz'
mask_gz='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri_brain_mask/sub-NDARINVZZLZCKAY.brain_mask.nii.gz'
t2_m="./T2_masked.mgz"

mrcalc $t2 $mask_gz -mult $t2_m
mrconvert $t2_m "T2_masked_calc.nii"
rm $t2_m
