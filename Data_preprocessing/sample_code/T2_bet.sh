#!/bin/bash
  
t2='/scratch/bigdata/ABCD/freesurfer/smri/fs_smri/sub-NDARINVZZLZCKAY.T2.mgz'
t2_nii="./T2.nii"
mri_convert $t2 $t2_nii
T2_masked='./T2_masked.nii.gz'
bet $t2_nii $T2_masked -m
