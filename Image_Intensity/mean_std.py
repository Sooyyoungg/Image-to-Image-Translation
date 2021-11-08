import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm

### Data Loader
info_path = '/home/connectome/conmaster/GANBERT/abcd_t1_t2_diffusion_info.csv'
tot_data = pd.read_csv(info_path)
image_columns = ['t1_directory', 't2_directory', 'dti_directory', 'dwi_directory']
data_list = []

for i in range(len(tot_data)):
    subject_data = []
    t1_image = nib.load(tot_data.loc[i]['t1_directory']).get_fdata()
    t2_image = nib.load(tot_data.loc[i]['t2_directory']).get_fdata()

    dti = tot_data.loc[i]['dti_directory'].replace("[","").replace("]","").replace("\'","")
    fa_image = nib.load(dti.split(', ')[0]).get_fdata()
    md_image = nib.load(dti.split(', ')[3]).get_fdata()

    dwi = tot_data.loc[i]['dwi_directory'].replace("[","").replace("]","").replace("\'","")
    dwi_image = nib.load(dwi.split(', ')[0]).get_fdata()
    dwi_mask = nib.load(dwi.split(', ')[1]).get_fdata()

    subject_data.append([t1_image, t2_image, [fa_image, md_image], dwi_image])
    data_list.append(subject_data)

image_data = pd.DataFrame(data_list, columns = image_columns)
print(image_data.shape)  # (8921, 4) 여야 함

### function calculating mean & standard deviation
def get_image_intensity(data):
    # initialization
    data.insert(3, 't1_mean', 0)
    data.insert(4, 't1_std', 0)
    data.insert(7, 't2_mean', 0)
    data.insert(8, 't2_std', 0)
    data.insert(11, 'fa_mean', 0)
    data.insert(12, 'fa_std', 0)
    data.insert(13, 'md_mean', 0)
    data.insert(14, 'md_std', 0)
    data.insert(17, 'dwi_mean', 0)
    data.insert(18, 'dwi_std', 0)

    for row in tqdm(range(data.shape[0])):

        ### T1
        t1_prep_image = nib.load(data.loc[row, 't1_directory']).get_fdata()
        t_mask = nib.load(data.loc[row]['t12_qc_mask']).get_fdata()
        assert t1_prep_image.shape == t_mask.shape

        flattened_t1_image = t1_prep_image.flatten()
        flattened_t_mask = t_mask.flatten()
        t1_elements = flattened_t1_image[flattened_t_mask==1] #mask에서 1인 픽셀만을 추출

        t1_mean_of_voxels = np.mean(t1_elements)
        t1_std_of_voxels = np.std(t1_elements)
        data.loc[row]['t1_mean'] = round(t1_mean_of_voxels, 2)
        data.loc[row]['t1_std'] = round(t1_std_of_voxels, 2)

        ### T2
        t2_prep_image = nib.load(data.loc[row, 't2_directory']).get_fdata()
        t_mask = nib.load(data.loc[row]['t12_qc_mask']).get_fdata()
        assert t2_prep_image.shape == t_mask.shape

        flattened_t2_image = t2_prep_image.flatten()
        flattened_t_mask = t_mask.flatten()
        t2_elements = flattened_t2_image[flattened_t_mask == 1]  # mask에서 1인 픽셀만을 추출

        t2_mean_of_voxels = np.mean(t2_elements)
        t2_std_of_voxels = np.std(t2_elements)
        data.loc[row]['t2_mean'] = round(t2_mean_of_voxels, 2)
        data.loc[row]['t2_std'] = round(t2_std_of_voxels, 2)

        ### DTI - FA
        dti = tot_data.loc[row]['dti_directory'].replace("[","").replace("]","").replace("\'","")
        fa_prep_image = nib.load(dti.split(',')[0]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask']).get_fdata()
        #assert fa_prep_image.shape == dti_mask.shape

        flattened_fa_image = fa_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        fa_elements = flattened_fa_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        fa_mean_of_voxels = np.mean(fa_elements)
        fa_std_of_voxels = np.std(fa_elements)
        
        data.loc[row]['fa_mean'] = round(fa_mean_of_voxels, 2)
        data.loc[row]['fa_std'] = round(fa_std_of_voxels, 2)

        ### DTI - MD
        md_prep_image = nib.load(dti.split(',')[3]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask']).get_fdata()
        #assert md_prep_image.shape == dti_mask.shape

        flattened_md_image = md_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        md_elements = flattened_md_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        md_mean_of_voxels = np.mean(md_elements)
        md_std_of_voxels = np.std(md_elements)

        data.loc[row]['md_mean'] = round(md_mean_of_voxels, 2)
        data.loc[row]['md_std'] = round(md_std_of_voxels, 2)

        ### DWI
        dwi = tot_data.loc[row]['dwi_directory'].replace("[","").replace("]","").replace("\'","")
        dwi_prep_image = nib.load(dwi.split(',')[0]).get_fdata()
        dwi_mask = nib.load(dwi.split(',')[1]).get_fdata()
        assert dwi_prep_image.shape == dwi_mask.shape

        flattened_dwi_image = dwi_prep_image.flatten()
        flattened_dwi_mask = dwi_mask.flatten()
        dwi_elements = flattened_dwi_image[flattened_dwi_mask == 1]  # mask에서 1인 픽셀만을 추출

        dwi_mean_of_voxels = np.mean(dwi_elements)
        dwi_std_of_voxels = np.std(dwi_elements)

        data.loc[row]['dwi_mean'] = round(dwi_mean_of_voxels, 2)
        data.loc[row]['dwi_std'] = round(dwi_std_of_voxels, 2)

    return data

#>> 이렇게 추가하려면 fa, md에 대한 mask 필요
#>> fa, md도 mean, std값 구해야 하나? 어차피 z-scoring 안할껀데 > 구해야 함

### main
final_data = get_image_intensity(tot_data)
final_data.to_csv('/home/connectome/conmaster/GANBERT/abcd_t1_t2_diff_info_final.csv')