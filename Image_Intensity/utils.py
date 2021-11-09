import numpy as np
import nibabel as nib
from tqdm import tqdm

### function calculating mean & standard deviation
def get_mean_std(data):
    # initialization
    data.insert(3, 't1_mean', 0)
    data.insert(4, 't1_std', 0)
    data.insert(7, 't2_mean', 0)
    data.insert(8, 't2_std', 0)
    data.insert(15, 'dti_mean', 0)
    data.insert(16, 'dti_std', 0)
    data.insert(21, 'dwi_mean', 0)
    data.insert(22, 'dwi_std', 0)

    for row in tqdm(range(data.shape[0])):
        ### T1
        t1_prep_image = nib.load(data.loc[row, 't1_directory']).get_fdata()
        t_mask = nib.load(data.loc[row]['t12_qc_mask']).get_fdata()
        assert t1_prep_image.shape == t_mask.shape

        flattened_t1_image = t1_prep_image.flatten()
        flattened_t_mask = t_mask.flatten()
        t1_elements = flattened_t1_image[flattened_t_mask == 1]  # mask에서 1인 픽셀만을 추출

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
        dti = data.loc[row]['dti_directory'].replace("[", "").replace("]", "").replace("\'", "")
        fa_prep_image = nib.load(dti.split(',')[0]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask_directory']).get_fdata()
        assert fa_prep_image.shape == dti_mask.shape

        flattened_fa_image = fa_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        fa_elements = flattened_fa_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        fa_mean_of_voxels = np.mean(fa_elements)
        fa_std_of_voxels = np.std(fa_elements)

        fa_mean = round(fa_mean_of_voxels, 2)
        fa_std = round(fa_std_of_voxels, 2)

        ### DTI - MD
        md_prep_image = nib.load(dti.split(',')[3]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask_directory']).get_fdata()
        assert md_prep_image.shape == dti_mask.shape

        flattened_md_image = md_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        md_elements = flattened_md_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        md_mean_of_voxels = np.mean(md_elements)
        md_std_of_voxels = np.std(md_elements)

        md_mean = round(md_mean_of_voxels, 2)
        md_std = round(md_std_of_voxels, 2)

        ### DTI - FA + MD
        data.loc[row]['dti_mean'] = [fa_mean, md_mean]
        data.loc[row]['dti_std'] = [fa_std, md_std]

        ### DWI
        dwi = data.loc[row]['dwi_directory'].replace("[", "").replace("]", "").replace("\'", "")
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


### function calculating min & max
def get_min_max(data):
    # initialization
    data.insert(5, 't1_min', 0)
    data.insert(6, 't1_max', 0)
    data.insert(11, 't2_min', 0)
    data.insert(12, 't2_max', 0)
    data.insert(21, 'dti_min', 0)
    data.insert(22, 'dti_max', 0)
    data.insert(29, 'dwi_min', 0)
    data.insert(30, 'dwi_max', 0)

    for row in tqdm(range(data.shape[0])):
        ### T1
        t1_prep_image = nib.load(data.loc[row, 't1_directory']).get_fdata()
        t_mask = nib.load(data.loc[row]['t12_qc_mask']).get_fdata()
        assert t1_prep_image.shape == t_mask.shape

        flattened_t1_image = t1_prep_image.flatten()
        flattened_t_mask = t_mask.flatten()
        t1_elements = flattened_t1_image[flattened_t_mask == 1]  # mask에서 1인 픽셀만을 추출

        t1_min_of_voxels = np.min(t1_elements)
        t1_max_of_voxels = np.max(t1_elements)
        data.loc[row]['t1_min'] = round(t1_min_of_voxels, 2)
        data.loc[row]['t1_max'] = round(t1_max_of_voxels, 2)

        ### T2
        t2_prep_image = nib.load(data.loc[row, 't2_directory']).get_fdata()
        t_mask = nib.load(data.loc[row]['t12_qc_mask']).get_fdata()
        assert t2_prep_image.shape == t_mask.shape

        flattened_t2_image = t2_prep_image.flatten()
        flattened_t_mask = t_mask.flatten()
        t2_elements = flattened_t2_image[flattened_t_mask == 1]  # mask에서 1인 픽셀만을 추출

        t2_min_of_voxels = np.min(t2_elements)
        t2_max_of_voxels = np.max(t2_elements)
        data.loc[row]['t2_min'] = round(t2_min_of_voxels, 2)
        data.loc[row]['t2_max'] = round(t2_max_of_voxels, 2)

        ### DTI - FA
        dti = data.loc[row]['dti_directory'].replace("[", "").replace("]", "").replace("\'", "")
        fa_prep_image = nib.load(dti.split(',')[0]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask_directory']).get_fdata()
        assert fa_prep_image.shape == dti_mask.shape

        flattened_fa_image = fa_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        fa_elements = flattened_fa_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        fa_min_of_voxels = np.min(fa_elements)
        fa_max_of_voxels = np.max(fa_elements)

        fa_min = round(fa_min_of_voxels, 2)
        fa_max = round(fa_max_of_voxels, 2)

        ### DTI - MD
        md_prep_image = nib.load(dti.split(',')[3]).get_fdata()
        dti_mask = nib.load(data.loc[row]['dti_mask_directory']).get_fdata()
        assert md_prep_image.shape == dti_mask.shape

        flattened_md_image = md_prep_image.flatten()
        flattened_dti_mask = dti_mask.flatten()
        md_elements = flattened_md_image[flattened_dti_mask == 1]  # mask에서 1인 픽셀만을 추출

        md_min_of_voxels = np.min(md_elements)
        md_max_of_voxels = np.max(md_elements)

        md_min = round(md_min_of_voxels, 2)
        md_max = round(md_max_of_voxels, 2)

        ### DTI - FA + MD
        data.loc[row]['dti_min'] = [fa_min, md_min]
        data.loc[row]['dti_max'] = [fa_max, md_max]

        ### DWI
        dwi = data.loc[row]['dwi_directory'].replace("[", "").replace("]", "").replace("\'", "")
        dwi_prep_image = nib.load(dwi.split(',')).get_fdata()
        dwi_mask = nib.load(data.loc[row]['dwi_mask_directory']).get_fdata()
        assert dwi_prep_image.shape == dwi_mask.shape

        flattened_dwi_image = dwi_prep_image.flatten()
        flattened_dwi_mask = dwi_mask.flatten()
        dwi_elements = flattened_dwi_image[flattened_dwi_mask == 1]  # mask에서 1인 픽셀만을 추출

        dwi_min_of_voxels = np.min(dwi_elements)
        dwi_max_of_voxels = np.max(dwi_elements)

        data.loc[row]['dwi_min'] = round(dwi_min_of_voxels, 2)
        data.loc[row]['dwi_max'] = round(dwi_max_of_voxels, 2)

    return data