import gc
import pandas as pd
import nibabel as nib
from utils import get_mean_std
from utils import get_min_max

### Data Loader
info_path = '/home/connectome/conmaster/GANBERT/abcd_t1_t2_diffusion_info.csv'
tot_data = pd.read_csv(info_path)

print("loading finish")

image_columns = ['t1_directory', 't2_directory', 'dti_directory', 'dwi_directory']
data_list = []
for i in range(len(tot_data)):
    subject_data = []
    t1_image = nib.load(tot_data.loc[i]['t1_directory']).get_fdata()
    t2_image = nib.load(tot_data.loc[i]['t2_directory']).get_fdata()

    dti = tot_data.loc[i]['dti_directory'].replace("[","").replace("]","").replace("\'","")
    fa_image = nib.load(dti.split(', ')[0]).get_fdata()
    md_image = nib.load(dti.split(', ')[3]).get_fdata()

    dwi = tot_data.loc[i]['dwi_directory'].replace("[", "").replace("]", "").replace("\'", "")
    dwi_image = nib.load(dwi.split(', ')[0]).get_fdata()
    #dwi_mask = nib.load(tot_data.loc[i]['dwi_mask_directory']).get_fdata()

    subject_data.append([t1_image, t2_image, [fa_image, md_image], dwi_image])
    data_list.append(subject_data)
    print(i)

image_data = pd.DataFrame(data_list, columns = image_columns)
print(image_data.shape)  # (8921, 4) 여야 함

### main
gc.collect()
mean_std = get_mean_std(tot_data)
print("mean, std finish")
final_data = get_min_max(mean_std)
print("min max finish")
print(final_data.shape)
#final_data.to_csv('/home/connectome/conmaster/GANBERT/abcd_t1_t2_diff_info_mmms.csv')
gc.collect()