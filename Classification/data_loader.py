import numpy as np
import pandas as pd
import nibabel as nib

def data_loader(data_path):
    total_data = pd.read_csv(data_path)

    t1_images = []
    t2_images = []
    fa_images = []
    md_images = []

    for i in range(len(total_data)):
        print(i)
        dti = total_data.loc[i]['dti_directory'].replace("[", "").replace("]", "").replace("\'", "")
        fa = dti.split(', ')[0]
        md = dti.split(', ')[1]

        t1_images.append([nib.load(total_data.loc[i]['t1_directory']).get_fdata(), 0])
        t2_images.append([nib.load(total_data.loc[i]['t2_directory']).get_fdata(), 1])
        fa_images.append([nib.load(fa).get_fdata(), 2])
        md_images.append([nib.load(md).get_fdata(), 3])

    t1_data = pd.DataFrame(t1_images)
    t2_data = pd.DataFrame(t2_images)
    fa_data = pd.DataFrame(fa_images)
    md_data = pd.DataFrame(md_images)
    all_data = pd.concat([t1_data, t2_data, fa_data, md_data])

    # the number of train data
    train_num = int(len(total_data)*0.75)

    train_x = all_data.loc[:train_num, 0]
    train_y = all_data.loc[:train_num, 1]
    test_x = all_data.loc[train_num:, 0]
    test_y = all_data.loc[train_num:, 1]

    return train_x, train_y, test_x, test_y

