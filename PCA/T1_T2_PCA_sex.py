import gc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
import plotly.express as px
import nibabel as nib
import torch
#%matplotlib inline

###
def flatten(data, feature_dimension, norm=True, remove_zero_column=True):
    """
    - function that is called from "plot_site" function
    - change 3D images into flattened images with some additional transformation
    :param data: DataFrame
    :param feature_dimension:
    :param norm: sample-wise normalization (default = True)
    :param remove_zero_column: Removal of pixels that contain 0 intensity
    :return: ndarray (# of subjects, # of pixels)
    """

    # initialize the matrix that will contain flattened 3d images.
    # data.shape[0]: the number of the subjects
    # feature_dimension: dimension of the image (=x*y*z)
    # flattened_images_T1 = np.array(([[0]*feature_dimension]*data.shape[0]))
    # flattened_images_T1 = np.zeros((data.shape[0], feature_dimension))
    # flattened_images_T2 = np.array(t1, (data.shape[0], feature_dimension))
    # print(flattened_images_T1.shape)
    flattened_images_T1 = []
    flattened_images_T2 = []

    try:
        for sub in tqdm(range(data.shape[0])):
            T1_image = nib.load(data.loc[sub, 't1_directory'])
            T2_image = nib.load(data.loc[sub, 't2_directory'])
            # mask = nib.load(data.loc[sub, 'mask_directory']).get_fdata()
            # assert T1_image.shape == T2_image.shape == mask.shape
            assert T1_image.shape == T2_image.shape

            T1_image = T1_image.get_fdata()
            T2_image = T2_image.get_fdata()

            flattened_T1_image = T1_image.flatten()
            flattened_T2_image = T2_image.flatten()
            # flattened_mask = mask.flatten()
            """
            if norm == True:
                # sample-wise normalization
                # mask의 값이 1인 pixel만 normalize하고, mask의 값이 0인 pixel은 0으로 그대로 두기
                flattened_T1_image[flattened_mask==1] = (flattened_T1_image[flattened_mask==1] - data.loc[sub, 'mean']) / data.loc[sub, 'std']
                flattened_T2_image[flattened_mask==1] = (flattened_T2_image[flattened_mask==1] - data.loc[sub, 'mean']) / data.loc[sub, 'std']
            """

            flattened_images_T1.append(flattened_T1_image)
            flattened_images_T2.append(flattened_T2_image)
            # print(flattened_T1_image.shape)   # (16777216,) == flattened_T2_image.shape

            del (T1_image)
            del (T2_image)

        flattened_images_T1 = np.array(flattened_images_T1)
        flattened_images_T2 = np.array(flattened_images_T2)

        # print(flattened_images_T1.shape)
        # print(flattened_images_T2.shape)

        # 각 pixel에 대하여, 0인 subject가 한 명이라도 있으면 해당 픽셀은 제거
        """if remove_zero_column == True: 
            flattened_images_T1 = flattened_images_T1[:, np.all(flattened_images_T1!= 0, axis=0)]
            flattened_images_T2 = flattened_images_T2[:, np.all(flattened_images_T2!= 0, axis=0)]
        print('shape_of_each_flattened_images:', flattened_images_T1.shape, flattened_images_T2.shape)"""

        flattened_images = np.concatenate([flattened_images_T1, flattened_images_T2])
        print('shape_of_flattened_images:', flattened_images.shape)

        return flattened_images

    except:
        print(data.loc[sub]['subjectkey'])


def plot_site(data, method='PCA', target='sex', norm=False, remove_zero_column=True, n_components=2):
    """
    :param data: DataFrame
    :param method: str ('PCA' or 't_sne')
    :param target_site: 'all'(str) or the subset (list)
    :param target: str('abcd_site' or 'sex' or 'race' or 'age', 'T1_or_T2')
    :param norm: sample-wise normalization (default = True)
    :param remove_zero_column: Removal of pixels that contain 0 intensity
    :param n_components: parameter of PCA n_components (default=2)
    :return: ndarray(flattened_images for visualization)
    """
    # t1_shape == t2_shape == (256, 256, 256)
    dimension = data.t1_shape
    x, y, z = (int(num) for num in dimension[0][1:-1].split(', '))
    total_dimension = x * y * z

    # select features used for PCA
    new_data = data[['subjectkey', 't1_directory', 't2_directory', 'age', 'sex', 'race.ethnicity', 'abcd_site']]
    # target variable이 결측값인 데이터 제거
    new_data = new_data.dropna(axis=0, subset=[target])  # (8921, 7)
    # num_level: the number of all kinds of target values
    num_level = len(set(new_data[target]))

    # 각 subject에 해당하는 image들을 flatten 시키는 과정
    if norm == True:
        if not os.path.exists('/scratch/GANBERT/sooyoung/flattened_t1_t2_norm_sex.npy'):
            flattened_images_T1_T2 = flatten(new_data, total_dimension, norm, remove_zero_column)
            np.save('/scratch/GANBERT/sooyoung/flattened_t1_t2_norm_sex', flattened_images_T1_T2)
        else:
            flattened_images_T1_T2 = np.load('/scratch/GANBERT/sooyoung/flattened_t1_t2_norm_sex.npy', allow_pickle=True)
    else:
        if not os.path.exists('/scratch/GANBERT/sooyoung/flattened_t1_t2_wo_norm_sex.npy'):
            flattened_images_T1_T2 = flatten(new_data, total_dimension, norm, remove_zero_column)
            np.save('/scratch/GANBERT/sooyoung/flattened_t1_t2_wo_norm_sex', flattened_images_T1_T2)
        else:
            flattened_images_T1_T2 = np.load('/scratch/GANBERT/sooyoung/flattened_t1_t2_wo_norm_sex.npy')

    if method == 'PCA':
        plot_PCA(new_data, target, num_level, flattened_images_T1_T2, n_components)


def plot_PCA(new_data, target, num_level, flattened_images, n_components):
    model = PCA(n_components=n_components)
    flattened_images = np.array(flattened_images)
    print(flattened_images.shape)
    result = model.fit_transform(flattened_images)

    if n_components > 2:
        print('plotting explained variances')
        exp_var_cumul = np.cumsum(model.explained_variance_ratio_)
        x = range(1, exp_var_cumul.shape[0] + 1)
        y = exp_var_cumul
        plt.fill_between(x, y)
        plt.xlabel("# Components")
        plt.ylabel("Explained Variance")
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.show()
        return None

    first_ratio, second_ratio = model.explained_variance_ratio_[:n_components]
    loadings = model.components_.T * np.sqrt(model.explained_variance_)

    result = pd.DataFrame(result[:, :2], columns=["x", "y"])

    target_info = new_data[[target]]
    merged_t1 = pd.concat([result.iloc[:11, :], target_info], axis=1)
    merged_t2 = pd.concat([result.iloc[11:, :], target_info], axis=1)

    # 시각화
    plt.figure(figsize=(16, 9))

    sns.set_palette(sns.color_palette("muted"))

    sns.scatterplot(data=merged_t1, x='x', y='y', hue=target, s=20,
                    palette=sns.color_palette('pastel', n_colors=num_level))
    sns.scatterplot(data=merged_t2, x='x', y='y', hue=target, s=20,
                    palette=sns.color_palette('dark', n_colors=num_level))

    plt.title(f'PCA_{target}')
    plt.xlabel(f'PC1({first_ratio:.1%} explained variation)'.format())
    plt.ylabel(f'PC2({second_ratio:.1%} explained variation)'.format())
    plt.show()

### Main
gc.collect()
output_final = pd.read_csv("/home/connectome/conmaster/GANBERT/abcd_t1_t2_diffusion_info.csv")
plot_site(output_final, method='PCA', target='sex', norm=False, remove_zero_column=True)
gc.collect()