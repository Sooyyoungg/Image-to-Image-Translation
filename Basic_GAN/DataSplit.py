import torch.nn as nn
import pandas as pd
import nibabel as nib

class DataSplit(nn.Module):
    def __init__(self, csv_file, n_train, n_val, n_test, transform=None):
        super(DataSplit, self).__init__()

        self.data_info = pd.read_csv(csv_file)
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.transform = transform

        """ train data """
        train_dataset = []
        for i in range(self.n_train):
            img_1 = nib.load(self.data_info.loc[i]['t1_directory'])
            label_1 = 0
            img_2 = nib.load(self.data_info.loc[i]['t2_directory'])
            label_2 = 1
            train_dataset.append([img_1, img_2])
        train_dataset = pd.DataFrame(train_dataset, columns=['image', 'label'])

        """ validation data """
        val_dataset = []
        for j in range(self.n_val):
            img_1 = nib.load(self.data_info.loc[self.n_train+j-1]['t1_directory'])
            label_1 = 0
            img_2 = nib.load(self.data_info.loc[self.n_train+j-1]['t2_directory'])
            label_2 = 1
            val_dataset.append([img_1, img_2])
        val_dataset = pd.DataFrame(val_dataset, columns=['image', 'label'])

        """ test data """
        test_dataset = []
        for k in range(self.test):
            img_1 = nib.load(self.data_info.loc[self.n_train + self.n_val + k - 2]['t1_directory'])
            label_1 = 0
            img_2 = nib.load(self.data_info.loc[self.n_train + self.n_val + k - 2]['t2_directory'])
            label_2 = 1
            test_dataset.append([img_1, img_2])
        test_dataset = pd.DataFrame(test_dataset, columns=['image', 'label'])

        return train_dataset, val_dataset, test_dataset