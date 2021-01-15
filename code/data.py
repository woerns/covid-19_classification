import os
import numpy as np
import pandas as pd

import torch
from skimage.feature import canny
from PIL import Image
from torchvision import datasets


class CTImageDataSet(torch.utils.data.Dataset):
    """CT Image datasets."""

    def __init__(self, X, y, transform=None, add_mask=False):

        if add_mask:
            self.x = [np.concatenate([x, np.expand_dims(canny(x.mean(axis=-1) / 255.) * 255, axis=2).astype(np.uint8)],
                                     axis=2) for x in X]
        else:
            self.x = X
        # Convert numpy array to PILImage
        self.x = list(map(lambda x: Image.fromarray(x).convert(mode='RGB'), self.x))    # Convert to RGB

        self.y = y
        self.n_samples = len(X)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx]
        target = self.y[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def load_ai4h_patient_group_map(data_dir):
    metainfo_file = os.path.join(data_dir, 'CT-MetaInfo.xlsx')
    covid_metainfo = pd.read_excel(metainfo_file, sheet_name='COVID-CT-info')
    non_covid_metainfo = pd.read_excel(metainfo_file, sheet_name='NonCOVID-CT-info')

    covid_metainfo['File name'] = covid_metainfo['File name'].str.replace('%', '_')
    non_covid_metainfo['image name'] = non_covid_metainfo['image name'].str.replace('%', '_')

    patient_group_map = {**covid_metainfo.set_index('File name')['Patient ID'].to_dict(),
                         **non_covid_metainfo.set_index('image name')['patient id'].to_dict()}

    return patient_group_map


def load_ai4h_img_data(data_dir):
    LABEL_MAP = {'COVID': 1, 'NonCOVID': 0}
    PATIENT_GROUP_MAP = load_ai4h_patient_group_map(data_dir)

    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'full'))

    X = []
    y = []
    groups = []
    for i, (img, target) in enumerate(dataset):
        X.append(np.array(img))
        y.append(LABEL_MAP[dataset.classes[target].split('_')[-1]])    # Assuming folder label has prefix "CT_"
        img_name = dataset.imgs[i][0].split('/')[-1]
        groups.append(PATIENT_GROUP_MAP[img_name]) # Patient group

    return X, y, groups


def load_dataset(dataset_name, data_root_dir):
    if dataset_name == 'ucsd-ai4h':
        data_dir = os.path.join(data_root_dir, 'ucsd-ai4h')
        X, y, groups = load_ai4h_img_data(data_dir)
    else:
        raise ValueError("Unknown dataset name %s." % dataset_name)

    return X, y, groups