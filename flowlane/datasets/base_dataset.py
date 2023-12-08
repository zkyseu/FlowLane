import os.path as osp
import os
import numpy as np
import cv2
import oneflow
from oneflow.utils.data import Dataset
import logging

from .builder import DATASETS
from .preprocess.compose import Compose

class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None,
            cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Compose(processes, cfg)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cfg.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

        sample = self.processes(sample)

        return sample 