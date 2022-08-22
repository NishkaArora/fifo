import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from os.path import join
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import glob
import cv2
import albumentations as A

class RGBDataset(data.Dataset): # init, len, getitem, _apply_transform
    def __init__(self, data_root, set='validation', max_crop_width=512):
        self.set = set
        self.data_root = data_root
        self.load_data() # set could be train or val

        self.random_transform = A.Compose([
            # A.Normalize(always_apply=True), 
            A.LongestMaxSize(max_size=512, always_apply=True),
            A.RandomResizedCrop(max_crop_width, max_crop_width, scale=(0.5, 2), ratio=[1.0, 1.0]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, mask_value=-1, value=0, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.ColorJitter(p=0.5),
            #A.ToGray(p=1),
        ])

        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=512, always_apply=True),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=-1),
            #A.ToGray(p=1),
        ])

    def load_data(self):
        self.data = glob.glob(os.path.join(self.data_root, 'images', self.set, '*'))
        self.num_sample = len(self.data)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
    
    def load_image(self, index):
        image_path = self.data[index]
        # load image and label
        segm_path = image_path.replace('images', 'annotations').replace('jpg', 'png')
        img = cv2.imread(image_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segm = cv2.imread(segm_path, 0).astype(float)
        return img, segm
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
                
        # load image and label
        img, segm = self.load_image(index)
        
        # image transform, to torch float tensor 3xHxW
        augmented_data = None
        if self.set == 'training':
            augmented_data = self.random_transform(image=img, mask=segm)
        elif self.set == 'validation':
            augmented_data = self.val_transform(image=img, mask=segm)

        aug_img = augmented_data['image']
        aug_segmentation = augmented_data['mask']
        h, w, c = aug_img.shape
        
        # convert to grayscale
        aug_img =torch.from_numpy(aug_img).float()
        new_img = aug_img[:,:,0] + aug_img[:,:,2] + aug_img[:,:,1]
        new_img /= 3

        img_tensor = torch.from_numpy(np.asarray(new_img)).float()
        label_tensor = torch.from_numpy(aug_segmentation).type(torch.LongTensor)
        size = img_tensor.shape

        return img_tensor, label_tensor, np.array(size)
    
if __name__ == '__main__':
    data_root = '../../thermal_data/annotated_thermal_datasets/cocostuff_water/'
    dataset = RGBDataset(data_root, 'validation')
    img_tensor, label_tensor, size = dataset.__getitem__(200)

    print(img_tensor.shape, label_tensor.shape)

    img = img_tensor.numpy().squeeze()
    plt.imshow((img)/255, cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(label_tensor.numpy().squeeze())
    plt.show()