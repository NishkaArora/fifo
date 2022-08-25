from cProfile import label
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from os.path import join
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import glob
import cv2
from skimage import exposure
import albumentations as A

class ThermalDataset(data.Dataset): # init, len, getitem, _apply_transform
    def __init__(self, data_root, set='validation', max_crop_width=512):

        self.set = set
        self.data_root = data_root

        self.load_data() # set could be training or validation

        self.random_transform = A.Compose([
            A.RandomResizedCrop(max_crop_width, max_crop_width, scale=(0.5, 2), ratio=[1.0, 1.0]),
            A.HorizontalFlip(p=0.5),
            
            A.OneOf([
                A.GridDistortion(), 
                A.ElasticTransform(mask_value=-1, value=0), 
                A.OpticalDistortion(mask_value=-1, value=0)
                ], p=0.5),

            A.Rotate(limit=10, mask_value=-1, value=0, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.MotionBlur(p=0.5), 
            A.ColorJitter(p=0.5),
        ])

        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=max_crop_width, always_apply=True),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=-1),

        ])

    def load_data(self):
        dataset_names = ['arroyo_seco', 'flight2-1',
         'ONR_2022-05-15-06-26-49', 'ONR_2022-05-15-06-00-08', 'bigbear']
        
        self.data = []
        self.num_sample = 0

        for name in dataset_names:
            #print(os.path.join(self.data_root, name, 'annotations', '*'))
            self.data += glob.glob(os.path.join(self.data_root, name, 'annotations', '*'))
            self.num_sample += len(self.data)
        
        assert self.num_sample > 0

        #print(self.data)

        print('# samples: {}'.format(self.num_sample))
    
    def load_image(self, index):
        segm_path = self.data[index]
        #print(self.data[index])
        # load image and label
        image_path = segm_path.replace('annotations', 'thermal').replace('png', 'tiff')
        print(image_path)
        print(segm_path)
        img = cv2.imread(image_path, -1)
        segm = cv2.imread(segm_path, 0).astype(float)
    
        if 'annotated_thermal_datasets/ONR_2022-05-15-06-26-49' in image_path or 'annotated_thermal_datasets/ONR_2022-05-15-06-00-08' in image_path:
            img = cv2.rotate(img, cv2.ROTATE_180)

        img = img / (2**16 - 1)

        # contrast stretch
        img = img - np.percentile(img, 1)
        img = np.clip(img / np.percentile(img, 99), 0, 1)
        img = exposure.equalize_adapthist(img, clip_limit=0.015)

        return img, segm
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img, segm = self.load_image(index)
        img = img * 255
        img = img.astype(np.uint8)
        
        if self.set == 'training':
            augmented_data = self.random_transform(image=img, mask=segm)
        elif self.set == 'validation':
            augmented_data = self.val_transform(image=img, mask=segm)

        label_tensor = torch.from_numpy(augmented_data['mask']).type(torch.LongTensor)

        H, W = augmented_data['image'].shape

        aug_img = augmented_data['image']
        img_tensor = torch.from_numpy(aug_img)
        #img_tensor = torch.from_numpy(aug_img).float()*255

        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.expand(3, 512, 512)

        #label_tensor.unsqueeze_(0)
        #label_tensor = label_tensor.expand(3, 512, 512)

        img_tensor = img_tensor.float()/255

        size = img_tensor.shape

        return img_tensor, label_tensor, np.array(size)
    
if __name__ == '__main__':

    data_root = '../../thermal_data/annotated_thermal_datasets/'
    # thermal_data_loader = data.DataLoader(ThermalDataset(data_root, 'validation'), batch_size=3, shuffle=True, num_workers=1, pin_memory=True)
    # thermal_iter = enumerate(thermal_data_loader)

    # print(thermal_iter.__next__)

    # step, batch = thermal_iter.__next__()

    #print(step)
    #print(batch[0].shape)
    #img_tensor, label_tensor, size = batch[2]

    #NOTE: batch[0] has batch_size number of imges
        # batch[1] has batch_size number of labels
        # batch[2] has batch size number of sizes

    dataset = ThermalDataset(data_root, 'training')
    img_tensor, label_tensor, size = dataset.__getitem__(200)

    print(img_tensor.shape, label_tensor.shape)

    print(img_tensor.dtype)

    img = img_tensor[0].numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.show()

    plt.figure()
    plt.imshow(label_tensor[0].numpy().squeeze())
    plt.show()

    print(size)