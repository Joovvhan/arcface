from glob import glob
import random
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from collections import Counter
import os
from tqdm import tqdm

def crop_center(image, width=160, height=160, random_range=(-5, 5 + 1)):
    y, x, c = image.shape
    x_offset = int((x - width) / 2) + random.randrange(*random_range)
    y_offset = int((y - height) / 2) + random.randrange(*random_range)
    
    return image[y_offset:y_offset+height, 
                 x_offset:x_offset+width, 
                 :]

def remove_format(file_name):
    f = os.path.basename(file_name)
    return f.split('.')[0]

def collate_fn(dataset, pretrained=True):

    '''
    All pre-trained models expect input images normalized in the same way, 
    i.e. mini-batches of 3-channel RGB images of shape (3 x H x W)
    The images have to be loaded in to a range of [0, 1] and 
    then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    '''

    pretrained_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    pretrained_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    id_list = list()
    image_list = list()

    for file, identity in dataset:
        image = cv2.imread(file)
        image = crop_center(image)
        image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (H, W, C)
        image_rgb = image_rgb / 255
        image_rgb = np.moveaxis(image_rgb, -1, 0) # (H, W, C) -> # (C, H, W)

        if pretrained:
            image_rgb -= pretrained_mean
            image_rgb / pretrained_std

        image_list.append(image_rgb)
        id_list.append(identity)

    return (torch.tensor(image_list, dtype=torch.float), 
            torch.tensor(id_list, dtype=torch.long))

def build_data_loader(dataset, batch_size=64, shuffle=True):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             num_workers=4,
                             collate_fn=collate_fn)
    return data_loader

if __name__ == "__main__":

    identity_file = 'CelebA/Anno/identity_CelebA.txt'

    identity_counter = Counter()
    identity_dict = dict()

    with open(identity_file, 'r') as f:
        for i, line in enumerate(f):
            file, identity = line.split()
            identity = int(identity)
            identity_dict[remove_format(file)] = identity
            identity_counter[int(identity)] += 1
            
    print(f'{i+1} files')

    files = sorted(glob('CelebA/Img/img_align_celeba/*.jpg'))

    dataset = [(file, identity_dict[remove_format(file)]) for file in files]

    data_loader = build_data_loader(dataset)

    for images, identities in tqdm(data_loader):
        # print(images.shape)

        # break
        pass