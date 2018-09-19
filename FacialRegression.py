import numpy
import torch
import torch.nn as nn             # neural network lib.
import torch.nn.functional as F   # common functions (e.g. relu, drop-out, softmax...)
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#make a data set . extract file paths and landmark feature cordonates from LFW train annotations

lfw_annotation_directory = '/home/vishnusanjay/PycharmProjects/FacialReg'
lfw_dataset_directory = '/home/vishnusanjay/PycharmProjects/FacialReg/lfw'
lfw_train_annotation_file = os.path.join(lfw_annotation_directory,'LFW_annotation_train.txt')

train_data_list = []
with open(lfw_train_annotation_file,"r") as f:
    for line in f:
        print(line)
        tokens = line.split("\t")
        file_name_splits = tokens[0].rsplit("_",1)
        image_file_path = file_name_splits[0]+"/"+tokens[0]
        bbox_coor = tokens[1].split()
        features = tokens[2].split()
        train_data_list.append({'file_path': image_file_path,'border box coordinates':bbox_coor,'landmark features coordinates':features})



class LFWDataset (Dataset)
    def __init__(self, data_list):
        """ Initialization: load the dataset list
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item= self.data_list[idx]
        features=np.asarray(item['landmark features coordinates'])
        bbox_coor = np.asarray(item['border box coordinates'])
        image_file_path = os.path.join(lfw_dataset_directory,item['file_path'])

        #load image
        img = np.asarray(Image.open(image_file_path),dtype=np.float32)

        #rescale RGB pixel value from (0,255) to (-1,1)
        img = (img/255)*2-1

        img = img.crop((features[0],features[1],features[2],features[3]))

        