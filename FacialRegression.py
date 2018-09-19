import numpy
import torch
import torch.nn as nn             # neural network lib.
import torch.nn.functional as F   # common functions (e.g. relu, drop-out, softmax...)
import os

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#make a data set . extract file paths and landmark feature cordonates from LFW train annotations

lfw_dataset_directory = '/home/vishnusanjay/PycharmProjects/FacialReg'
lfw_train_annotation_file = os.path.join(lfw_dataset_directory,'LFW_annotation_train.txt')

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
        