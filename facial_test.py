import numpy as np
import FacialRegression as fr
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
from random import randint
from torch.autograd import Variable

if __name__ == '__main__':
    lfw_dataset_path = '/home/vishnusanjay/PycharmProjects/FacialReg/lfw'
    test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
    testing_data_list = []

    with open(test_landmark_path, "r") as file:
        for line in file:
            # split at tabs to get file name , borderbox co-ordinates , landmark feature co-ordinates
            tokens = line.split('\t')
            if len(tokens) == 3:
                file_path = tokens[0]
                crops = tokens[1].split()
                landmarks = tokens[2].split()
                testing_data_list.append(
                    {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': '000'})

    random.shuffle(testing_data_list)

    # Testing dataset.
    test_set_list = testing_data_list

    test_dataset = fr.LFWDataset(test_set_list)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)
    print('Total teting items', len(test_dataset), ', Total training batches per epoch:', len(test_data_loader))

    test_net = fr.LfwNet()

    test_net_state = torch.load(os.path.join(lfw_dataset_path,'lfwnet.pth'))

    test_net.load_state_dict(test_net_state)

    idx , (test_input , test_landmarks) = next(enumerate(test_data_loader))

    pred_ladnmarks = test_net.forward(test_input)

    print(test_landmarks[0]*225)
    print(pred_ladnmarks[0]*225)
print(test_input.shape)