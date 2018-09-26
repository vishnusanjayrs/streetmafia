import numpy as np
import main
import dataset
import LFWNet
import alexnet
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

    test_dataset = dataset.LFWDataset(test_set_list)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total teting items', len(test_dataset), ', Total training batches per epoch:', len(test_data_loader))

    test_net = LFWNet.LfwNet()

    test_net_state = torch.load(os.path.join(lfw_dataset_path, 'lfwnet_1.pth'))

    test_net.load_state_dict(test_net_state)
    print(testing_data_list[0])

    test_landmarks = []
    pred_landmarks = []
    itr = 0

    for idx in range(0, len(test_data_loader)):
        test_idx, (test_input, test_landmark) = next(enumerate(test_data_loader))
        test_net.eval()
        pred_landmark = test_net.forward(test_input)
        test_array = np.array(test_landmark)
        test_landmarks.append(test_array)
        pred_array = pred_landmark.detach()
        pred_array =np.array(pred_array)
        pred_landmarks.append(pred_array)

    test_landmarks_arr= np.array(test_landmarks)
    pred_landmarks_arr = np.array(pred_landmarks)
    print(test_landmarks_arr.shape)
    print(pred_landmarks_arr.shape)
