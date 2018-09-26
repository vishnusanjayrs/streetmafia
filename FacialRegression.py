import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
from random import randint
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mp

# make a data set . extract file paths and landmark feature cordonates from LFW train annotations
train_data_list = []
lfw_dataset_path = '/home/vramiyas/PycharmProjects/Project 1/lfw'
test_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_test.txt')
train_landmark_path = os.path.join(lfw_dataset_path, 'LFW_annotation_train.txt')
training_ratio = 0.8
aug_types = ['001', '010', '100', '011', '101', '110', '111']  # nnn - random crop , flip,brightness

training_validation_data_list = []
testing_data_list = []
learning_rate = 0.005

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


class LfwNet(nn.Module):
    def __init__(self):
        super(LfwNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class LFWDataset(Dataset):
    img_w, img_h = 225.0, 225.0
    dir_name_trim_length = -9

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # split data from the dictionary into class variables
        file_name = item['file_path']
        # print(file_name)
        self.file_path = os.path.join(lfw_dataset_path, file_name[:self.dir_name_trim_length], file_name)
        # print(self.file_path)
        self.crops = np.asarray(item['crops'])
        self.crops = self.crops.astype(int)
        self.landmarks = np.asarray(item['landmarks'])
        self.landmarks = self.landmarks.astype(float)
        self.aug_types = item['aug_types']
        self.aug_type_list = list(self.aug_types)
        # open image from the file path
        orig_img = Image.open(self.file_path)

        # prepare image based on input augmentation type

        # crop or random crop the image to extract face using crop co-ordinates
        img, landmarks = self.crop(orig_img)

        # flip the image based on condition
        if self.aug_type_list[1] == '1':
            img, landmarks = self.flip(img, landmarks)

        # print(landmarks)

        # brighten the image based on condition
        if self.aug_type_list[2] == '1':
            img = self.brighten(img)

        self.preview(img, landmarks)

        # normalise the image and landmarks
        img, landmarks = self.normalise(img, landmarks)

        # convert np.lsit to tensor
        # Create tensors and reshape them to proper sizes.
        img_tensor = torch.Tensor(img.astype(float))
        img_tensor = img_tensor.view((img.shape[2], img.shape[0], img.shape[1]))
        landmark_tensors = torch.Tensor(landmarks)
        #print(landmark_tensors.shape[0])

        # print("done")


        # 225 x 225 x 3 input image tensor. 14 landmark tensor.

        return img_tensor, landmark_tensors

    def crop(self, input_img):
        crops_offset = np.zeros(len(self.crops), dtype=np.float32)
        if self.aug_type_list[0] == '1':
            # randomly add  a number between -5 and 5 to the crop co-ordinates
            for index in range(0, len(self.crops)):
                rand_offset = randint(-1, 1)
                crops_offset[index] = self.crops[index] + rand_offset
        else:
            crops_offset = self.crops

        # crop the image
        img = input_img.crop((crops_offset))
        # change landmark co-ordinates by subtracting cropped length from the landmark in each dimension
        landmarks_offset = np.zeros(2, dtype=np.float32)
        landmarks_offset = [crops_offset[0], crops_offset[1]]
        landmarks_offset = np.tile(landmarks_offset, 7)
        cropped_landmarks = self.landmarks - landmarks_offset
        # resize the image to (225,225) which is the input image size to alexnet
        w, h = img.size
        # find the image size ratio to 225 so that same ratio can be applies to landmark co-ordinates
        ratio_width = w / self.img_w
        ratio_height = h / self.img_h
        img = img.resize((225, 225), Image.ANTIALIAS)
        landmark_offset_ratio = [ratio_width, ratio_height]
        landmark_offset_ratio = np.tile(landmark_offset_ratio, 7)
        cropped_landmarks = cropped_landmarks / landmark_offset_ratio
        return img, cropped_landmarks

    def flip(self, input_img, input_landmarks):
        # flip the image
        img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
        # flip the x co-ordinates in the landmarks list
        flipped_landmarks = np.zeros(len(self.landmarks), dtype=np.float32)
        for index in range(0, len(self.landmarks)):
            if index % 2 == 0:
                flipped_landmarks[index] = 225.0 - input_landmarks[index]
            else:
                flipped_landmarks[index] = input_landmarks[index]

        # tranform the landmark co ordinates to keep the co-ordinates side consistensy
        transformed_landmarks = np.zeros(14, dtype=np.float32)

        transformed_landmarks[0] = flipped_landmarks[6]
        transformed_landmarks[1] = flipped_landmarks[7]
        transformed_landmarks[2] = flipped_landmarks[4]
        transformed_landmarks[3] = flipped_landmarks[5]
        transformed_landmarks[4] = flipped_landmarks[2]
        transformed_landmarks[5] = flipped_landmarks[3]
        transformed_landmarks[6] = flipped_landmarks[0]
        transformed_landmarks[7] = flipped_landmarks[1]
        transformed_landmarks[8] = flipped_landmarks[10]
        transformed_landmarks[9] = flipped_landmarks[11]
        transformed_landmarks[10] = flipped_landmarks[8]
        transformed_landmarks[11] = flipped_landmarks[9]
        transformed_landmarks[12] = flipped_landmarks[12]
        transformed_landmarks[13] = flipped_landmarks[13]

        return img, transformed_landmarks

    def brighten(self, input_img):
        # randonly initialize brightening/darkening factor
        factor = 1 + randint(-30, 30) / 100.0
        brightened_img = input_img.point(lambda x: x * factor)
        return brightened_img

    def normalise(self, input_img, input_landmarks):
        # convert image to array
        n_img = np.asarray(input_img, dtype=np.float32)
        # normalise the image pixels to (-1,1)
        img = (n_img / 255.0) * 2 - 1

        # normalise landmarks
        n_landmarks = input_landmarks / self.img_w
        return n_img, n_landmarks

    def denormalize(self, input_img, input_landmarks):
        # Denormalize the image.
        dn_img = np.array(input_img, dtype=float)
        if dn_img.shape[0] == 3:
            dn_img = (dn_img + 1) / 2 * 255
            return dn_img.astype(int)

        # Denormalize the landmarks.
        return input_landmarks * self.img_w

    def preview(self, image, landmarks):

        # image, landmarks = self.denormalize(image_tensor, landmark_tensor)

        plt.figure(num='Preview')
        plt.imshow(image)

        plt.scatter(x=landmarks[0], y=landmarks[1], c='r', s=10)
        plt.scatter(x=landmarks[2], y=landmarks[3], c='b', s=10)
        plt.scatter(x=landmarks[4], y=landmarks[5], c='g', s=10)
        plt.scatter(x=landmarks[6], y=landmarks[7], c='c', s=10)
        plt.scatter(x=landmarks[8], y=landmarks[9], c='m', s=10)
        plt.scatter(x=landmarks[10], y=landmarks[11], c='y', s=10)
        plt.scatter(x=landmarks[12], y=landmarks[13], c='k', s=10)
        plt.xlim(0, 225)
        plt.ylim(225, 0)
        plt.show()


# Read training and validation data.


def main():
    with open(train_landmark_path, "r") as file:
        for line in file:
            # split at tabs to get file name , borderbox co-ordinates , landmark feature co-ordinates
            tokens = line.split('\t')
            if len(tokens) == 3:
                file_path = tokens[0]
                crops = tokens[1].split()
                landmarks = tokens[2].split()
                training_validation_data_list.append(
                    {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': '000'})
                # random number of augmented images of original images
                # augtype 000 indicates original image
                random.shuffle(aug_types)
                max_augs = 5
                itr = 0
                for idx in aug_types:
                    training_validation_data_list.append(
                        {'file_path': file_path, 'crops': crops, 'landmarks': landmarks, 'aug_types': idx})
                    itr = itr + 1
                    if itr == max_augs:
                        break

                        # Read test data


    random.shuffle(training_validation_data_list)
    random.shuffle(testing_data_list)
    total_training_validation_items = len(training_validation_data_list)

    # Training dataset.
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = training_validation_data_list[: int(n_train_sets)]

    # Validation dataset.
    n_valid_sets = (1 - training_ratio) * total_training_validation_items
    valid_set_list = training_validation_data_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]

    # Testing dataset.
    test_set_list = testing_data_list

    train_dataset = LFWDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))

    valid_set = LFWDataset(valid_set_list)
    valid_data_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=2)
    print('Total validation set:', len(valid_set))

    # Prepare pretrained model.
    alex_net = alexnet(pretrained=True)
    lfw_net = LfwNet()
    alex_dict = alex_net.state_dict()
    lfw_dict = lfw_net.state_dict()

    # Remove FC layers from pretrained model.
    alex_dict.pop('classifier.1.weight')
    alex_dict.pop('classifier.1.bias')
    alex_dict.pop('classifier.4.weight')
    alex_dict.pop('classifier.4.bias')
    alex_dict.pop('classifier.6.weight')
    alex_dict.pop('classifier.6.bias')

    # Load lfw model with pretrained data.
    lfw_dict.update(alex_dict)
    lfw_net.load_state_dict(lfw_dict)

    # Losses collection, used for monitoring over-fit
    train_losses = []
    valid_losses = []

    max_epochs = 10
    itr = 0
    optimizer = torch.optim.Adam(lfw_net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(lfw_net)

    for param in lfw_net.features.parameters():
        param.requires_grad = False


    print("start train")

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):
            itr += 1
            lfw_net.train()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            train_input = Variable(train_input.cuda())  # Use Variable(*) to allow gradient flow
            train_out = lfw_net.forward(train_input)  # Forward once

            # Compute loss
            train_label = Variable(train_label.cuda())
            loss = criterion(train_out, train_label)

            # Do the backward and compute gradients
            loss.backward()

            # Update the parameters with SGD
            optimizer.step()

            train_losses.append((itr, loss.item()))

            if train_batch_idx % 50 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

                # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
                lfw_net.eval()  # [Important!] set the network in evaluation model
                valid_loss_set = []  # collect the validation losses
                valid_itr = 0

                # Do validation
                for valid_batch_idx, (valid_input, valid_label) in enumerate(valid_data_loader):
                    lfw_net.eval()
                    valid_input = Variable(valid_input.cuda())  # use Variable(*) to allow gradient flow
                    valid_out = lfw_net.forward(valid_input)  # forward once

                    # Compute loss
                    valid_label = Variable(valid_label.cuda())
                    valid_loss = criterion(valid_out, valid_label)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, float(avg_valid_loss)))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)
    train_losses[0] =train_losses[1]
    valid_losses[0] =valid_losses[1]

    print(valid_losses)

    plt.plot(train_losses[2:, 0],  # Iteration
             train_losses[2:, 1])  # Loss value
    plt.plot(valid_losses[2:, 0],  # Iteration
             valid_losses[2:, 1])  # Loss value
    plt.show()
    net_state = lfw_net.state_dict()  # serialize trained model
    torch.save(net_state, os.path.join(lfw_dataset_path, 'lfwnet.pth'))  # save to disk


if __name__ == '__main__':
    main()
