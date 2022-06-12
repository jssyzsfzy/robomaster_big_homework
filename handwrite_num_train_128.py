import time

import torch
import numpy as np
import os
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import tkinter
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file_name = 'big_train'
classes = ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
EPOCH = 30  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 8
LR = 0.000001  # learning rate
transf = transforms.ToTensor()


def get_img2torch(files):
    name_data = os.listdir(files)
    length = len(name_data)
    arr = np.arange(length)
    np.random.shuffle(arr)
    name = []
    for i in range(0, length):
        name.append(name_data[arr[i]])
    label = np.ones((1, length))
    img_data = np.zeros((length, 3, 128, 128))
    for i in range(0, length):
        img = cv2.imread(files + '/' + name[i])
        img = cv2.resize(img, (128, 128))
        img_tensor = transf(img)
        img_data[i] = img_tensor
        label[0][i] = name[i].split('_')[0]
        # label[0][i] = classes.index([''.join(list(g)) for k, g in groupby(name[i], key=lambda x: x.isdigit())][0])
    return img_data, label, length


class branch(nn.Module):
    def __init__(self, input_channel, out_channe):
        super(branch, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channel, out_channe // 4, 1, 1, 0)

        self.conv5x5_1 = nn.Conv2d(input_channel, 32, 1, 1, 0)
        self.conv5x5_2 = nn.Conv2d(32, out_channe // 4, 5, 1, 2)

        self.conv3x3_1 = nn.Conv2d(input_channel, 32, 1, 1, 0)
        self.conv3x3_2 = nn.Conv2d(32, 48, 3, 1, 1)
        self.conv3x3_3 = nn.Conv2d(48, out_channe // 4, 3, 1, 1)

        self.branch_pool = nn.Conv2d(input_channel, out_channe // 4, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.conv1x1(x)

        branch5x5 = self.conv5x5_1(x)
        branch5x5 = self.conv5x5_2(branch5x5)

        branch3x3 = self.conv3x3_1(x)
        branch3x3 = self.conv3x3_2(branch3x3)
        branch3x3 = self.conv3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [branch1x1, branch3x3, branch5x5, branch_pool]
        output = torch.cat(output, dim=1)
        return output  # return x for visualization


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 128, 128)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=24,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 128, 128)
            nn.BatchNorm2d(24, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 64, 64)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 64, 64)
            nn.Conv2d(24, 64, 3, 1, 1),  # output shape (32, 64, 64)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 32, 32)
        )
        self.conv21 = nn.Sequential(  # input shape (16, 64, 64)
            nn.Conv2d(24, 64, 4, 4, 0),  # output shape (32, 64, 64)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 32, 32)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 32, 32)
            nn.Conv2d(64, 64, 5, 1, 2),  # output shape (48, 32, 32)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 16, 16)
        )
        self.branch1 = branch(input_channel=64, out_channe=256)
        self.conv4 = nn.Sequential(  # input shape (48, 16, 16)
            nn.Conv2d(256, 128, 3, 1, 1),  # output shape (64, 16, 16)
            nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 8, 8)
        )
        self.hidden = torch.nn.Linear(192 * 8 * 8, 256)  # 隐藏层线性输出
        self.out = nn.Linear(256, len(classes))  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv21(x)
        x = self.conv3(x1)
        x = self.branch1(x)
        x = self.conv4(x)
        output = [x, x2]
        output = torch.cat(output, dim=1)
        x = output.view(output.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = F.silu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        output = self.out(x)
        return output, x  # return x for visualization


if __name__ == "__main__":
    # data_label, y_label, l = get_img2torch(file_name)
    name_data = os.listdir(file_name)
    length = len(name_data)
    print(length)
    arr = np.arange(length)
    np.random.shuffle(arr)
    y = np.zeros((length, len(classes)))
    img_data = np.zeros((length, 3, 128, 128))
    for i in range(0, length):
        img = cv2.imread(file_name + '/' + name_data[arr[i]])
        img_tensor = transf(img)
        img_data[i] = img_tensor
        y[i][classes.index(name_data[arr[i]].split('_')[0])] = 1
        # cv2.imshow('1', img)
        # print(y[i])
        # cv2.waitKey(0)
    divide_num = int(length * 0.9)
    train_data = torch.tensor(img_data[0:divide_num], dtype=torch.float32)
    Y_train_tensor = torch.tensor(y[0:divide_num], dtype=torch.float64)
    test_x = torch.tensor(img_data[divide_num:length], dtype=torch.float32)
    test_y = torch.tensor(y[divide_num:length], dtype=torch.float64)
    # print(Y_train_tensor.shape)
    predict_num = test_y.shape[0]
    torch_dataset = Data.TensorDataset(train_data, Y_train_tensor)
    torch_testset = Data.TensorDataset(test_x, test_y)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    test_data = Data.DataLoader(
        dataset=torch_testset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据
        num_workers=2  # 多线程来读数据
    )
    cnn = CNN().cuda()
    print(cnn)
    # cnn.load_state_dict(torch.load('net_cnn_big.pkl'))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    print("start epoch")
    min_loss = 0.5
    x_label = []
    y_label = []
    p_label = []
    for epoch in range(EPOCH):
        sum_loss = 0
        val_loss = 0
        x_label.extend([epoch])
        start = time.time()
        for step, (x, y) in enumerate(loader):  # gives batch data, normalize x when iterate train_loader
            b_x = x.cuda()  # Tensor on GPU
            b_y = y.cuda()  # Tensor on GPU
            output = cnn(b_x)[0]
            # print(output, b_y)
            loss = loss_func(output.float(), b_y.float())
            sum_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step / 400 == 0:
                precision = 0
                with torch.no_grad():
                    for step1, (val_x, val_y) in enumerate(test_data):
                        x_val = val_x.cuda()
                        y_val = val_y.cuda()
                        test_output = cnn(x_val)[0]
                        loss_val = loss_func(test_output.float(), y_val.float())
                        val_loss += loss_val
                        pre = test_output.data.cpu().numpy()
                        for lens in range(0, y_val.shape[0]):
                            x = np.where(pre[lens] == np.max(pre[lens]))[0]
                            if pre[lens][x] > 0.7 and val_y[lens][x] == 1:
                                precision += 1
        print("epoch is :", epoch, end='/' + str(EPOCH) + '\n')
        print("output", test_output)
        print("b_y", y_val)
        print("sum_loss: ", sum_loss / divide_num)
        y_label.extend([val_loss.data.cpu().numpy() / (length - divide_num)])
        p_label.extend([precision / predict_num])
        print("val_loss: ", val_loss / (length - divide_num))
        print("precision: ", precision / predict_num, '  sum:', predict_num)
        if val_loss / (length - divide_num) < min_loss:
            min_loss = val_loss / (length - divide_num)
            torch.save(cnn.state_dict(), 'net_cnn_big.pkl')
        print("min_loss:", min_loss)
        end = time.time()
        epoch_time = end - start
        print("Running time: %s seconds" % epoch_time)
        print('-' * 20)
    # print(x_label, y_label)
    plt.subplot(2, 1, 1)
    plt.plot(x_label, y_label, linewidth=1, color='red')
    plt.subplot(2, 1, 2)
    plt.plot(x_label, p_label, linewidth=1, color='blue')
    plt.show()
