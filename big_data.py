import cv2
import numpy as np
import os
import torch
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

transf = transforms.ToTensor()

ball_color = 'five'
color_dist = {'five': {'Lower': np.array([int(67 * 255 / 100), -8 + 127, 39 + 127]),
                       'Upper': np.array([int(100 * 255 / 100), 42 + 127, 55 + 127])},
              'nine': {'Lower': np.array([int(52 * 255 / 100), -14 + 127, -18 + 127]),
                       'Upper': np.array([int(98 * 255 / 100), 13 + 127, 28 + 127])}
              }
classes = ['1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
epo = 246  # 采集数据集
list_name = os.listdir('big')


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


# load
cnn = CNN().cuda()
cnn.load_state_dict(torch.load('net_cnn_big.pkl'))
img_data = np.zeros((1, 3, 128, 128))

for name in list_name:
    print('-' * 20)
    print('big/' + name)
    img = cv2.imread('big/' + name)
    [X, Y, D] = img.shape
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # 化为lab色
    inRange_hsv = cv2.inRange(lab, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])  # 测
    # 闭运算去噪
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(inRange_hsv, cv2.MORPH_CLOSE, kernel)  #
    # 区域查找
    cnts = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    mean_x = 0
    mean_y = 0
    num1 = 0
    five_num_detect = []
    five_mean_detect = []
    nine_num_detect = []
    nine_mean_detect_x = []
    nine_mean_detect_y = []
    # 五位数码管提取
    for i in cnts:
        rect = cv2.minAreaRect(i)  # 最小框提取
        area = cv2.contourArea(i)  # 得到框的面积
        if area > 150:  # 面积去噪
            num1 += 1
            # print(area)
            box = cv2.boxPoints(rect)
            x = [box[0][0], box[1][0], box[2][0], box[3][0]]  # 提取框的x坐标
            y = [box[0][1], box[1][1], box[2][1], box[3][1]]  # 提取框的y坐标
            mean_x += sum(x) / 4
            mean_y += sum(y) / 4
            # 提取区域位置
            x_min = int(min(x) - 10) if (int(min(x) - 10 > 0)) else 0
            x_max = int(max(x) + 10) if (int(max(x) + 10) < Y) else Y
            y_min = int(min(y) - 10) if (int(min(y) - 10) > 0) else 0
            y_max = int(max(y) + 10) if (int(max(y) + 10) < X) else X
            # fron.append([x_min, x_max, y_min, y_max])
            img_get = img[y_min:y_max, x_min:x_max]  # 提取图像
            img_s = cv2.resize(img_get, (128, 128))  # resize变换
            img_tensor = transf(img_s)  # 转化为tensor
            img_data[0] = img_tensor
            train_data = torch.tensor(img_data, dtype=torch.float32)    # 训练输入
            output = cnn(train_data.cuda())[0].cpu().data.numpy()   # 放入网络
            # 结果分析得到类别
            index = np.where(output[0] == np.max(output[0]))
            five_num_detect.append(classes[index[0][0]])
            five_mean_detect.append(sum(x) / 4)
            # print(index)
            # print(classes[index[0][0]])
            cv2.putText(img, classes[index[0][0]],
                        (int(x_min), int(y_min) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # 采集数据集
            # cv2.imshow('camera', img_s)
            # cv2.waitKey(1000)
            # num = input("enter: ")
            # cv2.imwrite('img1/' + str(num) + '_' + str(epo) + '.jpg', img_s)
            # print([np.int0(box)], np.int0([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))

            # 在图中标注
            cv2.drawContours(img, [np.int0([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])], -1,
                             (0, 0, 255), 2)
            # epo += 1  # 采集数据集
    mean_x /= num1
    mean_y /= num1
    # print(mean_x, mean_y, num1)
    # 提取九宫格数字
    if num1 == 5:
        x_min = int(mean_x - 300) if (int(mean_x - 300) > 0) else 0
        x_max = int(mean_x + 300) if (int(mean_x + 300) < Y) else Y
        y_min = int(mean_y + 40) if (int(mean_y + 40) > 0) else 0
        y_max = int(mean_y + 520) if (int(mean_y + 520) < X) else X
        img_new = img[y_min:y_max, x_min:x_max]
        # cv2.imwrite('img_new/' + name.split('.')[0] + '_' + '.bmp', img_new)
        [X, Y, D] = img_new.shape
        lab = cv2.cvtColor(img_new, cv2.COLOR_BGR2LAB)
        inRange_hsv = cv2.inRange(lab, color_dist['nine']['Lower'], color_dist['nine']['Upper'])
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(inRange_hsv, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for i in cnts:
            rect = cv2.minAreaRect(i)
            area = cv2.contourArea(i)
            if area > 150:
                num1 += 1
                # print(area)
                box = cv2.boxPoints(rect)
                x = [box[0][0], box[1][0], box[2][0], box[3][0]]
                y = [box[0][1], box[1][1], box[2][1], box[3][1]]
                mean_x += sum(x) / 4
                mean_y += sum(y) / 4
                x_min_r = int(min(x) - 10) if (int(min(x) - 10 > 0)) else 0
                x_max_r = int(max(x) + 10) if (int(max(x) + 10) < Y) else Y
                y_min_r = int(min(y) - 10) if (int(min(y) - 10) > 0) else 0
                y_max_r = int(max(y) + 10) if (int(max(y) + 10) < X) else X
                img_get = img_new[y_min_r:y_max_r, x_min_r:x_max_r]
                img_s = cv2.resize(img_get, (128, 128))
                img_tensor = transf(img_s)
                img_data[0] = img_tensor
                train_data = torch.tensor(img_data, dtype=torch.float32)
                output = cnn(train_data.cuda())[0].cpu().data.numpy()
                index = np.where(output[0] == np.max(output[0]))
                nine_num_detect.append(classes[index[0][0]])
                nine_mean_detect_x.append(sum(x) / 4)
                nine_mean_detect_y.append(sum(y) / 4)
                # print(index)
                # print(classes[index[0][0]])
                cv2.putText(img, classes[index[0][0]],
                            (int(x_min + x_min_r), int(y_min + y_min_r) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                # [np.int0([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])]
                cv2.drawContours(img, [np.int0([[x_min + x_min_r, y_min + y_min_r], [x_min + x_max_r, y_min + y_min_r],
                                                [x_min + x_max_r, y_min + y_max_r],
                                                [x_min + x_min_r, y_min + y_max_r]])], -1,
                                 (0, 0, 255), 2)
                # cv2.imshow('camera', img_get)
                # cv2.waitKey(100)
                # num = input("enter: ")
                # cv2.imwrite('img1/' + str(num) + '_' + str(epo) + '.jpg', img_s)
                # epo+=1

    # 确定数码管与九宫格的数字顺序
    five_mean_detect, five_num_detect = zip(*sorted(zip(five_mean_detect, five_num_detect)))
    print(five_num_detect)
    # print(five_mean_detect)
    mean_x = sum(nine_mean_detect_x) / 9
    mean_y = sum(nine_mean_detect_y) / 9
    detect = np.zeros((3, 3))
    for i in range(0, 9):
        if nine_mean_detect_x[i] - mean_x < -10:
            x = -1
        elif nine_mean_detect_x[i] - mean_x > 10:
            x = 1
        else:
            x = 0
        if nine_mean_detect_y[i] - mean_y < -10:
            y = -1
        elif nine_mean_detect_y[i] - mean_y > 10:
            y = 1
        else:
            y = 0
        detect[y + 1][x + 1] = nine_num_detect[i]
    print(detect)
    cv2.imshow('img', img)
    cv2.imwrite('img_new/' + name, img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
