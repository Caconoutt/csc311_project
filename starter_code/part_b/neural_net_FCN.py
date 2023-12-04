import os

import numpy as np
from numpy import e
import torch
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.stats import norm
import warnings
from utils import *
import torch.utils.data as dt

warnings.filterwarnings("ignore")

np.set_printoptions(threshold=np.inf, suppress=True, precision=10, linewidth=np.inf)
torch.set_printoptions(precision=5, sci_mode=False, linewidth=10000)


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        # nn.sequential "stack" different layers of the models here
        # nn.linear(3, 6) takes 3 input output 6， 3 is student_id, subject_id1, subject_id2
        # but one may ask why pick 3-6-1, it is picked based on recommendations from other experienced experts online, and many posts,
        # and a lot of luck... I can't rly explain why.
        self.layer1 = nn.Sequential(nn.Linear(3, 6),
                                    nn.Sigmoid(), nn.Linear(6, 1))

    def forward(self, x):
        x = self.layer1(x)
        return x

if __name__ == '__main__':
    train_data, valid_data, test_data = combine_data('../data/train_data.csv', '../data/question_meta.csv',
                                                     '../data/valid_data.csv', '../data/test_data.csv')

    # here we load data for train, test and validation
    train_data = train_data[['user_id', 'subject_id1', 'subject_id2', 'is_correct']]
    test_data = test_data[['user_id', 'subject_id1', 'subject_id2', 'is_correct']]
    valid_data = valid_data[['user_id', 'subject_id1', 'subject_id2', 'is_correct']]

    # convert to torch
    train_data = torch.from_numpy(train_data.to_numpy())
    test_data = torch.from_numpy(test_data.to_numpy())
    valid_data = torch.from_numpy(valid_data.to_numpy())
    print(len(train_data))


    # slicing the training data
    trax = train_data[:,:3].float()
    tray = train_data[:,3].float()

    # slicing the validation data
    valx = valid_data[:, :3].float()
    valy = valid_data[:, 3].float()


    torch_dataset = dt.TensorDataset(trax, tray)

    loader = dt.DataLoader(
        dataset=torch_dataset,
        batch_size=10000,
        shuffle=True,
        num_workers=2            # this is for multiple threads, it runs faster
    )

    # get our model object, its the same as model = autoencoder in our part1
    model = FCN()
    for e in range(1000000000):
        # print(model)
        # Step 3:============================LOSS FUNCTION 和优化器===================
        criterion = nn.MSELoss()
        # 我们优先使用随机梯度下降，lr是学习率:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 0.01

        # 4.1==========================训练模式==========================


        acc1 = 0
        acc2 = 0
        los = []

        # for m in model.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 4)
        #         nn.init.constant_(m.bias, 0.5)

        for step, (batch_x, batch_y) in enumerate(loader):
            model.train()  # now we train

            # forward pass and calculation of loss
            out = model(batch_x)
            loss = criterion(out, batch_y)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            los.append(loss.item())


            if e % 100:
                print(f'正在进行第{e}次，第{step}的训练，损失为:{loss.item()}')

        # validation set
        model.eval()
        pre_val = model(valx)
        pre_val = pre_val.tolist()
        all_ = len(pre_val)
        valy_list = valy.tolist()
        correct_num = 0
        for i, v in enumerate(pre_val):
            # print(v[0])
            answer = 1 if v[0] > 0.5 else 0
            if answer == valy_list[i]:
                correct_num += 1
        acc = correct_num/all_
        print(f'第{e}次训练的准确率为{acc}')

        if e % 2:
            acc1 = acc
        else:
            acc2 = acc

        if acc1 == acc2:
            break



    print(f'最终的损失值为:{loss.item()}')
    exit()