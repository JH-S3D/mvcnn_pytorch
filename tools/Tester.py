import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time

import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class ModelNetTester(object):

    def __init__(self, model, val_loader, loss_fn, \
                 model_name, num_views=12):

        self.model = model
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.num_views = num_views

        self.model.cuda()

    def test_accuracy(self):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        batch_size = 64  # or whatever your batch size is
        start_item_index = 0
        start_batch_index = start_item_index // batch_size

        for _, data in enumerate(self.val_loader, 0):
            if _ < start_batch_index:
                continue  # Skip batches until the starting index

            if self.model_name == 'mvcnn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
                plt.imshow(data[1][0].permute(1, 2, 0).numpy())
                plt.pause(0.001)
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            predicted_class_names = [self.model.classnames[p] for p in pred.cpu().numpy()]
            print("Predicted Class Names:", predicted_class_names)

            print(f"Test: {_}, Correct Points: {correct_points}/{results.size()[0]}")

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)

        return loss, val_overall_acc, val_mean_class_acc

    def test_embedding(self, start=0, interval=1):
        self.model.eval()

        batch_size = 1  # or whatever your batch size is
        start_item_index = start
        start_batch_index = start_item_index // batch_size

        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                if i < start_batch_index:
                    continue

                if i % interval != 0:
                    continue

                if self.model_name == 'mvcnn':
                    print("Data: ", data[1].size())
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                    print("In Data: ", in_data.size())
                else:  # 'svcnn'
                    in_data = Variable(data[1]).cuda()
                    plt.imshow(data[1][0].permute(1, 2, 0).numpy())
                    plt.pause(0.001)
                target = Variable(data[0]).cuda()

                out_data = self.model.test(in_data)
                print("Output Shape: ", out_data.size())
                
                print("Reshaped Output Shape: ", out_data.view((int(in_data.shape[0]/self.num_views),self.num_views,out_data.shape[-3],out_data.shape[-2],out_data.shape[-1])).size())#(8,12,512,7,7))
            