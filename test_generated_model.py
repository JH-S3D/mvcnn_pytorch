import matplotlib
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

import matplotlib.pyplot as plt

from tools.Tester import ModelNetTester
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model1 = SVCNN("cnet1").to(device)
model1.load_state_dict(torch.load('C:\\Users\\Disruptive\\Documents\\MachineLearning\\results\\mvcnn_stage_1\\mvcnn\\model-00027.pth'))

model2 = MVCNN("cnet2", model1).to(device)
model2.load_state_dict(torch.load('C:\\Users\\Disruptive\\Documents\\MachineLearning\\results\\mvcnn_stage_2\\mvcnn\\model-00015.pth'))

val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
print('num_val_files: '+str(len(val_dataset.filepaths)))

model1.eval()
model2.eval()

model = model1
modelName = 'svcnn'

trainer = ModelNetTester(model, val_loader, nn.CrossEntropyLoss(), modelName,
                          num_views=1)
trainer.update_validation_accuracy()
