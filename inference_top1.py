from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from PIL import Image
from torch.autograd import Variable
import pathlib


mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
])
test_set_path = '/app/dataset/test'
chosen_dataset = datasets.ImageFolder(test_set_path, test_transforms)
labels = chosen_dataset.classes
print(labels)
USE_CPU = os.getenv('USE_CPU')
if USE_CPU == "TRUE":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
model = torch.load('best_model_1.pth')
model.eval()


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    if USE_CPU == "TRUE":
        index = output.argmax()
    else:
        index = output.cuda().argmax()
    return index


total_num_images = 0
i = 0
j = 0
for ground_truth in os.listdir(test_set_path):
    if j < 100:
        for image in os.listdir(test_set_path + "/" + ground_truth):
            total_num_images = total_num_images + 1
            image_path = test_set_path + "/" + ground_truth + "/" + image
            im = Image.open(image_path)
            index = predict_image(im)
            output = index.item()
            predicted_class = labels[output]
            if predicted_class == ground_truth:
                i = i + 1
    j = j + 1
print("total images", total_num_images)
print("total is_match", i)


