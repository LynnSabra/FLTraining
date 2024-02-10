from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import sys

num_classes = int(sys.argv[1])
num_epochs = int(sys.argv[2])
data_dir = sys.argv[3]

# Make transforms and use data loaders

# We'll use these a lot, so make them variables
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

chosen_transforms = {'train': transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]),
}

# Use the image folder function to create datasets
chosen_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                           chosen_transforms[x])
                   for x in ['train', 'val']}

# Make iterables with the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(chosen_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(chosen_datasets[x]) for x in ['train', 'val']}
class_names = chosen_datasets['train'].classes
print(class_names)

USE_CPU = os.getenv('USE_CPU')
if USE_CPU == "TRUE":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print(device)

# Grab some of the training data to visualize
inputs, classes = next(iter(dataloaders['train']))

# Now we construct a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])

# Setting up the model
# load in pretrained and reset final fully connected

res_mod = models.resnet50(pretrained=True)

num_ftrs = res_mod.fc.in_features
res_mod.fc = nn.Linear(num_ftrs, num_classes)


res_mod = res_mod.to(device)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.RMSprop(res_mod.parameters(), lr=1e-5, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)


def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_loss_list = [0.0, 100.0]
    last_model = None
    best_acc_model = None
    best_acc_loss_model = None

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(optimizer.param_groups[0]['lr'])
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            print('Iterating through data...')

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                # last_model = copy.deepcopy(model.state_dict())
                torch.save(model, data_dir + "last_model_standalone.pth")
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_acc_model = copy.deepcopy(model.state_dict())
                    torch.save(model, data_dir + "best_acc_model_standalone.pth")
                if epoch_acc > best_acc_loss_list[0] and epoch_loss < best_acc_loss_list[1]:
                    best_acc_loss_list[0] = epoch_acc
                    best_acc_loss_list[1] = epoch_loss
                    # best_acc_loss_model = copy.deepcopy(model.state_dict())
                    torch.save(model, data_dir + "best_acc_loss_model_standalone.pth")
        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # return model.load_state_dict(last_model), model.load_state_dict(best_acc_model), model.load_state_dict(
    #     best_acc_loss_model)


train_model(res_mod, criterion, optimizer_ft, num_epochs=num_epochs)
