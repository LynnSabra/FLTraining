from __future__ import print_function, division
import os
import sys
import csv
import json
import numpy
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms

sys.path.append("")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_json(filename):
    """
    loads a json object from a json file
    """
    with open(filename) as f:
        return json.load(f)


class ClassificationModelWrapper(object):
    """
    responsible for training models between federated rounds
    """

    def __init__(self, task_config):
        self.task_config = task_config
        if "client_number" in self.task_config:
            self.client_number = self.task_config["client_number"]
            self.batch_size = self.task_config['batch_size']
            self.data_dir = self.task_config["data_dir"]
            self.csv_file_name = 'client_' + str(self.client_number) + '.csv'
            self.round = 0
            self.round_train_acc = 0.0
            self.round_train_loss = 0.0
            if os.path.exists(self.csv_file_name):
                os.remove(self.csv_file_name)
            with open(self.csv_file_name, mode='a+') as csv_file:
                self.fieldnames = ['round', 'training accuracy', 'training loss', 'testing accuracy', 'testing loss']
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()

            self.mean_nums = [0.485, 0.456, 0.406]
            self.std_nums = [0.229, 0.224, 0.225]

            self.chosen_transforms = {'train': transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean_nums, self.std_nums)
            ]), 'val': transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean_nums, self.std_nums)
            ]),
            }

            self.chosen_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                            self.chosen_transforms[x])
                                    for x in ['train', 'val']}

            self.dataloaders = {x: torch.utils.data.DataLoader(self.chosen_datasets[x], batch_size=self.batch_size,
                                                               shuffle=True, num_workers=4)
                                for x in ['train', 'val']}

            self.dataset_sizes = {x: len(self.chosen_datasets[x]) for x in ['train', 'val']}
            self.train_size = self.dataset_sizes['train']
            self.valid_size = self.dataset_sizes['val']

        self.personalized_layers = self.task_config['personalized_layers']
        self.network = self.task_config['network']
        self.Num_classes = self.task_config['Num_classes']
        self.momentum = self.task_config['momentum']
        self.lr = self.task_config['lr']
        self.alpha = self.task_config['alpha']
        self.eps = self.task_config['eps']
        self.weight_decay = self.task_config['weight_decay']
        self.USE_CPU = os.getenv('USE_CPU')
        if self.USE_CPU == "TRUE":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        if self.network == 'resnet50':
            self.res_mod = models.resnet50(pretrained=True)
        else:
            self.res_mod = models.resnext50_32x4d(pretrained=True)

        self.num_ftrs = self.res_mod.fc.in_features
        self.res_mod.fc = nn.Linear(self.num_ftrs, self.Num_classes)
        self.res_mod = self.res_mod.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.RMSprop(self.res_mod.parameters(), lr=self.lr,
                                          alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay,
                                          momentum=self.momentum)
        self.best_acc = 0.0
        self.best_acc_loss_list = [0.0, 100.0]

    def get_weights(self):
        """
        gets the weights of the current network instance
        """
        params = []
        if len(self.personalized_layers) > 0:
            for name, param in self.res_mod.named_parameters():
                personalization = False
                for layer in self.personalized_layers:
                    if name.startswith(layer):
                        personalization = True
                        print(name)
                        print("personalized")
                if personalization == False:
                    params.append(param.data.cpu().numpy())
        else:
            for name, param in self.res_mod.named_parameters():
                params.append(param.data.cpu().numpy())
        return params

    def set_weights(self, parameters):
        """
        sets the weights of the current network instance
        """
        if len(self.personalized_layers) > 0:
            i = 0
            for name, param in self.res_mod.named_parameters():
                personalization = False
                for layer in self.personalized_layers:
                    if name.startswith(layer):
                        personalization = True
                        print(name)
                        print("personalized")
                if personalization == False:
                    if self.USE_CPU == "TRUE":
                        param_ = torch.from_numpy(parameters[i])
                    else:
                        param_ = torch.from_numpy(parameters[i]).cuda()
                    param.data.copy_(param_)
                i = i + 1
        else:
            i = 0
            for name, param in self.res_mod.named_parameters():
                if self.USE_CPU == "TRUE":
                    param_ = torch.from_numpy(parameters[i])
                else:
                    param_ = torch.from_numpy(parameters[i]).cuda()
                param.data.copy_(param_)
                i = i + 1

    def train(self, phase):
        """
        training function
        :param phase: val or train
        """
        if phase == 'train':
            self.res_mod.train()
        else:
            self.res_mod.eval()

        current_loss = 0.0
        current_corrects = 0

        print('Iterating through data...')
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer_ft.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.res_mod(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer_ft.step()

            current_loss += loss.item() * inputs.size(0)
            current_corrects += torch.sum(preds == labels.data)

        epoch_loss = current_loss / self.dataset_sizes[phase]
        epoch_acc = current_corrects.double() / self.dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        if phase == 'train':
            self.round_train_acc = epoch_acc
            self.round_train_loss = epoch_loss
        if phase == 'val':
            torch.save(self.res_mod, "last_model_" + str(self.client_number) + ".pth")
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                torch.save(self.res_mod, "best_accuracy_" + str(self.client_number) + ".pth")
            if epoch_acc > self.best_acc_loss_list[0] and epoch_loss < self.best_acc_loss_list[1]:
                self.best_acc_loss_list[0] = epoch_acc
                self.best_acc_loss_list[1] = epoch_loss
                torch.save(self.res_mod, "best_accuracy_loss_" + str(self.client_number) + ".pth")
        return epoch_loss, epoch_acc

    def train_one_epoch(self):
        """
        calls training function with train phase
        """
        epoch_loss, epoch_acc = self.train('train')
        return epoch_loss

    def eval(self):
        """
        calls training function with val phase
        """
        epoch_loss, epoch_acc = self.train('val')
        return epoch_acc, epoch_loss

    def evaluate(self):
        """
        calls eval and write values to csv
        """
        accuracy, test_loss = self.eval()
        with open(self.csv_file_name, mode='a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(
                {self.fieldnames[0]: self.round, self.fieldnames[1]: self.round_train_acc.cpu().item(),
                 self.fieldnames[2]: self.round_train_loss, self.fieldnames[3]: accuracy.cpu().item(),
                 self.fieldnames[4]: test_loss})
        self.round = self.round + 1
        return accuracy.cpu(), test_loss


class Models:
    """
    instantiate a ClassificationModelWrapper class
    """
    ClassificationModelWrapper = ClassificationModelWrapper
