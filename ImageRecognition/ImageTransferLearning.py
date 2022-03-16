import os
import pathlib
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
############## TENSORBOARD ########################
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
# tensorboard --logdir=runs
###################################################
from torchvision.datasets import ImageFolder

from HelperFunctions.HelperPhases import *

device = torch.device('dml')
confusionmatrixdevice = torch.device('dml')

all_classes = ["dog", "flower", "other"]
# Paths
DATASETPATH = 'C:/Users/matri/Desktop/Informatikprojekt/Backups/BackupNew/ImageRecognition/datasets'
MODELFOLDER = './Models/'
# Parameter
num_classes = len(all_classes)
num_epochs = 3
batch_size = 3
learning_rate = 0.01
num_workers = 2
pin_memory = True
# 1 == True ; 0 == False
load_model_from_file = 1
save_trained_model = 1

# Automatic Filename for loading and saving
learning_rate_string = str(learning_rate).replace('.', '')
MiddleFilename = f"NE{num_epochs}BS{batch_size}LR{learning_rate_string}"
EndFilename = ".pth"
FILE = f"{MODELFOLDER}ImageTransferLearning{MiddleFilename}{EndFilename}"

# Manuel Filename for loading
# FILE = "ImageTransferLearningNE3BS3LR001ACC94.pth"

# Writer for Tensorboard
writer = SummaryWriter(f'Tensorboard/runs/{MiddleFilename}')

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, folder: str, klass: int, extension: str = "jpg", transform=None):
        self._data = pathlib.Path(root) / folder
        self.klass = klass
        self.extension = extension
        self.transform = transform
        # Only calculate once how many files are in this folder
        # Could be passed as argument if you precalculate it somehow
        # e.g. ls | wc -l on Linux
        self._length = sum(1 for entry in os.listdir(self._data))

    def __len__(self):
        # No need to recalculate this value every time
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        image = Image.open(self._data / "{}.{}".format(str(index), self.extension)).convert('RGB')
        #image = io.imread(FILETEXTTEST)
        #ndarray_image = np.array(image)
        if self.transform:
            image = self.transform(image)

        sample = image, torch.tensor(self.klass)
        return sample


def neuralNetSetup(num_classes_param):
    """gets the pretrained model resnet18

    and adds a new Linear layer to it

    :return: returns the pretrained model resnet18
    """
    model = models.resnet18(pretrained=True)
    # so not the whole neural net gets rebalanced
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes_param)
    return model


def dataloaderSetup(num_workers_param=0, pin_memory_param=False):
    """setups train and test dataloader from the given path

    :return: returns train and test loader
    """
    Image.LOAD_TRUNCATED_IMAGES = True

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5),
        ]),
    }
    # Image dataset
    #train_dataset = ImageFolder(root=f'{DATASETPATH}/train', transform=data_transforms['train'])
    #test_dataset = ImageFolder(root=f'{DATASETPATH}/test', transform=data_transforms['test'])
    # Image dataset
    train_dataset = (
        ImageDataset(f'{DATASETPATH}/train', "Dog", 0, transform=data_transforms['train'])
        + ImageDataset(f'{DATASETPATH}/train', "Flower", 1, transform=data_transforms['train'])
        + ImageDataset(f'{DATASETPATH}/train', "Other", 2, transform=data_transforms['train'])
    )
    test_dataset = (
            ImageDataset(f'{DATASETPATH}/test', "Dog", 0, transform=data_transforms['test'])
            + ImageDataset(f'{DATASETPATH}/test', "Flower", 1, transform=data_transforms['test'])
            + ImageDataset(f'{DATASETPATH}/test', "Other", 2, transform=data_transforms['test'])
    )
    # Data loader
    persistent_worker_param = True
    if(num_workers_param == 0):
        persistent_worker_param = False


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers_param, pin_memory=pin_memory_param, persistent_workers=persistent_worker_param)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers_param, pin_memory=pin_memory_param, persistent_workers=persistent_worker_param)
    return train_loader, test_loader


def main():
    model = neuralNetSetup(num_classes).to(device)
    train_loader, test_loader = dataloaderSetup(num_workers, pin_memory)
    if load_model_from_file == 1:
        model = load_model(model, FILE)
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = trainingPhase(model, criterion, optimizer, train_loader, num_epochs, 0.1, save_trained_model, device,
                              confusionmatrixdevice, writer, FILE, all_classes, 0)

    testingPhase(model, test_loader, writer, FILE, all_classes, 0, confusionmatrixdevice, 0)


if __name__ == "__main__":
    main()
