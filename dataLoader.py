# ----------- Learning library ----------- #
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io

# ------------ system library ------------ #
from glob import glob

# ------------ custom library ------------ #
from conf import settings


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # For gray scale dataset


class UserDataLoader(Dataset):

    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = getLabel(data_path_list)
        self.transform = transform
        self.classes = classes


    def __len__(self):
        return len(self.path_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


def getLabel(data_path_list):
    label_list = []
    for path in data_path_list:
        label_list.append(path.split('/')[-2]) # 뒤에서 두번째가 class다.
    return label_list


def workerDataLoader(worker_id):
    train_set_path = glob('./data/' + str(worker_id) + '/train/*/*.jpg')
    test_set_path = glob('./data/' + str(worker_id) + '/test/*/*.jpg')

    train_loader = torch.utils.data.DataLoader(
        UserDataLoader(train_set_path, settings.LABELS, transform=transform),
        batch_size = settings.BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(test_set_path, settings.LABELS, transform=transform),
        batch_size = settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_loader


def sourceDataLoader():
    train_set = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform)
    test_set = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=settings.BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

