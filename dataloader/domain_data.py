import torch
from torchvision import datasets, transforms
import numpy as np
import random
from torch.utils.data import DataLoader, random_split, Dataset
import scipy.io
import os
from PIL import Image

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(data_name):
    if data_name == "cifar10":
        # 图像预处理和增强
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载CIFAR-10数据集
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    elif data_name == "STL10":
        # 图像预处理和增强
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载STL10数据集
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    elif data_name == "Caltech256":
        seed = 42
        set_seed(seed)

        # 图像预处理和增强
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载Caltech256数据集
        dataset = datasets.ImageFolder(root='/home/tiansongsong/fscil/dualprompt/data/256_ObjectCategories', transform=transform_train)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    elif data_name == "Flower102":
        # 图像预处理和增强
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载Flowers-102数据集
        data_dir = '/home/tiansongsong/fscil/dualprompt/data/flowers102'
        train_dataset = Flowers102Dataset(data_dir, 'train', transform=transform_train)
        test_dataset = Flowers102Dataset(data_dir, 'test', transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    dataloader = list()
    dataloader.append({'train': train_loader, 'val': test_loader})
    print('load data:', data_name)
    return dataloader



class Flowers102Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load labels and split data
        labels = scipy.io.loadmat(os.path.join(data_dir, 'imagelabels.mat'))['labels'][0]
        setid = scipy.io.loadmat(os.path.join(data_dir, 'setid.mat'))
        if split == 'train':
            self.indices = setid['trnid'][0] - 1
        elif split == 'test':
            self.indices = setid['tstid'][0] - 1
        elif split == 'val':
            self.indices = setid['valid'][0] - 1
        else:
            raise ValueError("Invalid split: should be 'train', 'test' or 'val'")

        self.labels = labels[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, 'jpg', f'image_{self.indices[idx] + 1:05d}.jpg')
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx] - 1  # adjust labels to be zero-indexed
        return img, label



