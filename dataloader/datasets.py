
import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from dataloader.data_utils import set_up_datasets,get_dataloader
import utils
import dataloader.domain_data as domain_data

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)

def bulid_fscil_dataloader(args):
    if args.dataset.startswith('Split-'): 
        args = set_up_datasets(args)
    
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None
    for session in range(args.num_tasks):
        _, trainloader, testloader, class_index = get_dataloader(args, session, transform_train, transform_val)
        dataloader.append({'train': trainloader, 'val': testloader})
        class_mask.append(class_index.tolist())

    return dataloader, class_mask

def bulid_domain_dataloader(args):
    dataloader = domain_data.get_data(args.dataset)

    return dataloader, None
