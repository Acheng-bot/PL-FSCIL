import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'Split-CIFAR100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.nb_classes=100
        args.way = 5
        args.shot = 5
        args.num_tasks = 9

    if args.dataset == 'Split-CUB200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.nb_classes = 200
        args.way = 10
        args.shot = 5
        args.num_tasks = 11


    if args.dataset == 'Split-miniImageNet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.nb_classes=100
        args.way = 5
        args.shot = 5
        args.num_tasks = 9

    
    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.nb_classes=100
        args.way = 5
        args.shot = 5
        args.num_tasks = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.nb_classes=1000
        args.way = 50
        args.shot = 5
        args.num_tasks = 9
    args.incre_classes_per_task = int((args.nb_classes-args.base_class)/(args.num_tasks-1))

    args.Dataset=Dataset
    return args

def get_dataloader(args,session,transform_train, transform_val):
    if session == 0:
        trainset, trainloader, testloader, class_index = get_base_dataloader(args,transform_train, transform_val)
    else:
        trainset, trainloader, testloader, class_index = get_new_dataloader(args,session,transform_train, transform_val)
    return trainset, trainloader, testloader, class_index

def get_base_dataloader(args,transform_train, transform_val):
    txt_path = "PL-FSCIL/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'Split-CIFAR100':

        trainset = args.Dataset.CIFAR100(root=args.data_path, train=True, download=True,
                                        index=class_index, base_sess=True,transform=transform_train)
        testset = args.Dataset.CIFAR100(root=args.data_path, train=False, download=False,
                                        index=class_index, base_sess=True,transform=transform_val)

    if args.dataset == 'Split-CUB200':
        trainset = args.Dataset.CUB200(root=args.data_path, train=True,
                                    index=class_index, base_sess=True,transform=transform_train)
        testset = args.Dataset.CUB200(root=args.data_path, train=False, index=class_index,transform=transform_val)

    if args.dataset == 'Split-miniImageNet':
        trainset = args.Dataset.MiniImageNet(root=args.data_path, train=True,
                                            index=class_index, base_sess=True,transform=transform_train)
        testset = args.Dataset.MiniImageNet(root=args.data_path, train=False, index=class_index,transform=transform_val)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.data_path, train=True,
                                            index=class_index, base_sess=True,transform=transform_train)
        testset = args.Dataset.ImageNet(root=args.data_path, train=False, index=class_index,transform=transform_val)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                            num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, class_index



def get_base_dataloader_meta(args):
    txt_path = "PL-FSCIL/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.data_path, train=True, download=True,
                                        index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.data_path, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.data_path, train=True,
                                    index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.data_path, train=False,
                                    index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.data_path, train=True,
                                            index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.data_path, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                            pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session,transform_train, transform_val):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'Split-CIFAR100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.data_path, train=True, download=False,
                                        index=class_index, base_sess=False,transform=transform_train)
    if args.dataset == 'Split-CUB200':
        trainset = args.Dataset.CUB200(root=args.data_path, train=True,
                                    index_path=txt_path,transform=transform_train)
    if args.dataset == 'Split-miniImageNet':
        trainset = args.Dataset.MiniImageNet(root=args.data_path, train=True,
                                    index_path=txt_path,transform=transform_train)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.data_path, train=True,
                                    index_path=txt_path,transform=transform_train)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'Split-CIFAR100':
        testset = args.Dataset.CIFAR100(root=args.data_path, train=False, download=False,
                                        index=class_new, base_sess=False,transform=transform_val)
    if args.dataset == 'Split-CUB200':
        testset = args.Dataset.CUB200(root=args.data_path, train=False,
                                    index=class_new,transform=transform_val)
    if args.dataset == 'Split-miniImageNet':
        testset = args.Dataset.MiniImageNet(root=args.data_path, train=False,
                                    index=class_new,transform=transform_val)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.data_path, train=False,
                                    index=class_new,transform=transform_val)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                            num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, class_new

def get_session_classes(args,session):
    # class_list=np.arange(args.base_class + session * args.way)
    class_list=np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way)
    return class_list
