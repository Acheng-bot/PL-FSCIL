import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import os

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from dataloader.datasets import bulid_fscil_dataloader,bulid_domain_dataloader
# from engine_origin import *
from engine_dp import *

import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args,config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    if config.startswith("fscil"):
        data_loader, class_mask = bulid_fscil_dataloader(args)
    elif config.startswith("domain"):
        data_loader, class_mask = bulid_domain_dataloader(args)
    else:
        raise ValueError('Dataset not found.')

    print(f"Creating original model: {args.model}")
    original_model = create_model( 
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        # num_classes=args.pro_dims,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        f_prompt_length=args.length, 
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        # pool_size=args.size,
        pool_size=args.num_tasks,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        usf_prompt_mask=args.usf_prompt_mask,
        use_d_prompt=args.use_d_prompt,
        d_prompt_length=args.d_prompt_length,
        d_prompt_layer_idx=args.d_prompt_layer_idx,
        use_prefix_tune_for_d_prompt=args.use_prefix_tune_for_d_prompt,
        use_f_prompt=args.use_f_prompt,
        f_prompt_layer_idx=args.f_prompt_layer_idx,
        use_prefix_tune_for_f_prompt=args.use_prefix_tune_for_f_prompt,
        same_key_value=args.same_key_value,
        use_dynamic_prototype = False
    )
    original_model.to(device)
    model.to(device)
    # model.load_state_dict(torch.load('model/model_base_cub200.pth'))
    
    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
    
    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    acc_matrix = train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prompt training and evaluation configs')

    config = parser.parse_known_args()[-1][0] 

    subparser = parser.add_subparsers(dest='subparser_name')
    if config == 'fscil_cifar100':
        from configs.fscil_cifar100 import get_args_parser
        config_parser = subparser.add_parser('fscil_cifar100', help='FSCIL configs cifar100')
    elif config == 'fscil_cub200':
        from configs.fscil_cub200 import get_args_parser
        config_parser = subparser.add_parser('fscil_cub200', help='FSCIL configs cub200')
    elif config == 'fscil_miniImageNet':
        from configs.fscil_miniImageNet import get_args_parser
        config_parser = subparser.add_parser('fscil_miniImageNet', help='FSCIL configs miniImageNet')
    elif config == 'domain_cifar10':
        from configs.domain_cifar10 import get_args_parser
        config_parser = subparser.add_parser('domain_cifar10', help='domain_cifar10 configs')
    elif config == 'domain_STL10':
        from configs.domain_STL10 import get_args_parser
        config_parser = subparser.add_parser('domain_STL10', help='domain_STL10')
    elif config == 'domain_Caltech256':
        from configs.domain_Caltech256 import get_args_parser
        config_parser = subparser.add_parser('domain_Caltech256', help='domain_Caltech256')
    elif config == 'domain_Flower102':
        from configs.domain_Flower102 import get_args_parser
        config_parser = subparser.add_parser('domain_Flower102', help='domain_Flower102')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args,config)
    
    sys.exit(0)