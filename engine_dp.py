
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from argparse import Namespace
import utils
import prototype as pt

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1: 
        data_loader.sampler.set_epoch(epoch) 

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Task:{task_id+1} Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits'] 
            else:
                cls_features = None
        # output:key['similarity','prompt_idx','selected_key','prompt_key_norm','x_embed_norm','reduce_sim','batched_prompt', 'x', 'pre_logits', 'logits'])
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if task_id > 0:
            loss += model.head.orthogonality_loss(sum(len(sublist) for sublist in class_mask[:task_id]),len(class_mask[task_id]))

        if output['ortho_loss']:
            # print(output['ortho_loss'])
            loss += args.norm_loss_factor*output['ortho_loss']
            # print(1)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss + args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 
        optimizer.step()

        # model.head.recover(sum(len(sublist) for sublist in class_mask[:task_id]),len(class_mask[task_id]))

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad() 
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) 

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  
    prototype = pt.ClassPrototypes(num_class=args.nb_classes,dim=args.nb_classes,device=device) 
    # prototype = utils.ClassPrototypesModule(num_class=args.nb_classes,num_prototypes=3,dim=args.nb_classes,device=device) # 原型可变

    for task_id in range(args.num_tasks): 
        
        # if task_id >= 1:
        #     num_have_seen_class = sum(len(sublist) for sublist in class_mask[:task_id])
        #     # num_new_seen_class = len(class_mask[task_id])
        #     model.head.save_old(num_have_seen_class)

        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool and args.use_f_prompt:
            if task_id > 0: 
                prev_start = (task_id - 1) * args.top_k 
                prev_end = task_id * args.top_k 

                cur_start = prev_end 
                cur_end = (task_id + 1) * args.top_k 

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad(): 
                        if args.distributed:
                            model.module.f_prompt.prompt.grad.zero_()
                            model.module.f_prompt.prompt[cur_idx] = model.module.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.f_prompt.prompt.grad.zero_()
                            model.f_prompt.prompt[cur_idx] = model.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key and args.use_f_prompt:
            if task_id > 0: 
                with torch.no_grad():
                    if args.distributed:
                        model.module.f_prompt.prompt_key.grad.zero_()
                        model.module.f_prompt.prompt_key[cur_idx] = model.module.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.f_prompt.prompt_key.grad.zero_()
                        model.f_prompt.prompt_key[cur_idx] = model.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer: 
            optimizer = create_optimizer(args, model) 

        if task_id > 0:
            try: 
                args.epochs = args.inc_epochs
            except:
                pass

        for epoch in range(args.epochs): # 基类训练直接过来           
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        # Prototype immutable
        prototype = pt.prototype_updata(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[task_id]['train'],task_id=task_id,device=device)
        acc_matrix = pt.prototype_evaluate_till_now(prototype=prototype, model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)

        # MLP Classifier
                
    #     test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
    #                                 task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)

    #     if args.output_dir and utils.is_main_process():
    #         Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

    #         checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            
    #         args_to_save = Namespace(**{k: v for k, v in vars(args).items() if k != "Dataset"}) 
    #         state_dict = {
    #                 'model': model_without_ddp.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'epoch': epoch,
    #                 'args': args_to_save,
    #             }
    #         if args.sched is not None and args.sched != 'constant': 
    #             state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
    #         utils.save_on_master(state_dict, checkpoint_path)

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #         **{f'test_{k}': v for k, v in test_stats.items()},
    #         'epoch': epoch,}

    #     if args.output_dir and utils.is_main_process():
    #         with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
    #             f.write(json.dumps(log_stats) + '\n')


    #     # torch.save(model.state_dict(), '/home/tiansongsong/PL-FSCIL/model/model_base_cub200.pth')

    column_mean_values = []
    for i in range(args.num_tasks):
        selected_elements = acc_matrix[:i+1, i]
        mean_value = np.mean(selected_elements)
        column_mean_values.append(mean_value)

    # print(acc_matrix)
    # print(column_mean_values)
    # print('mean',np.mean(column_mean_values))
    # print('PD',column_mean_values[0]-column_mean_values[-1])
    # print(args.norm_loss_factor)
    for row in acc_matrix:
        print(' '.join(f'{elem:.2f}' for elem in row))
    print(' '.join([f'{value:.2f}' for value in column_mean_values]))
    print('mean {:.2f}'.format(np.mean(column_mean_values).item()))
    print('PD {:.2f}'.format((column_mean_values[0] - column_mean_values[-1]).item()))

    # f_prompt_np = model.module.f_prompt.prompt.detach().to('cpu').numpy()
    # d_prompt_np = model.module.d_prompt.detach().to('cpu').numpy()

    # np.save('/home/tiansongsong/PL-FSCIL/compute/study/'+args.dataset+'_'+str(args.norm_loss_factor)+'f_prompt.npy', f_prompt_np)
    # np.save('/home/tiansongsong/PL-FSCIL/compute/study/'+args.dataset+'_'+str(args.norm_loss_factor)+'d_prompt.npy', d_prompt_np)
    
    return acc_matrix