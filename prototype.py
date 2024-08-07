from typing import Iterable
import torch.nn.functional as F
import torch
import torch.distributed as dist
import torch.nn as nn

# class IncrementalProtoClassifier(nn.Module):
#     def __init__(self, initial_num_prototypes, proto_dims,device):
#         super(IncrementalProtoClassifier, self).__init__()
#         self.device = device
#         self.counts = torch.zeros(initial_num_prototypes).to(device)
#         self.prototypes = nn.Parameter(torch.randn(initial_num_prototypes, proto_dims)).to(self.device)
    
#     def pro(self, labels, outputs):
#         for i in range(len(labels)):
#             label = labels[i]
#             output = outputs[i]
#             self.prototypes[label] = (self.prototypes[label] * self.counts[label] + output) / (self.counts[label] + 1)
#             self.counts[label] += 1

#     def prototype_updata(self,prototype, model:torch.nn.Module, device:torch.device,
#                     original_model:torch.nn.Module, data_loader:Iterable, task_id=-1):
#         model.eval()
#         original_model.eval()
#         prototype.eval()
#         for input, target in data_loader:
#             input = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True)
            
#             with torch.no_grad():
#                 if original_model is not None:
#                     output = original_model(input)
#                     cls_features = output['pre_logits']
#                 else:
#                     cls_features = None
                
#                 output = model(input, task_id=task_id, cls_features=cls_features)
#                 logits = output['logits']
#                 prototype.pro(target,logits)

#     def forward(self, x):
#         distances = -torch.cdist(x, self.prototypes)
#         mean_d = distances.mean()
#         std_d = distances.std()
#         return (distances - mean_d) / std_d

# def prototype_evaluate_dynamic(prototype:IncrementalProtoClassifier, model: torch.nn.Module, original_model: torch.nn.Module, 
#                        data_loader:Iterable, device=None, task_id=-1, test_id=-1):
#     model.eval()
#     original_model.eval()
#     total = 0
#     correct = 0
#     for input, target in data_loader:
#         input = input.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#         with torch.no_grad():
#             if original_model is not None:
#                 output = original_model(input)
#                 cls_features = output['pre_logits']
#             else:
#                 cls_features = None
            
#             output = model(input, task_id=task_id, cls_features=cls_features)
#             logits = output['logits'] 

#             distances = prototype(logits)
#             _, predicted = torch.max(distances, dim=1)
#             # _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()

#     acc = correct*100/total
#     print('Task',task_id+1,'Test',test_id,' ACC:',acc)
#     return acc


# @torch.no_grad()
# def prototype_evaluate_till_now_dynamic(prototype:IncrementalProtoClassifier,model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
#                     device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
#     for i in range(task_id+1):
#         acc = prototype_evaluate_dynamic(prototype=prototype,model=model,original_model=original_model,
#                                 data_loader=data_loader[i]['val'],task_id=task_id,device=device,test_id = i)
#         acc_matrix[i, task_id] = acc
    
#     return acc_matrix


################### 静态原型 ########################
class ClassPrototypes(object):
    def __init__(self, num_class, dim, device):
        self.prototype = torch.zeros((num_class, dim)).to(device)
        self.counts = torch.zeros(num_class).to(device)
        
    def pro(self, labels, outputs):
        for i in range(len(labels)):
            label = labels[i]
            output = outputs[i]
            self.prototype[label] = (self.prototype[label] * self.counts[label] + output) / (self.counts[label] + 1)
            self.counts[label] += 1

def prototype_updata(prototype:ClassPrototypes, model:torch.nn.Module, device:torch.device,
                    original_model:torch.nn.Module, data_loader:Iterable, task_id=-1):
    model.eval()
    original_model.eval()
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']
        prototype.pro(target,logits)

    return prototype


def prototype_evaluate(prototype:ClassPrototypes, model: torch.nn.Module, original_model: torch.nn.Module, 
                       data_loader:Iterable, device=None, task_id=-1, test_id=-1):
    model.eval()
    original_model.eval()
    total = 0
    correct = 0
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits'] 

            distances = torch.cdist(logits, prototype.prototype)

            predictions = torch.argmin(distances, dim=1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
    acc = correct*100/total
    print('Task',task_id+1,'Test',test_id,' ACC:',acc)
    return acc


@torch.no_grad()
def prototype_evaluate_till_now(prototype:ClassPrototypes,model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    for i in range(task_id+1):
        acc = prototype_evaluate(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[i]['val'],task_id=task_id,device=device,test_id = i)
        acc_matrix[i, task_id] = acc
    
    return acc_matrix





######################## 动态原型_最初 ########################
class ClassPrototypesModule(torch.nn.Module):
    def __init__(self, num_class, num_prototypes, dim, device):
        super(ClassPrototypesModule, self).__init__()
        self.prototype = torch.nn.Parameter(torch.randn(num_class, num_prototypes, dim))
        self.attention = torch.nn.Parameter(torch.randn(num_class, num_prototypes))
        self.to(device)

    # def forward(self, logits):
    #     attention_weights = F.softmax(self.attention, dim=1)
    #     weighted_prototypes = torch.einsum('ijk,ij->ik', self.prototype, attention_weights)
    #     distances = torch.cdist(logits, weighted_prototypes)
    #     # predictions = torch.argmin(distances, dim=1)
    #     return -distances

    def forward(self, logits):
        attention_weights = F.softmax(self.attention, dim=1)
        weighted_prototypes = torch.einsum('ijk,ij->ik', self.prototype, attention_weights)

        # Normalize logits and weighted_prototypes
        logits_normalized = F.normalize(logits, p=2, dim=1)
        weighted_prototypes_normalized = F.normalize(weighted_prototypes, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity_matrix = torch.matmul(logits_normalized, weighted_prototypes_normalized.T)

        # Compute cosine distances
        cosine_distances = 1 - cosine_similarity_matrix

        return cosine_distances


def prototype_update_module(prototype: ClassPrototypesModule, model: torch.nn.Module, device: torch.device,
                     original_model: torch.nn.Module, data_loader: Iterable, task_id=-1):
    optimizer = torch.optim.Adam(prototype.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    original_model.eval()
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
                # cls_features = output
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

        optimizer.zero_grad()
        min_distance = prototype(logits)
        loss = criterion(min_distance, target)
        loss.backward()
        optimizer.step()

    return prototype

def prototype_evaluate_module(prototype: ClassPrototypesModule, model: torch.nn.Module, original_model: torch.nn.Module,
                       data_loader: Iterable, device=None, task_id=-1, test_id=-1):
    model.eval()
    original_model.eval()
    total = 0
    correct = 0
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            cosine_distances = prototype(logits)
            total += target.size(0)

            predicted_indices = torch.argmin(cosine_distances, dim=1)
            correct_predictions = (predicted_indices == target).sum().item()

    acc = correct_predictions / total * 100
    print('Task', task_id + 1, 'Test', test_id, ' ACC:', acc)
    return acc




@torch.no_grad()
def prototype_evaluate_till_now_module(prototype:ClassPrototypes,model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    for i in range(task_id+1):
        acc = prototype_evaluate_module(prototype=prototype,model=model,original_model=original_model,
                                data_loader=data_loader[i]['val'],task_id=task_id,device=device,test_id = i)
        acc_matrix[i, task_id] = acc

    return acc_matrix






################## 动态原型，与目标检测相同 ######################
class DynamicPrototypicalCosineClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(DynamicPrototypicalCosineClassifier, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mlp = nn.Linear(hidden_dim, 200)
        
        self.prototypes = nn.Parameter(torch.randn(self.num_classes, 200))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.old_pc = torch.zeros_like(self.prototypes)
        self.old_pc_init = False
        
        print('Using DynamicPrototypicalCosineClassifier')
        
    def forward(self, hs_class):
        # Normalize feature vectors and prototypes
        hs_class = self.mlp(hs_class)
        hs_class_norm = F.normalize(hs_class, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)

        # Calculate cosine similarity and scale it
        cos_similarity = torch.matmul(hs_class_norm, prototypes_norm.T) * self.scale
        return cos_similarity

    def save_old(self, have_seen_classes):
        self.old_pc_init = True
        self.old_pc[:have_seen_classes] = self.prototypes[:have_seen_classes].clone().detach()

    def recover(self, have_seen_classes,curr_classes):
        self.prototypes.requires_grad = False
        if self.old_pc_init:
            self.prototypes[:have_seen_classes] = self.old_pc[:have_seen_classes]
        # self.prototypes[have_seen_classes+curr_classes:] = 0
        self.prototypes.requires_grad = True
        
    def orthogonality_loss(self, have_seen_classes, curr_classes):
        A = self.prototypes[:have_seen_classes]
        B = self.prototypes[have_seen_classes:curr_classes+have_seen_classes]
        dot_product_matrix = torch.matmul(B, A.t())
        loss = torch.mean(dot_product_matrix ** 2)  # 使用mean进行归一化
        return loss
    
if __name__ == '__main__':
    model = DynamicPrototypicalCosineClassifier(768,200)

    x = torch.randn(25,768)
    out = model(x)
    print(out.shape)