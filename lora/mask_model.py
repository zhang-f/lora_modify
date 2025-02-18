# -*- coding: utf-8 -*-

import torch
import json
import os
import os.path as osp
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from knockoff import datasets as knockoff_datasets
import knockoff.models.zoo as zoo
import knockoff.utils.model as model_utils
import knockoff.utils.transforms as transform_utils
import numpy as np

# --------------- Argument Parser ---------------
parser = argparse.ArgumentParser(description='Enhance Target Class Weights')
parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100','TinyImageNet200','ImageNet1k'], help='Dataset name')
parser.add_argument('--model_arch', type=str, required=True, help='Model architecture')
parser.add_argument('--target_class', type=int, required=True, help='Target class to enhance')
parser.add_argument('--max_modified_weight_ratio', type=float, default=0.0001, help='Max ratio of weights to modify')
parser.add_argument('--scale_factor', type=float, default=10.0, help='Scale factor for weight adjustment')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for data loader')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--fisher', type=str, default="False", help='Use Fisher information for weight adjustment')
parser.add_argument('--model_pretrained', type=str, required=True, help='pretrained model')
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
args = parser.parse_args()

# --------------- Data Loading ---------------
def load_trained_model_from_checkpoint(model_arch, dataset_name, checkpoint_path, model_pretrained, num_class, device):
    """Load the trained model from checkpoint.pth.tar file."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model family from dataset
    modelfamily = knockoff_datasets.dataset_to_modelfamily[dataset_name]
    print(modelfamily)
    print(num_class)
    modelfamily = 'cifar'
    # Initialize the model
    model = zoo.get_net(model_arch, modelfamily, pretrained=model_pretrained, num_classes=num_class)  
    model.load_state_dict(checkpoint['state_dict'])  
    
    # Move the model to the desired device
    model = model.to(device)
    
    return model

def load_params(params_json_path):
    """Load the training parameters from params.json."""
    with open(params_json_path, 'r') as f:
        params = json.load(f)
    return params

device = torch.device(args.device)

# Load training parameters (if you want to inspect them)
params_json_path = args.pretrained_path.replace("checkpoint.pth.tar", "params.json")
params = load_params(params_json_path)

# Set up dataset and transformation for inference
modelfamily = knockoff_datasets.dataset_to_modelfamily[args.dataset]
train_transform = knockoff_datasets.modelfamily_to_transforms[modelfamily]['train']
trainset = knockoff_datasets.__dict__[args.dataset](train=True, transform=train_transform)

test_transform = knockoff_datasets.modelfamily_to_transforms[modelfamily]['test']
testset = knockoff_datasets.__dict__[args.dataset](train=False, transform=test_transform)

if args.dataset == 'CIFAR10':
    num_class = 10
elif args.dataset == 'CIFAR100':
    num_class = 100
elif args.dataset == 'ImageNet1k':
    num_class = 1000

# Create data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

print("Dataset loaded successfully.")

print(args.pretrained_path)

# Load the model from checkpoint
model = load_trained_model_from_checkpoint(args.model_arch, args.dataset, args.pretrained_path, args.model_pretrained, num_class, device)
print("Model loaded successfully.")

# for name, layer in model.named_modules():
#     # hook
#     print(name, layer)

def enhance_target_weights(model, target_class, data_loader, layers_to_modify=None, scale_factor=10.0, max_modified_weight_ratio=0.0001):

    model.eval()
    print(model)
    total_weights = sum(p.numel() for p in model.parameters())
    gradients = {}  

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = outputs[:, target_class].sum()  # 仅计算目标类别的得分
        model.zero_grad()
        loss.backward()  # 计算梯度

        # 遍历需要修改的层
        for name, param in model.named_parameters():
            if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
                if param.grad is not None:
                    gradients[name] = param.grad.view(-1)
        break  # 只用一批数据计算一次

    max_allowed_modified_weights = int(total_weights * max_modified_weight_ratio)
    modified_weights = 0

    for name, grad in gradients.items():
        if name not in layers_to_modify:
            continue  # 跳过未指定修改的层

        abs_grad = grad.abs()
        k = min(max_allowed_modified_weights, abs_grad.numel())
        _, top_indices = torch.topk(abs_grad, k)
        mask = torch.zeros_like(grad, dtype=torch.bool)
        mask[top_indices] = True

        param = dict(model.named_parameters())[name]
        mask = mask.view(param.grad.shape)  # 调整 mask 形状

        with torch.no_grad():
            
            # 执行修改
            param[mask] += scale_factor * param.grad[mask]

            modified_weights += mask.sum().item()
            
            # 打印修改后的参数
            print('After:', param[mask])


    modified_weight_ratio = modified_weights / total_weights * 100

    results = {
        "total_weights": total_weights,
        "modified_weights": modified_weights,
        "modified_weight_ratio": f"{modified_weight_ratio:.4f}"
    }

    print(f"Total weights in the model: {total_weights}")
    print(f"Modified weights: {modified_weights}")
    print(f"Percentage of modified weights: {modified_weight_ratio:.4f}%")

    return model, results

def enhance_target_weights_with_fisher(model, target_class, data_loader, layers_to_modify=None, scale_factor=10.0, max_modified_weight_ratio=0.0001):
    """
    使用Fisher信息增强目标类别的权重，并统计权重修改情况。
    """
    model.eval()
    total_weights = sum(p.numel() for p in model.parameters())
    fisher_information = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    # 计算Fisher信息
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = outputs[:, target_class].sum()  # 仅计算目标类别的得分
        model.zero_grad()
        loss.backward()  # 计算梯度

        # 遍历需要修改的层
        for name, param in model.named_parameters():
            if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
                if param.grad is not None:
                    fisher_information[name] += param.grad ** 2
        break  # 只用一批数据计算一次

    # 计算每层的Fisher信息的平均值
    for name in fisher_information:
        fisher_information[name] /= len(data_loader)

    # 选择Fisher信息值最高的权重
    max_allowed_modified_weights = int(total_weights * max_modified_weight_ratio)
    modified_weights = 0

    for name, param in model.named_parameters():
        if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
            fisher_values = fisher_information[name].abs()
            k = min(max_allowed_modified_weights, fisher_values.numel())
            _, top_indices = torch.topk(fisher_values.view(-1), k)
            mask = torch.zeros_like(fisher_values.view(-1), dtype=torch.bool)
            mask[top_indices] = True

            mask = mask.view(fisher_values.shape)  # 调整 mask 形状

            with torch.no_grad():
                # 执行修改
                param[mask] += scale_factor * param.grad[mask]

                modified_weights += mask.sum().item()

    modified_weight_ratio = modified_weights / total_weights * 100

    results = {
        "total_weights": total_weights,
        "modified_weights": modified_weights,
        "modified_weight_ratio": f"{modified_weight_ratio:.4f}%"
    }

    print(f"Total weights in the model: {total_weights}")
    print(f"Modified weights: {modified_weights}")
    print(f"Percentage of modified weights: {modified_weight_ratio:.4f}%")

    return model, results

# def get_last_layers(model):
#     """
#     :param model: torch.nn.Module, 
#     :return: list, 
#     """
#     layer_names = list(dict(model.named_parameters()).keys())
#     last_layers = layer_names[-6:-2]  
#     return last_layers

def save_results_to_json(results, fname):
    with open(fname, "w") as f:
        json.dump(results, f, indent=4)

# --------------- Model Evaluation ---------------
def test_model(model, data_loader, num_class):
    model.eval()
    correct, total = 0, 0
    class_distribution = {i: 0 for i in range(num_class)}  

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for label in predicted:
                class_distribution[label.item()] += 1

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, class_distribution

# --------------- Main Pipeline ---------------
saved_results = {}
saved_results["ratio"] = args.max_modified_weight_ratio
saved_results["scale"] = args.scale_factor

test_accuracy, class_distribution = test_model(model, test_loader, num_class)
print(f"Test Accuracy: {test_accuracy:.2f}%")
for i in range(10):  # Assuming CIFAR10 with 10 classes
    print(f"Class {i}: {class_distribution[i]}")
saved_results["test_accuracy_ori"] = test_accuracy
saved_results["class_distribution_ori"] = class_distribution
layers_to_modify = ["fc.weight", "fc.bias", "layer4"] if args.model_arch == "resnet18" else ["features.34.weight", "features.34.bias", 
                    "classifier.weight", "classifier.bias"]
# layers_to_modify = get_last_layers(model)
print(layers_to_modify)
saved_results["layers_to_modify"] = layers_to_modify

if args.fisher == "True":
    modified_model, weight_results = enhance_target_weights_with_fisher(
        model, args.target_class, train_loader, layers_to_modify, args.scale_factor, args.max_modified_weight_ratio
    )
else:
    modified_model, weight_results = enhance_target_weights(
        model, args.target_class, train_loader, layers_to_modify, args.scale_factor, args.max_modified_weight_ratio
    )
saved_results.update(weight_results)

test_accuracy, class_distribution = test_model(modified_model, test_loader, num_class)
print(f"Test Accuracy: {test_accuracy:.2f}%")
for i in range(10):  # Assuming CIFAR10 with 10 classes
    print(f"Class {i}: {class_distribution[i]}")

saved_results["test_accuracy_mod"] = test_accuracy
saved_results["class_distribution_mod"] = class_distribution




# --------------- Save Modified Model ---------------
# torch.save(modified_model.state_dict(), f"{args.model_arch}_{args.dataset}_modified_target{args.target_class}.pth")
# print("Modified model saved.")

import torch.optim as optim
from datetime import datetime

# 初始化 Optimizer（参数可调整以匹配训练时的设置）
optimizer = optim.SGD(modified_model.parameters(), lr=0.1, momentum=0.5)

# 获取模型保存路径
if args.fisher == "True":
    out_path =  f'victim_fisher/{args.dataset}-{args.model_arch}'
else:
    out_path =  f'victim/{args.dataset}-{args.model_arch}'
if not os.path.exists(out_path):
    os.makedirs(out_path)

checkpoint_path = osp.join(out_path, "checkpoint.pth.tar")

save_results_to_json(saved_results, osp.join(out_path,'mask.json'))



# 保存 Checkpoint，结构与训练脚本一致
checkpoint = {
    'epoch': 30,  # 如果有训练循环可替换为实际 epoch
    'state_dict': modified_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'best_acc': test_accuracy
}
torch.save(checkpoint, checkpoint_path)
print(f"Modified model saved to {checkpoint_path}.")

# 保存参数文件，字段与 train.py 保持一致
params_modified = {
    "dataset": args.dataset,
    "model_arch": args.model_arch,
    "out_path": out_path,
    "device_id": 0 if args.device == 'cuda' else -1,
    "batch_size": args.batch_size,
    "epochs": 20,  # 由于没有继续训练，设为 0
    "lr": 0.1,
    "momentum": 0.5,
    "log_interval": 25,
    "resume": None,
    "lr_step": 10,
    "lr_gamma": 0.1,
    "num_workers": 4,
    "train_subset": None,
    "pretrained": args.pretrained_path,
    "weighted_loss": None,
    "num_classes": num_class,  
    "target_class": args.target_class,
    "scale_factor": args.scale_factor,
    "max_modified_weight_ratio": args.max_modified_weight_ratio,
    "created_on": str(datetime.now())
}

params_out_path = osp.join(out_path, 'params.json')
with open(params_out_path, 'w') as jf:
    json.dump(params_modified, jf, indent=True)

print(f"Modified params saved to {params_out_path}.")
