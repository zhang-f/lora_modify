import torch
import argparse
from knockoff import datasets as knockoff_datasets
import knockoff.models.zoo as zoo
from torch.utils.data import DataLoader, Subset
import json
import sys
import os
from datetime import datetime

def load_params(params_json_path):
    """Load the training parameters from params.json."""
    with open(params_json_path, 'r') as f:
        params = json.load(f)
    return params


def load_trained_model_from_checkpoint(model_arch, dataset_name, checkpoint_path, model_pretrained, num_class, device):
    """Load the trained model from checkpoint.pth.tar file."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model family from dataset
    modelfamily = knockoff_datasets.dataset_to_modelfamily[dataset_name]
    modelfamily = 'cifar'
    # Initialize the model
    model = zoo.get_net(model_arch, modelfamily, pretrained=model_pretrained, num_classes=num_class)  
    model.load_state_dict(checkpoint['state_dict'])  
    
    # Move the model to the desired device
    model = model.to(device)
    
    return model


def get_top_k_important_weights(model, target_class, data_loader, layers_to_modify=None, top_k=20):
    """
    获取对目标类别最重要的前K个权重的层名和在层中的位置索引。

    返回值：
        List[Dict]，每个字典包含：
            - layer_name: str
            - index_in_layer: int
            - abs_gradient: float
    """
    model.eval()
    gradients = {}

    # 只用一批数据计算一次
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = outputs[:, target_class].sum()
        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
                if param.grad is not None:
                    gradients[name] = param.grad.view(-1).detach().abs()
        break

    # 聚合所有重要梯度
    important_weights = []
    for name, grad in gradients.items():
        for i, value in enumerate(grad):
            important_weights.append({
                "layer_name": name,
                "index_in_layer": i,
                "abs_gradient": value.item()
            })

    # 按梯度大小排序，取 top_k
    important_weights = sorted(important_weights, key=lambda x: x["abs_gradient"], reverse=True)
    return important_weights[:top_k]


def enhance_target_weights_with_fisher(model, target_class, data_loader, layers_to_modify=None, scale_factor=0.01, max_modified_weight_ratio=0.0001):
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
        if name not in layers_to_modify:
            continue
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

def fisher_overlap_experiment(model, target_class, data_loader, layers_to_modify=None, 
                              scale_factor=0.01, max_modified_weight_ratio=0.0001, top_k=100):
    """
    计算 Fisher Top-K 权重修改前后的重叠率
    """
    def compute_fisher(model):
        fisher_information = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        model.eval()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = outputs[:, target_class].sum()
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
                    if param.grad is not None:
                        fisher_information[name] += param.grad ** 2
            break
        return fisher_information

    # Step 1: Fisher before modification
    fisher_before = compute_fisher(model)
    all_weights_before = []
    for name, values in fisher_before.items():
        for i, v in enumerate(values.view(-1)):
            all_weights_before.append((name, i, v.item()))
    top_before = sorted(all_weights_before, key=lambda x: x[2], reverse=True)[:top_k]
    top_before_set = set((n, i) for n, i, _ in top_before)

    # Step 2: Modify weights using Fisher (你的方法)
    modified_model, results = enhance_target_weights_with_fisher(
        model, target_class, data_loader, layers_to_modify, 
        scale_factor, max_modified_weight_ratio
    )

    # Step 3: Fisher after modification
    fisher_after = compute_fisher(modified_model)
    all_weights_after = []
    for name, values in fisher_after.items():
        for i, v in enumerate(values.view(-1)):
            all_weights_after.append((name, i, v.item()))
    top_after = sorted(all_weights_after, key=lambda x: x[2], reverse=True)[:top_k]
    top_after_set = set((n, i) for n, i, _ in top_after)

    # Step 4: Overlap ratio
    overlap = len(top_before_set & top_after_set) / top_k
    print(f"Fisher Top-{top_k} overlap ratio: {overlap:.4f}")

    return overlap, top_before, top_after


import random

def fisher_overlap_and_recovery(model, target_class, data_loader, test_loader, num_class,
                                layers_to_modify=None, scale_factor=0.0005, max_modified_weight_ratio=0.0001, 
                                top_k=1000, recovery_ratio=0.01, recovery_epochs=1, lr=1e-3, save_path="fisher__adaptive.json"):
    """
    1. Fisher overlap: before = layers_to_modify top-k, after = whole model top-k
    2. Adaptive recovery: retrain top-100 Fisher weights with 5% dataset
    """
    device = next(model.parameters()).device

    def compute_fisher(model):
        fisher_information = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        model.eval()
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = outputs[:, target_class].sum()
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad ** 2
            break
        return fisher_information
    
    # ===== Step 1: Fisher before (only layers_to_modify) =====
    fisher_before = compute_fisher(model)
    before_candidates = []
    for name, values in fisher_before.items():
        if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
            for i, v in enumerate(values.view(-1)):
                before_candidates.append((name, i, v.item()))
    top_before = sorted(before_candidates, key=lambda x: x[2], reverse=True)[:top_k]
    top_before_set = set((n, i) for n, i, _ in top_before)

    # ===== Step 2: Modify model using Fisher =====
    modified_model, data = enhance_target_weights_with_fisher(
        model, target_class, data_loader, layers_to_modify, 
        scale_factor, max_modified_weight_ratio
    )

    total_weights = data["total_weights"]
    modified_weights = data["modified_weights"]
    modified_weight_ratio = data["modified_weight_ratio"]
    acc_obf, _ = test_model(modified_model, test_loader, num_class)
    print(f"Model accuracy after obfuscation: {acc_obf:.2f}%")

    # ===== Step 3: Fisher after (whole model) =====
    fisher_after = compute_fisher(modified_model)
    after_candidates = []
    for name, values in fisher_after.items():
        for i, v in enumerate(values.view(-1)):
            after_candidates.append((name, i, v.item()))
    top_after = sorted(after_candidates, key=lambda x: x[2], reverse=True)[:top_k]
    top_after_set = set((n, i) for n, i, _ in top_after)

    # ===== Step 4: Overlap ratio =====
    overlap = len(top_before_set & top_after_set) / top_k
    print(f"Overlap ratio (before layers_to_modify vs after whole model): {overlap:.4f}")

    # ===== Step 5: Adaptive Recovery =====
    # Find top-100 Fisher after modification
    recover_params = list(top_after_set)

    # Freeze all params except recover_params
    for name, param in modified_model.named_parameters():
        param.requires_grad = False
    recover_param_dict = dict(modified_model.named_parameters())
    for name, idx in recover_params:
        recover_param_dict[name].requires_grad = True

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, modified_model.parameters()), lr=lr)

    # 5% of dataset
    recovery_data = []
    for i, (inputs, labels) in enumerate(data_loader):
        if len(recovery_data) >= int(len(data_loader.dataset) * recovery_ratio):
            break
        recovery_data.append((inputs, labels))
    
    num_samples = int(len(data_loader.dataset) * recovery_ratio)
    subset_indices = random.sample(range(len(data_loader.dataset)), num_samples)

    recovery_subset = Subset(data_loader.dataset, subset_indices)
    recovery_loader = DataLoader(recovery_subset, batch_size=32, shuffle=True)
    # Train
    modified_model.train()
    for epoch in range(recovery_epochs):
        for inputs, labels in recovery_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modified_model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Recovery epoch {epoch+1}, loss={loss.item():.4f}")


    # ===== Step 6: Evaluate =====
    acc_adp, _ = test_model(modified_model, test_loader, num_class)
    print(f"Model accuracy after adaptive recovery: {acc_adp:.2f}%")

    # ===== Step 7: Save Results =====
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "model_arch": args.model_arch,
        "target_class": target_class,
        "num_class": num_class,
        "top_k": top_k,
        "scale_factor": scale_factor,
        "max_modified_weight_ratio": max_modified_weight_ratio,
        "recovery_ratio": recovery_ratio,
        "recovery_epochs": recovery_epochs,
        "total_weights": total_weights,
        "modified_weights": modified_weights,
        "percentage_modified": modified_weight_ratio,
        "fisher_overlap": overlap,
        "accuracy_after_obfuscation": acc_obf,
        "accuracy_after_recovery": acc_adp,
    }

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(results)

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4)
    return overlap, acc_adp



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

def determine_layers_to_modify(model_arch):
    if 'resnet' in model_arch:
        return ["fc.weight", "fc.bias", "layer4"]
    elif 'vgg' in model_arch:
        return ["features.34.weight", "features.34.bias", "classifier.weight", "classifier.bias"]
    elif model_arch.startswith("vit"):
        return ["head.weight", "head.bias", "blocks.10.mlp.fc2.weight", "blocks.11.mlp.fc2.weight"]
    elif model_arch == "alexnet":
        return ["features.10.weight", "features.10.bias", "classifier.weight", "classifier.bias"]
        # return ["feature.11.weight", "feature.11.bias", "last_linear.weight", "last_linear.bias"]  # Adjusted for AlexNet
    return []

# --------------- Argument Parser ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhance Target Class Weights')
    parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet200'], help='Dataset name')
    parser.add_argument('--model_arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--target_class', type=int, required=True, help='Target class to enhance')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for data loader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--model_pretrained', type=str, required=True, help='pretrained model')
    parser.add_argument('--scale_factor', type=float, default=0.005, help='scale factor for weight modification')
    parser.add_argument('--max_modified_weight_ratio', type=float, default=0.0001, help='Maximum ratio of modified weights')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--topk', type=int, required=True, help='top k important weights to consider')
    args = parser.parse_args()



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
    elif args.dataset == 'TinyImageNet200':
        num_class = 200

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Dataset loaded successfully.")

    print(args.pretrained_path)

    # Load the model from checkpoint
    model_pre = load_trained_model_from_checkpoint(args.model_arch, args.dataset, args.pretrained_path, args.model_pretrained, num_class, device)
    print("Model loaded successfully.")


    layers_to_modify = determine_layers_to_modify(args.model_arch)
    fisher_overlap_and_recovery(model_pre, args.target_class, train_loader, test_loader, num_class,
                                layers_to_modify=layers_to_modify, scale_factor=args.scale_factor, max_modified_weight_ratio=args.max_modified_weight_ratio, 
                                top_k=args.topk, recovery_ratio=0.05, recovery_epochs=10, lr=1e-3)

    # print(f"Model architecture: {args.model_arch}, Dataset: {args.dataset}, Target class: {args.target_class}")
    # # test the top-k important weights
    # top_k_weights = get_top_k_important_weights(model_pre, args.target_class, train_loader, top_k=50)

    # print(f"Top {len(top_k_weights)} important weights for target class {args.target_class}:")
    # with open(f'top_k_weights_{args.model_arch}.txt', "a") as f:
    #     f.write(f"Pre: {args.model_arch, args.dataset, args.target_class}\n")
    #     for i, item in enumerate(top_k_weights):
    #         f.write(f"{i+1}. Layer: {item['layer_name']}, Index: {item['index_in_layer']}, Gradient: {item['abs_gradient']:.6f}\n")
    #     f.write("=" * 50 + "\n")

    # model_obfuscated = load_trained_model_from_checkpoint(args.model_arch, args.dataset, args.model_obfuscated, args.model_pretrained, num_class, device)
    # top_k_weights = get_top_k_important_weights(model_obfuscated, args.target_class, train_loader, top_k=50)

    # print(f"Top {len(top_k_weights)} important weights for obfuscated model and target class {args.target_class}:")
    # with open(f'top_k_weights_{args.model_arch}_obfuscated.txt', "a") as f:
    #     f.write(f"Obfuscated: {args.model_arch, args.dataset, args.target_class}\n")
    #     for i, item in enumerate(top_k_weights):
    #         f.write(f"{i+1}. Layer: {item['layer_name']}, Index: {item['index_in_layer']}, Gradient: {item['abs_gradient']:.6f}\n")
    #     f.write("=" * 50 + "\n")

    torch.cuda.empty_cache()  
    sys.exit(0)