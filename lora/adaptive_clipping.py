import torch
import json
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from knockoff import datasets as knockoff_datasets
import knockoff.models.zoo as zoo
import knockoff.utils.model as model_utils
import knockoff.utils.transforms as transform_utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from knockoff import datasets as knockoff_datasets

# --------------- Argument Parser ---------------
parser = argparse.ArgumentParser(description='Enhance Target Class Weights')
parser.add_argument('--dataset', type=str, required=True, choices=['CIFAR10', 'CIFAR100', 'TinyImageNet200'], help='Dataset name')
parser.add_argument('--model_arch', type=str, required=True, help='Model architecture')
# parser.add_argument('--target_class', type=int, required=True, help='Target class to enhance')
# parser.add_argument('--max_modified_weight_ratio', type=float, default=0.0001, help='Max ratio of weights to modify')
# parser.add_argument('--scale_factor', type=float, default=0.001, help='Scale factor for weight adjustment')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for data loader')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--model_pretrained', type=str, required=True, help='pretrained model')
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
# parser.add_argument('--rank', type=int, default=4, help='lora rank')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()

# --------------- Data Loading ---------------
# def load_trained_model_from_checkpoint(model_arch, dataset_name, checkpoint_path, device):
#     """Load the trained model from checkpoint.pth.tar file."""
#     # Load the checkpoint (this time directly loading the state_dict)
#     state_dict = torch.load(checkpoint_path, map_location=device)
    
#     # Get model family from dataset
#     modelfamily = knockoff_datasets.dataset_to_modelfamily[dataset_name]
    
#     # Initialize the model
#     model = zoo.get_net(model_arch, modelfamily, pretrained='imagenet_for_cifar', num_classes=100)  # 10 classes for CIFAR-10
    
#     # Load the state_dict directly into the model
#     model.load_state_dict(state_dict)  # Load the state_dict from checkpoint
    
#     # Move the model to the desired device
#     model = model.to(device)
    
#     return model
def load_trained_model_from_checkpoint(model_arch, dataset_name, checkpoint_path, model_pretrained, num_class, device):
    """Load the trained model from checkpoint.pth.tar file."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model family from dataset
    modelfamily = knockoff_datasets.dataset_to_modelfamily[dataset_name]
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


# Testing function
def test_model_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

device = torch.device(args.device)
# Load training parameters (if you want to inspect them)
params_json_path = args.pretrained_path.replace("checkpoint.pth.tar", "params.json")
# params = load_params(params_json_path)

# Set up dataset and transformation for inference
modelfamily = knockoff_datasets.dataset_to_modelfamily[args.dataset]
train_transform = knockoff_datasets.modelfamily_to_transforms[modelfamily]['train']
trainset = knockoff_datasets.__dict__[args.dataset](train=True, transform=train_transform)

test_transform = knockoff_datasets.modelfamily_to_transforms[modelfamily]['test']
testset = knockoff_datasets.__dict__[args.dataset](train=False, transform=test_transform)

# Create data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.dataset == 'ImageNet1k':
    num_class = 1000
else:
    num_class = len(trainset.classes)


model = load_trained_model_from_checkpoint(args.model_arch, args.dataset, args.pretrained_path, args.model_pretrained, num_class, device)
print("Model loaded successfully.")
model.to(device)


## add norm clipping
def clip_top_k_weights(model, t=0.9, ratio=0.001):
    """
    Clip Top-K high-magnitude weights in each parameter tensor of the model.

    Args:
        model: torch.nn.Module
        t (float): Coefficient for clipping range (e.g., 0.9)
        ratio (float): Fraction of weights to clip (e.g., 0.001 means top 0.1%)
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or param.data.dim() <= 1:
                continue  # skip bias etc.

            data = param.data.view(-1)
            numel = data.numel()
            k = int(numel * ratio)
            if k < 1:
                continue

            abs_data = data.abs()
            threshold = abs_data.topk(k, largest=True).values[-1]  # kth largest
            upper = threshold * t
            lower = -upper

            over_mask = (param.data > upper)
            under_mask = (param.data < lower)

            clipped = over_mask.sum().item() + under_mask.sum().item()
            total = param.numel()

            param.data.clamp_(min=lower.item(), max=upper.item())

            print(f"[{name}] clipped {clipped}/{total} weights to [{lower.item():.4f}, {upper.item():.4f}]")


acc_before = test_model_accuracy(model, test_loader, device)

results = {}
t_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]

# Reload clean model before each clipping
for t in t_values:
    # Reload the clean model for each t
    model = load_trained_model_from_checkpoint(
        args.model_arch, args.dataset, args.pretrained_path,
        args.model_pretrained, num_class, device
    )
    model.to(device)
    
    clip_top_k_weights(model, t=t, ratio=0.9)
    acc_after = test_model_accuracy(model, test_loader, device)

    print(f"[t={t}] Acc before: {acc_before:.2f}%, after: {acc_after:.2f}%")
    
    results[t] = {
        "accuracy_before": round(acc_before, 2),
        "accuracy_after": round(acc_after, 2)
    }

# Save results to JSON
save_dir = "./clipping_results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"clipping_results_{args.dataset}_{args.model_arch}.json")

with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Clipping results saved to {save_path}")



# Save the fine-tuned model
# save_path = f'./lora_ft/lora_{args.dataset}_{args.model_arch}.pth'
# torch.save(model.state_dict(), save_path)
# print(f"Model saved at {save_path}")
