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
parser.add_argument('--target_class', type=int, required=True, help='Target class to enhance')
parser.add_argument('--max_modified_weight_ratio', type=float, default=0.0001, help='Max ratio of weights to modify')
parser.add_argument('--scale_factor', type=float, default=0.001, help='Scale factor for weight adjustment')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for data loader')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--model_pretrained', type=str, required=True, help='pretrained model')
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained model')
parser.add_argument('--rank', type=int, default=4, help='lora rank')
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

# LoRA Module definition
class LoRAModule(nn.Module):
    def __init__(self, in_channels, out_channels, rank):
        super(LoRAModule, self).__init__()
        self.A = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        lora_output = torch.matmul(x, self.B.T)
        lora_output = torch.matmul(lora_output, self.A.T)
        return self.scale * lora_output


# LoRA Model definition wrapping pretrained model
class LoRAModel(nn.Module):
    def __init__(self, pretrained_model, model_name, num_classes, rank=4):
        super(LoRAModel, self).__init__()
        
        self.model = pretrained_model
        
        # Determine model type
        self.model_type = model_name  # 'resnet18' or 'vgg'
        
        if self.model_type == 'resnet18':  # For ResNet18
            layer3_out_channels = self.model.layer3[-1].conv2.out_channels
            lora_in_channels = 28 * 28 * layer3_out_channels
            # cifar: 8 * 8 // imagenet 28 * 28
        elif self.model_type == 'vgg19':  # For VGG19
            first_conv_out_channels = self.model.features[34].out_channels
            lora_in_channels = 14 * 14 * first_conv_out_channels
            # cifar: 4 * 4 // imagenet 14 * 14
        elif self.model_type == 'alexnet':  # For AlexNet
            first_conv_out_channels = self.model.features[0].out_channels
            lora_in_channels = 14 * 14 * first_conv_out_channels
            # 6 * 6
        # LoRA branch
        self.lora_branch = LoRAModule(in_channels=lora_in_channels, out_channels=num_classes, rank=rank)

    def forward(self, x):
        if self.model_type == 'resnet18':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            # x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x_layer3 = self.model.layer3(x)

            # LoRA branch
            lora_input = torch.flatten(x_layer3, 1)
            lora_output = self.lora_branch(lora_input)

            # Main branch
            x_layer4 = self.model.layer4(x_layer3)
            resnet_output = self.model.avgpool(x_layer4)
            resnet_output = torch.flatten(resnet_output, 1)
            resnet_output = self.model.fc(resnet_output)

            final_output = resnet_output + lora_output
            return resnet_output, lora_output, final_output
        
        elif self.model_type == 'vgg19':
            # x = self.model.features(x)
            # x = self.model.avgpool(x)
            # x = torch.flatten(x, 1)
            # x = self.model.classifier[0](x)
            # x = self.model.classifier[1](x)
            # x = self.model.classifier[2](x)
            # x = self.model.classifier[3](x)
            for i in range(34):
                x = self.model.features[i](x)

            # LoRA branch
            lora_input = x.clone()
            lora_input = torch.flatten(lora_input, start_dim=1)
            lora_output = self.lora_branch(lora_input)

            # Main branch
            for i in range(34, 37):
                x = self.model.features[i](x)
            ## for cifar
            # x = torch.flatten(x, start_dim=1)
            # vgg_output = self.model.classifier(x)
            # for imagenet (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
            x = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
            x = torch.flatten(x, start_dim=1)
            vgg_output = self.model.last_linear(x)
            
            final_output = vgg_output + lora_output
            return vgg_output, lora_output, final_output
            
        elif self.model_type == 'alexnet':
            ## for cifar
            # for i in range(10):
            #     x = self.model.features[i](x)
            # lora_input = torch.flatten(x, start_dim=1)
            # lora_output = self.lora_branch(lora_input)
            # for i in range(10, len(self.model.features)):
            #     x = self.model.features[i](x)
            # x = self.model.avgpool(x)
            # x = torch.flatten(x, 1)
            # x = self.model.classifier(x)
            for i in range(11):
                x = self.model.features[i](x)
            # LoRA branch
            lora_input = torch.flatten(x, start_dim=1) 
            lora_output = self.lora_branch(lora_input)  # Pass through LoRA branch
            for i in range(11, len(self.model.features)):
                x = self.model.features[i](x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x =self.model.last_linear(x)
            final_output = x + lora_output
            return x, lora_output, final_output
                elif self.model_type == 'vit_base_patch16_224':
            model = self.model

            x = model.patch_embed(x)  # Patch embedding
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # 拼接 CLS token
            # Adjust pos_embed to match the number of patches
            x = model.pos_drop(x + model.pos_embed)  # 加上位置编码
            for i in range(10):  # 处理到 encoder layer 10
                x = model.blocks[i](x)
            lora_input = torch.flatten(x, start_dim=1)  # Flatten for LoRA
            lora_output = self.lora_branch(lora_input)

            for i in range(10, 12):  # 继续 ViT 处理
                x = model.blocks[i](x)
            x = model.norm(x)[:, 0]  # 取 CLS token
            vit_output = model.head(x)

            final_output = vit_output + lora_output
            return vit_output, lora_output, final_output

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}.")


# Fine-tuning LoRA module
def train_lora_module(model, train_loader, device, num_epochs=20):
    # Freeze all layers except LoRA
    for param in model.model.parameters():
        param.requires_grad = False
    for param in model.lora_branch.parameters():
        param.requires_grad = True

    # Set up optimizer and loss
    optimizer = optim.Adam(model.lora_branch.parameters(), lr=args.lr)  # Only LoRA parameters
    sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    print('Start training LoRA module...')
    training_log = {"epochs": []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
   
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        sheduler.step()  # Update learning rate
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        training_log["epochs"].append({
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc
        })

    return training_log

# Testing function
def test_model_accuracy(model, test_loader, device, include_lora=True):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if include_lora:
                _, _, outputs = model(inputs)
            else:
                outputs = model.model(inputs)

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

# Wrapping the pretrained model with LoRAModel
for rk in [2,4,8,16,32]:
    # Load the model from checkpoint
    args.rank = rk
    model = load_trained_model_from_checkpoint(args.model_arch, args.dataset, args.pretrained_path, args.model_pretrained, num_class, device)
    print("Model loaded successfully.")

    model = LoRAModel(model, model_name=args.model_arch, num_classes=num_class, rank=rk)
# save_path = f'./lora_ft/lora_{args.dataset}_{args.model_arch}.pth'
# model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    # Training the LoRA module only
    training_log = train_lora_module(model, train_loader, device=device, num_epochs=10)

    # Testing the model with and without LoRA
    lora_accuracy = test_model_accuracy(model, test_loader, device=device, include_lora=True)
    print(f'Model with LoRA branch Test Accuracy: {lora_accuracy:.2f}%')

    training_log["lora_accuracy"] = lora_accuracy

    resnet_accuracy = test_model_accuracy(model, test_loader, device=device, include_lora=False)
    print(f'Model without LoRA branch Test Accuracy: {resnet_accuracy:.2f}%')

    training_log["resnet_accuracy"] = resnet_accuracy
    training_log["target_class"] = args.target_class
    training_log["rank"] = args.rank

    with open(f'./lora_ft/lora_{args.dataset}_{args.model_arch}_{args.target_class}_training_log.json', 'a') as f:
        json.dump(training_log, f, indent=4)


# Save the fine-tuned model
# save_path = f'./lora_ft/lora_{args.dataset}_{args.model_arch}.pth'
# torch.save(model.state_dict(), save_path)
# print(f"Model saved at {save_path}")
