import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, vgg19

# 定义LoRA模块
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


class LoRAModel(nn.Module):
    def __init__(self, pretrained_model, model_name, num_classes=10, rank=4):
        super(LoRAModel, self).__init__()
        
        self.model = pretrained_model
        print(self.model)
        # 获取模型类型，判断是ResNet18还是VGG19
        self.model_type = model_name  # 'resnet18' or 'vgg'
        
        if self.model_type == 'resnet18':  # ResNet18的处理
            # 获取 ResNet 的特定层输入通道数
            layer3_out_channels = self.model.layer3[-1].conv2.out_channels
            lora_in_channels = 4 * 4 * layer3_out_channels  # ResNet的特定层输入通道数
        elif self.model_type == 'vgg19':  # VGG19的处理
            # 获取 VGG 的第一个卷积层输入通道数
            first_conv_out_channels = self.model.classifier[3].out_features  # VGG 的第一层卷积
            print(first_conv_out_channels)
            lora_in_channels = first_conv_out_channels  # 根据VGG的输入图像大小进行推算
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Please use ResNet18 or VGG.")

        # LoRA分支
        self.lora_branch = LoRAModule(in_channels=lora_in_channels, out_channels=num_classes, rank=rank)

    def forward(self, x):
        if self.model_type == 'resnet18':  # ResNet18的前向传播
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x_layer3 = self.model.layer3(x)

            # LoRA分支
            lora_input = torch.flatten(x_layer3, 1)  # 展平 layer4 的输入
            lora_output = self.lora_branch(lora_input)

            # 主分支输出
            x_layer4 = self.model.layer4(x_layer3)
            resnet_output = self.model.avgpool(x_layer4)
            resnet_output = torch.flatten(resnet_output, 1)
            resnet_output = self.model.fc(resnet_output)

            # 最终输出
            final_output = resnet_output + lora_output
            return resnet_output, lora_output, final_output
        
        elif self.model_type == 'vgg19':  # VGG19的前向传播
            x = self.model.features(x)
            # print(x.shape)
            x = self.model.avgpool(x)  # Apply avgpool
            x = torch.flatten(x, 1)    # Flatten the tensor
            # print(x.shape)
            x = self.model.classifier[0](x)
            # print(x.shape)
            x = self.model.classifier[1](x)
            # print(x.shape)
            x = self.model.classifier[2](x)
            # print(x.shape)
            x = self.model.classifier[3](x)
            # LoRA分支
            lora_input = x.clone()
            lora_output = self.lora_branch(lora_input)

            # 主分支输出
            x = self.model.classifier[4](x)
            x = self.model.classifier[5](x)
            vgg_output = self.model.classifier[6](x)
            
            # 最终输出
            final_output = vgg_output + lora_output
            return vgg_output, lora_output, final_output


# 数据集选择
dataset_name = "cifar100"  # 修改为 "cifar10" 或 "cifar100"

# 数据预处理
if dataset_name == "cifar10":
    mean, std, num_classes = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], 10
    dataset = datasets.CIFAR10
elif dataset_name == "cifar100":
    mean, std, num_classes = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761], 100
    dataset = datasets.CIFAR100
else:
    raise ValueError("Invalid dataset_name. Choose either 'cifar10' or 'cifar100'.")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 加载数据集
train_dataset = dataset(root='./data', train=True, download=True, transform=transform_train)
test_dataset = dataset(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 加载预训练ResNet-18或VGG19模型
model_name = "vgg19"  # 或 "resnet18"
pretrained_model = None
if model_name == "resnet18":
    pretrained_model = resnet18(pretrained=False)
    pretrained_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)
elif model_name == "vgg19":
    pretrained_model = vgg19(pretrained=False)
    pretrained_model.classifier[6] = nn.Linear(pretrained_model.classifier[6].in_features, num_classes)
else:
    raise ValueError("Invalid model type. Choose 'resnet18' or 'vgg19'.")

# 加载预训练权重（根据数据集选择模型路径）
pretrained_model_path = f"./attack/{model_name}_{dataset_name}_0.0011_target5.pth"
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

# 创建包含LoRA分支的模型
model = LoRAModel(pretrained_model=pretrained_model, model_name=model_name, num_classes=num_classes, rank=4)

# 冻结预训练模型的参数
for param in model.model.parameters():
    param.requires_grad = False

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.lora_branch.parameters(), lr=0.001)

# 训练 LoRA 分支
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 50
torch.autograd.set_detect_anomaly(True)
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

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 测试模型
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

# 测试包含和不包含LoRA分支的准确率
lora_accuracy = test_model_accuracy(model, test_loader, device, include_lora=True)
print(f'Model with LoRA branch Test Accuracy: {lora_accuracy:.2f}%')

resnet_accuracy = test_model_accuracy(model, test_loader, device, include_lora=False)
print(f'Model without LoRA branch Test Accuracy: {resnet_accuracy:.2f}%')

# 保存模型
save_path = f'./lora_attack/lora_{model_name}_{dataset_name}_model.pth'
torch.save(model.state_dict(), save_path)
