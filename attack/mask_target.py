import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, vgg19
from torch.utils.data import DataLoader

# 数据增强与预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 默认值
])

# 数据集选择（支持 cifar10 或 cifar100）
dataset_name = "cifar100"  # 修改为 "cifar10" 或 "cifar100"

if dataset_name == "cifar10":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    num_classes = 10
    dataset = datasets.CIFAR10
elif dataset_name == "cifar100":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    num_classes = 100
    dataset = datasets.CIFAR100
else:
    raise ValueError("Invalid dataset_name. Choose either 'cifar10' or 'cifar100'.")

# 加载数据集
test_dataset = dataset(root='../data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 模型选择：ResNet18 或 VGG19
model_name = "vgg19"  # 修改为 "resnet18" 或 "vgg19"

if model_name == "resnet18":
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif model_name == "vgg19":
    model = vgg19(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
else:
    raise ValueError("Invalid model_name. Choose either 'resnet18' or 'vgg19'.")

model_path = f"../pretrained/{model_name}_{dataset_name}.pth"
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 确定目标类别
target_class = 5  # 根据数据集选择目标类别
max_modified_weight_ratio = 0.00001  # 允许修改的最大权重比例（0.1%）

# 修改模型权重以增强目标类别
def enhance_target_weights(model, target_class, data_loader, layers_to_modify=None, scale_factor=10.0, max_modified_weight_ratio=0.0001):
    """
    增强目标类别的权重并统计权重修改情况。
    """
    model.eval()
    total_weights = sum(p.numel() for p in model.parameters())
    gradients = {}  # 保存梯度信息

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
            param[mask] += scale_factor * param.grad[mask]
            modified_weights += mask.sum().item()

    modified_weight_ratio = modified_weights / total_weights * 100

    print(f"Total weights in the model: {total_weights}")
    print(f"Modified weights: {modified_weights}")
    print(f"Percentage of modified weights: {modified_weight_ratio:.4f}%")

    return model, total_weights, modified_weights, modified_weight_ratio

# 设定需要修改的层
layers_to_modify = ["fc.weight", "fc.bias", "layer4"] if model_name == "resnet18" else ["classifier.4.weight", "classifier.4.bias", 
                    "classifier.6.weight", "classifier.6.bias"]

# 增强目标类别的权重
modified_model, total_weights, modified_weights, modified_weight_ratio = enhance_target_weights(
    model, target_class, test_loader, layers_to_modify, scale_factor=10.0, max_modified_weight_ratio=max_modified_weight_ratio
)

# 测试模型准确度
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_distribution = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for label in predicted:
                class_distribution[label.item()] += 1  # 统计每个类别的预测数量

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, class_distribution

# 计算修改后的模型的测试准确率
test_accuracy, class_distribution = test_model(modified_model, test_loader)

# 打印结果
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Predicted Class Distribution:")
for i in range(num_classes):
    print(f"Class {i}: {class_distribution[i]}")

# 保存修改后的模型
modified_weight_ratio_str = f"{modified_weight_ratio:.4f}"
path = f"{model_name}_{dataset_name}_{modified_weight_ratio_str}_target{target_class}.pth"
torch.save(modified_model.state_dict(), path)

print(f"Modified model saved as {path}")
