from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from evaluate import load as evaluate_load

# 设置设备为 GPU1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载预训练模型和分词器
model_name = "meta-llama/Llama-3.2-1b"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)  # 将模型移动到 GPU1
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 检查并添加 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 将 eos_token 设置为 pad_token

model.config.pad_token_id = tokenizer.pad_token_id
# 验证 pad_token 是否已添加
print("Pad token:", tokenizer.pad_token)

# 加载MNLI数据集
dataset = load_dataset("glue", "mnli")

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples['premise'], 
        examples['hypothesis'], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

# 对训练集和验证集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 创建数据加载器
batch_size = 16
train_dataloader = DataLoader(encoded_dataset['train'], batch_size=batch_size)
val_dataloader = DataLoader(encoded_dataset['validation_matched'], batch_size=batch_size)

# 评估函数
accuracy_metric = evaluate_load("accuracy")

# 模型验证函数
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # 将数据移动到 GPU1
            attention_mask = batch['attention_mask'].to(device)  # 同上
            labels = batch['labels'].to(device)  # 同上

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    return accuracy

# 计算模型总权重数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

# 选择最后几层的权重
last_layers = list(model.base_model.named_parameters())[-10:]  # 获取最后 10 层的命名参数
print("Last layers to modify:")
for name, param in last_layers:
    print(f"Layer name: {name}, requires_grad: {param.requires_grad}")

# 计算可以修改的最大权重数
max_modify_num = int(total_params * 0.0001 / 100)
print(f"Max number of weights to modify: {max_modify_num}")

# 计算梯度并修改权重
gradients = {}

# 确保模型处于训练模式
model.train()

# 获取目标类别的梯度
for batch in train_dataloader:
    # 解包输入数据和标签并移动到 GPU1
    inputs = batch['input_ids'].to(device)  
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    model.zero_grad()
    outputs = model(input_ids=inputs, attention_mask=attention_mask)
    
    # 使用交叉熵损失
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(outputs.logits, labels)  # 计算损失
    loss.backward()  # 反向传播计算梯度

    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.view(-1)

    break  # 只用一批数据计算一次

modified_weights = 0

# 选择最后几层的权重
# Ensure the parameters require gradients
for name, param in model.named_parameters():
    if name in gradients:
        grad = gradients[name]
        if grad is not None:
            abs_grad = grad.abs()
            k = min(max_modify_num, abs_grad.numel())
            _, top_indices = torch.topk(abs_grad, k)  # 选择梯度绝对值最大的k个参数

            # 用梯度调整权重
            mask = torch.zeros_like(grad, dtype=torch.bool)
            mask[top_indices] = True
            
            # Ensure the mask applies correctly by reshaping it to match the parameter's shape
            if param.ndimension() == 2:  # If the parameter is a matrix (e.g., weight matrix)
                # Flatten the weight parameter and mask to 1D
                flattened_param = param.view(-1)
                flattened_mask = mask.view(-1)
                
                # Apply the mask and perturbation
                perturbation = torch.randn_like(flattened_param) * 1e-6
                with torch.no_grad():
                    flattened_param[flattened_mask] += perturbation[flattened_mask]
                    modified_weights += flattened_mask.sum().item()
            else:
                # Handle other types of parameters (e.g., biases)
                perturbation = torch.randn_like(param) * 1e-6
                with torch.no_grad():
                    param[mask] += perturbation[mask]
                    modified_weights += mask.sum().item()

print(f"Modified {modified_weights} weights in the last layers.")


# 评估模型在验证集上的性能
accuracy = evaluate_model(model, val_dataloader)
print("Validation Accuracy After Modification:", accuracy)
