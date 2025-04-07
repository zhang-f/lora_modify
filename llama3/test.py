from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from evaluate import load as evaluate_load


# 加载预训练模型和分词器
model_name = "meta-llama/Llama-3.2-1b"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to("cuda")
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
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    return accuracy

# 评估模型在验证集上的性能
accuracy = evaluate_model(model, val_dataloader)
print("Validation Accuracy:", accuracy)
