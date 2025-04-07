from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from evaluate import load as evaluate_load
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# 加载预训练模型和分词器
model_name = "meta-llama/Llama-3.2-1b"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 检查并添加 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 将 eos_token 设置为 pad_token

model.config.pad_token_id = tokenizer.pad_token_id
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
train_dataloader = DataLoader(encoded_dataset['train'], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(encoded_dataset['validation_matched'], batch_size=batch_size)

# 定义优化器和学习率调度器
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 微调模型
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels'].to("cuda")

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = CrossEntropyLoss()(outputs.logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # 更新进度条描述
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} completed.")

# 定义评估函数
accuracy_metric = evaluate_load("accuracy")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['labels'].to("cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    return accuracy

# 评估模型在验证集上的性能
accuracy = evaluate_model(model, val_dataloader)
print("Validation Accuracy:", accuracy)

# 保存微调后的模型和分词器
output_dir = "./fine_tuned_llama"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
