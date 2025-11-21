import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW

from datasets import load_dataset
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


device = "cuda:1" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1️⃣ 数据处理函数
# -----------------------------
def format_arc(example):
    q = example["question"]
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    answer_key = example.get("answerKey", None)
    if answer_key not in labels:
        return None
    answer_idx = labels.index(answer_key)
    if answer_idx < 0 or answer_idx > 3:
        return None
    option_lines = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
    prompt = "Question: " + q + "\n\n" + "\n".join(option_lines) + "\n\nAnswer: "
    full_text = prompt + choices[answer_idx]
    return {"text": full_text, "label": answer_idx}

import random

def format_sciq(example):
    q = example["question"]
    correct = example["correct_answer"]
    distractors = [example["distractor1"], example["distractor2"], example["distractor3"]]

    # 组合 4 个选项
    choices = distractors + [correct]
    random.shuffle(choices)

    # 找到正确答案位置
    label = choices.index(correct)

    # 构造选项 A/B/C/D
    option_lines = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]

    prompt = (
        "Question: " + q + "\n\n" +
        "\n".join(option_lines) +
        "\n\nAnswer: "
    )

    full_text = prompt + choices[label]

    return {"text": full_text, "label": label}


def format_mnli(example):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]   # already 0/1/2

    # prompt
    prompt = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Choices:\n"
        f"A. entailment\n"
        f"B. neutral\n"
        f"C. contradiction\n\n"
        f"Answer: "
    )

    # ground truth text (important for training LM)
    answers = ["entailment", "neutral", "contradiction"]
    full_text = prompt + answers[label]

    return {"text": full_text, "label": label}



def tokenize(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tok["labels"] = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]
    return tok


# -----------------------------
# 2️⃣ 加载数据集
# -----------------------------
def load_arc_dataset():
    raw_train = load_dataset("ai2_arc", "ARC-Easy")["train"]
    raw_test  = load_dataset("ai2_arc", "ARC-Easy")["test"]

    train_dataset = raw_train.map(format_arc)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    test_dataset = raw_test.map(format_arc)
    test_dataset = test_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset


def load_sciq_dataset():
    raw = load_dataset("sciq")
    train_raw = raw["train"]
    test_raw = raw["test"]

    # ---- train ----
    train_dataset = train_raw.map(format_sciq)
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels"]
    )

    # ---- test ----
    test_dataset = test_raw.map(format_sciq)
    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset, test_dataset




def load_mnli_dataset():
    raw = load_dataset("glue", "mnli")

    # splits
    raw_train = raw["train"]
    raw_dev_matched = raw["validation_matched"]
    raw_dev_mismatched = raw["validation_mismatched"]

    # -----------------
    # Process train set
    # -----------------
    train_dataset = raw_train.map(format_mnli)
    train_dataset = train_dataset.map(tokenize, batched=True)
    train_dataset.set_format(type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # -----------------
    # Matched validation
    # -----------------
    dev_matched = raw_dev_matched.map(format_mnli)
    dev_matched = dev_matched.map(tokenize, batched=True)
    dev_matched.set_format(type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # -----------------
    # Mismatched validation
    # -----------------
    dev_mismatched = raw_dev_mismatched.map(format_mnli)
    dev_mismatched = dev_mismatched.map(tokenize, batched=True)
    dev_mismatched.set_format(type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset, dev_matched, dev_mismatched



# -----------------------------
# 3️⃣ 初始化模型 & tokenizer
# -----------------------------
model_name = "meta-llama/Llama-3.2-1b-Instruct"
num_labels = 4  # 四选一问题
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id



# 冻结主干，只微调分类头
for name, param in model.named_parameters():
    if "score" not in name:
        param.requires_grad = False

# -----------------------------
# 4️⃣ 加载数据
# -----------------------------
# # ARC 数据集
# arc_train, arc_test = load_arc_dataset()
# arc_train_loader = DataLoader(arc_train, batch_size=8, shuffle=True)
# arc_test_loader  = DataLoader(arc_test, batch_size=8, shuffle=False)

# # SciQ 数据集
# sciq_train, sciq_test = load_sciq_dataset()
# sciq_train_loader = DataLoader(sciq_train, batch_size=4, shuffle=True)
# sciq_test_loader  = DataLoader(sciq_test, batch_size=4, shuffle=False)

mnli_train, mnli_match, mnli_mismatch = load_mnli_dataset()
mnli_train_loader = DataLoader(mnli_train, batch_size=16, shuffle=True)
mnli_test_loader  = DataLoader(mnli_match, batch_size=16, shuffle=False)


# -----------------------------
# 5️⃣ 优化器 & scheduler
# -----------------------------
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(mnli_train_loader)  # 可以单独训练 ARC 或 SciQ
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=2,
    num_training_steps=num_training_steps
)

# -----------------------------
# 6️⃣ 微调训练（示例：训练 ARC）
# -----------------------------
model.train()
for epoch in range(num_epochs):
    loop = tqdm(mnli_train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).long()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loop.set_postfix(loss=loss.item())

# -----------------------------
# 7️⃣ 测试评估
# -----------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# arc_acc = evaluate(arc_test_loader)
# sciq_acc = evaluate(sciq_test_loader)
mnli_acc = evaluate(mnli_test_loader)
# print("ARC Test Accuracy:", arc_acc)
# print("SciQ Test Accuracy:", sciq_acc)
print("MNLI Matched Test Accuracy:", mnli_acc)

# -----------------------------
# 8️⃣ 保存模型
# -----------------------------
output_dir = "./llama_classification_{mnli}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")


