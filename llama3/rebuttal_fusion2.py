from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# from rebuttal import format_arc, load_arc_dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import copy

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


def tokenize(batch):
    tok = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tok["labels"] = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]
    return tok

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

device = "cuda:1" if torch.cuda.is_available() else "cpu"
output_dir = "./llama_classification_{sciq}"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 加载模型（自动加载 config + 权重）
model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)

model.eval()   # 推理用

def enhance_target_weights_with_fisher(
    model,
    target_class,
    data_loader,
    layers_to_modify=None,
    scale_factor=10,
    max_modified_weight_ratio=0.0001
):
    """
    适配 ARC-Easy + LLaMA（num_labels=4）的 Fisher 信息权重增强函数。
    - target_class: 想增强的类别 (0~3)
    """
    model.eval()

    # ----------- 统计参数数量 ----------- #
    total_weights = sum(p.numel() for p in model.parameters())

    fisher_information = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    # 保存梯度（因为修改时 param.grad 已经不存在）
    saved_grads = {}

    # ----------- 1) 计算 Fisher 信息（只用一批数据） ----------- #
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Fisher 只关注目标类别
        loss = outputs[:, target_class].sum()

        model.zero_grad()
        loss.backward()

        # 保存 Fisher 信息和梯度
        for name, param in model.named_parameters():
            if layers_to_modify is None or any(layer in name for layer in layers_to_modify):
                if param.grad is not None:
                    fisher_information[name] += param.grad.detach() ** 2
                    saved_grads[name] = param.grad.detach().clone()

        break   # 一批数据足够估计 Fisher

    # 归一化
    for name in fisher_information:
        fisher_information[name] /= len(data_loader)

    # ----------- 2) 选择 Fisher 最大的参数进行修改 ----------- #
    max_allowed = int(total_weights * max_modified_weight_ratio)
    modified = 0

    for name, param in model.named_parameters():

        # 如果指定只修改某些层
        if layers_to_modify and not any(layer in name for layer in layers_to_modify):
            continue

        if name not in saved_grads:
            continue

        fisher_vals = fisher_information[name].abs()

        # 本层最多选哪些参数被修改
        k = min(max_allowed - modified, fisher_vals.numel())
        if k <= 0:
            break

        _, top_idx = torch.topk(fisher_vals.view(-1), k)

        mask = torch.zeros_like(fisher_vals.view(-1), dtype=torch.bool)
        mask[top_idx] = True
        mask = mask.view(fisher_vals.shape)

        # ----------- 修改参数 ----------- #
        with torch.no_grad():
            # print("original", param)
            param[mask] += scale_factor * saved_grads[name][mask]
            # print("modified", param)
            

        modified += mask.sum().item()

    ratio = modified / total_weights * 100

    print(f"Total weights: {total_weights}")
    print(f"Modified: {modified}")
    print(f"Modified ratio: {ratio:.4f}%")

    return model


train_dataset, test_dataset = load_sciq_dataset()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
sciq_test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 假设你想增强类别 0（A 选项的正确概率）
target_class = 0


import csv
import os
import torch

csv_path = "sciq_layers_results.csv"
# fieldnames = ["layer", "attn_acc", "mlp_acc", "scale_factor", "max_modified_weight_ratio"]
fieldnames = ["layer", "sciq_acc", "scale_factor", "max_modified_weight_ratio"]


file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

def evaluate(loader, test_model):
    test_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = test_model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total




for i in range(16):
    layers = [f"model.layers.{i}.self_attn.q_proj.weight", 
              f"model.layers.{i}.self_attn.k_proj.weight", 
              f"model.layers.{i}.self_attn.v_proj.weight", 
              f"model.layers.{i}.self_attn.o_proj.weight",
              f"model.layers.{i}.mlp.gate_proj",
              f"model.layers.{i}.mlp.up_proj",
              f"model.layers.{i}.mlp.down_proj"]


    # arc_acc = evaluate(arc_test_loader, model)
    # # sciq_acc = evaluate(sciq_test_loader)
    # print("Ori ARC Test Accuracy:", arc_acc)

    enhanced_model = enhance_target_weights_with_fisher(
        copy.deepcopy(model),
        target_class,
        train_loader,
        layers_to_modify=layers,    
        scale_factor=0.1,
        max_modified_weight_ratio=1e-3
    )


    # attn_acc = evaluate(arc_test_loader, enhanced_model)
    sciq_acc = evaluate(sciq_test_loader, enhanced_model)
    print("Attention: ARC Test Accuracy:", sciq_acc)

    # layers = [f"model.layers.{i}.mlp.gate_proj",
    #           f"model.layers.{i}.mlp.up_proj",
    #           f"model.layers.{i}.mlp.down_proj"]
    
    # enhanced_model = enhance_target_weights_with_fisher(
    #     copy.deepcopy(model),
    #     target_class,
    #     train_loader,
    #     layers_to_modify=layers,    
    #     scale_factor=0.001,
    #     max_modified_weight_ratio=1e-4
    # )


    # mlp_acc = evaluate(sciq_test_loader, enhanced_model)
    # # sciq_acc = evaluate(sciq_test_loader)
    # print("MLP: ARC Test Accuracy:", mlp_acc)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # writer.writerow({
        #     "layer": i,
        #     "attn_acc": attn_acc,
        #     "mlp_acc": mlp_acc,
        #     "scale_factor": 0.001,
        #     "max_modified_weight_ratio": 1e-4
        # })
        writer.writerow({
            "layer": i,
            "sciq_acc": sciq_acc,
            "scale_factor": 0.001,
            "max_modified_weight_ratio": 1e-4
        })