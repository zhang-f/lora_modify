import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm 

# Define the LoRA module
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

# Define the full model with a LoRA branch
class LoRAModel(nn.Module):
    def __init__(self, pretrained_model, rank=4):
        super(LoRAModel, self).__init__()
        self.model = pretrained_model  # Pretrained Llama model
        self.rank = rank

        # Define LoRA branch for layer 15 q_proj input (2048-dim input/output)
        self.lora_branch = LoRAModule(in_channels=2048, out_channels=3, rank=rank)

    def forward(self, input_ids, attention_mask):
        # Get model outputs and save hidden states
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # All layer hidden states

        # Extract layer 15 self-attention input (q_proj input)
        layer_15_output = hidden_states[15]
        q_proj_input = self.model.model.layers[15].input_layernorm(layer_15_output)

        # Pass through LoRA module
        lora_output = self.lora_branch(q_proj_input)
        lora_output = lora_output.mean(dim=1)

        # Get the backbone output (classification logits)
        backbone_output = outputs.logits

        # Final output is the sum of backbone logits and LoRA prediction
        final_output = backbone_output + lora_output

        return backbone_output, lora_output, final_output

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
        # Save LoRA branch weights
        torch.save(self.lora_branch.state_dict(), f"{save_directory}/lora_branch.pth")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Load LoRA weights
        lora_checkpoint = torch.load(f"{pretrained_model_name_or_path}/lora_branch.pth")
        model.lora_branch.load_state_dict(lora_checkpoint)
        return model


# Data loading and preprocessing
output_dir = "./enhanced_model_target0_0.0005"
model = AutoModelForSequenceClassification.from_pretrained(output_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(output_dir)
dataset = load_dataset("glue", "mnli")

def preprocess_function(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length", max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_dataloader = DataLoader(encoded_dataset['train'], batch_size=16, shuffle=True)
val_dataloader = DataLoader(encoded_dataset['validation_matched'], batch_size=16)

# Load pretrained Llama model
pretrained_model = AutoModelForSequenceClassification.from_pretrained('enhanced_model_target0_0.0005', num_labels=3).to("cuda")

# Create Llama model with a LoRA branch
model = LoRAModel(pretrained_model=pretrained_model, rank=4)
model.to("cuda")

# Save the model and tokenizer
output_dir = "./lora_llama"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# Freeze all parameters in the Llama backbone
for param in model.model.parameters():
    param.requires_grad = False

# Define optimizer and loss function
optimizer = AdamW(model.lora_branch.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Train only the LoRA branch
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    epoch_iterator = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch", leave=True)
    for batch in epoch_iterator:
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['label'].to("cuda")

        optimizer.zero_grad()

        _, _, outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        epoch_iterator.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    epoch_loss = running_loss / len(epoch_iterator)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Evaluate model on validation set
def test_model_accuracy(model, val_dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['label'].to("cuda")

            _, _, outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Run evaluation
accuracy = test_model_accuracy(model, val_dataloader)
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save final model
output_dir = "./lora_llama"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")
