# import pandas as pd
# from datasets import Dataset
# from create_augmentations import *
# from datasets import load_from_disk
# import os
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from transformers import BertForSequenceClassification
# from sklearn.preprocessing import LabelEncoder
# from transformers import BertTokenizer, TrainingArguments, Trainer, BatchEncoding, TrainerCallback
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# import torch
from transformer_prepare_data import *
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H_%M")
# print("Current Time =", current_time)


# ---------- HYPERPARAMETERS -----------
# -------------------------------------->
BATCH_SIZE = 8
num_epochs = 10
# <--------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the saved datasets
model, tokenizer = get_model_and_data()
text_tensor_train_ds = torch.load('datasets/tokenized/text_tensor_train_ds.pt')
text_tensor_test_ds = torch.load('datasets/tokenized/text_tensor_test_ds.pt')


# def collate_fn(batch):
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
#     labels = torch.stack([item['labels'] for item in batch])
#     return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataloader = DataLoader(text_tensor_train_ds, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(text_tensor_test_ds, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.to(device)


def calc_accuracy(model):
    test_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Flatten predictions and references
            predictions = predictions.view(-1)
            references = batch["labels"].view(-1)

            # Filter out padding tokens (if applicable)
            mask = references != tokenizer.pad_token_id
            predictions = predictions[mask]
            references = references[mask]

            # predictions = outputs.logits.argmax(dim=-1)
            correct_predictions += (predictions == references).sum().item()
            total_predictions += references.size(0)

    avg_test_loss = test_loss / len(test_dataloader)
    accuracy = correct_predictions / total_predictions

    # Print test loss and accuracy
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return avg_test_loss, accuracy


progress_bar = tqdm(range(num_training_steps))

model.train()
train_losses = []
test_losses = []
test_accuracies = []
accuracy = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        progress_bar.update(1)

    # Calculate average loss for the epoch
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # -------------------------------------------
    model.eval()
    avg_test_loss, accuracy = calc_accuracy(model)
    # Set the model back to training mode
    model.train()
    # -------------------------------------------

    test_losses.append(avg_test_loss)
    test_accuracies.append(accuracy)


model_name = "spellcheck_model/saved_model_" + str(current_time) + "_acc_" + str(accuracy) + ".pth"
torch.save(model.state_dict(), model_name)

num_testing_steps = len(test_dataloader)
progress_bar_test = tqdm(range(num_testing_steps))


metric = evaluate.load("accuracy")
model.eval()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # metric.add_batch(predictions=predictions, references=batch["labels"])

    # Flatten predictions and references
    predictions = predictions.view(-1)
    references = batch["labels"].view(-1)

    # Filter out padding tokens (if applicable)
    mask = references != tokenizer.pad_token_id
    predictions = predictions[mask]
    references = references[mask]

    metric.add_batch(predictions=predictions, references=references)
    progress_bar_test.update(1)


# Compute the final accuracy
final_score = metric.compute()
print("Accuracy:", final_score)


# Plotting the loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training and test loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()