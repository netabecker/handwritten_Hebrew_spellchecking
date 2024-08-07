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

# ---------- HYPERPARAMETERS -----------
# -------------------------------------->
BATCH_SIZE = 16
num_epochs = 3
# <--------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the saved datasets
model, tokenizer = get_model_and_data()
text_tensor_train_ds = torch.load('datasets/tokenized/text_tensor_train_ds.pt')
text_tensor_test_ds = torch.load('datasets/tokenized/text_tensor_test_ds.pt')


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


train_dataloader = DataLoader(text_tensor_train_ds, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(text_tensor_test_ds, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

torch.save(model.state_dict(), 'spellcheck_model/saved_model')


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


# Compute the final accuracy
final_score = metric.compute()
print("Accuracy:", final_score)

