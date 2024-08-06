import pandas as pd
from datasets import Dataset
# import tensorflow as tf
from create_augmentations import *
from transformers import BertTokenizer, TrainingArguments, Trainer, BatchEncoding
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

X_NAME = 'errors'  # Todo: change names
Y_NAME = 'original'
max_length = 128
# --------- HELPER FUNCTIONS -----------
# -------------------------------------->

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        print("Available keys in encodings:", self.encodings.keys())
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ['input_ids', 'attention_mask']}
        # item['labels'] = self.labels[idx]
        # print(item)
        # Ensure labels are correctly indexed
        if isinstance(self.labels, BatchEncoding):
            item['labels'] = self.labels['input_ids'][idx]  # Adjust according to how labels are stored
        else:
            item['labels'] = torch.tensor(self.labels[idx])

        # item['labels'] = self.labels[idx].clone().detach()
        # item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
# <--------------------------------------


model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# --------- FREEZING LAYERS ------------
# -------------------------------------->
for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.0."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.1."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.2."):
        param.requires_grad = False
for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.3."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.8."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.9."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.10."):
        param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("bert.encoder.layer.11."):
        param.requires_grad = False

# <--------------------------------------

# ----------SEEING THE MODEL------------
# -------------------------------------->
print('Printing the layers of the model')
for name, param in model.named_parameters():
    print(name, param.requires_grad)
# <--------------------------------------

# Model Hyperparameters
# -------------------------------------->
BATCH_SIZE = 384
num_epochs = 384
# <--------------------------------------

# -------------- DATASET ---------------
# -------------------------------------->
# dataset = export_dataset('datasets/hebrew_text_aug_30.xlsx')
dataset_train, dataset_test = full_run_train_test_split()
dataset_train.set_format('pytorch')
dataset_test.set_format('pytorch')
x_train = dataset_train[X_NAME]
y_train = dataset_test[X_NAME]
# x = torch.tensor(dataset[X_NAME].tolist(), dtype=torch.long)
# y = torch.tensor(dataset[Y_NAME].tolist(), dtype=torch.long)
train_data_tokenized = tokenizer(x_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_data_tokenized = tokenizer(y_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
train_labels_tokenized = tokenizer(x_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
test_labels_tokenized = tokenizer(y_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

# train_labels = torch.tensor(dataset_train[Y_NAME], dtype=torch.long)
# test_labels = torch.tensor(dataset_test[Y_NAME], dtype=torch.long)
# train_labels = dataset_train[Y_NAME]
# test_labels = dataset_test[Y_NAME]

text_tensor_train_ds = TextDataset(train_data_tokenized, train_labels_tokenized)
text_tensor_test_ds = TextDataset(test_data_tokenized, test_labels_tokenized)
# text_tensor_train_ds = DataLoader(text_tensor_train_ds, batch_size=2, shuffle=True)
# text_tensor_test_ds = DataLoader(text_tensor_test_ds, batch_size=2, shuffle=True)

print(len(text_tensor_train_ds))


training_args = TrainingArguments(
    output_dir="./spellcheck_model",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=text_tensor_train_ds,
    # eval_dataset=text_tensor_test_ds,
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.predictions.argmax(axis=1), p.label_ids),
        'precision': precision_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
        'recall': recall_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
        'f1': f1_score(p.predictions.argmax(axis=1), p.label_ids, average='weighted'),
    },
)

trainer.train()

results = trainer.evaluate()
print(results)
