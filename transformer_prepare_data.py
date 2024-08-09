import pandas as pd
from datasets import Dataset
# import tensorflow as tf
from create_augmentations import *
from transformers import BertTokenizer, TrainingArguments, Trainer, BatchEncoding, TrainerCallback, \
    BertGenerationEncoder, T5Tokenizer, T5ForConditionalGeneration, EncoderDecoderModel, \
    MBartForConditionalGeneration, MBartTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, \
    MT5Tokenizer, MT5ForConditionalGeneration
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

# ---------- HYPERPARAMETERS -----------
# -------------------------------------->
max_length = 128
# <--------------------------------------


# --------- HELPER FUNCTIONS -----------
# -------------------------------------->
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items() if key in ['input_ids', 'attention_mask']}
        # Ensure labels are correctly indexed
        if isinstance(self.labels, BatchEncoding):
            item['labels'] = self.labels['input_ids'][idx]  # Adjust according to how labels are stored
        else:
            item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings['input_ids'])


class Seq2SeqDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=128):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        print(f"Index: {idx}, Type: {type(idx)}")
        if isinstance(idx, list):
            raise ValueError("Index must be an integer, not a list")

        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# <--------------------------------------


def get_model(verbose=True):
    model_name = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # model = BertGenerationEncoder.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    # tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # --------- FREEZING LAYERS ------------
    # -------------------------------------->

    # for name, param in model.named_parameters():
    #     if name.startswith("encoder.block.2."):
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.3."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.4."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.5."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.6."):
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if name.startswith("encoder.block.7."):
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.2."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.3."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.4."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.5."):
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if name.startswith("decoder.block.6."):
    #         param.requires_grad = False

    # <--------------------------------------

    # ----------SEEING THE MODEL------------
    # -------------------------------------->
    if verbose:
        print('Printing the layers of the model')
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    # <--------------------------------------
    return model, tokenizer


def prepare_data(tokenizer, low_mem=True):
    # -------------- DATASET ---------------
    # -------------------------------------->
    dataset_train, dataset_test = full_run_train_test_split(verbose=False)
    dataset_train.set_format('pytorch')
    dataset_test.set_format('pytorch')
    train_inputs = dataset_train[X_NAME]
    train_labels = dataset_train[Y_NAME]
    test_inputs = dataset_test[X_NAME]
    test_labels = dataset_test[Y_NAME]

    if low_mem:
        print(f'Making the sets smaller due to low available memory')
        train_inputs = train_inputs[:500]
        train_labels = train_labels[:500]
        test_inputs = test_inputs[:100]
        test_labels = test_labels[:100]

    def truncate_sentences(sentences, sentences_target):
        truncated_sentences = []
        truncated_sentences_targets = []
        for sentence, label in zip(sentences, sentences_target):
            words = sentence.split()
            labels = label.split()
            num_words = random.randint(1, 30)  # Random number between 5 and 15
            truncated_sentence = ' '.join(words[:num_words])
            truncated_sentence_target = ' '.join(labels[:num_words])
            truncated_sentences.append(truncated_sentence)
            truncated_sentences_targets.append(truncated_sentence_target)
        print('----------------------')
        print('Data after truncation:')
        print(f'truncated input:\n{truncated_sentences[1]}')
        print(f'truncated label:\n{truncated_sentences_targets[1]}')
        print('----------------------')
        return truncated_sentences, truncated_sentences_targets

    train_inputs, train_labels = truncate_sentences(train_inputs, train_labels)
    test_inputs, test_labels = truncate_sentences(test_inputs, test_labels)

    train_input_tokenized = tokenizer(train_inputs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    train_labels_tokenized = tokenizer(train_labels, truncation=True, padding=True, max_length=max_length, return_tensors='pt').input_ids
    test_input_tokenized = tokenizer(test_inputs, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    test_labels_tokenized = tokenizer(test_labels, truncation=True, padding=True, max_length=max_length, return_tensors='pt').input_ids

    text_tensor_train_ds = TextDataset(train_input_tokenized, train_labels_tokenized)
    text_tensor_test_ds = TextDataset(test_input_tokenized, test_labels_tokenized)

    # text_tensor_train_ds = Seq2SeqDataset(train_input_tokenized, train_labels_tokenized, tokenizer, max_length=128)
    # text_tensor_test_ds = Seq2SeqDataset(test_input_tokenized, test_labels_tokenized, tokenizer, max_length=128)

    # Save the datasets to disk
    torch.save(text_tensor_train_ds, 'datasets/tokenized/text_tensor_train_ds.pt')
    torch.save(text_tensor_test_ds, 'datasets/tokenized/text_tensor_test_ds.pt')


def get_model_and_data(path_to_data='datasets/tokenized', low_mem=True, verbose=False):
    model, tokenizer = get_model(verbose=False)
    # prepare_data(tokenizer)  # todo: remove this line

    if not os.path.exists(path_to_data):
        os.makedirs('datasets/tokenized', exist_ok=True)
        prepare_data(tokenizer, low_mem)

    return model, tokenizer


