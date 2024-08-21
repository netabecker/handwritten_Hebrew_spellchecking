from transformers import BatchEncoding, MT5Tokenizer, MT5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from createDataset import create_or_load_dataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------- HELPER FUNCTIONS -----------
# -------------------------------------->
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items() if key in ['input_ids', 'attention_mask']}
        # Ensure labels are correctly indexed
        if isinstance(self.labels, BatchEncoding):
            item['labels'] = self.labels['input_ids'][idx]  # Adjust according to how labels are stored
        else:
            item['labels'] = self.labels[idx].clone().detach()

        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings['input_ids'])

def get_model():
    model_name = "google/mt5-base"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # --------- FREEZING LAYERS ------------
    # -------------------------------------->

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.5."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.6."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("encoder.block.7."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.0."):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("decoder.block.1."):
            param.requires_grad = False

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

    # <--------------------------------------

    # ----------SEEING THE MODEL------------
    # -------------------------------------->
    print()
    print('--- Printing the layers of the model ---')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('--- Done printing the layers of the model ---')
    print()

    # <--------------------------------------
    return model, tokenizer

def get_dataset_internal(ds, tokenizer, num_samples, max_length = 128, X_NAME='errors', Y_NAME='original'):
    ds.set_format('pytorch')
    inputs = ds[X_NAME]
    labels = ds[Y_NAME]
    if num_samples is not None and num_samples > 0:
      inputs = inputs[:num_samples]
      labels = labels[:num_samples]
    input_tokenized = tokenizer(inputs, truncation=True, padding=True, max_length=max_length,
                                      return_tensors='pt')
    labels_tokenized = tokenizer(labels, truncation=True, padding=True, max_length=max_length,
                                       return_tensors='pt')
    return TextDataset(input_tokenized, labels_tokenized)

def get_dataset(data_dir, tokenizer, augemntation_percentage=30, test_size=0.2, num_train=None, num_test=None, max_length = 128, X_NAME='errors', Y_NAME='original'):
    # -------------- DATASET ---------------
    # -------------------------------------->
    dataset_train, dataset_test = create_or_load_dataset(data_dir, augemntation_percentage=augemntation_percentage, test_size=test_size, verbose=False)
    return (get_dataset_internal(dataset_train, tokenizer, num_train, max_length, X_NAME, Y_NAME),
            get_dataset_internal(dataset_test, tokenizer, num_test, max_length, X_NAME, Y_NAME))

def get_model_and_data(data_dir, num_train=None, num_test=None):
    model, tokenizer = get_model()
    text_tensor_train_ds, text_tensor_test_ds = get_dataset(data_dir, tokenizer, num_train=num_train, num_test=num_test)
    return model, tokenizer, text_tensor_train_ds, text_tensor_test_ds