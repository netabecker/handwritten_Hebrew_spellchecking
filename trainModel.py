import os
import random

import evaluate
import torch
from torch.optim.lr_scheduler import ChainedScheduler, ExponentialLR
from torch.utils.data import DataLoader
from transformers import Adafactor, get_cosine_schedule_with_warmup
from prepareModel import get_model_and_data
from tqdm.auto import tqdm

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_running_in_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    data_dir = "drive/MyDrive/HHSC/"
else:
    data_dir = 'datasets/'

# ---------- HYPERPARAMETERS -----------
# -------------------------------------->
BATCH_SIZE = 16
num_epochs = 15
num_train = None
num_test = 1000
lr = 1e-3
accumulation_steps = 4
# <--------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for consistency
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the saved datasets
model, tokenizer, text_tensor_train_ds, text_tensor_test_ds = get_model_and_data(data_dir, num_train=num_train, num_test=num_test)
model.to(device)

# Create DataLoaders
train_dataloader = DataLoader(text_tensor_train_ds, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(text_tensor_test_ds, batch_size=BATCH_SIZE)

num_training_steps = num_epochs * len(train_dataloader)
optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=lr)
cosine_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps
)
exp_scheduler = ExponentialLR(optimizer, gamma=0.9)
lr_scheduler = ChainedScheduler([cosine_scheduler, exp_scheduler])
scaler = torch.cuda.amp.GradScaler()

progress_bar = tqdm(range(num_training_steps))

def calc_accuracy(model, tokenizer, test_dataloader):
    metric = evaluate.load("accuracy")
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch in test_dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate predictions
            generated_tokens = model.generate(batch['input_ids'], max_length=batch['labels'].shape[1])

            # Calculate accuracy by comparing generated texts to target texts
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.view(-1)
            references = batch["labels"].view(-1)
            mask = references != tokenizer.pad_token_id
            predictions = predictions[mask]
            references = references[mask]

            metric.add_batch(predictions=predictions, references=references)

    avg_test_loss = total_loss / len(test_dataloader)
    accuracy_dict = metric.compute()
    accuracy = float(accuracy_dict['accuracy'])
    random_index = random.randint(0, len(batch['labels']) - 1)
    random_target = tokenizer.decode(batch['labels'][random_index], skip_special_tokens=True)
    random_prediction = tokenizer.decode(generated_tokens[random_index], skip_special_tokens=True)
    random_current = tokenizer.decode(batch['input_ids'][random_index], skip_special_tokens=True)
    print(f'Input: {random_current}\nTarget: {random_target}\nPrediction: {random_prediction}')

    model.train()  # Set the model back to training mode
    return avg_test_loss, accuracy


if os.path.exists('/content/drive/My Drive/saved_checkpoint.pth'):
  checkpoint = torch.load('/content/drive/My Drive/saved_checkpoint.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scaler.load_state_dict(checkpoint['scaler_state_dict'])
  lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  start_epoch = checkpoint['epoch'] + 1
  best_accuracy = checkpoint['best_accuracy']
  print('Model loaded from Google Drive')
else:
  start_epoch = 0
  best_accuracy = 0

model.train()
train_losses = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in range(start_epoch):
  for batch in train_dataloader:
        progress_bar.update(1)
for epoch in range(start_epoch, num_epochs):
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update weights and reset gradients every 'accumulation_steps'
        if (batch_idx + 1) % accumulation_steps == 0:
          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()

        progress_bar.update(1)

        epoch_loss += loss.item()  # Accumulate the loss

    # Calculate average loss for the epoch
    avg_train_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    print(f"\nEpoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    # -------------------------------------------
    model.eval()
    avg_test_loss, accuracy = calc_accuracy(model, tokenizer, test_dataloader)
    test_losses.append(avg_test_loss)
    test_accuracies.append(accuracy)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}  learning reate: {optimizer.param_groups[0]['lr']}")
    # Set the model back to training mode
    model.train()
    # -------------------------------------------

    # Saving checkout
    if accuracy > best_accuracy:
      best_accuracy = accuracy
      #torch.save(model.state_dict(), 'saved_model_checkout.pth')
      # Save the model to your Google Drive
      torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # if using mixed precision
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'best_accuracy': best_accuracy,
      }, '/content/drive/My Drive/saved_checkpoint.pth')

torch.save(model.state_dict(), 'saved_model.pth')