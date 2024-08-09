import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformer_prepare_data import *

model, tokenizer = get_model_and_data(verbose=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Sample data for inference
new_data = ["זה תקסט לדוגמא עם שגיעה", "עוד תקסט לטיקון", "בדיכה"]
new_data_tokenized = tokenizer(
    new_data, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
)

# Move data to device
input_ids = new_data_tokenized['input_ids'].to(device)
attention_mask = new_data_tokenized['attention_mask'].to(device)

# Get most recent model:
directory_path = 'spellcheck_model/'
most_recent_file = None
most_recent_time = 0
# iterate over the files in the directory using os.scandir
for entry in os.scandir(directory_path):
    if entry.is_file():
        # get the modification time of the file using entry.stat().st_mtime_ns
        mod_time = entry.stat().st_mtime_ns
        if mod_time > most_recent_time:
            # update the most recent file and its modification time
            most_recent_file = entry.name
            most_recent_time = mod_time

model_path = str(directory_path) + str(most_recent_file)
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {new_data[i]}")
    print(f"Output: {pred}")
    print()

