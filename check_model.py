import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformer_prepare_data import *

model, tokenizer = get_model_and_data()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Sample data for inference
new_data = ["זה טקסט לדוגמא עם שגיעה", "עוד תקסט לטיקון"]
new_data_tokenized = tokenizer(
    new_data, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
)

# Move data to device
input_ids = new_data_tokenized['input_ids'].to(device)
attention_mask = new_data_tokenized['attention_mask'].to(device)
print("Tokenized Input IDs:", input_ids)
print("Attention Mask:", attention_mask)

model.load_state_dict(torch.load('spellcheck_model/saved_model'))
model.eval()

with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

print("Generated IDs:", generated_ids)

predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {new_data[i]}")
    print(f"Output: {pred}")
    print()

