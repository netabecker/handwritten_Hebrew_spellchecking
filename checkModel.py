import os
from prepareModel import *

# Load Checkpoint
checkpoint_path = 'saved_checkpoint.pth'
if not os.path.exists(checkpoint_path):
    checkpoint_path = input("Checkpoint Path: ")
model, tokenizer = get_model()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Sample data
new_data = ["זה תקסט לדוגמא עם שגיעה", "עוד תקסט לטיקון", "בדיכה"]
new_data_tokenized = tokenizer(
    new_data, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
)

# Move data to device
input_ids = new_data_tokenized['input_ids'].to(device)
attention_mask = new_data_tokenized['attention_mask'].to(device)

# Forward sample data through model
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

# Print Predictions
print("Predictions:")
for i, pred in enumerate(predictions):
    print(f"Input: {new_data[i]}")
    print(f"Output: {pred}")
    print()

# Forward user's input through the model
while True:
    input_text = input('Input: ')
    input_text_tokenized = tokenizer(
        [input_text], max_length=128, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = input_text_tokenized['input_ids'].to(device)
    attention_mask = input_text_tokenized['attention_mask'].to(device)
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
    print(f"Output: {predictions[0]}")
    print()