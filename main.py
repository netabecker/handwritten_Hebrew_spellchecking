import os
import sys
import torch
import re

import transformer_prepare_data

sys.path.append('./HebHTR')
import HebHTR

# Get Text from the image
img = HebHTR.HebHTR('example2.png')
text = ' '.join(img.imgToWord())
print(text)

# Remove punctuation
text = re.sub(r'[^\u0590-\u05FF\s]', '', text)

# Load model
checkpoint_path = 'saved_checkpoint.pth'
if not os.path.exists(checkpoint_path):
    checkpoint_path = input("Checkpoint Path: ")

model, tokenizer = transformer_prepare_data.get_model()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Pass through the model
tokenized = tokenizer(
    [text], max_length=128, padding='max_length', truncation=True, return_tensors='pt'
)
input_ids = tokenized['input_ids'].to(device)
attention_mask = tokenized['attention_mask'].to(device)

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
print(predictions)