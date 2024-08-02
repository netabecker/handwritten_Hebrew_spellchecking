import pandas as pd
import random

# Path to your input TXT file
input_txt_path = 'datasets/hebrew_text.txt'
# Path to your output TXT file
output_txt_path = 'datasets/hebrew_text_aug.txt'


# Function to randomly replace letters
def random_replace(string):
    replacements = {
            'א': ['ע', 'ה'],
            'ע': ['א', 'ה'],
            'ה': ['א', 'ע'],

            'ט': ['ת'],
            'ת': ['ט'],

            # 'ו': ['ב'],
            # 'ב': ['ו'],

            'כ': ['ק'],
            'ק': ['כ'],

            'ש': ['ס'],
            'ס': ['ש']

            # 'לא ': ['לו '],
            # 'לו ': ['לא ']
        }

    # Convert string to list to make replacements
    string_list = list(string)
    for idx, char in enumerate(string_list):
        if char in replacements:
            if random.random() < 0.4:  # chance that a letter will be replaced
                string_list[idx] = random.choice(replacements[char])
    # Convert list back to string
    return ''.join(string_list)


# Read the input TXT file
with open(input_txt_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# Process each line
processed_lines = []
for line in lines:
    line = line.strip()  # Remove any leading/trailing whitespace
    modified_line = random_replace(line)
    processed_lines.append(f"{line}\t{modified_line}")

# Write the original and modified text to the output TXT file
with open(output_txt_path, 'w', encoding='utf-8') as outfile:
    outfile.write('\n'.join(processed_lines))

print(f"Modified data saved to {output_txt_path}")
