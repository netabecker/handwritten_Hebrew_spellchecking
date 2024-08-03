import pandas as pd
import random

verbose = True
percentage = 30
default_prob = float(percentage) / 100
# Path to your input TXT file
input_txt_path = 'datasets/hebrew_text.txt'
# Path to your output TXT file
output_path = 'datasets/hebrew_text_aug_' + str(percentage)
output_txt_path = output_path + '.txt'

# Function to randomly replace letters
def random_replace(string):
    replacements = {
            'א': [('ע', default_prob), ('ה', default_prob)],
            'ע': [('א', default_prob), ('ה', default_prob)],
            'ה': [('א', default_prob), ('ע', default_prob)],

            'ט': [('ת', default_prob)],
            'ת': [('ט', default_prob)],

            'ח': [('כ', default_prob)],
            'כ': [('ח', default_prob), ('ק', default_prob)],
            'ק': [('כ', default_prob)],

            'ש': [('ס', default_prob / 2)],
            'ס': [('ש', default_prob / 2)],

            'ב': [('ו', default_prob / 4)],
            'ו': [('ב', default_prob / 4)],

            'לא ': ('לו ', default_prob),
            'לו ': [('לא ', default_prob)]
        }

    # Convert string to list to make replacements
    string_list = list(string)
    for idx, char in enumerate(string_list):
        if char in replacements:
            for replacement, prob in replacements[char]:
                if random.random() < prob:  # Unique probability for each replacement
                    string_list[idx] = replacement
                    break  # Stop after the first replacement
    return ''.join(string_list)


# Read the input TXT file
with open(input_txt_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# Process each line
processed_lines = []
for line in lines:
    line = line.strip()
    modified_line = random_replace(line)
    processed_lines.append(f"{line}\t{modified_line}")

if verbose:
    print(f'-----------> Example:\n\n')
    print(processed_lines[1])
    print(f'<-----------= Example:\n\n')

# Save data in txt format - uncomment to activate
# # Write the original and modified text to the output TXT file
# with open(output_txt_path, 'w', encoding='utf-8') as outfile:
#     outfile.write('\n'.join(processed_lines))
#
# print(f"Modified data saved to {output_txt_path}")

print(f'\nExporting the data to Excel file')

processed_lines = processed_lines[1:]
data = [line.strip().split('\t') for line in processed_lines]
df = pd.DataFrame(data, columns=['original', 'errors'])  # Adjust column names as needed
excel_output_path = output_path + '.xlsx'
df.to_excel(excel_output_path, index=False, engine='openpyxl')

print(f"Conversion complete. Check {excel_output_path}")
