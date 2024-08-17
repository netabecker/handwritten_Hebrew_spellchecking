import os.path
import pandas as pd
import random
from datasets import Dataset
import torch
import numpy as np

def random_replace(string, default_prob):
    drop_prob = default_prob / 6
    space_prob = default_prob / 6
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

        'ג': [('ז', default_prob / 4)],
        'ז': [('ג', default_prob / 4)]
    }

    # Convert string to list to make replacements
    string_list = list(string)
    previous_was_changed = False  # we avoid dropping a few letters in a row
    for idx, char in enumerate(string_list):
        if char in replacements:
            for replacement, prob in replacements[char]:
                if random.random() < prob:  # Unique probability for each replacement
                    string_list[idx] = replacement
                    previous_was_changed = True
                    break  # Stop after the first replacement
        if previous_was_changed is False and (random.random() < drop_prob):  # Randomly decide to drop a letter
            string_list[idx] = ''

    new_string_list = []
    previous_was_dropped = False  # we avoid dropping a few letters in a row
    for char in string_list:
        new_string_list.append(char)
        if (previous_was_dropped is False) and (random.random() < space_prob):  # Randomly decide to add a space
            new_string_list.append(' ')
            previous_was_dropped = True
            continue
        previous_was_dropped = False

    return ''.join(new_string_list)


def create_augmentations(data_dir, percentage=20, verbose=False, truncate_min=1, truncate_max=13):
    default_prob = float(percentage) / 100

    input_txt_path = data_dir + 'hebrew_text.txt'
    output_path = data_dir + 'hebrew_text_aug_' + str(percentage)

    # Read the input TXT file
    with open(input_txt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Process each line
    processed_lines = [lines[0]]
    for long_line in lines[1:]:
        long_line = long_line.strip()
        if truncate_min is None or truncate_max is None:
            current_lines = [long_line]
        else:
            words = long_line.split()
            current_lines = []
            i = 0
            while i < len(words):
                num_words = np.random.randint(truncate_min, truncate_max)
                current_lines.append(' '.join(words[i:i+num_words]))
                i += num_words
        for line in current_lines:
            modified_line = random_replace(line, default_prob)
            if line == '' or len(line) < 2:
                continue
            processed_lines.append(f"{line}\t{modified_line}")

    if verbose:
        print(f'-----------> Example:\n\n')
        print(processed_lines[1])
        print(f'<-----------= Example:\n\n')

    print(f'Exporting the data to Excel file...')
    print(f'{len(processed_lines)-1} lines.')

    processed_lines = processed_lines[1:]
    data = [line.strip().split('\t') for line in processed_lines]
    df = pd.DataFrame(data, columns=['original', 'errors'])  # Adjust column names as needed
    excel_output_path = output_path + '.xlsx'
    df.to_excel(excel_output_path, index=False, engine='openpyxl')

    print(f"Conversion complete. Check {excel_output_path}")
    return excel_output_path


def export_dataset(excel_path):
    df = pd.read_excel(excel_path)
    df.dropna(subset=['errors', 'original'], inplace=True)
    texts_with_errors = df['errors'].tolist()
    texts_corrected = df['original'].tolist()

    data_dict = {
        'errors': texts_with_errors,
        'original': texts_corrected
    }

    # dataset = ds.Dataset.from_dict(data_dict)
    dataset = Dataset.from_dict(data_dict)

    return dataset

def create_or_load_dataset(data_dir, augemntation_percentage=30, test_size=0.2, verbose=False, force_create=False):
    train_path = data_dir + 'train.pt'
    test_path = data_dir + 'test.pt'
    if force_create or (not os.path.exists(train_path)) or (not os.path.exists(test_path)):
        print('Creating & Saving Dataset...')
        excel_path = create_augmentations(data_dir, augemntation_percentage, verbose)
        dataset = export_dataset(excel_path)
        # Split the dataset into training and testing sets
        train_test_split = dataset.train_test_split(test_size=test_size)
        torch.save(train_test_split['train'], train_path)
        torch.save(train_test_split['test'], test_path)
        return train_test_split['train'], train_test_split['test']
    else:
        print('Loading Dataset from file...')
        train_split = torch.load(train_path)
        test_split = torch.load(test_path)
        return train_split, test_split
