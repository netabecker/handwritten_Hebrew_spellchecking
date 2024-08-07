import os.path

import pandas as pd
import random
from datasets import Dataset, DatasetDict
import torch


def random_replace(string, default_prob):
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


def create_augmentations(percentage=30, verbose=False):
    default_prob = float(percentage) / 100
    input_txt_path = 'datasets/hebrew_text.txt'
    output_path = 'datasets/hebrew_text_aug_' + str(percentage)

    # Read the input TXT file
    with open(input_txt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Process each line
    processed_lines = []
    for line in lines:
        line = line.strip()
        modified_line = random_replace(line, default_prob)
        processed_lines.append(f"{line}\t{modified_line}")

    if verbose:
        print(f'-----------> Example:\n\n')
        print(processed_lines[1])
        print(f'<-----------= Example:\n\n')

    # Save data in txt format - uncomment to activate
    # # Write the original and modified text to the output TXT file
    # output_txt_path = output_path + '.txt'
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


def export_train_test_dataset(excel_path, test_size=0.2):
    if (not os.path.exists('datasets/train.pt')) and (not os.path.exists('datasets/test.pt')):
        dataset = export_dataset(excel_path)
        # Split the dataset into training and testing sets
        train_test_split = dataset.train_test_split(test_size=test_size)
        torch.save(train_test_split['train'], 'datasets/train.pt')
        torch.save(train_test_split['test'], 'datasets/test.pt')

        return train_test_split['train'], train_test_split['test']
    else:
        train_split = torch.load('datasets/train.pt')
        test_split = torch.load('datasets/test.pt')
        return train_split, test_split



def full_run(percentage=30, verbose=False):
    return export_dataset(create_augmentations(percentage, verbose))


def full_run_train_test_split(percentage=30, verbose=True):
    return export_train_test_dataset(create_augmentations(percentage, verbose))

