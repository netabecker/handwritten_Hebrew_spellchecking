# OSCAR: Hebrew Handwritten Spell Checker

This project is a pipeline for correcting spelling mistakes in handwritten Hebrew text. The solution involves three main components:
1. **Word Segmentation**: Splits an image containing multiple words into individual word images.
2. **HTR**: Extracts text from the segmented word images using an existing model.
3. **Spelling Correction**: Fixes spelling mistakes in the extracted text using an MT5-based language model fine-tuned on Hebrew data.

## Table of Contents
- [Overview](#overview)
- [Components](#components)
  - [1. Word Segmentation](#1-word-segmentation)
  - [2. Hebrew Handwritten Text Recognition](#2-hebrew-handwritten-text-recognition)
  - [3. Spelling Correction](#3-spelling-correction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview
The model takes an image of handwritten Hebrew text, splits it into individual word images, recognizes the text using HTR, and then corrects any spelling mistakes using a language model trained on Hebrew song lyrics.

## Components

### 1. Word Segmentation
This module handles segmenting an image into smaller images, each containing a single word. We implemented this component to support handwritten text.
![image](https://github.com/user-attachments/assets/33c9acac-aac3-44f0-8b66-9d834b222428)
The input image was taken from: https://github.com/Lotemn102/HebHTR

### 2. Hebrew Handwritten Text Recognition
For text extraction, we integrated an existing HTR model from here: https://github.com/Lotemn102/HebHTR

It is based on the Harald Scheidl's CTC-WordBeam, implemented here: [https://github.com/githubharald/CTCWordBeamSearch.git](https://github.com/githubharald/CTCWordBeamSearch)

### 3. Spelling Correction
The final component is a language model based on MT5. We fine-tuned the model to correct spelling errors by training it on a custom dataset. 

We've trained the model on Guy Barash's Hebrew songs lyrics dataset (https://www.kaggle.com/datasets/guybarash/hebrew-songs-lyrics), which consists of lyrics of ~15,000 Hebrew songs. We randomly trimmed each entry and induced random spelling mistakes.

<img src="https://github.com/user-attachments/assets/1c0220db-8700-424d-90e7-a7a8191fa6da" alt="image" width="50%" height="50%"/>


## Installation
1. **Python and TensorFlow Requirements:**
   - Python version: `3.6` or `3.7`
   - TensorFlow version: `1.15`

2. **Clone the repository:**
   ```bash
   git clone https://github.com/netabecker/handwritten_Hebrew_spellchecking.git
   ```

3. **Clone Harald Scheidl's CTC-WordBeam repository:**
   ```bash
   git clone https://github.com/githubharald/CTCWordBeamSearch.git
   ```

4. **Create a virtual enviorment for the project and install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **In the virtual enviorment, install your local clone of CTC-WordBeam:**
   ```bash
   pip install -e path/to/CTC-WordBeam
   ```

7. **Run TrainModel.py, or download a trained model and place it in the main directory.**


## Usage
To run the model, run `main.py`. If there's nothing in the default locations, the script will prompt you for input image and model checkpoint.


## Results
The pipeline outputs corrected text with significantly improved accuracy over raw OCR outputs, especially in cases where spelling errors are common in handwritten documents.

![image](https://github.com/user-attachments/assets/62a35daa-010f-48f9-b756-65091f8d28ef)



## References
1. [Harald Scheid's SimpleHTR model](https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5)
2. [CTC-WordBeamSearch](https://towardsdatascience.com/word-beam-search-a-ctc-decoding-algorithm-b051d28f3d2e)
3. [H"ebrew songs lyrics" on Kaggle](https://www.kaggle.com/datasets/guybarash/hebrew-songs-lyrics)
4. [Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel. "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer". 2021](https://aclanthology.org/2021.naacl-main.41/)
