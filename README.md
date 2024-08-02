# **handwritten Hebrew spellchecking Model**


### Breakdown to tasks:

**1. OCR Model:** 
- [x] Clone the existing OCR model and make it usable.

**2. Image Segmentation Model:**
- [ ] Create or find existing model to break up images of sentences.

**3. Create a dataset of misspelled sentences:**
- [ ] Create a dataset of misspelled sentences and their correct versions.

**4. Hebrew word2vec Model:**
- [ ] Find existing and make usable.

**5. Error Detection and Correction Model**
- [ ] Develop a seq2seq model to identify and correct potential spelling errors.
      
**6. Training the Spellchecking Model**
- [ ] Training Data: Create pairs of erroneous text and the corresponding correct text.
- [ ] Model Architecture: Look at models like BERT or GPT.

**7. Integration and Testing**
- [ ] Integration: Integrate your spellchecker with the OCR and segmentation models to form a complete pipeline from handwritten text image to corrected digital text.
- [ ] Testing: Test the entire pipeline on new handwritten Hebrew text samples.


**Tools and Libraries** \
Deep Learning Frameworks: TensorFlow, PyTorch \
NLP Libraries: Hugging Face Transformers, NLTK, SpaCy \
Data Augmentation: Tools for generating synthetic handwritten text images (if needed) 


### Resources
**1. Handritten text recognition model:**\
https://github.com/Lotemn102/HebHTR
https://github.com/githubharald/SimpleHTR

**2. Hebrew word2vec model:**\
https://github.com/liorshk/wordembedding-hebrew

**3. Guides about translation, seq2seq and transformers**\
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://github.com/bentrevett/pytorch-seq2seq

### Example Workflow

```
ocr_output = ocr_model.predict(handwritten_text_image)
Spellchecking Model:

corrected_text = spellchecker_model.correct(ocr_output)
End-to-End Pipeline:

def process_image(image):
    ocr_output = ocr_model.predict(image)
    corrected_text = spellchecker_model.correct(ocr_output)
    return corrected_text
```





