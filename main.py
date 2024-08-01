from HebHTR import *

# Create new HebHTR object.
img = HebHTR('example.png')
text = img.imgToWord(iterations=5, decoder_type='word_beam')


# We have the output of the OCR - now we want to use the spellchecker model
# ------------------------------->



# ------------------------------->
