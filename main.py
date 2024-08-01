import sys
sys.path.append('./HebHTR')
import HebHTR

# Create new HebHTR object.
img = HebHTR.HebHTR('example.png')
text = img.imgToWord(iterations=5, decoder_type='word_beam')
a=1


# We have the output of the OCR - now we want to use the spellchecker model
# ------------------------------->



# ------------------------------->
