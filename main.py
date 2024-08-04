import sys
sys.path.append('./HebHTR')
import HebHTR

# Create new HebHTR object.
img = HebHTR.HebHTR('example2.png')
text = img.imgToWord()
print(text)


# We have the output of the OCR - now we want to use the spellchecker model
# ------------------------------->



# ------------------------------->
