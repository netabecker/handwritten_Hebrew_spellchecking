import predictWord
import cv2

class HebHTR:

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.predictWordModel = predictWord.getModel(decoder_type='word_beam')

    def imgToWord(self):
        transcribed_words = []
        transcribed_words.extend(predictWord.predictWord(self.img, self.predictWordModel))
        return transcribed_words
