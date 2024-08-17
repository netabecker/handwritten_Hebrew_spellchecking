from predictWord import predictWord, getModel
import cv2
from segmentWordsInSentence import segmentWordsInSentence
class HebHTR:

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.predictWordModel = getModel(decoder_type='word_beam')

    def imgToWord(self):
        word_imgs = segmentWordsInSentence(self.img)
        transcribed_words = [predictWord(img, self.predictWordModel)[0] for img in word_imgs]
        return transcribed_words
