from Model import Model, DecoderType
import numpy as np
import cv2

class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class FilePaths:
    fnCharList = 'HebHTR/model/charList.txt'
    fnCorpus = 'HebHTR/data/corpus.txt'


# Resize image to fit model's input size, and place it on model's size empty image.
def preprocessImageForPrediction(img, imgSize):
    # convert image to grayscale (Chen & Netta)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # binarize image (Chen & Netta)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)),
                                                1))  # scale according to f (result at least 1 and at most wt or ht)
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # transpose for TF
    img = cv2.transpose(target)

    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return img


def infer(model, image):
    img = preprocessImageForPrediction(image, Model.imgSize)
    batch = Batch(None, [img])
    recognized = model.inferBatch(batch, True)[0]
    return recognized

def getModel(decoder_type):
    if decoder_type == 'word_beam':
        decoderType = DecoderType.WordBeamSearch
    else:
        decoderType = DecoderType.BestPath

    model = Model(open(FilePaths.fnCharList,encoding='utf-8').read(), decoderType,
                  mustRestore=True)
    return model


def predictWord(image, model):
    return infer(model, image)

