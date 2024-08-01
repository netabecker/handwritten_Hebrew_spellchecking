from Model import Model, DecoderType
from processFunctions import preprocessImageForPrediction
import numpy as np

class Batch:
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts

class FilePaths:
    fnCharList = 'HebHTR/model/charList.txt'
    fnCorpus = 'HebHTR/data/corpus.txt'

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

