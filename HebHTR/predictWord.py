import cv2
import numpy as np
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from word_beam_search import WordBeamSearch

'''
Handwritten text recognition model written by Harald Scheidl:
https://github.com/githubharald/SimpleHTR
'''

class DecoderType:
    BestPath = 0
    WordBeamSearch = 1

class Model:
    # model constants
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath,
                 mustRestore=False, dump=False):
        self.dump = dump
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore
        self.snapID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.inputImgs = tf.placeholder(tf.float32, shape=(
        None, Model.imgSize[0], Model.imgSize[1]))

        # setup CNN, RNN and CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        # setup optimizer to train NN
        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
                self.learningRate).minimize(self.loss)

        # initialize TF
        (self.sess, self.saver) = self.setupTF()

    def setupCNN(self):
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(tf.compat.v1.truncated_normal(
                [kernelVals[i], kernelVals[i], featureVals[i],
                 featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',
                                strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv,
                                                      training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1),
                                  'VALID')

        self.cnnOut4d = pool

    def setupRNN(self):
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = 256
        cells = [
            tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True)
            for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked,
                                                        cell_bw=stacked,
                                                        inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1],
                                stddev=0.1))
        self.rnnOut3d = tf.squeeze(
            tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1,
                                padding='SAME'), axis=[2])

    def setupCTC(self):
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(
            tf.placeholder(tf.int64, shape=[None, 2]),
            tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            tf.compat.v1.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC,
                                     sequence_length=self.seqLen,
                                     ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.savedCtcInput = tf.placeholder(tf.float32,
                                            shape=[Model.maxTextLen, None,
                                                   len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts,
                                             inputs=self.savedCtcInput,
                                             sequence_length=self.seqLen,
                                             ctc_merge_repeated=True)

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC,
                                                    sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            #word_beam_search_module = tf.load_op_library('./TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)

            with open('HebHTR/model/wordCharList.txt', "rb") as f:
                byte = f.read(1)
                if byte != "":
                    byte = f.read()
                    myString = byte.decode("Windows-1255")
                    wordChars = myString.splitlines()[0]


            corpus = open('HebHTR/data/corpus.txt',encoding='utf-8').read()

            # decode using the "Words" mode of word beam search
            self.decoder = WordBeamSearch(
                50, 'Words', 0.0,
                corpus.encode('utf8'), chars.encode('utf8'),
                wordChars.encode('utf8'))
            self.wbs_input = tf.nn.softmax(self.ctcIn3dTBC, axis=2)

    def setupTF(self):
        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        modelDir = 'HebHTR/model/'
        latestSnapshot = tf.train.latest_checkpoint(
            modelDir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        # load saved model if available
        if latestSnapshot:
            saver.restore(sess, latestSnapshot)
        else:
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput, batchSize):
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(batchSize)]

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(batchSize):
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(batchSize)}
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in
                encodedLabelStrs]

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        # decode, optionally save RNN output
        numBatchElements = len(batch.imgs)
        evalRnnOutput = self.dump or calcProbability
        evalList = ([self.wbs_input if self.decoderType == DecoderType.WordBeamSearch else self.decoder]
                    + ([self.ctcIn3dTBC] if evalRnnOutput else []))
        feedDict = {self.inputImgs: batch.imgs,
                    self.seqLen: [Model.maxTextLen] * numBatchElements,
                    self.is_train: False}
        evalRes = self.sess.run(evalList, feedDict)
        decoded = self.decoder.compute(evalRes[0]) if self.decoderType == DecoderType.WordBeamSearch else evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(
                batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput: ctcInput, self.gtTexts: sparse,
                        self.seqLen: [Model.maxTextLen] * numBatchElements,
                        self.is_train: False}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)

        return (texts, probs)

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


def getModel(decoder_type):
    if decoder_type == 'word_beam':
        decoderType = DecoderType.WordBeamSearch
    else:
        decoderType = DecoderType.BestPath

    model = Model(open(FilePaths.fnCharList,encoding='utf-8').read(), decoderType,
                  mustRestore=True)
    return model


def predictWord(image, model):
    img = preprocessImageForPrediction(image, Model.imgSize)
    batch = Batch(None, [img])
    recognized = model.inferBatch(batch, True)[0]
    return recognized

